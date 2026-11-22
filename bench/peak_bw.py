"""Measure the achievable peak DRAM *read* bandwidth of this GPU.

Why: batch-1 decode is memory bound -- every weight byte is read once per
token and used for ~1 multiply-add -- so the right yardstick for a decode
kernel is "what fraction of the bandwidth the chip can actually deliver".
The marketing number for an M1 Pro is ~200 GB/s; a real read kernel gets
less. We measure it instead of trusting it.

How:
  * One big buffer (default 1 GiB), far larger than the ~24 MB system level
    cache (SLC), so every read really comes from DRAM.
  * A tiny custom Metal kernel: every thread walks the buffer with a
    grid-stride loop (thread t reads elements t, t+T, t+2T, ... where T is
    the total thread count), so neighbouring threads read neighbouring
    addresses (coalesced). Each thread XOR-folds what it read into one
    value and writes it out -- that store is what keeps the compiler from
    deleting the loads as dead code. The output is T*4 bytes, negligible.
  * Sweep the vector width per load (uint / uint2 / uint4 = 4/8/16 bytes)
    and the total thread count; report the best configuration's median.
  * For comparison also time MLX's own `mx.sum` (a read-only reduction) and a
    copy `a + 0` (read + write: counts 2x the buffer size).

Each measurement: a few warmup launches, then `trials` timed rounds; in one
round `launches` back-to-back launches are enqueued and evaluated with a
single `mx.eval`, so the CPU<->GPU sync round-trip (~0.1 ms) is amortised
over ~20 ms of GPU work. We report the median round.

Output: bench/results/peak_bw.json  {"peak_read_gbps": ..., ...details}
bench.py reads `peak_read_gbps` for its "% of peak" column.

Usage:  .venv/bin/python bench/peak_bw.py [--mb 1024] [--trials 15]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import quiet_trial  # noqa: E402

import mlx.core as mx
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# VTYPE: the Metal vector type loaded per iteration; NVEC: buffer length in
# units of VTYPE (a template constant); the buffer is passed as plain uint32.
_READ_SRC = """
    uint tid      = thread_position_in_grid.x;
    uint nthreads = threads_per_grid.x;
    const device VTYPE* p = reinterpret_cast<const device VTYPE*>(a);

    VTYPE acc = VTYPE(0);
    for (uint i = tid; i < NVEC; i += nthreads) {
        acc ^= p[i];
    }
    out[tid] = FOLD_EXPR; // one uint per thread: keeps the loads alive
"""

# How to fold each vector type down to one uint.
_VEC_TYPES = {
    4: ("uint", "acc"),
    8: ("uint2", "acc.x ^ acc.y"),
    16: ("uint4", "acc.x ^ acc.y ^ acc.z ^ acc.w"),
}

_kernels = {}


def _read_kernel(vec_bytes: int):
    """One mx.fast.metal_kernel per vector width (the source differs)."""
    if vec_bytes not in _kernels:
        vec, fold = _VEC_TYPES[vec_bytes]
        src = _READ_SRC.replace("VTYPE", vec).replace("FOLD_EXPR", fold)
        _kernels[vec_bytes] = mx.fast.metal_kernel(
            name=f"peak_read_{vec}", input_names=["a"], output_names=["out"], source=src)
    return _kernels[vec_bytes]


def time_rounds(launch, launches: int, trials: int, warmup: int = 2) -> list:
    """Seconds per launch for each timed round (`launches` launches per eval)."""
    for _ in range(warmup):
        mx.eval([launch() for _ in range(launches)])
    def one_round():
        outs = [launch() for _ in range(launches)]   # build the lazy graph first
        t0 = time.perf_counter()
        mx.eval(outs)                                 # ... then time only the GPU run
        return (time.perf_counter() - t0) / launches

    # quiet_trial: redo any round that overlapped another GPU job (bench.py)
    return [quiet_trial(one_round) for _ in range(trials)]


def summarize(nbytes: int, secs: list) -> dict:
    s = np.array(secs)
    med = float(np.median(s))
    return {
        "median_ms": med * 1e3,
        "min_ms": float(s.min()) * 1e3,
        "p10_ms": float(np.percentile(s, 10)) * 1e3,
        "p90_ms": float(np.percentile(s, 90)) * 1e3,
        "gbps_median": nbytes / med / 1e9,
        "gbps_best": nbytes / float(s.min()) / 1e9,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=1024, help="buffer size in MiB (default 1024)")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--launches", type=int, default=4, help="launches per timed eval")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "peak_bw.json"))
    args = ap.parse_args()

    n_words = args.mb * (1 << 20) // 4
    nbytes = n_words * 4
    # arange, not zeros: make sure the pages hold real, distinct data.
    a = mx.arange(n_words, dtype=mx.uint32)
    mx.eval(a)
    print(f"buffer: {nbytes / 2**20:.0f} MiB, {args.trials} trials x {args.launches} launches")

    # ---- custom read kernel sweep ------------------------------------------
    sweep = []
    tg = 256
    for vec_bytes in (4, 8, 16):
        kern = _read_kernel(vec_bytes)
        nvec = nbytes // vec_bytes
        for threads in (1 << 14, 1 << 16, 1 << 18, 1 << 20):
            def launch(kern=kern, nvec=nvec, threads=threads):
                return kern(inputs=[a], template=[("NVEC", nvec)],
                            grid=(threads, 1, 1), threadgroup=(tg, 1, 1),
                            output_shapes=[(threads,)], output_dtypes=[mx.uint32])[0]
            r = summarize(nbytes, time_rounds(launch, args.launches, args.trials))
            r.update(vec_bytes=vec_bytes, threads=threads, threadgroup=tg)
            sweep.append(r)
            print(f"  read kernel  vec={vec_bytes:2d}B threads={threads:8d}: "
                  f"{r['gbps_median']:6.1f} GB/s median  ({r['gbps_best']:6.1f} best)")

    best = max(sweep, key=lambda r: r["gbps_median"])

    # ---- MLX built-ins for comparison --------------------------------------
    af = a.view(mx.float32)   # same bytes, read as float32 for a float sum
    r_sum = summarize(nbytes, time_rounds(lambda: mx.sum(af), args.launches, args.trials))
    print(f"  mx.sum (float32)      : {r_sum['gbps_median']:6.1f} GB/s median")
    # A copy reads N bytes and writes N bytes: count both.
    r_copy = summarize(2 * nbytes, time_rounds(lambda: a + 0, args.launches, args.trials))
    print(f"  copy a+0 (read+write) : {r_copy['gbps_median']:6.1f} GB/s median (counting 2x bytes)")

    info = mx.device_info()
    result = {
        "peak_read_gbps": best["gbps_median"],
        "note": "median over trials of the best custom read-kernel config; "
                "use this as the 100% line for decode kernels",
        "best_config": best,
        "sweep": sweep,
        "mx_sum": r_sum,
        "copy_read_plus_write": r_copy,
        "buffer_bytes": nbytes,
        "trials": args.trials,
        "launches_per_eval": args.launches,
        "chip": info.get("device_name"),
        "mlx_version": mx.__version__,
        "macos": platform.mac_ver()[0],
        "loadavg_1m": os.getloadavg()[0],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\npeak read bandwidth: {best['gbps_median']:.1f} GB/s "
          f"(vec={best['vec_bytes']}B, threads={best['threads']})  -> {args.out}")


if __name__ == "__main__":
    main()
