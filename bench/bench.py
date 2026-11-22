"""Benchmark LittleBit kernels against MLX baselines for batch-1 decode.

For every (layer shape x bits-per-weight x residual on/off) we build a random
LittleBit layer (random sign bits, random positive scales -- no downloads),
check every kernel's output against the reference once, then time it.
Baselines are timed at the same (in, out).

Implementations timed
---------------------
  kernels (kind="kernel"):  every entry of littlebit_kernel.kernels.KERNELS
  baselines (kind="baseline"):
    fp16           y = x @ W.T with a dense fp16 W (out, in)
    q4, q2         mx.quantized_matmul, 4-/2-bit affine, group_size 64
                   (weights from mx.quantize) -- what mlx-lm uses for
                   quantized models
    unpack_matmul  what upstream LittleBit does at inference: U and V
                   materialised as dense fp16 +-1 matrices, two fp16 matmuls
                   plus the scale multiplies (prebuilt once, NOT unpacked per
                   call). Depends on the rank, so it is timed per bpw config.
  fp16/q4/q2 do not depend on bpw, so they are timed once per shape and stored
  with bpw = residual = rank = null.

Methodology (read this before trusting a number)
------------------------------------------------
1. In-graph batched timing (the primary number, `median_us`).
   mlx-lm decodes by building the whole lazy graph for a token and
   evaluating it at once, so a single projection never pays a full
   Python -> GPU -> Python round trip of its own. Timing `f(x); mx.eval()`
   per call would mostly measure that round trip (~50-150 us), not the
   kernel. So we build N independent calls (N = 16..256, chosen so a batch
   is ~20 ms of GPU work), each with its own x so nothing can be merged, and
   time a single `mx.eval` over all N outputs: us/call = elapsed / N.
   Graph construction (Python) happens before the timer starts; the timer
   covers MLX's command encoding + GPU execution.

   Caveat: MLX encodes with Metal's *concurrent* dispatch mode and only puts
   a barrier where an op reads a previous op's output. N independent calls
   can therefore overlap on the GPU, which hides launch latency and tail
   effects. Real decode is a dependency chain (layer i+1 needs layer i), so
   we also record:
2. Chained timing (`chain_us`). Same as (1), but call k's input depends on
   call k-1's output: x_k = xs[k] + 0 * y_{k-1}[:, :1] (one tiny fused
   elementwise "glue" kernel). This forces the GPU to run the calls one
   after another, like decode. It includes the glue kernel and the barrier;
   `glue_us` (a chain of glue kernels alone) is recorded per shape so you
   can see how big that part is. Do not just subtract it: the dependency
   latency is paid once per link in both chains.
3. Sync per call (`sync_us`). `y = f(x); mx.eval(y)` timed per call,
   including graph build. The worst case: pure launch + round-trip overhead.

Cache effects: weight rotation.
   An M1 Pro has a ~24 MB system level cache (SLC). Many sub-1-bit layers
   are only tens of KB to a few MB, so calling the same weights N times in a
   row would read them from cache and report impossible GB/s. In real decode
   each layer's weights are read once per token, with every other layer's
   weights streamed in between. So for each implementation we allocate C
   distinct copies of the weights with C * bytes_per_copy >= --rotate-mb
   (default 128 MB, > 5x the SLC), capped at --max-copies, and call k uses
   copy k % C, continuing round the ring across trials. Baselines get the
   same treatment. The copies hold identical values (the GPU cannot know
   that; nothing dedups memory). Use --rotate-mb 0 for a hot-cache number.

Statistics: --warmup untimed batches, then --trials timed batches; we report
the median, p10, p90 and min of the per-batch us/call.

Bytes and bandwidth:
   bytes = weight bytes the op must read (kernels: packed sign bits + fp16
           scale vectors [+ bias], i.e. DeviceLayer.nbytes(); baselines:
           their own weights -- fp16 W, quantized w + scales + biases, or the
           fp16 sign matrices + scales) + x bytes + y bytes.
   gbps  = bytes / median time;   pct_peak = gbps / peak_read_gbps, where
   the peak comes from bench/results/peak_bw.json (run bench/peak_bw.py).
   Intermediates (e.g. the rank-r vector between the two stages) are not
   counted -- a kernel that spills them to DRAM pays for it in time.

Correctness gate: before timing, each kernel (and unpack_matmul) is run once
and compared with reference_forward_mlx (fp32, on fp16-rounded scales). A
kernel whose max abs error exceeds --tol * max|ref| is skipped (recorded with
status "wrong"): a fast wrong kernel is worthless.

Output: bench/results/<YYYYmmdd-HHMMSS>_<label>.json (metadata + flat list of
records) and a compact table on stdout.

Usage examples:
  .venv/bin/python bench/bench.py --quick --label smoke
  .venv/bin/python bench/bench.py --kernels naive --models qwen3-0.6b --bpw 0.1,1.0 --residual off
  .venv/bin/python bench/bench.py --no-baselines --kernels naive,tiled --label tiled_v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

from littlebit_kernel.format import LittleBitLayer, effective_bits
from littlebit_kernel.kernels import KERNELS, DeviceLayer, round_layer_scales
from littlebit_kernel.reference import reference_forward_mlx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapes import MODELS, get_shapes  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
REPO_DIR = os.path.dirname(BENCH_DIR)

BASELINES = ["fp16", "q4", "q2", "unpack_matmul"]
DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}

# ==========================================================================
# timing primitives
# ==========================================================================


@mx.compile
def _glue(x, prev_col):
    """x + 0 * prev: a data dependency on the previous call, value unchanged.

    Compiled so MLX fuses it into one elementwise kernel.
    """
    return x + 0 * prev_col


class Timer:
    """Holds the knobs shared by every measurement."""

    def __init__(self, trials=25, warmup=3, target_batch_ms=20.0, max_calls=256,
                 min_calls=16, chain=True, sync=True):
        self.trials = trials
        self.warmup = warmup
        self.target_batch_ms = target_batch_ms
        self.max_calls = max_calls
        self.min_calls = min_calls
        self.chain = chain
        self.sync = sync


def _stats(per_call_s) -> dict:
    a = np.asarray(per_call_s) * 1e6
    return {
        "median_us": float(np.median(a)),
        "p10_us": float(np.percentile(a, 10)),
        "p90_us": float(np.percentile(a, 90)),
        "min_us": float(a.min()),
    }


def measure(call, xs, n_copies, timer: Timer) -> dict:
    """Time `call(x, copy_index)`; see the module docstring for the method.

    xs: list of distinct inputs (len >= max_calls).  Returns a dict of stats.
    """
    ring = [0]  # next weight copy to use; keeps rotating across batches

    def next_copy():
        c = ring[0]
        ring[0] = (c + 1) % n_copies
        return c

    def batch_independent(n):
        outs = [call(xs[k], next_copy()) for k in range(n)]
        t0 = time.perf_counter()
        mx.eval(outs)
        return (time.perf_counter() - t0) / n

    def batch_chained(n):
        outs, prev = [], None
        for k in range(n):
            x = xs[k] if prev is None else _glue(xs[k], prev[:, :1])
            prev = call(x, next_copy())
            outs.append(prev)
        t0 = time.perf_counter()
        mx.eval(outs)
        return (time.perf_counter() - t0) / n

    # Calibrate N: aim for ~target_batch_ms of work per timed eval.
    batch_independent(8)
    est = batch_independent(8)
    n = int(timer.target_batch_ms * 1e-3 / max(est, 1e-7))
    n = max(timer.min_calls, min(timer.max_calls, n, len(xs)))

    for _ in range(timer.warmup):
        batch_independent(n)
    res = _stats([batch_independent(n) for _ in range(timer.trials)])
    res["n_calls"] = n

    if timer.chain:
        batch_chained(n)
        ch = _stats([batch_chained(n) for _ in range(max(5, timer.trials // 2))])
        res["chain_us"] = ch["median_us"]

    if timer.sync:
        per = []
        for k in range(32):
            c = next_copy()
            t0 = time.perf_counter()
            mx.eval(call(xs[k], c))
            per.append(time.perf_counter() - t0)
        res["sync_us"] = _stats(per[4:])["median_us"]   # first few = warmup
    return res


def measure_glue(xs, timer: Timer) -> float:
    """us per link of a chain made only of glue kernels (for context)."""
    n = min(timer.max_calls, len(xs))

    def run():
        outs, prev = [], xs[0]
        for k in range(n):
            prev = _glue(xs[k], prev[:, :1])
            outs.append(prev)
        t0 = time.perf_counter()
        mx.eval(outs)
        return (time.perf_counter() - t0) / n

    for _ in range(timer.warmup):
        run()
    return _stats([run() for _ in range(max(5, timer.trials // 2))])["median_us"]


def n_copies_for(nbytes_per_copy: int, rotate_mb: float, max_copies: int) -> int:
    if rotate_mb <= 0:
        return 1
    return max(1, min(max_copies, math.ceil(rotate_mb * 2**20 / nbytes_per_copy)))


def _nbytes(arrays) -> int:
    return sum(a.nbytes for a in arrays if a is not None)

# ==========================================================================
# implementations: each builder returns (call(x, copy_idx), weight_bytes, n_copies)
# ==========================================================================


def build_kernel_copies(layer: LittleBitLayer, n_copies_fn):
    """C independent DeviceLayers (distinct GPU buffers, same values)."""
    first = DeviceLayer.from_layer(layer)
    wbytes = first.nbytes()
    C = n_copies_fn(wbytes)
    copies = [first] + [DeviceLayer.from_layer(layer) for _ in range(C - 1)]
    return copies, wbytes


def build_dense(in_f, out_f, dtype, n_copies_fn):
    C = n_copies_fn(in_f * out_f * dtype.size)
    Ws = []
    for _ in range(C):
        W = (mx.random.normal((out_f, in_f)) * (1.0 / math.sqrt(in_f))).astype(dtype)
        mx.eval(W)
        Ws.append(W)
    return (lambda x, c: x @ Ws[c].T), Ws[0].nbytes, C


def build_quant(in_f, out_f, dtype, bits, n_copies_fn, group_size=64):
    def one():
        W = (mx.random.normal((out_f, in_f)) * (1.0 / math.sqrt(in_f))).astype(dtype)
        q = mx.quantize(W, group_size=group_size, bits=bits)   # (w, scales, biases)
        mx.eval(q)
        return q

    first = one()
    wbytes = _nbytes(first)
    C = n_copies_fn(wbytes)
    Qs = [first] + [one() for _ in range(C - 1)]

    def call(x, c):
        w, s, b = Qs[c]
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=group_size, bits=bits)
    return call, wbytes, C


def build_unpack_matmul(layer: LittleBitLayer, dtype, n_copies_fn):
    """Upstream-style: dense +-1 sign matrices in fp16, prebuilt, two matmuls."""
    rl = round_layer_scales(layer)

    def one():
        paths = []
        for p in rl.paths:
            paths.append(dict(
                V=mx.array(p.V_signs()).astype(dtype),             # (r, in)
                U=mx.array(p.U_signs()).astype(dtype),             # (out, r)
                v2=mx.array(p.v2).astype(dtype),
                mid=mx.array(p.mid_scale).astype(dtype),
                u1=mx.array(p.u1).astype(dtype)))
        bias = None if rl.bias is None else mx.array(rl.bias).astype(dtype)
        mx.eval([list(d.values()) for d in paths], bias if bias is not None else [])
        return paths, bias

    first = one()
    wbytes = sum(_nbytes(d.values()) for d in first[0]) + _nbytes([first[1]])
    C = n_copies_fn(wbytes)
    copies = [first] + [one() for _ in range(C - 1)]

    def call(x, c):
        paths, bias = copies[c]
        y = None
        for d in paths:
            t = ((x * d["v2"]) @ d["V"].T) * d["mid"]
            o = (t @ d["U"].T) * d["u1"]
            y = o if y is None else y + o
        return y if bias is None else y + bias
    return call, wbytes, C

# ==========================================================================
# correctness gate
# ==========================================================================


def check(out, x, layer: LittleBitLayer, tol: float):
    """(ok, rel_err): max|out - ref| / max|ref| against the fp32 reference."""
    ref = reference_forward_mlx(x.astype(mx.float32), round_layer_scales(layer), dtype=mx.float32)
    out = out.astype(mx.float32)
    if out.shape != ref.shape:
        return False, float("inf")
    err = mx.max(mx.abs(out - ref)).item() / max(mx.max(mx.abs(ref)).item(), 1e-12)
    return (err <= tol and math.isfinite(err)), err

# ==========================================================================
# metadata
# ==========================================================================


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", REPO_DIR, *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def metadata(args, peak) -> dict:
    status = _git("status", "--porcelain")
    info = mx.device_info()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        cpu = None
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "mlx_version": mx.__version__,
        "chip": info.get("device_name") or cpu,
        "cpu": cpu,
        "memory_gb": info.get("memory_size", 0) / 2**30,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "loadavg_start": os.getloadavg(),
        "args": vars(args),
        "peak_bw": peak,
    }


def load_peak():
    path = os.path.join(RESULTS_DIR, "peak_bw.json")
    if not os.path.exists(path):
        print("! no bench/results/peak_bw.json -- run bench/peak_bw.py; pct_peak will be null")
        return None
    with open(path) as f:
        d = json.load(f)
    return {"peak_read_gbps": d["peak_read_gbps"], "measured_at": d.get("timestamp"), "file": path}

# ==========================================================================
# main sweep
# ==========================================================================


def record(base: dict, impl: str, kind: str, stats: dict, nbytes: int, peak) -> dict:
    r = dict(base, impl=impl, kind=kind, status="ok", bytes=nbytes, **stats)
    r["gbps"] = nbytes / (r["median_us"] * 1e-6) / 1e9
    r["pct_peak"] = 100 * r["gbps"] / peak["peak_read_gbps"] if peak else None
    return r


def run(args) -> dict:
    dtype = DTYPES[args.dtype]
    itemsize = dtype.size
    peak = load_peak()
    timer = Timer(trials=args.trials, warmup=args.warmup, target_batch_ms=args.target_ms,
                  max_calls=args.max_calls, chain=args.chain, sync=args.sync)
    ncopies = lambda nb: n_copies_for(nb, args.rotate_mb, args.max_copies)

    kernels = list(KERNELS) if args.kernels == "all" else [k for k in args.kernels.split(",") if k]
    for k in kernels:
        if k not in KERNELS:
            sys.exit(f"unknown kernel {k!r}; available: {list(KERNELS)}")
    models = None if args.models == "all" else args.models.split(",")
    shapes = get_shapes(models, quick=args.quick)
    bpws = [float(b) for b in args.bpw.split(",")]
    residuals = {"both": [False, True], "on": [True], "off": [False]}[args.residual]

    meta = metadata(args, peak)
    records = []
    t_start = time.time()

    for model, proj, in_f, out_f in shapes:
        print(f"\n=== {model} {proj}  in={in_f} out={out_f}")
        # Distinct inputs, one per call in a batch (so nothing is merged).
        xs = [mx.random.normal((1, in_f)).astype(dtype) for _ in range(args.max_calls)]
        mx.eval(xs)
        io_bytes = (in_f + out_f) * itemsize
        shape_base = dict(model=model, proj=proj, **{"in": in_f, "out": out_f})
        glue_us = measure_glue(xs, timer) if args.chain else None

        # ---- bpw-independent baselines: once per shape ----------------------
        if args.baselines:
            builders = {
                "fp16": lambda: build_dense(in_f, out_f, dtype, ncopies),
                "q4": lambda: build_quant(in_f, out_f, dtype, 4, ncopies),
                "q2": lambda: build_quant(in_f, out_f, dtype, 2, ncopies),
            }
            for name, build in builders.items():
                call, wbytes, C = build()
                st = measure(call, xs, C, timer)
                base = dict(shape_base, bpw=None, residual=None, rank=None, eff_bits=None,
                            n_copies=C, glue_us=glue_us)
                records.append(record(base, name, "baseline", st, wbytes + io_bytes, peak))
                _print_row(records[-1])
                del call
                mx.clear_cache()

        # ---- LittleBit configs ---------------------------------------------
        for bpw in bpws:
            for res in residuals:
                layer = LittleBitLayer.random(in_f, out_f, eff_bit=bpw, residual=res, seed=0)
                rank = layer.main.rank
                base = dict(shape_base, bpw=bpw, residual=res, rank=rank,
                            eff_bits=effective_bits(in_f, out_f, rank, res), glue_us=glue_us)
                print(f"--- bpw={bpw} residual={'on' if res else 'off'} rank={rank} "
                      f"(actual {base['eff_bits']:.3f} bpw)")
                cfg_records = []

                for kname in kernels:
                    fwd = KERNELS[kname]
                    copies, wbytes = build_kernel_copies(layer, ncopies)
                    ok, err = check(fwd(xs[0], copies[0]), xs[0], layer, args.tol)
                    if not ok:
                        print(f"  ! {kname}: WRONG output (rel err {err:.3g} > tol {args.tol}) -- skipped")
                        cfg_records.append(dict(base, impl=kname, kind="kernel", status="wrong",
                                                rel_err=err, median_us=None))
                        continue
                    st = measure(lambda x, c: fwd(x, copies[c]), xs, len(copies), timer)
                    r = record(dict(base, n_copies=len(copies), rel_err=err), kname, "kernel",
                               st, wbytes + io_bytes, peak)
                    cfg_records.append(r)
                    del copies
                    mx.clear_cache()

                if args.baselines:
                    call, wbytes, C = build_unpack_matmul(layer, dtype, ncopies)
                    ok, err = check(call(xs[0], 0), xs[0], layer, args.tol)
                    if not ok:
                        print(f"  ! unpack_matmul: WRONG (rel err {err:.3g}) -- harness bug?")
                    st = measure(call, xs, C, timer)
                    cfg_records.append(record(dict(base, n_copies=C, rel_err=err), "unpack_matmul",
                                              "baseline", st, wbytes + io_bytes, peak))
                    del call
                    mx.clear_cache()

                # speedup of each kernel vs each baseline (baseline_us / kernel_us)
                bl = {r["impl"]: r["median_us"] for r in records
                      if r["kind"] == "baseline" and r["in"] == in_f and r["out"] == out_f
                      and r["model"] == model and r["bpw"] is None}
                bl.update({r["impl"]: r["median_us"] for r in cfg_records if r["kind"] == "baseline"})
                for r in cfg_records:
                    if r["kind"] == "kernel" and r["status"] == "ok":
                        r["speedup_vs"] = {b: us / r["median_us"] for b, us in bl.items()}
                    _print_row(r)
                records.extend(cfg_records)
                del layer

    meta["elapsed_s"] = time.time() - t_start
    meta["loadavg_end"] = os.getloadavg()
    return {"meta": meta, "records": records}


def _print_row(r):
    if r.get("status") != "ok":
        return
    pct = f"{r['pct_peak']:5.1f}%" if r.get("pct_peak") is not None else "   n/a"
    extra = ""
    if r.get("chain_us") is not None:
        extra += f" chain {r['chain_us']:8.1f}"
    if r.get("sync_us") is not None:
        extra += f" sync {r['sync_us']:8.1f}"
    sp = ""
    if r.get("speedup_vs"):
        sp = "  x" + " ".join(f"{k}={v:.2f}" for k, v in r["speedup_vs"].items())
    print(f"  {r['impl']:14s} {r['median_us']:9.1f} us [{r['p10_us']:8.1f}-{r['p90_us']:8.1f}]"
          f" {r['bytes'] / 1e6:8.2f} MB {r['gbps']:7.1f} GB/s {pct}{extra}"
          f"  (C={r.get('n_copies')}, N={r.get('n_calls')}){sp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernels", default="all", help=f"comma list or 'all' (available: {','.join(KERNELS)})")
    ap.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True,
                    help="time fp16 / q4 / q2 / unpack_matmul baselines (default on)")
    ap.add_argument("--models", default="all", help=f"comma list or 'all' ({','.join(MODELS)})")
    ap.add_argument("--quick", action="store_true", help="only the quick shape subset (see shapes.py)")
    ap.add_argument("--bpw", default="0.1,0.3,0.55,1.0", help="comma list of bits per weight")
    ap.add_argument("--residual", default="both", choices=["both", "on", "off"])
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES))
    ap.add_argument("--label", default="run")
    ap.add_argument("--trials", type=int, default=25, help="timed batches per measurement")
    ap.add_argument("--warmup", type=int, default=3, help="untimed batches first")
    ap.add_argument("--target-ms", type=float, default=20.0, help="aim for this much work per timed batch")
    ap.add_argument("--max-calls", type=int, default=256, help="max calls per batch (N)")
    ap.add_argument("--rotate-mb", type=float, default=128.0,
                    help="rotate over weight copies totalling >= this many MB (0 = no rotation, hot cache)")
    ap.add_argument("--max-copies", type=int, default=4096, help="cap on number of weight copies")
    ap.add_argument("--chain", action=argparse.BooleanOptionalAction, default=True,
                    help="also measure dependent-chain time (chain_us)")
    ap.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True,
                    help="also measure eval-per-call time (sync_us)")
    ap.add_argument("--tol", type=float, default=2e-2, help="max |err| / max |ref| for the correctness gate")
    ap.add_argument("--out", default=None, help="output JSON path (default bench/results/<time>_<label>.json)")
    args = ap.parse_args()

    result = run(args)
    out = args.out or os.path.join(RESULTS_DIR, time.strftime("%Y%m%d-%H%M%S") + f"_{args.label}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nsaved {len(result['records'])} records -> {out}  ({result['meta']['elapsed_s']:.0f} s)")


if __name__ == "__main__":
    main()
