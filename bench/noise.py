"""Estimate run-to-run noise, so compare.py can tell a real change from luck.

We repeat the *same* measurement K times as independent repetitions (fresh
weight copies each time, i.e. new GPU allocations) on three representative
configs:

    small : qwen3-0.6b k_proj   (1024 -> 1024)   @ 0.1  bpw, residual off
    mid   : qwen3-4b   gate_proj (2560 -> 9728)  @ 0.55 bpw, residual on
    large : llama3-8b  down_proj (14336 -> 4096) @ 1.0  bpw, residual off

for one kernel (default naive) and one baseline (default fp16). Repetitions
are interleaved (rep 1 of every config, then rep 2, ...) so slow drift --
thermal throttling, background load -- shows up as spread instead of
hiding inside one config's block.

For each (config, impl, metric) we report:
    cv          = std / mean of the K medians  (coefficient of variation)
    rel_spread  = (max - min) / median of the K medians

and the recommended noise band (relative):
    noise_band = max(0.02, 2 * worst cv)          (per metric)

compare.py treats a change smaller than the band as "~ (noise)".
Saved to bench/results/noise.json. Re-run it whenever the machine setup
changes (macOS update, different power mode, etc.). Run it on a quiet
machine: nothing else on the GPU.

Usage:  .venv/bin/python bench/noise.py [--reps 8] [--kernel naive] [--baseline fp16]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
from littlebit_kernel.format import LittleBitLayer  # noqa: E402
from littlebit_kernel.kernels import KERNELS  # noqa: E402

CONFIGS = [
    # name, model, proj, in, out, bpw, residual
    ("small", "qwen3-0.6b", "k_proj", 1024, 1024, 0.1, False),
    ("mid", "qwen3-4b", "gate_proj", 2560, 9728, 0.55, True),
    ("large", "llama3-8b", "down_proj", 14336, 4096, 1.0, False),
]
METRICS = ["median_us", "chain_us"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--kernel", default="naive")
    ap.add_argument("--baseline", default="fp16", choices=["fp16", "q4", "q2", "unpack_matmul"])
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--rotate-mb", type=float, default=128.0)
    ap.add_argument("--max-copies", type=int, default=4096)
    ap.add_argument("--configs", default="small,mid,large")
    ap.add_argument("--out", default=os.path.join(bench.RESULTS_DIR, "noise.json"))
    args = ap.parse_args()

    dtype = mx.float16
    timer = bench.Timer(trials=args.trials, chain=True, sync=False)
    ncopies = lambda nb: bench.n_copies_for(nb, args.rotate_mb, args.max_copies)
    fwd = KERNELS[args.kernel]
    configs = [c for c in CONFIGS if c[0] in args.configs.split(",")]

    # The numpy layers are deterministic; build once (that part is CPU only).
    layers = {c[0]: LittleBitLayer.random(c[3], c[4], eff_bit=c[5], residual=c[6], seed=0)
              for c in configs}
    inputs = {c[0]: [mx.random.normal((1, c[3])).astype(dtype) for _ in range(timer.max_calls)]
              for c in configs}
    mx.eval(list(inputs.values()))

    # samples[(config, impl)][metric] = [median of rep 1, rep 2, ...]
    samples = {}
    t0 = time.time()
    for rep in range(args.reps):
        for name, model, proj, in_f, out_f, bpw, res in configs:
            layer, xs = layers[name], inputs[name]

            # kernel: fresh device copies every repetition
            copies, _ = bench.build_kernel_copies(layer, ncopies)
            st = bench.measure(lambda x, c: fwd(x, copies[c]), xs, len(copies), timer)
            del copies
            for m in METRICS:
                samples.setdefault((name, args.kernel), {}).setdefault(m, []).append(st[m])

            # baseline
            if args.baseline == "fp16":
                call, _, C = bench.build_dense(in_f, out_f, dtype, ncopies)
            elif args.baseline in ("q4", "q2"):
                call, _, C = bench.build_quant(in_f, out_f, dtype, int(args.baseline[1]), ncopies)
            else:
                call, _, C = bench.build_unpack_matmul(layer, dtype, ncopies)
            st_b = bench.measure(call, xs, C, timer)
            del call
            mx.clear_cache()
            for m in METRICS:
                samples.setdefault((name, args.baseline), {}).setdefault(m, []).append(st_b[m])

            print(f"rep {rep + 1}/{args.reps} {name:5s}: {args.kernel} {st['median_us']:8.1f} us "
                  f"(chain {st['chain_us']:8.1f})   {args.baseline} {st_b['median_us']:8.1f} us "
                  f"(chain {st_b['chain_us']:8.1f})")

    # ---- summarise ---------------------------------------------------------
    rows = []
    worst = {m: 0.0 for m in METRICS}
    print(f"\n{'config':6s} {'impl':14s} {'metric':10s} {'median':>9s} {'cv':>7s} {'spread':>7s}")
    for (name, impl), by_metric in samples.items():
        for m, vals in by_metric.items():
            v = np.array(vals)
            cv = float(v.std(ddof=1) / v.mean()) if len(v) > 1 else 0.0
            spread = float((v.max() - v.min()) / np.median(v))
            worst[m] = max(worst[m], cv)
            rows.append(dict(config=name, impl=impl, metric=m, values_us=vals,
                             median_us=float(np.median(v)), cv=cv, rel_spread=spread))
            print(f"{name:6s} {impl:14s} {m:10s} {np.median(v):9.1f} {100 * cv:6.2f}% {100 * spread:6.1f}%")

    bands = {m: max(0.02, 2 * worst[m]) for m in METRICS}
    result = {
        "noise_band": bands["median_us"],   # the default metric's band
        "noise_band_by_metric": bands,
        "worst_cv_by_metric": worst,
        "max_rel_spread": max(r["rel_spread"] for r in rows),
        "rule": "noise_band = max(0.02, 2 * worst cv over configs/impls), per metric",
        "reps": args.reps,
        "configs": [dict(zip(["name", "model", "proj", "in", "out", "bpw", "residual"], c)) for c in configs],
        "rows": rows,
        "meta": bench.metadata(args, bench.load_peak()),
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("\nnoise band: " + ", ".join(f"{m} {100 * b:.1f}%" for m, b in bands.items())
          + f"   (max rel spread {100 * result['max_rel_spread']:.1f}%)  -> {args.out}")


if __name__ == "__main__":
    main()
