"""Interleaved A/B(/C...) comparison of kernel implementations.

    .venv/bin/python bench/ab.py --impls lut,lut3,fused --quick
    .venv/bin/python bench/ab.py --impls lut3 --cfg 'lut3:TG2=128,RPT2=1' --cfg 'lut3:TG2=64'

Why this exists: on this machine the GPU's state drifts between runs (clock
ramping, thermal state, background apps), so "run A, then later run B" can
differ by 2x for the *same* kernel.  Here every round times each candidate
once, back to back, in a shuffled order, so drift hits all of them equally;
we then look at the per-round *ratio* to the first candidate (paired
comparison).  A candidate counts as better only if the 80% band of its
per-round ratios lies entirely beyond the noise band.

Timing per candidate per round = one dependent chain of N calls (each
call's input depends on the previous output, like decode), divided by N,
over enough weight copies to defeat the cache (see bench/README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import _glue, quiet_trial  # noqa: E402
from shapes import get_shapes  # noqa: E402

from littlebit_kernel.format import LittleBitLayer  # noqa: E402
from littlebit_kernel.kernels import KERNELS, DeviceLayer  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_cfgs(specs):
    """'impl:K=V,K=V' -> list of (label, impl, cfg dict)."""
    out = []
    for s in specs:
        impl, _, kv = s.partition(":")
        cfg = {k: int(v) for k, v in (p.split("=") for p in kv.split(",") if p)}
        out.append((s, impl, cfg))
    return out


def chain_time(fn, xs, n_copies, n):
    ring = time.perf_counter_ns() % n_copies  # start at a random copy
    outs, prev = [], None
    for k in range(n):
        x = xs[k] if prev is None else _glue(xs[k], prev[:, :1])
        prev = fn(x, (ring + k) % n_copies)
        outs.append(prev)
    t0 = time.perf_counter()
    mx.eval(outs)
    return (time.perf_counter() - t0) / n


def run(args):
    cands = [(i, i, {}) for i in args.impls.split(",") if i] + parse_cfgs(args.cfg)
    shapes = get_shapes(args.models.split(","), quick=args.quick)
    records = []
    for (model, proj, in_f, out_f) in shapes:
        for bpw in [float(b) for b in args.bpw.split(",")]:
            for res in {"off": [False], "on": [True], "both": [False, True]}[args.residual]:
                L = LittleBitLayer.random(in_f, out_f, eff_bit=bpw, residual=res)
                D0 = DeviceLayer.from_layer(L)
                C = int(min(args.max_copies, max(2, args.rotate_mb * 2**20 // D0.nbytes())))
                Ds = [D0] + [DeviceLayer.from_layer(L) for _ in range(C - 1)]
                xs = [mx.random.normal((1, in_f)).astype(mx.float16) for _ in range(args.n)]
                mx.eval(xs)
                fns = []
                for label, impl, cfg in cands:
                    f = KERNELS[impl]
                    fn = (lambda x, c, f=f, cfg=cfg: f(x, Ds[c], cfg)) if cfg else (lambda x, c, f=f: f(x, Ds[c]))
                    for _ in range(2):           # warm up + build per-copy caches
                        chain_time(fn, xs, C, args.n)
                    fns.append((label, fn))
                times = {lab: [] for lab, _ in fns}
                for _ in range(args.rounds):
                    order = list(fns)
                    random.shuffle(order)
                    for lab, fn in order:
                        times[lab].append(quiet_trial(chain_time, fn, xs, C, args.n))
                base = np.array(times[fns[0][0]])
                head = f"{model} {proj} {in_f}x{out_f} bpw={bpw} res={'on' if res else 'off'} r={L.main.rank}"
                print(head, flush=True)
                for lab, _ in fns:
                    t = np.array(times[lab]) * 1e6
                    ratio = base / np.array(times[lab])   # >1: faster than first candidate
                    rec = dict(model=model, proj=proj, **{"in": in_f, "out": out_f}, bpw=bpw, residual=res,
                               rank=L.main.rank, impl=lab, chain_us=float(np.median(t)),
                               p10_us=float(np.percentile(t, 10)), p90_us=float(np.percentile(t, 90)),
                               speedup_vs_first=float(np.median(ratio)),
                               speedup_p10=float(np.percentile(ratio, 10)),
                               speedup_p90=float(np.percentile(ratio, 90)))
                    records.append(rec)
                    print(f"   {lab:32s} {rec['chain_us']:8.1f} us [{rec['p10_us']:7.1f}-{rec['p90_us']:7.1f}]"
                          f"  x{rec['speedup_vs_first']:.2f} [{rec['speedup_p10']:.2f}-{rec['speedup_p90']:.2f}]",
                          flush=True)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impls", default="lut3")
    ap.add_argument("--cfg", action="append", default=[], help="impl:K=V,... (kernel config override)")
    ap.add_argument("--models", default="qwen3-0.6b,qwen3-4b,llama3-8b")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bpw", default="0.1,0.55,1.0")
    ap.add_argument("--residual", default="off", choices=["off", "on", "both"])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--n", type=int, default=64, help="calls per chain")
    ap.add_argument("--rotate-mb", type=int, default=128)
    ap.add_argument("--max-copies", type=int, default=256)
    ap.add_argument("--label", default="ab")
    args = ap.parse_args()
    recs = run(args)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, time.strftime("%Y%m%d-%H%M%S") + f"_{args.label}.json")
    with open(path, "w") as f:
        json.dump(dict(meta=dict(kind="ab", args=vars(args), mlx=mx.__version__), records=recs), f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
