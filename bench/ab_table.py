"""Markdown table from an `ab.py` result file whose FIRST candidate is the kernel.

    .venv/bin/python bench/ab_table.py bench/results/<time>_final.json

Speedups are *paired*: for every round, kernel time and baseline time were
measured back to back, so `speedup vs X` = median over rounds of
t_X / t_kernel (ab.py stores t_first / t_X per round; we invert it).
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict


def main(path):
    d = json.load(open(path))
    recs = d["records"]
    first = recs[0]["impl"]
    groups = OrderedDict()
    for r in recs:
        key = (r["model"], r["proj"], r["in"], r["out"], r["bpw"], r["residual"])
        groups.setdefault(key, {})[r["impl"]] = r
    bases = [b for b in ("fp16", "q4", "q2", "unpack_matmul") if any(b in g for g in groups.values())]
    hdr = ["layer", "in×out", "bpw", "r", f"{first} µs", "MB", "GB/s", "% peak"] + \
          [f"vs {b}" for b in bases] + [f"{b} µs" for b in bases]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for (model, proj, i, o, bpw, res), g in groups.items():
        k = g[first]
        row = [f"{model} {proj}" + (" +res" if res else ""), f"{i}×{o}", f"{bpw:g}", str(k["rank"]),
               f"{k['chain_us']:.0f}", f"{k['weight_bytes'] / 2**20:.2f}", f"{k['gbps']:.0f}",
               f"{k.get('pct_peak', float('nan')):.0f}%"]
        for b in bases:
            if b in g:
                s = 1.0 / g[b]["speedup_vs_first"]
                lo, hi = 1.0 / g[b]["speedup_p90"], 1.0 / g[b]["speedup_p10"]
                mark = "**" if lo > 1 else ""
                row.append(f"{mark}{s:.2f}x{mark}")
            else:
                row.append("–")
        for b in bases:
            row.append(f"{g[b]['chain_us']:.0f}" if b in g else "–")
        print("| " + " | ".join(row) + " |")
    print(f"\nBold = the kernel was faster in at least 90% of rounds (paired). "
          f"Times are per call in a dependent chain (decode-like), cold cache.")


if __name__ == "__main__":
    main(sys.argv[1])
