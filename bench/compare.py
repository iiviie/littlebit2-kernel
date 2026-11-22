"""Compare a benchmark run against earlier runs and against its baselines.

    compare.py NEW.json                    # vs best earlier result per config (default)
                                           # metric defaults to chain_us (see bench/README.md)
    compare.py NEW.json --against OLD.json # vs one specific earlier file
    compare.py NEW.json --markdown         # README-ready table of the best kernel
    compare.py NEW.json --metric median_us # batched independent calls (overlap on GPU)

A "config" is (in, out, bpw, residual, impl). Model/proj names are only
labels. Earlier files are only used if they were run with the same dtype and
the same --rotate-mb (a hot-cache run is not comparable to a rotated one).

Ratios are old_time / new_time, so > 1 means NEW is faster. A change whose
|new/old - 1| is below the noise band (bench/results/noise.json, written by
noise.py) is shown as "~ (noise)". Baselines are compared too: they never
change, so their ratio shows how much the machine itself drifted between
runs -- if fp16 moved 8%, don't celebrate a 5% kernel win.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NON_RUN_FILES = {"peak_bw.json", "noise.json"}
BASELINE_COLS = ["fp16", "q4", "q2", "unpack_matmul"]


def load(path):
    with open(path) as f:
        return json.load(f)


def run_settings(d):
    a = d["meta"]["args"]
    return (a.get("dtype"), a.get("rotate_mb"))


def key(r):
    return (r["in"], r["out"], r["bpw"], r["residual"], r["impl"])


def ok_records(d, metric):
    return [r for r in d["records"] if r.get("status") == "ok" and r.get(metric) is not None]


def earlier_runs(new_path, new):
    """All result files run before NEW with the same settings, oldest first."""
    out = []
    t_new = new["meta"]["timestamp"]
    for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        if os.path.basename(p) in NON_RUN_FILES or os.path.abspath(p) == os.path.abspath(new_path):
            continue
        try:
            d = load(p)
        except (json.JSONDecodeError, OSError):
            continue
        if "records" not in d or d["meta"]["timestamp"] >= t_new:
            continue
        if run_settings(d) != run_settings(new):
            continue
        out.append((p, d))
    return out


def noise_band(metric):
    path = os.path.join(RESULTS_DIR, "noise.json")
    if not os.path.exists(path):
        return 0.03, "default 3% (no noise.json; run bench/noise.py)"
    d = load(path)
    band = d.get("noise_band_by_metric", {}).get(metric, d["noise_band"])
    return band, f"{100 * band:.1f}% from noise.json"


def verdict(ratio, band):
    """ratio = old/new. Text like '1.25x faster' or '~ (noise)'."""
    if abs(1 / ratio - 1) < band:
        return "~ (noise)"
    return f"{ratio:.2f}x faster" if ratio > 1 else f"{1 / ratio:.2f}x SLOWER"


def fmt_cfg(r):
    bpw = "-" if r["bpw"] is None else f"{r['bpw']:g}"
    res = "-" if r["residual"] is None else ("res" if r["residual"] else "nores")
    return f"{r['model']} {r['proj']} ({r['in']}x{r['out']}) bpw={bpw} {res}"


def baselines_for(records, r, metric):
    """{baseline name: time} matching kernel record r (same shape, same file)."""
    out = {}
    for b in records:
        if b["kind"] != "baseline" or b["in"] != r["in"] or b["out"] != r["out"]:
            continue
        if b["bpw"] is None or (b["bpw"] == r["bpw"] and b["residual"] == r["residual"]):
            out[b["impl"]] = b[metric]
    return out


def compare(new_path, against, metric):
    new = load(new_path)
    band, band_src = noise_band(metric)
    if against:
        olds = [(against, load(against))]
    else:
        olds = earlier_runs(new_path, new)
    # previous best per config: key -> (time, file label)
    prev = {}
    for p, d in olds:
        for r in ok_records(d, metric):
            k = key(r)
            if k not in prev or r[metric] < prev[k][0]:
                prev[k] = (r[metric], os.path.basename(p))

    recs = ok_records(new, metric)
    m = new["meta"]
    print(f"NEW: {os.path.basename(new_path)}  commit {str(m.get('git_commit'))[:8]}"
          f"{' (dirty)' if m.get('git_dirty') else ''}  metric={metric}  noise band {band_src}")
    print(f"previous: {'file ' + against if against else f'best of {len(olds)} earlier run(s)'}\n")

    # ---- vs previous -------------------------------------------------------
    print(f"{'config':58s} {'impl':14s} {'new us':>9s} {'prev us':>9s}  {'change':16s} prev file")
    for r in recs:
        p = prev.get(key(r))
        if p is None:
            change, old_s, src = "new config", "-", ""
        else:
            change, old_s, src = verdict(p[0] / r[metric], band), f"{p[0]:9.1f}", p[1]
        print(f"{fmt_cfg(r):58s} {r['impl']:14s} {r[metric]:9.1f} {old_s:>9s}  {change:16s} {src}")

    # ---- kernels vs baselines (same file) ------------------------------------
    print(f"\nkernels vs baselines in this run (baseline_time / kernel_time, > 1 = kernel faster):")
    print(f"{'config':58s} {'kernel':10s} " + " ".join(f"{b:>13s}" for b in BASELINE_COLS))
    for r in recs:
        if r["kind"] != "kernel":
            continue
        bl = baselines_for(recs, r, metric)
        cells = []
        for b in BASELINE_COLS:
            if b not in bl:
                cells.append(f"{'-':>13s}")
                continue
            ratio = bl[b] / r[metric]
            tag = "~" if abs(ratio - 1) < band else ""
            cells.append(f"{tag + f'{ratio:.2f}x':>13s}")
        print(f"{fmt_cfg(r):58s} {r['impl']:10s} " + " ".join(cells))


def markdown(new_path, metric):
    """README table: best kernel per (model/proj, bpw, residual)."""
    new = load(new_path)
    recs = ok_records(new, metric)
    peak = (new["meta"].get("peak_bw") or {}).get("peak_read_gbps")
    best = {}
    for r in recs:
        if r["kind"] != "kernel":
            continue
        k = (r["model"], r["proj"], r["bpw"], r["residual"])
        if k not in best or r[metric] < best[k][metric]:
            best[k] = r
    m = new["meta"]
    print(f"Batch-1 decode, {m['args']['dtype']}, {m.get('chip')}, MLX {m.get('mlx_version')}, "
          f"commit `{str(m.get('git_commit'))[:8]}`{' (dirty)' if m.get('git_dirty') else ''}. "
          f"Time = {'dependent-chain' if metric == 'chain_us' else 'in-graph batched'} median per call; "
          f"%peak vs measured {peak:.0f} GB/s read bandwidth. Speedup = baseline time / kernel time.\n"
          if peak else "")
    cols = ["layer", "shape", "bpw", "residual", "kernel", "µs", "GB/s", "%peak"] + \
           [f"vs {b}" for b in BASELINE_COLS]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * 5 + ["---:"] * (len(cols) - 5)) + "|")
    for (model, proj, bpw, res), r in best.items():
        gbps = r["bytes"] / (r[metric] * 1e-6) / 1e9
        pct = f"{100 * gbps / peak:.0f}%" if peak else "-"
        bl = baselines_for(recs, r, metric)
        sp = [f"{bl[b] / r[metric]:.2f}x" if b in bl else "-" for b in BASELINE_COLS]
        print("| " + " | ".join([f"{model} {proj}", f"{r['in']}→{r['out']}", f"{bpw:g}",
                                  "on" if res else "off", r["impl"], f"{r[metric]:.1f}",
                                  f"{gbps:.0f}", pct] + sp) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new", help="result JSON to examine")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--against", help="compare with this one earlier file")
    g.add_argument("--best", action="store_true", help="compare with the best of all earlier runs (default)")
    ap.add_argument("--markdown", action="store_true", help="print a README-ready table instead")
    ap.add_argument("--metric", default="chain_us", choices=["median_us", "chain_us", "min_us", "sync_us"],
                    help="which time to compare (default: dependent-chain time, closest to real decode)")
    args = ap.parse_args()
    if args.markdown:
        markdown(args.new, args.metric)
    else:
        compare(args.new, args.against, args.metric)


if __name__ == "__main__":
    main()
