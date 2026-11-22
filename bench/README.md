# Benchmarks

Batch-1 decode benchmarks for the LittleBit kernels (`x` is `(1, in)` fp16)
against MLX baselines, on the real projection shapes of Qwen3-0.6B, Qwen3-4B
and Llama-3-8B.

| file | what it does |
|---|---|
| `peak_bw.py` | measures the achievable DRAM read bandwidth → `results/peak_bw.json` |
| `shapes.py` | per-projection `(in, out)` shapes, deduplicated, plus a `--quick` subset |
| `bench.py` | the sweep: kernels + baselines → `results/<time>_<label>.json` |
| `noise.py` | run-to-run noise estimate → `results/noise.json` |
| `compare.py` | new run vs earlier runs / vs baselines; `--markdown` table |

## The tool to trust: interleaved A/B (`ab.py`)

On this machine the *same* kernel can measure 200 µs in one run and 280 µs
in the next (thermals, charging, GPU clock ramping, background apps). A
sequential "run A, later run B" comparison is therefore unreliable.
`ab.py` times every candidate once per round, in shuffled order, back to
back, and reports the **per-round ratio** to the first candidate with its
10th-90th percentile band. Baselines (`fp16`, `q4`, `q2`,
`unpack_matmul`) can be candidates too, so speedups vs baselines are
paired as well. Timing is always the dependent-chain time over rotated
weight copies (see below). `ab_table.py` turns an `ab.py` result into the
README table.

```sh
.venv/bin/python bench/ab.py --impls lut5,lut4 --quick                       # kernel vs kernel
.venv/bin/python bench/ab.py --impls lut4 --cfg lut4:KC1=256 --quick         # config sweep
.venv/bin/python bench/ab.py --impls auto,fp16,q4,q2,unpack_matmul --label final
.venv/bin/python bench/ab_table.py bench/results/<time>_final.json
```

Every timed trial checks for other GPU jobs (`BENCH_BUSY_PATTERN`, default
`demucs`; a regex for `pgrep -f`) before and after, and is redone if one
overlapped.

## Running (bench.py: single-run sweep)

Run everything one at a time on a quiet machine: nothing else on the GPU.
Two timing processes at once disturb each other.

```sh
.venv/bin/python bench/peak_bw.py                      # once per machine (~10 s)
.venv/bin/python bench/noise.py                        # once per machine (~1-2 min)
.venv/bin/python bench/bench.py --quick --label try1   # fast loop (~20 s)
.venv/bin/python bench/bench.py --label full           # all shapes x bpw x residual
.venv/bin/python bench/compare.py bench/results/<new>.json            # vs best earlier
.venv/bin/python bench/compare.py bench/results/<new>.json --markdown # README table
```

Useful `bench.py` flags: `--kernels naive,foo`, `--no-baselines`,
`--models qwen3-0.6b`, `--bpw 0.1,1.0`, `--residual off`, `--rotate-mb 0`
(hot cache), `--no-chain`, `--no-sync`, `--trials 25`.

## Methodology

**Bandwidth is the yardstick.** At batch 1 every weight byte is read once
per token and used for about one multiply-add, so a decode kernel is memory
bound. `peak_bw.py` measures what the chip can really read (a grid-stride
Metal read kernel over a 1 GiB buffer, sweeping load width and thread
count, median of trials). Each result then gets
`gbps = bytes / time` and `%peak = gbps / peak_read_gbps`, where
`bytes` = weights the op must read (packed sign bits + fp16 scales for
LittleBit; fp16 `W`, or quantized `w + scales + biases`, or fp16 sign
matrices for baselines) + `x` + `y`.

**In-graph batched timing (`median_us`).** mlx-lm builds the whole graph for
a token and evaluates it in one go, so one projection never pays a full
Python→GPU→Python round trip by itself. Timing `f(x); mx.eval()` per call
measures mostly that round trip. Instead we build N independent calls (each
with its own `x`, so nothing is merged) and time one `mx.eval` of all N
outputs; µs/call = elapsed / N. N is picked so one batch is ~20 ms.

**Chained timing (`chain_us`): read this.** MLX dispatches in Metal's
*concurrent* mode and only puts a barrier where one op reads another's
output. So N *independent* calls can overlap on the GPU. A kernel that
launches only a few hundred threads (low occupancy) then looks far better
than it is, because several calls run side by side. In real decode, layer
i+1 waits for layer i. `chain_us` makes each call depend on the previous one
(`x_k = xs[k] + 0 * y_{k-1}[:, :1]`, one tiny fused kernel) so calls run in
order. In smoke tests the naive kernel on Llama `down_proj` @ 1 bpw was
~340 µs batched but ~880 µs chained, while fp16 moved only ~7%. **When you
work on occupancy or launch structure, judge by `chain_us`**
(`compare.py --metric chain_us`). `glue_us` (a chain of glue kernels alone)
is stored per shape for context. `sync_us` (eval after every call) is the
worst case: pure launch + round-trip overhead.

**Weight rotation (cache effects).** The M1 Pro has a ~24 MB system level
cache. A sub-1-bit layer is often only tens of KB, so calling the same
weights N times reads them from cache and reports impossible GB/s. In decode
each layer is read once per token with every other layer in between. So
each implementation gets C distinct copies of its weights, with
C × bytes ≥ `--rotate-mb` (128 MB by default, > 5× the cache), capped at
`--max-copies` (4096), and call k uses copy k mod C. Baselines get the same.
Note the cap: a 12 KB layer × 4096 copies is only ~50 MB, still more than
twice the cache.

**Statistics.** Warmup batches, then `--trials` timed batches; we store
median, p10, p90, min.

**Correctness gate.** Before timing, each kernel's output is checked against
`reference_forward_mlx` (fp32, fp16-rounded scales). A kernel whose
`max|err| / max|ref|` exceeds `--tol` (2e-2) is skipped with status
`"wrong"`.

## Noise and comparing runs

`noise.py` repeats the same measurement 8 times (fresh allocations,
interleaved) on a small, a mid and a large config for naive + fp16, and
stores `noise_band = max(2%, 2 × worst coefficient of variation)` for
each metric. `compare.py` marks any change smaller than that band as
`~ (noise)`. It also compares the *baselines* with earlier runs: they never
change, so if fp16 moved 8% the machine drifted and a 5% kernel "win" means
nothing. Only runs with the same `--dtype` and `--rotate-mb` are compared.
