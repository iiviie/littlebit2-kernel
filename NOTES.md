# Optimisation log

Every variant lives in its own file under `src/littlebit_kernel/kernels/`
and is registered in `KERNELS`, so it is automatically covered by the
correctness gate (`tests/test_kernels.py`) and the benchmark. A change counts
as a win only if it passes the gate **and** beats the previous best by more
than the measured noise band (`bench/results/noise.json`), compared with
`bench/compare.py` on the dependent-chain time (`chain_us`, see below).

Hardware: Apple M1 Pro (16-core GPU), 16 GB, macOS 15.8, MLX 0.32.4.

## Measurement notes

* **Batched vs chained timing.** MLX encodes independent ops so the GPU may
  overlap them. Timing N independent calls therefore hides poor occupancy
  (the naive kernel looked 2.5-4x better batched than chained). Real decode
  is a dependency chain, so `chain_us` (each call's input depends on the
  previous output through a tiny glue op) is the headline metric. It
  slightly overstates cost (one glue op + barrier per link; `glue_us` is
  recorded so you can see how much).
* **Cache rotation.** A sub-1-bit layer is often smaller than the M1 Pro's
  system cache, so repeating the same layer measures cache bandwidth, not
  DRAM. The harness rotates over enough copies to touch >= 128 MB.
* **Other GPU jobs.** Another project on this machine runs demucs on MPS
  in a loop, for minutes at a time; overlap made numbers swing by 2x.
  Every timed trial now checks for such processes before and after and is
  redone if one overlapped (`bench.quiet_trial`; `BENCH_BUSY_PATTERN`).
  **Bug found on the way:** the first version called `pgrep -E`, which
  macOS `pgrep` doesn't have; it silently returned "not busy", so the first
  round of results was contaminated. They were deleted and remeasured.
* **Drift between runs.** Even on a quiet GPU, the *same* kernel measured
  203 µs in one run and 276 µs in another (thermals, charging, clock
  ramping). So sequential "before vs after" runs are not trustworthy here.
  `bench/ab.py` interleaves the candidates inside every round (shuffled
  order) and reports the per-round *ratio*. A change is accepted when the
  10th-90th percentile band of that ratio lies above 1 (it wins in at
  least 90% of rounds), not when one median beats another.
* **Dispatch latency.** An *empty* kernel in a dependency chain costs
  ~14 µs per link here (the empty kernel plus the 1-element glue op), i.e.
  ~6-7 µs per dependent dispatch. For small sub-1-bit layers that is most
  of the time, so dispatch count matters as much as the kernel body.
* **Peak read bandwidth** (bench/peak_bw.py, 16-byte loads, 1M threads,
  1 GiB buffer): **158.7 GB/s** (MLX `sum` reaches about the same). The
  200 GB/s spec is not reachable for reads.

## Variants

### naive (baseline kernel)
One thread per output row, walks its row's words bit by bit, `acc += bit ? -a : a`.
Two dispatches per path (stage 1 over r rows, stage 2 over out rows), plus
MLX adds for residual/bias and a cast. Correct, slow: lanes of a SIMD-group
read 32 different rows (uncoalesced) and every row recomputes `x*v2`.

### simd
A SIMD-group (or a power-of-two slice of one, for short rows) per row; lanes
read consecutive words (coalesced), `simd_sum`/`simd_shuffle_xor` reduction.
Still ~4 instructions per weight bit.

### lut (T-MAC style)
Back-of-envelope: at 200 GB/s a 1-bit kernel must consume 1.6 T weight
bits/s. The M1 Pro GPU retires ~2.6 T simple ops/s, so anything above ~1.5
ops per bit is compute-bound. Per-bit add/subtract is ~4 ops/bit.

So: precompute, for each group of 4 activations, the 16 possible signed
sums; then each 4-bit nibble of weights costs one table lookup + add
(~1 op/bit). Tables live in threadgroup memory (16 KB per 1024 inputs).
Bits are repacked once to word-major (`bitsT[w, row]`) so lane = row reads
are coalesced and all 32 lanes hit the same 16-entry table (no bank
conflicts). Split-K over the inner dim for parallelism, partials combined
with float atomics; residual path and bias fold into the same atomics.
Cost: 5 dispatches/layer (MLX zero-fills for atomic outputs, and a cast).

### lut2
Same math, 2 dispatches/layer: stage 1 writes per-chunk partials with plain
stores; stage 2 sums them while building its table, loops over r instead of
splitting it, adds residual + bias and writes fp16 directly.

### fused (single dispatch) -- rejected
Every threadgroup recomputes stage 1 into threadgroup memory, then does its
slice of stage 2. Lost everywhere in the interleaved A/B: 0.5-0.7x of `lut`
even on the smallest layers (0.6B k_proj @0.1 bpw), and 0.1x on big ones.
With one TG per 256 output rows, a 1024-row layer only launches 4
threadgroups, and each of them repeats all of stage 1. Saving one
dispatch (~7 µs) doesn't pay for that.

### lut2 / lut3 -- mixed, superseded
2 dispatches instead of 5 helped the small layers (0.6B k_proj @0.1 bpw:
42 µs vs 55 µs chained) but was 0.5-0.6x of `lut` on the big ones. Stage 2
no longer split its inner dimension, so a 4096-row output had only 32-64
threadgroups, each with a 16 KB table, which starved occupancy. `lut3`'s
rows-per-thread and smaller tables recovered part of it (0.85x of `lut`).

### lut4: weight prefetch -- accepted (big layers)
Diagnosis: `lut` stage 1 on Llama-3-8B down_proj @1 bpw took about the same
time with a hot and a cold cache (135 vs 144 µs), so DRAM wasn't the
limit. The loop loaded one word, waited, then did 8 lookups. `lut4`
issues all KC/32 word loads of a thread before the table build.
A/B vs `lut` (paired ratio, median [p10-p90]): with KC=512 or 256,
Llama down @1 bpw x1.34-1.43, @0.55 x1.53-1.61, 4B gate @0.55 x1.07-1.30.
Small layers: no change (~1.0). With KC=1024 (32 prefetched words) it was
*not* faster: register pressure.

Chunk-size sweep (KC1/KC2 in {256, 512, 1024}, TG in {128, 256}): no
setting won consistently across shapes (bands overlap). Kept 512/512.

### table build in registers -- accepted
Ablation of `lut4` stage 1 (Llama down @1 bpw, hot cache, chained µs):
real 136 | no float atomics 102 | no table build 35-43 | no weight loads 146.
The table build is the biggest cost. The original wrote T[0] and then
`T[p] = T[p & (p-1)] - 2 a_i`: 15 *dependent* threadgroup-memory
round trips per group, in every threadgroup. Building the 16 entries in
registers (fully unrolled, so indices are constants) and storing 4 float4s:
136-171 -> 100-124 µs. (Caveat: "no table build" also lets the compiler
drop the lookups of never-written memory, so 35 µs is not a real floor.)
Remaining cost isn't the activation loads or the stores (ablated each,
no single one explains it): it is instruction latency/throughput of the
lookup loop at the occupancy we get.

### more accumulators -- rejected
2 / 4 / 8 / 16 independent accumulators in the lookup loop: 100 / 148 /
150 / 267 µs. More registers -> fewer resident SIMD-groups -> worse.

### threadgroup size -- no effect
TG 128 / 256 / 512 / 1024 with KC 512: 122 / 118 / 118 / 130 µs.

### lut5
`lut4`'s prefetch + register tables, but split-K partials instead of
atomics (no zero-fill dispatches, no cast). 2 dispatches for most layers,
3 (+ a tiny reduce) when stage 2 must split r for occupancy. The gate caught
a bug in the first version (a 160-wide chunk halved to 80, not a multiple
of 32 -> wrong results for r=136); fixed by using power-of-two chunks.
