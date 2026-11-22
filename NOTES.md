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

### lut5 vs lut4 -> `auto` (accepted)
Interleaved A/B (`results/*_ab_lut5.json`, quick shapes, residual on/off):
lut5 beats lut4 by 1.1-1.5x on small layers (0.6B, 4B @0.1 bpw), and lut4
beats lut5 by 1.2-1.6x on large ones (Llama down/gate @0.55-1 bpw). lut5's
stage 2 re-sums up to ~28 stage-1 partials per table entry in *every*
threadgroup, which gets expensive when K and r are large. `auto` picks by
packed-weight size. A first full-table run with the switch at 1 MB showed
0.6-0.9 MB layers on lut5 running slower than bigger neighbours on lut4
(4B q_proj 0.69 MB: 125 µs vs 87 µs at 1.25 MB), so the switch moved to
512 KB and the table was re-run (`results/20261122-184955_final.json`).

## Final per-layer standing (auto vs baselines, paired, chained, cold cache)

* Llama-3-8B / Qwen3-4B-sized layers: 1.7-8x vs fp16, 1.3-2.9x vs q4,
  1.2-2.7x vs q2, 1.5-4.3x vs upstream's unpack-then-matmul.
* Qwen3-0.6B-sized layers: 0.9-1.9x vs fp16, ~0.9-1.6x vs q4/q2. Parity
  above 0.3 bpw: every implementation is dominated by fixed per-call cost.
* Achieved bandwidth: at most ~27% of the 158.7 GB/s peak (fp16 reaches ~94%,
  q4 ~74%). The win comes from reading far fewer bytes. The kernel is
  still bound by instruction latency, not by DRAM.

## End to end (Qwen3-0.6B in mlx-lm)

First run (`results/20261122-190230_e2e_qwen06.json`): fp16 117 tok/s,
q4 197, q2 204, LittleBit 86-101. Slower than fp16! Investigation:

* A pure chain of 196 distinct 1024x1024 layers (`y = f(y)`, no glue op)
  costs per layer: fp16 32 µs, q4 28 µs, LittleBit 60 µs. Of LittleBit's
  60 µs, ~20-26 µs is *CPU* graph-building: each layer makes 2-3
  `mx.fast.metal_kernel` calls, and every call builds template lists and
  looks up the specialised kernel. A q4 linear costs 2.2 µs to build.
* Fix: `mx.compile` the layer's forward (traced once per shape, replayed
  from C++). Build cost 19.5 -> 8.3 µs per call, with identical output.
  End to end: LittleBit 0.1 / 0.55 / 1.0 bpw = 138 / 111 / 125 tok/s
  (q4 179 in the same run). Now faster than fp16, still behind q4.
* What's left is structural: 2 GPU dispatches per layer (vs 1) at ~6-7 µs
  dependent-dispatch latency, x196 layers.

## Not done / next ideas (ranked)

1. **Group layers that share an input** (q/k/v, gate/up) into one launch:
   per block, 14 dispatches -> 8. The kernels already handle several
   "paths" sharing x (that is how the residual path works). Stage 2 would
   need per-path output segments instead of a sum. This is the main lever
   for small models.
2. **Lookup loop throughput** (big layers, ~27% of peak): try tables in
   registers via `simd_shuffle` (no threadgroup memory -> more occupancy),
   half-precision tables for fp16 activations (would need a looser gate),
   and fewer, longer-lived threadgroups.
3. **One dispatch for tiny layers.** The redundant-stage-1 approach
   (`fused`) failed; a grid-wide barrier via atomics is possible but relies
   on co-residency of threadgroups, so it is risky.
4. **Bigger-model end to end.** Per-layer data says the 4B/8B models are
   where LittleBit pays off; see README for what was measured.
