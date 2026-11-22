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
* **Other GPU jobs.** Another project on this machine periodically runs
  demucs on MPS; any overlap made numbers swing by 2x. `bench/quiet.sh`
  waits for an idle GPU and discards runs that overlapped another job.

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
