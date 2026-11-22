# littlebit2-kernel

Fused GPU kernels for running **LittleBit / LittleBit-2** sub-1-bit LLM
layers on **Apple Silicon**, written as MLX custom Metal kernels
(`mx.fast.metal_kernel`). Target: batch-1 decode (local chat).

LittleBit ([NeurIPS 2025, arXiv 2506.13771](https://arxiv.org/abs/2506.13771))
and LittleBit-2 ([ICML 2026, arXiv 2603.00042](https://arxiv.org/abs/2603.00042))
compress each linear layer to 0.1–1.0 bits per weight. The official code
([SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)) stores
the weights packed at 1 bit, but at inference it unpacks them to dense
±1 matrices and runs ordinary float matmuls, so none of the compression
turns into speed. The paper reports an 11.6x kernel speedup on CUDA, but
that kernel isn't public, and nothing existed for Mac. This repo is an open
implementation that works directly on the packed bits.

> Status: research code. Correctness is checked against the upstream
> implementation (see Tests). Speed numbers below are from one M1 Pro.

## 1. The format

A LittleBit linear layer with weight `W` (out × in) is replaced by

```
W  ≈  diag(u1) · sign(U) · diag(u2 ⊙ v1) · sign(V) · diag(v2)

  sign(U): out × r   ±1        u1: out     u2, v1: r     v2: in
  sign(V): r × in    ±1        (fp16 scale vectors)
```

so for an activation `x`

```
y = ((((x ⊙ v2) @ sign(V)ᵀ) ⊙ (v1 ⊙ u2)) @ sign(U)ᵀ) ⊙ u1
```

`r` (the latent rank, `split_dim`) is chosen from the bit budget:
`r ≈ (in·out·bpw − 16·(in+out)) / (in+out+16)`, rounded down to a multiple
of 8 (see `format.split_dim_for`). With `residual=True` a second,
independent path (`U_R, V_R, u1_R, …`) is added to the output; it gets half
the rank. LittleBit-2 changes only the *initialisation* (Joint-ITQ rotation
of the latent factors), so the inference format is identical.

**Packing** (verified bit-exact against upstream `binary_packer`): each row
of `sign(U)`/`sign(V)` is cut into 32-entry chunks → one int32 word each,
entry k → bit k (LSB first), `+1 → 0`, `−1 → 1`, last word padded with `+1`.
Checkpoint keys: `U_packed`, `U_shape`, `V_packed`, `V_shape`, `u1`, `u2`,
`v1`, `v2` (+ `_R` versions, + `bias`). `convert.layer_from_state_dict`
reads them.

**The identity the kernels use.** For a row of signs `s_k = 1 − 2·b_k`,

```
Σ_k s_k·a_k  =  Σ_k a_k  −  2·Σ_{k: b_k=1} a_k
```

A binary dot product needs no multiplications, only adds of activations
selected by bits.

## 2. Kernel design

Batch-1 decode is two binary matrix-vector products per path:

```
stage 1 (r rows,   in/32 words each):  t = (v1⊙u2) ⊙ (sign(V) @ (x ⊙ v2))
stage 2 (out rows, r/32 words each):   y = u1 ⊙ (sign(U) @ t)   (+ residual path, + bias)
```

Every variant lives in its own file in
[`src/littlebit_kernel/kernels/`](src/littlebit_kernel/kernels), with a
long comment explaining it. They are listed here in the order they were
built; [NOTES.md](NOTES.md) has the measurements behind each step.

| kernel | idea |
|---|---|
| `naive` | one thread per row, add/subtract per bit. The correctness baseline. |
| `simd` | a SIMD-group (32 lanes) per row, coalesced word reads, `simd_sum` |
| `lut` | **T-MAC-style lookup tables**: for every group of 4 activations precompute the 16 possible signed sums, so each 4-bit nibble of weights is one table lookup + add instead of 4 bit-ops. Bits are repacked word-major so 32 lanes = 32 rows read one 128-byte line and share one table (no bank conflicts). Split-K with float atomics; the residual path and bias fold into the same atomics. |
| `lut2` | `lut` in 2 dispatches instead of 5 (no zero-fills, no atomics, no cast) |
| `lut3` | `lut2` + several rows per thread, tunable table chunk |
| `lut4` | `lut` + **weight prefetch**: each thread issues all its word loads before building the table, so DRAM latency overlaps the table build |
| `fused` | whole layer in **one** dispatch: every threadgroup recomputes stage 1 (redundantly) into threadgroup memory |

Why lookup tables: at ~160 GB/s a 1-bit kernel has to consume ~1.3 T
weight bits per second. Per-bit add/subtract costs ~4 GPU instructions per
bit, more than the M1 Pro GPU can issue, so the bit-by-bit kernels are
compute-bound long before they are memory-bound. With tables it is ~1
instruction per bit.

## 3. Results

All numbers are from one Apple M1 Pro (16-core GPU, 16 GB), MLX 0.32.4,
fp16 activations, batch 1. The measured peak read bandwidth is
**158.7 GB/s** (`bench/peak_bw.py`; the 200 GB/s spec isn't reachable).
LittleBit = the `auto` kernel (`lut5` below 512 KB of packed weights, `lut4`
above). Layers use synthetic random bits at the exact rank the bit budget
gives (speed doesn't depend on the bit values).

### Per layer: LittleBit vs MLX fp16, 4-bit, 2-bit, and upstream's unpack-then-matmul

Each layer's time is the cost per call inside a *dependent chain* (each
call's input depends on the previous output, as in decode), with weights
rotated over ≥128 MB of copies so they come from DRAM, not cache. The
kernel and the four baselines are timed **interleaved in the same rounds**.
Speedups are the median of per-round ratios, and **bold** means LittleBit
was faster in ≥ 90% of rounds. (See bench/README.md for why: on this
machine the same kernel can drift 30% between runs.)

| layer (in×out) | bpw | LittleBit µs | fp16 µs | q4 µs | q2 µs | vs fp16 | vs q4 | vs q2 | vs unpack+matmul |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b k_proj/v_proj (1024×1024) | 0.1 | 39 | 50 | 48 | 48 | **1.3x** | **1.2x** | **1.2x** | **1.8x** |
| qwen3-0.6b k_proj/v_proj (1024×1024) | 0.55 | 53 | 49 | 47 | 46 | 1.0x | 0.9x | 0.9x | **1.5x** |
| qwen3-0.6b k_proj/v_proj (1024×1024) | 1 | 52 | 50 | 48 | 47 | 0.9x | 0.9x | 0.9x | **1.4x** |
| qwen3-0.6b gate/up_proj (1024×3072) | 0.1 | 42 | 78 | 69 | 66 | **1.9x** | **1.6x** | **1.6x** | **1.8x** |
| qwen3-0.6b gate/up_proj (1024×3072) | 0.55 | 58 | 78 | 67 | 67 | **1.4x** | 1.2x | 1.2x | **1.5x** |
| qwen3-0.6b gate/up_proj (1024×3072) | 1 | 72 | 80 | 71 | 69 | 1.2x | 1.0x | 0.9x | **1.3x** |
| qwen3-4b q_proj (2560×4096) | 0.1 | 59 | 170 | 109 | 103 | **2.8x** | **2.0x** | **1.8x** | **1.5x** |
| qwen3-4b q_proj (2560×4096) | 0.55 | 81 | 166 | 118 | 103 | **2.0x** | **1.4x** | **1.3x** | **1.7x** |
| qwen3-4b q_proj (2560×4096) | 1 | 97 | 166 | 97 | 101 | **1.7x** | 1.0x | 1.1x | **1.9x** |
| qwen3-4b down_proj (9728×2560) | 0.1 | 71 | 342 | 174 | 176 | **4.9x** | **2.2x** | **2.5x** | **1.6x** |
| qwen3-4b down_proj (9728×2560) | 0.55 | 112 | 336 | 160 | 168 | **3.1x** | **1.6x** | **1.5x** | **2.1x** |
| qwen3-4b down_proj (9728×2560) | 1 | 115 | 331 | 159 | 171 | **2.9x** | **1.5x** | **1.6x** | **3.2x** |
| llama3-8b q_proj/o_proj (4096×4096) | 0.1 | 69 | 242 | 153 | 134 | **3.5x** | **2.1x** | **1.9x** | **1.6x** |
| llama3-8b q_proj/o_proj (4096×4096) | 0.55 | 96 | 259 | 146 | 144 | **2.7x** | **1.4x** | **1.5x** | **1.9x** |
| llama3-8b q_proj/o_proj (4096×4096) | 1 | 114 | 252 | 143 | 126 | **2.3x** | 1.3x | 1.2x | **2.4x** |
| llama3-8b down_proj (14336×4096) | 0.1 | 98 | 783 | 282 | 270 | **8.1x** | **2.9x** | **2.7x** | **1.7x** |
| llama3-8b down_proj (14336×4096) | 0.55 | 131 | 775 | 291 | 273 | **6.0x** | **2.2x** | **2.0x** | **3.5x** |
| llama3-8b down_proj (14336×4096) | 1 | 175 | 785 | 283 | 250 | **4.5x** | **1.7x** | **1.4x** | **4.3x** |

The full table (all 7 projections × 3 models × bpw 0.1/0.3/0.55/1.0, with
GB/s and % of peak) is below, from `bench/results/20261122-184955_final.json`.

<details><summary>Full per-layer table</summary>

| layer | in×out | bpw | r | auto µs | MB | GB/s | % peak | vs fp16 | vs q4 | vs q2 | vs unpack_matmul | fp16 µs | q4 µs | q2 µs | unpack_matmul µs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-0.6b q_proj | 1024×2048 | 0.1 | 48 | 45 | 0.03 | 1 | 1% | 1.41x | **1.40x** | 1.38x | 1.82x | 59 | 57 | 62 | 73 |
| qwen3-0.6b q_proj | 1024×2048 | 0.3 | 184 | 50 | 0.08 | 2 | 1% | **1.31x** | 1.01x | 1.11x | **1.72x** | 63 | 55 | 55 | 81 |
| qwen3-0.6b q_proj | 1024×2048 | 0.55 | 352 | 56 | 0.14 | 3 | 2% | 1.15x | 0.95x | 1.04x | **1.58x** | 61 | 56 | 58 | 83 |
| qwen3-0.6b q_proj | 1024×2048 | 1 | 656 | 62 | 0.25 | 4 | 3% | 1.02x | 0.93x | 0.88x | 1.45x | 64 | 59 | 57 | 94 |
| qwen3-0.6b k_proj/v_proj | 1024×1024 | 0.1 | 32 | 39 | 0.01 | 1 | 0% | **1.28x** | **1.19x** | **1.19x** | **1.77x** | 50 | 48 | 48 | 72 |
| qwen3-0.6b k_proj/v_proj | 1024×1024 | 0.3 | 136 | 50 | 0.04 | 1 | 1% | 1.00x | 0.92x | 0.96x | **1.46x** | 50 | 48 | 47 | 76 |
| qwen3-0.6b k_proj/v_proj | 1024×1024 | 0.55 | 256 | 53 | 0.07 | 1 | 1% | 0.98x | 0.90x | 0.92x | **1.47x** | 49 | 47 | 46 | 72 |
| qwen3-0.6b k_proj/v_proj | 1024×1024 | 1 | 488 | 52 | 0.13 | 3 | 2% | 0.93x | 0.95x | 0.90x | **1.40x** | 50 | 48 | 47 | 71 |
| qwen3-0.6b o_proj | 2048×1024 | 0.1 | 48 | 42 | 0.03 | 1 | 1% | **1.54x** | **1.36x** | **1.34x** | **1.74x** | 65 | 60 | 58 | 76 |
| qwen3-0.6b o_proj | 2048×1024 | 0.3 | 184 | 53 | 0.07 | 2 | 1% | **1.22x** | 1.13x | 1.08x | **1.40x** | 65 | 60 | 58 | 83 |
| qwen3-0.6b o_proj | 2048×1024 | 0.55 | 352 | 58 | 0.14 | 3 | 2% | **1.14x** | 1.07x | 1.01x | **1.43x** | 66 | 64 | 62 | 87 |
| qwen3-0.6b o_proj | 2048×1024 | 1 | 656 | 67 | 0.25 | 4 | 3% | 0.97x | 0.89x | 0.81x | **1.36x** | 65 | 59 | 57 | 89 |
| qwen3-0.6b gate_proj/up_proj | 1024×3072 | 0.1 | 56 | 42 | 0.04 | 1 | 1% | **1.86x** | **1.63x** | **1.58x** | **1.76x** | 78 | 69 | 66 | 80 |
| qwen3-0.6b gate_proj/up_proj | 1024×3072 | 0.3 | 208 | 56 | 0.12 | 2 | 2% | **1.35x** | **1.20x** | **1.23x** | **1.45x** | 74 | 70 | 70 | 84 |
| qwen3-0.6b gate_proj/up_proj | 1024×3072 | 0.55 | 400 | 58 | 0.21 | 4 | 3% | **1.40x** | 1.17x | 1.15x | **1.54x** | 78 | 67 | 67 | 95 |
| qwen3-0.6b gate_proj/up_proj | 1024×3072 | 1 | 744 | 72 | 0.38 | 6 | 4% | 1.15x | 0.96x | 0.92x | **1.31x** | 80 | 71 | 69 | 111 |
| qwen3-0.6b down_proj | 3072×1024 | 0.1 | 56 | 49 | 0.04 | 1 | 1% | **1.70x** | **1.40x** | **1.48x** | **1.66x** | 79 | 68 | 68 | 82 |
| qwen3-0.6b down_proj | 3072×1024 | 0.3 | 208 | 57 | 0.11 | 2 | 1% | **1.44x** | 1.26x | 1.17x | **1.52x** | 79 | 71 | 69 | 92 |
| qwen3-0.6b down_proj | 3072×1024 | 0.55 | 400 | 65 | 0.21 | 4 | 2% | **1.17x** | 1.01x | 1.07x | **1.41x** | 79 | 64 | 68 | 98 |
| qwen3-0.6b down_proj | 3072×1024 | 1 | 744 | 68 | 0.38 | 6 | 4% | **1.16x** | 0.94x | 1.07x | **1.46x** | 79 | 61 | 67 | 101 |
| qwen3-4b q_proj | 2560×4096 | 0.1 | 136 | 59 | 0.13 | 3 | 2% | **2.84x** | **2.01x** | **1.77x** | **1.50x** | 170 | 109 | 103 | 92 |
| qwen3-4b q_proj | 2560×4096 | 0.3 | 448 | 86 | 0.37 | 5 | 3% | **1.95x** | **1.42x** | **1.15x** | **1.26x** | 169 | 118 | 95 | 110 |
| qwen3-4b q_proj | 2560×4096 | 0.55 | 848 | 81 | 0.69 | 9 | 6% | **2.04x** | **1.44x** | **1.25x** | **1.74x** | 166 | 118 | 103 | 140 |
| qwen3-4b q_proj | 2560×4096 | 1 | 1552 | 97 | 1.25 | 14 | 9% | **1.72x** | 1.02x | 1.05x | **1.95x** | 166 | 97 | 101 | 193 |
| qwen3-4b k_proj/v_proj | 2560×1024 | 0.1 | 56 | 49 | 0.03 | 1 | 1% | **1.63x** | **1.60x** | **1.51x** | **2.09x** | 73 | 65 | 63 | 90 |
| qwen3-4b k_proj/v_proj | 2560×1024 | 0.3 | 200 | 62 | 0.10 | 2 | 1% | **1.26x** | 1.16x | 1.16x | **1.36x** | 73 | 64 | 68 | 88 |
| qwen3-4b k_proj/v_proj | 2560×1024 | 0.55 | 384 | 66 | 0.17 | 3 | 2% | **1.11x** | 0.98x | 0.94x | **1.37x** | 72 | 65 | 63 | 91 |
| qwen3-4b k_proj/v_proj | 2560×1024 | 1 | 712 | 77 | 0.32 | 5 | 3% | 0.92x | 0.81x | 0.81x | **1.23x** | 73 | 64 | 62 | 94 |
| qwen3-4b o_proj | 4096×2560 | 0.1 | 136 | 64 | 0.13 | 3 | 2% | **2.56x** | **1.50x** | **1.73x** | 1.53x | 172 | 107 | 110 | 100 |
| qwen3-4b o_proj | 4096×2560 | 0.3 | 448 | 71 | 0.37 | 6 | 4% | **2.35x** | **1.41x** | **1.47x** | **1.57x** | 170 | 102 | 110 | 116 |
| qwen3-4b o_proj | 4096×2560 | 0.55 | 848 | 84 | 0.69 | 9 | 6% | **2.06x** | 1.31x | 1.21x | **1.58x** | 177 | 112 | 110 | 136 |
| qwen3-4b o_proj | 4096×2560 | 1 | 1552 | 99 | 1.25 | 14 | 9% | **1.78x** | **1.29x** | 1.16x | **2.09x** | 171 | 123 | 109 | 200 |
| qwen3-4b gate_proj/up_proj | 2560×9728 | 0.1 | 184 | 68 | 0.30 | 5 | 3% | **4.87x** | **2.74x** | **2.42x** | **1.76x** | 332 | 185 | 168 | 112 |
| qwen3-4b gate_proj/up_proj | 2560×9728 | 0.3 | 584 | 109 | 0.91 | 9 | 6% | **3.19x** | **1.55x** | **1.63x** | **1.41x** | 327 | 166 | 175 | 156 |
| qwen3-4b gate_proj/up_proj | 2560×9728 | 0.55 | 1096 | 111 | 1.66 | 16 | 10% | **3.04x** | **1.39x** | **1.39x** | **2.10x** | 335 | 172 | 155 | 230 |
| qwen3-4b gate_proj/up_proj | 2560×9728 | 1 | 2008 | 127 | 2.98 | 25 | 16% | **2.85x** | **1.54x** | **1.32x** | **2.97x** | 336 | 185 | 183 | 347 |
| qwen3-4b down_proj | 9728×2560 | 0.1 | 184 | 71 | 0.30 | 5 | 3% | **4.92x** | **2.22x** | **2.53x** | **1.61x** | 342 | 174 | 176 | 115 |
| qwen3-4b down_proj | 9728×2560 | 0.3 | 584 | 88 | 0.89 | 11 | 7% | **3.91x** | **1.84x** | **1.97x** | **1.89x** | 345 | 157 | 166 | 167 |
| qwen3-4b down_proj | 9728×2560 | 0.55 | 1096 | 112 | 1.64 | 16 | 10% | **3.12x** | **1.58x** | **1.51x** | **2.10x** | 336 | 160 | 168 | 231 |
| qwen3-4b down_proj | 9728×2560 | 1 | 2008 | 115 | 2.97 | 28 | 17% | **2.87x** | **1.53x** | **1.57x** | **3.22x** | 331 | 159 | 171 | 358 |
| llama3-8b q_proj/o_proj | 4096×4096 | 0.1 | 184 | 69 | 0.20 | 3 | 2% | **3.54x** | **2.11x** | **1.89x** | **1.65x** | 242 | 153 | 134 | 110 |
| llama3-8b q_proj/o_proj | 4096×4096 | 0.3 | 592 | 97 | 0.60 | 7 | 4% | **2.76x** | 1.38x | **1.32x** | 1.35x | 252 | 142 | 147 | 130 |
| llama3-8b q_proj/o_proj | 4096×4096 | 0.55 | 1104 | 96 | 1.10 | 12 | 8% | **2.68x** | **1.37x** | **1.45x** | **1.93x** | 259 | 146 | 144 | 188 |
| llama3-8b q_proj/o_proj | 4096×4096 | 1 | 2024 | 114 | 2.01 | 19 | 12% | **2.26x** | 1.35x | 1.18x | **2.42x** | 252 | 143 | 126 | 262 |
| llama3-8b k_proj/v_proj | 4096×1024 | 0.1 | 64 | 51 | 0.05 | 1 | 1% | **1.66x** | 1.36x | **1.53x** | **1.60x** | 93 | 78 | 79 | 81 |
| llama3-8b k_proj/v_proj | 4096×1024 | 0.3 | 224 | 67 | 0.15 | 3 | 2% | **1.50x** | 1.17x | 1.16x | **1.65x** | 92 | 80 | 79 | 111 |
| llama3-8b k_proj/v_proj | 4096×1024 | 0.55 | 432 | 72 | 0.28 | 4 | 3% | **1.29x** | 1.11x | 1.19x | **1.41x** | 93 | 79 | 81 | 95 |
| llama3-8b k_proj/v_proj¹ | 4096×1024 | 1 | 800 | 302 | 0.50 | 2 | 1% | 0.35x | 0.28x | 0.32x | 0.42x | 94 | 85 | 88 | 120 |
| llama3-8b gate_proj/up_proj | 4096×14336 | 0.1 | 296 | 107 | 0.73 | 8 | 5% | **6.75x** | **2.62x** | **2.69x** | **1.31x** | 725 | 284 | 284 | 144 |
| llama3-8b gate_proj/up_proj | 4096×14336 | 0.3 | 936 | 115 | 2.13 | 20 | 13% | **6.25x** | **2.50x** | **2.47x** | **2.47x** | 715 | 298 | 284 | 279 |
| llama3-8b gate_proj/up_proj | 4096×14336 | 0.55 | 1728 | 141 | 3.84 | 29 | 18% | **5.21x** | **1.84x** | **1.80x** | **3.23x** | 732 | 271 | 272 | 446 |
| llama3-8b gate_proj/up_proj | 4096×14336 | 1 | 3160 | 190 | 7.00 | 39 | 25% | **4.15x** | **1.47x** | **1.46x** | **4.17x** | 751 | 297 | 293 | 773 |
| llama3-8b down_proj | 14336×4096 | 0.1 | 296 | 98 | 0.70 | 8 | 5% | **8.11x** | **2.89x** | **2.74x** | **1.73x** | 783 | 282 | 270 | 169 |
| llama3-8b down_proj | 14336×4096 | 0.3 | 936 | 108 | 2.11 | 21 | 13% | **7.29x** | **2.75x** | **2.58x** | **2.65x** | 776 | 294 | 273 | 284 |
| llama3-8b down_proj | 14336×4096 | 0.55 | 1728 | 131 | 3.84 | 31 | 20% | **5.97x** | **2.25x** | **1.95x** | **3.54x** | 775 | 291 | 273 | 452 |
| llama3-8b down_proj | 14336×4096 | 1 | 3160 | 175 | 6.99 | 42 | 27% | **4.49x** | **1.71x** | **1.38x** | **4.31x** | 785 | 283 | 250 | 771 |

Bold = the kernel was faster in at least 90% of rounds (paired). Times are per call in a dependent chain (decode-like), cold cache.

¹ One outlier: Llama-3-8B k/v_proj at 1 bpw shows 302 µs. A targeted
re-run (`bench/results/20261122-185034_ab_llama_kv.json`) measured the
same kernel at 58-72 µs, so that row was hit by interference. It is kept
in the table as measured.

</details>

**Where it wins.** On 4B- and 8B-sized layers LittleBit is **2-8x faster
than fp16, 1.3-2.9x faster than MLX 4-bit and 1.2-2.7x faster than 2-bit**,
and 1.5-4.3x faster than upstream's own inference path (unpack to dense ±1
fp16, two matmuls). The speedup is biggest where the layer is largest and
the bit rate lowest, i.e. where the 1-bit format reads far fewer bytes than
anything else.

**Where it doesn't.** On Qwen3-0.6B-sized layers (1024-wide) everything,
fp16 included, takes 40-80 µs per call, and LittleBit is roughly at parity
with 4-/2-bit (0.9-1.2x) above 0.3 bpw. At this size a call is dominated by
fixed costs: each dependent GPU dispatch costs ~6-7 µs here, and LittleBit
needs 2 (vs 1 for a quantized matmul). The weights are only 10-400 KB, so
there is little memory traffic left to save.

**Honest bandwidth numbers.** The kernel reaches only **up to ~27% of peak
bandwidth** (42 GB/s on Llama down_proj at 1 bpw). fp16 gets ~94% and q4
~74% on big layers. LittleBit still wins because it reads 4-16x fewer
bytes, not because it reads them efficiently. The time goes to the lookup
loop's instruction latency at the occupancy we reach, plus the table builds
and dispatch overhead (see [NOTES.md](NOTES.md)). There is headroom left.

### End to end: Qwen3-0.6B in mlx-lm

All 196 q/k/v/o/gate/up/down projections are replaced by `LittleBitLinear`.
The embeddings and the tied LM head stay fp16. Greedy decode of 128 tokens,
median of 5 runs, `bench/e2e_qwen.py`. LittleBit layers use SVD init
without QAT (the text is gibberish, but the shapes and the speed are those of
a real checkpoint).

| variant | linear weights | decode tok/s | vs fp16 | vs q4 | prompt tok/s | peak MLX memory |
|---|---|---|---|---|---|---|
| fp16 | 840 MB | 118.5 | 1.00x | 0.60x | 636 | 1226 MB |
| MLX 4-bit (q4) | 236 MB | 196.6 | 1.66x | 1.00x | 690 | 812 MB |
| MLX 2-bit (q2) | 131 MB | 209.4 | 1.77x | 1.07x | 655 | 705 MB |
| LittleBit 0.1 bpw | 5.3 MB | 143.9 | 1.21x | 0.73x | 930 | 355 MB |
| LittleBit 0.1 bpw, grouped | 5.6 MB | **166.9** | 1.41x | 0.85x | 959 | 371 MB |
| LittleBit 0.3 bpw, grouped | 16.6 MB | 155.4 | 1.31x | 0.79x | 708 | 457 MB |
| LittleBit 0.55 bpw | 28.8 MB | 127.4 | 1.08x | 0.65x | 583 | 495 MB |
| LittleBit 0.55 bpw, grouped | 30.2 MB | 159.4 | 1.35x | 0.81x | 615 | 515 MB |
| LittleBit 1.0 bpw | 53.0 MB | 125.8 | 1.06x | 0.64x | 484 | 563 MB |
| LittleBit 1.0 bpw, grouped | 55.2 MB | 153.5 | 1.30x | 0.78x | 526 | 570 MB |

(`bench/results/20261122-192303_e2e_qwen06_final.json`, all variants in one
run. "Grouped" = q/k/v and gate/up, which read the same input, run as one
launch pair per group (`nn.group_shared_inputs`). That is 8 dispatches per
block instead of 14; the smaller ranks are zero-padded to the largest, hence
the slightly bigger weights. The q2 model itself produces garbage text too,
and quality isn't the point here. "Peak MLX memory" includes the ~300 MB
fp16 embedding/LM head and the KV cache.)

**At 0.6B, LittleBit end to end is faster than fp16 (1.3-1.4x grouped)
but still slower than MLX 4-bit (~0.8x)**, while holding the linear layers
in 5-55 MB instead of 236 MB. The per-layer kernels are at parity with q4
at this size. The gap comes from fixed per-call costs that one quantized
matmul doesn't pay: 2 GPU dispatches per layer or group instead of 1, and
~8 µs of Python/MLX graph-building per call (20 µs before the forward was
wrapped in `mx.compile`; q4 needs ~2 µs). Grouping same-input layers
recovered 10-25% of decode speed.


## 4. Running it

Requires macOS on Apple Silicon and [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync                                   # Python 3.12 env with mlx, mlx-lm, torch, pytest
git clone --depth 1 https://github.com/SamsungLabs/LittleBit third_party/LittleBit   # test oracle (optional, gitignored)

uv run pytest -q                          # full correctness suite (~1 min)
uv run pytest -q -m "not slow"            # skip the largest shapes
```

Benchmarks (one at a time, on an otherwise idle GPU; see
[bench/README.md](bench/README.md) for the method):

```sh
uv run python bench/peak_bw.py                              # measure peak read bandwidth
uv run python bench/ab.py --impls lut,lut4 --quick          # interleaved A/B of kernels
uv run python bench/bench.py --kernels lut4 --label run1    # full table vs fp16/q4/q2/unpack
uv run python bench/compare.py bench/results/<run>.json --markdown
uv run python bench/e2e_qwen.py                             # Qwen3-0.6B tokens/s in mlx-lm
```

Using it from code:

```python
import mlx.core as mx
from littlebit_kernel.convert import layer_from_state_dict   # from an upstream checkpoint
from littlebit_kernel.init import littlebit_from_dense       # or: SVD (+ITQ) init from a dense W
from littlebit_kernel.nn import LittleBitLinear, littlebitify

layer = littlebit_from_dense(W, eff_bit=0.55, use_itq=True)   # W: (out, in) numpy
lin = LittleBitLinear(layer)                                   # an mlx.nn.Module
y = lin(mx.random.normal((1, W.shape[1])).astype(mx.float16))
```

## 5. Tests

`tests/test_kernels.py` is the gate every kernel must pass: all registered
kernels × shapes including odd sizes (37×45, 1000×777, 33×4097) × bpw
{0.1, 0.3, 0.55, 1.0} × residual on/off × fp16/fp32 activations × batch
1 and 3, against a float64 NumPy reference, with tolerances 1e-4 (fp32)
and 3e-3 (fp16), relative to the output's max. A test that flips one sign
bit checks that the tolerance actually catches errors.
`tests/test_upstream_oracle.py` builds layers with **upstream's own**
`LittleBitLinear` (SVD init, with and without ITQ, residual on/off) from
real Qwen3-0.6B weights. It checks our converter, our reference and every
kernel against upstream's `forward`. `tests/test_format.py` checks packing
and the rank formula against upstream bit for bit.

## 6. Layout

```
src/littlebit_kernel/
  format.py      packing, rank/bit-budget math, LittleBitLayer containers
  reference.py   float64 NumPy reference + MLX unpack-then-matmul
  convert.py     upstream packed state_dict -> LittleBitLayer
  init.py        SVD (+ Joint-ITQ) initialisation from a dense weight
  kernels/       one file per kernel variant, registry in __init__.py
  nn.py          MLX LittleBitLinear module + mlx-lm swap helper
tests/           correctness gate + upstream oracle
bench/           peak bandwidth, A/B tool, full benchmark, compare, e2e
NOTES.md         optimisation log, including what didn't work
```

## License note

The upstream LittleBit code is CC BY-NC 4.0. This repository contains
none of it: the format was reimplemented from its description, and the
upstream repo is only *imported at test time* from an untracked
`third_party/` clone, as a reference to check against.
