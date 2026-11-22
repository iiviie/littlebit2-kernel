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

RESULTS_PLACEHOLDER

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
