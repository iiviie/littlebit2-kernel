"""THE CORRECTNESS GATE: every kernel in KERNELS vs the float64 reference.

Any function registered in `littlebit_kernel.kernels.KERNELS` is picked up
here automatically.  A kernel is "correct" if, on the same packed bits and the
same fp16 scales, it matches `reference_forward` (float64 NumPy) within:

  * fp32 activations: rel_err <= 1e-4.  Everything is exact except float32
    rounding during accumulation; with in/r up to ~14k terms the worst case
    we see is ~1e-6, so 1e-4 leaves room for any summation order.
  * fp16 activations: rel_err <= 3e-3.  The kernel accumulates in fp32, but
    its output is rounded to fp16, which alone costs up to 2**-11 ~= 4.9e-4
    of each value (<= 4.9e-4 of max|y|).  The reference is fed the *same*
    fp16-rounded x, so input rounding is not counted against the kernel.
    3e-3 is ~6x the output rounding: loose enough for a kernel that keeps an
    intermediate (e.g. the rank-r vector t) in fp16, which adds another
    ~2**-11 per stage, but tight enough that a single wrong sign bit on an
    important input fails (see test_gate_detects_a_single_flipped_bit).

rel_err(y, ref) = max|y - ref| / max|ref|  (see conftest.py).

Run just the fast part with:  pytest -m "not slow"
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np
import pytest

from conftest import rel_err, to_np
from littlebit_kernel.format import LittleBitLayer, LittleBitPath, pack_signs
from littlebit_kernel.kernels import KERNELS, DeviceLayer, round_layer_scales
from littlebit_kernel.reference import reference_forward

TOL = {"float32": 1e-4, "float16": 3e-3}
MX_DTYPE = {"float32": mx.float32, "float16": mx.float16}
NP_DTYPE = {"float32": np.float32, "float16": np.float16}

# (in_features, out_features).  Odd sizes exercise partial last words (in or
# r not a multiple of 32) and grids that are not a multiple of the threadgroup.
SHAPES = [
    (1024, 1024),
    (1024, 3072),   # up_proj-like
    (3072, 1024),   # down_proj-like
    (2048, 1024),
    (1000, 777),
    (37, 45),       # tiny: r clamps to min_split_dim=8 at low bpw
    (33, 4097),     # in = 32 + 1, out just past a power of two
    pytest.param(4096, 14336, marks=pytest.mark.slow),  # Llama-8B up_proj
]
BPWS = [0.1, 0.3, 0.55, 1.0]
M_MAX = 3  # we test batch M in {1, 3}; one cached x of 3 rows serves both


# --------------------------------------------------------------------------
# cached test cases: building layers and float64 references is the slow part
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _layer(in_f, out_f, bpw, residual, bias=False):
    """Random layer with fp16-representable scales, so the kernel (which
    stores scales in fp16) and the float64 reference see identical numbers."""
    seed = int(in_f * 7 + out_f * 13 + bpw * 1000 + residual)
    L = LittleBitLayer.random(in_f, out_f, eff_bit=bpw, residual=residual, seed=seed, bias=bias)
    return round_layer_scales(L)


@lru_cache(maxsize=None)
def _device_layer(in_f, out_f, bpw, residual, bias=False):
    return DeviceLayer.from_layer(_layer(in_f, out_f, bpw, residual, bias))


@lru_cache(maxsize=None)
def _x_and_ref(in_f, out_f, bpw, residual, dtype):
    """(M_MAX, in) activations, already rounded to `dtype`, and the reference."""
    rng = np.random.default_rng(in_f + out_f)
    x = rng.standard_normal((M_MAX, in_f)).astype(NP_DTYPE[dtype])
    ref = reference_forward(x.astype(np.float64), _layer(in_f, out_f, bpw, residual))
    return x, ref


# --------------------------------------------------------------------------
# the main sweep
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 3])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("residual", [False, True], ids=["single", "residual"])
@pytest.mark.parametrize("bpw", BPWS)
@pytest.mark.parametrize("in_f,out_f", SHAPES)
@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_kernel_matches_reference(kernel, in_f, out_f, bpw, residual, dtype, M):
    x_np, ref = _x_and_ref(in_f, out_f, bpw, residual, dtype)
    x = mx.array(x_np[:M])
    y = KERNELS[kernel](x, _device_layer(in_f, out_f, bpw, residual))
    mx.eval(y)

    assert y.dtype == MX_DTYPE[dtype], f"output dtype {y.dtype}, input {dtype}"
    assert y.shape == (M, out_f)
    yn = to_np(y)
    assert np.isfinite(yn).all()
    err = rel_err(yn, ref[:M])
    assert err <= TOL[dtype], f"{kernel}: rel_err {err:.2e} > {TOL[dtype]:.0e}"


# --------------------------------------------------------------------------
# input shapes and bias
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_3d_input_and_bias(kernel, dtype, bias):
    """(batch, seq, in) input keeps its leading dims; bias is added once."""
    in_f, out_f = 1000, 777
    L = _layer(in_f, out_f, 0.55, True, bias)
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 3, in_f)).astype(NP_DTYPE[dtype])
    ref = reference_forward(x_np.astype(np.float64), L)

    y = KERNELS[kernel](mx.array(x_np), _device_layer(in_f, out_f, 0.55, True, bias))
    mx.eval(y)
    assert y.shape == (2, 3, out_f)
    assert y.dtype == MX_DTYPE[dtype]
    assert rel_err(y, ref) <= TOL[dtype]
    if bias:  # make sure the bias is really in there (it is O(1), like y)
        no_bias = reference_forward(x_np.astype(np.float64), _layer(in_f, out_f, 0.55, True, False))
        assert rel_err(y, no_bias) > 10 * TOL[dtype]


# --------------------------------------------------------------------------
# scales can be negative
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_negative_scales(kernel, dtype):
    """LittleBitLayer.random only makes positive scales, but real checkpoints
    don't: upstream gets (u1, u2) and (v1, v2) from an SVD whose singular
    vectors have an arbitrary sign, so e.g. u1 and u2 can both be entirely
    negative (we see this on Qwen3 layers).  Flip signs at random here so a
    kernel that assumes positive scales (abs/log tricks, sign folding) fails.
    """
    in_f, out_f = 1000, 777
    rng = np.random.default_rng(5)
    base = _layer(in_f, out_f, 0.55, True, True)
    flip = lambda a: (a * np.where(rng.random(a.shape) < 0.5, -1, 1)).astype(np.float32)
    paths = [LittleBitPath(p.U_packed, p.V_packed, flip(p.u1), flip(p.u2), flip(p.v1), flip(p.v2))
             for p in base.paths]
    L = LittleBitLayer(main=paths[0], residual=paths[1], bias=base.bias)  # still fp16-exact
    x_np = rng.standard_normal((3, in_f)).astype(NP_DTYPE[dtype])
    ref = reference_forward(x_np.astype(np.float64), L)
    y = KERNELS[kernel](mx.array(x_np), DeviceLayer.from_layer(L))
    mx.eval(y)
    assert rel_err(y, ref) <= TOL[dtype]


# --------------------------------------------------------------------------
# fp16 overflow: big activations must not overflow inside the accumulation
# --------------------------------------------------------------------------


def _all_plus_one_V_layer(in_f, out_f, r, seed=0):
    """A layer whose V is all +1, so stage 1 is a plain sum over `in`.

    With x ~ 1000 (all positive) the stage-1 sum is ~ in * 1000 = 4e6, far
    past fp16's max (65504).  mid = 1/in brings it back to ~1000, and u1 is
    chosen so the output is ~1000 too.  So the inputs and the final output
    fit in fp16 but the running sum does not: a kernel that accumulates in
    fp16 returns inf here; one that accumulates in fp32 is fine.
    """
    rng = np.random.default_rng(seed)
    main = LittleBitPath(
        U_packed=pack_signs(np.where(rng.standard_normal((out_f, r)) < 0, -1, 1)),
        V_packed=np.zeros((r, (in_f + 31) // 32), dtype=np.int32),  # all bits 0 = all +1
        u1=np.full(out_f, 1.0 / np.sqrt(r), np.float32),
        u2=np.ones(r, np.float32),
        v1=np.full(r, 1.0 / in_f, np.float32),
        v2=np.ones(in_f, np.float32),
    )
    return round_layer_scales(LittleBitLayer(main=main))


@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_fp16_large_activations_do_not_overflow(kernel):
    in_f, out_f, r = 4096, 1024, 512
    rng = np.random.default_rng(0)

    # Case 1: adversarial -- all-positive x and an all-+1 V row.
    L = _all_plus_one_V_layer(in_f, out_f, r)
    x_np = rng.uniform(500, 1500, (2, in_f)).astype(np.float16)
    ref = reference_forward(x_np.astype(np.float64), L)
    assert np.abs(ref).max() < 60000 and x_np.astype(np.float64).sum(axis=1).min() > 1e6
    y = KERNELS[kernel](mx.array(x_np), DeviceLayer.from_layer(L))
    mx.eval(y)
    yn = to_np(y)
    assert np.isfinite(yn).all(), "fp16 overflow (inf/nan) in the output"
    assert rel_err(yn, ref) <= TOL["float16"]

    # Case 2: ordinary random layer, x ~ N(0, 1000^2).
    L = _layer(in_f, out_f, 1.0, False)
    x_np = (1000 * rng.standard_normal((2, in_f))).astype(np.float16)
    ref = reference_forward(x_np.astype(np.float64), L)
    y = KERNELS[kernel](mx.array(x_np), _device_layer(in_f, out_f, 1.0, False))
    mx.eval(y)
    yn = to_np(y)
    assert np.isfinite(yn).all()
    assert rel_err(yn, ref) <= TOL["float16"]


# --------------------------------------------------------------------------
# sanity: the gate can actually fail
# --------------------------------------------------------------------------


def test_gate_detects_a_single_flipped_bit():
    """One flipped sign bit must push rel_err well over the fp16 tolerance,
    otherwise the tolerances above would be meaningless.

    We flip U[i, j] where i is the output with the largest |y| and j the
    latent with the largest |t|: that changes y[i] by 2 * u1[i] * t[j].
    """
    L = _layer(1000, 777, 0.55, False)
    p = L.main
    x = np.random.default_rng(0).standard_normal((1, 1000))
    ref = reference_forward(x, L)
    t = (x * p.v2) @ p.V_signs().T.astype(np.float64) * p.mid_scale  # (1, r)
    i, j = int(np.argmax(np.abs(ref))), int(np.argmax(np.abs(t)))
    U = p.U_packed.copy()
    U.view(np.uint32)[i, j // 32] ^= np.uint32(1 << (j % 32))
    bad = LittleBitLayer(main=LittleBitPath(U, p.V_packed, p.u1, p.u2, p.v1, p.v2))
    assert rel_err(reference_forward(x, bad), ref) > 3 * TOL["float16"]
