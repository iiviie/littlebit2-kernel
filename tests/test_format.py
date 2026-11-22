"""Our on-disk format helpers (format.py) vs the upstream implementation.

These are the foundations everything else stands on: if packing or the rank
arithmetic differ by a single bit, a real checkpoint would load as garbage.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from littlebit_kernel.format import (
    LittleBitLayer, effective_bits, pack_signs, sign_pm1, split_dim_for, unpack_signs,
)

COLS = [1, 31, 32, 33, 64, 100, 1000]


# --------------------------------------------------------------------------
# packing: self-contained checks (no upstream needed)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cols", COLS)
def test_pack_roundtrip(cols):
    rng = np.random.default_rng(cols)
    s = np.where(rng.standard_normal((5, cols)) < 0, -1, 1).astype(np.int8)
    p = pack_signs(s)
    assert p.dtype == np.int32
    assert p.shape == (5, (cols + 31) // 32)
    np.testing.assert_array_equal(unpack_signs(p, cols), s)


def test_pack_bit_layout_by_hand():
    """LSB first, -1 -> bit 1, +1 -> bit 0, padding is +1 (bit 0)."""
    s = np.ones((1, 33), dtype=np.int8)
    s[0, 0] = -1   # bit 0 of word 0
    s[0, 31] = -1  # bit 31 of word 0 -> the int32 is negative
    s[0, 32] = -1  # bit 0 of word 1; bits 1..31 of word 1 are padding
    p = pack_signs(s)
    assert p.view(np.uint32).tolist() == [[(1 << 0) | (1 << 31), 1]]
    assert p[0, 0] < 0  # stored as signed int32, like upstream


def test_padding_bits_are_zero():
    """All -1 row of 33 cols: word 1 must be exactly 1 (31 pad bits = +1 = 0)."""
    p = pack_signs(-np.ones((2, 33), dtype=np.int8)).view(np.uint32)
    assert (p[:, 0] == 0xFFFFFFFF).all()
    assert (p[:, 1] == 1).all()


def test_sign_zero_is_plus_one():
    a = np.array([0.0, -0.0, 1e-30, -1e-30, 3.0, -3.0, np.inf, -np.inf], dtype=np.float32)
    np.testing.assert_array_equal(sign_pm1(a), [1, 1, 1, -1, 1, -1, 1, -1])
    # pack_signs applies the same convention to raw zeros
    z = np.array([[0.0, -0.0, -1.0, 1.0]])
    np.testing.assert_array_equal(unpack_signs(pack_signs(z), 4), [[1, 1, -1, 1]])


# --------------------------------------------------------------------------
# packing: bit-exact against upstream
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rows,cols", [(1, 1), (7, 31), (3, 32), (5, 33), (64, 100), (9, 1000), (2, 4097)])
def test_pack_matches_upstream(upstream, rows, cols):
    import torch

    rng = np.random.default_rng(rows * 10007 + cols)
    s = np.where(rng.standard_normal((rows, cols)) < 0, -1, 1).astype(np.int8)
    ours = pack_signs(s)
    theirs = upstream.binary_packer(torch.from_numpy(s)).numpy()
    assert theirs.dtype == np.int32 and ours.dtype == np.int32
    np.testing.assert_array_equal(ours, theirs)

    # and each side unpacks the other's words
    np.testing.assert_array_equal(unpack_signs(theirs, cols), s)
    back = upstream.binary_unpacker(torch.from_numpy(ours), (rows, cols)).numpy()
    np.testing.assert_array_equal(back, s)


def test_sign_matches_upstream_binariser(upstream):
    """Our sign_pm1 == upstream STEBinary forward, including +-0 and tiny values."""
    import torch

    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000).astype(np.float32)
    a[:8] = [0.0, -0.0, 1e-38, -1e-38, 1e-45, -1e-45, np.inf, -np.inf]
    theirs = upstream.STEBinary(torch.from_numpy(a.copy())).numpy()
    np.testing.assert_array_equal(sign_pm1(a), theirs.astype(np.int8))


# --------------------------------------------------------------------------
# rank / bit-budget arithmetic vs upstream's static methods
# --------------------------------------------------------------------------

DIMS = [37, 128, 1000, 1024, 2048, 3072, 4096, 14336]
EFF_BITS = [0.1, 0.3, 0.55, 0.8, 1.0]
RATIOS = [0.5, 1.0, 1.3]


def _upstream_split_dim(up, a, b, eff, residual, ratio, min_split_dim=8):
    """Exactly the sequence upstream's __quant_convert__ runs."""
    est = up.LittleBitLinear._estimate_split_dim(a, b, eff, residual)
    if est:
        est *= ratio
    return up.LittleBitLinear._finalize_split_dim(est, 1024, min_split_dim)


def test_split_dim_and_eff_bits_match_upstream(upstream):
    n = 0
    for a, b, eff, residual, ratio in itertools.product(DIMS, DIMS, EFF_BITS, [False, True], RATIOS):
        r_up = _upstream_split_dim(upstream, a, b, eff, residual, ratio)
        r_ours = split_dim_for(a, b, eff, residual, ratio_factor=ratio)
        assert r_ours == r_up, (a, b, eff, residual, ratio, r_ours, r_up)
        assert r_ours % 8 == 0 and r_ours >= 8
        e_up = upstream.LittleBitLinear._compute_eff_bits(a, b, r_ours, residual)
        assert effective_bits(a, b, r_ours, residual) == pytest.approx(e_up, rel=1e-12)
        n += 1
    assert n == len(DIMS) ** 2 * len(EFF_BITS) * 2 * len(RATIOS)


def test_min_split_dim_respected(upstream):
    """Tiny layers at low bpw have a negative raw estimate -> clamp to min."""
    for msd in (8, 16, 32):
        r_up = _upstream_split_dim(upstream, 37, 45, 0.1, True, 1.0, min_split_dim=msd)
        assert split_dim_for(37, 45, 0.1, True, min_split_dim=msd) == r_up == msd


def test_effective_bits_close_to_target():
    """Rounding r down to a multiple of 8 never overshoots the budget, and
    undershoots by less than the cost of 8 more ranks."""
    a = b = 4096
    for eff, residual in itertools.product(EFF_BITS, [False, True]):
        r = split_dim_for(a, b, eff, residual)
        got = effective_bits(a, b, r, residual)
        paths = 2 if residual else 1
        one_step = 8 * paths * (a + b + 16) / (a * b)  # bits/weight of 8 ranks
        assert eff - one_step < got <= eff + 1e-12


def test_random_layer_shapes():
    L = LittleBitLayer.random(1000, 777, eff_bit=0.55, residual=True, bias=True)
    r = split_dim_for(1000, 777, 0.55, True)
    for p in L.paths:
        assert p.U_packed.shape == (777, (r + 31) // 32)
        assert p.V_packed.shape == (r, (1000 + 31) // 32)
        assert p.U_packed.dtype == p.V_packed.dtype == np.int32
    assert L.bias.shape == (777,)
