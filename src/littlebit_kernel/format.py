"""The LittleBit weight format, written from the paper/format description.

A LittleBit linear layer replaces a dense weight W (out x in) with

    W  ~=  diag(u1) . sign(U) . diag(u2 * v1) . sign(V) . diag(v2)

where sign(U) is (out x r) and sign(V) is (r x in), both {-1, +1}, and the
four vectors u1 (out), u2 (r), v1 (r), v2 (in) carry all the magnitude.
`r` is the latent rank ("split_dim"). So for an activation x:

    y = ((((x * v2) @ sign(V).T) * (v1 * u2)) @ sign(U).T) * u1

A layer may also carry a second, independent "residual" path with its own
U_R, V_R, u1_R, ...; its output is simply added.

On disk, sign(U) and sign(V) are stored 1 bit per entry:

  * each row is cut into 32-entry chunks, one int32 word per chunk;
  * entry k of the chunk goes to bit k (LSB first);
  * +1 -> bit 0, -1 -> bit 1;
  * the last chunk of a row is padded with +1 (bit 0).

Packed shape: (rows, ceil(cols / 32)) int32.  We keep the same layout on the
GPU (as uint32) so a checkpoint can be used without repacking.

The key identity the kernels use: for a row of signs s_k = 1 - 2*b_k,

    sum_k s_k * a_k = sum_k a_k  -  2 * sum_{k : b_k = 1} a_k

so a binary dot product needs no multiplies: one plain sum of `a`
(shared by every row) plus a masked sum per row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# rank (split_dim) bookkeeping -- same arithmetic as the paper's bit budget
# --------------------------------------------------------------------------


def estimate_split_dim(in_f: int, out_f: int, eff_bit: float, residual: bool) -> float:
    """Float rank that spends `eff_bit` bits per weight.

    Bit budget used by LittleBit: one path costs r*(in + out + 16) bits for
    the signs plus the rank-sized scales, and 16*(in + out) bits for the
    in/out-sized fp16 scales.  A residual layer pays for two paths.
    """
    base = in_f + out_f + 16
    if residual:
        return (in_f * out_f * eff_bit - 32 * (in_f + out_f)) / (2 * base)
    return (in_f * out_f * eff_bit - 16 * (in_f + out_f)) / base


def split_dim_for(in_f: int, out_f: int, eff_bit: float, residual: bool,
                  ratio_factor: float = 1.0, min_split_dim: int = 8) -> int:
    """Integer rank: estimate, scale, round down to a multiple of 8, clamp."""
    r = estimate_split_dim(in_f, out_f, eff_bit, residual) * ratio_factor
    r = (int(r) // 8) * 8
    if r == 0:
        r = min_split_dim
    return max(r, min_split_dim)


def effective_bits(in_f: int, out_f: int, r: int, residual: bool) -> float:
    if residual:
        return (r * 2 * (in_f + out_f + 16) + 32 * (in_f + out_f)) / (in_f * out_f)
    return (r * (in_f + out_f + 16) + 16 * (in_f + out_f)) / (in_f * out_f)


# --------------------------------------------------------------------------
# bit packing (NumPy, vectorised)
# --------------------------------------------------------------------------


def words_per_row(cols: int) -> int:
    return (cols + 31) // 32


def pack_signs(signs: np.ndarray) -> np.ndarray:
    """{-1,+1} (rows, cols) -> int32 (rows, ceil(cols/32)), LSB first, -1 -> 1.

    Zeros are treated as +1 (sign(0) := +1, same convention as the
    binariser used in training).
    """
    signs = np.asarray(signs)
    rows, cols = signs.shape
    w = words_per_row(cols)
    bits = np.zeros((rows, w * 32), dtype=np.uint8)  # padding = bit 0 = +1
    bits[:, :cols] = signs < 0
    # (rows, w, 32) -> weight bit k by 2**k, LSB first
    bits = bits.reshape(rows, w, 32).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(32, dtype=np.uint64))
    words = (bits * weights).sum(axis=2).astype(np.uint32)
    return words.view(np.int32)


def unpack_signs(packed: np.ndarray, cols: int) -> np.ndarray:
    """int32/uint32 (rows, words) -> int8 {-1,+1} (rows, cols)."""
    packed = np.ascontiguousarray(packed).view(np.uint32)
    rows, w = packed.shape
    shifts = np.arange(32, dtype=np.uint32)
    bits = (packed[:, :, None] >> shifts) & np.uint32(1)  # (rows, w, 32)
    bits = bits.reshape(rows, w * 32)[:, :cols]
    return (1 - 2 * bits.astype(np.int8)).astype(np.int8)


def sign_pm1(a: np.ndarray) -> np.ndarray:
    """sign with sign(0) = +1, as int8."""
    return np.where(np.asarray(a) < 0, -1, 1).astype(np.int8)


# --------------------------------------------------------------------------
# a container for one path / one layer
# --------------------------------------------------------------------------


@dataclass
class LittleBitPath:
    """One factorised path, already binarised and packed.

    U_packed: (out, ceil(r/32)) int32     V_packed: (r, ceil(in/32)) int32
    u1: (out,)  u2: (r,)  v1: (r,)  v2: (in,)   -- float32 NumPy
    """

    U_packed: np.ndarray
    V_packed: np.ndarray
    u1: np.ndarray
    u2: np.ndarray
    v1: np.ndarray
    v2: np.ndarray

    @property
    def out_features(self) -> int:
        return self.u1.shape[0]

    @property
    def in_features(self) -> int:
        return self.v2.shape[0]

    @property
    def rank(self) -> int:
        return self.u2.shape[0]

    @property
    def mid_scale(self) -> np.ndarray:
        """v1 * u2: the one scale vector between the two binary matmuls."""
        return (self.v1 * self.u2).astype(np.float32)

    def U_signs(self) -> np.ndarray:
        return unpack_signs(self.U_packed, self.rank)

    def V_signs(self) -> np.ndarray:
        return unpack_signs(self.V_packed, self.in_features)

    @classmethod
    def from_float(cls, U, V, u1, u2, v1, v2) -> "LittleBitPath":
        """Build from (latent, un-binarised) float factors + scale vectors."""
        f = lambda t: np.asarray(t, dtype=np.float32).reshape(-1)
        return cls(
            U_packed=pack_signs(sign_pm1(U)),
            V_packed=pack_signs(sign_pm1(V)),
            u1=f(u1), u2=f(u2), v1=f(v1), v2=f(v2),
        )

    @classmethod
    def random(cls, in_f: int, out_f: int, r: int, seed: int = 0,
               scale_spread: float = 0.5) -> "LittleBitPath":
        """Random bits + positive random scales (for tests and benchmarks).

        Scale magnitudes are chosen so a unit-variance x gives an O(1) y:
        each binary matmul multiplies the variance by its inner length.
        """
        rng = np.random.default_rng(seed)
        U = rng.standard_normal((out_f, r))
        V = rng.standard_normal((r, in_f))

        def pos(n, base):
            return (base * np.exp(scale_spread * rng.standard_normal(n))).astype(np.float32)

        return cls.from_float(
            U, V,
            u1=pos(out_f, 1.0 / np.sqrt(r)),
            u2=pos(r, 1.0),
            v1=pos(r, 1.0 / np.sqrt(in_f)),
            v2=pos(in_f, 1.0),
        )


@dataclass
class LittleBitLayer:
    """A main path plus an optional residual path, and an optional bias."""

    main: LittleBitPath
    residual: Optional[LittleBitPath] = None
    bias: Optional[np.ndarray] = None

    @property
    def in_features(self) -> int:
        return self.main.in_features

    @property
    def out_features(self) -> int:
        return self.main.out_features

    @property
    def paths(self):
        return [self.main] + ([self.residual] if self.residual is not None else [])

    @classmethod
    def random(cls, in_f: int, out_f: int, eff_bit: float = None, residual: bool = False,
               r: int = None, seed: int = 0, bias: bool = False) -> "LittleBitLayer":
        if r is None:
            r = split_dim_for(in_f, out_f, eff_bit, residual)
        main = LittleBitPath.random(in_f, out_f, r, seed=seed)
        res = LittleBitPath.random(in_f, out_f, r, seed=seed + 1) if residual else None
        b = None
        if bias:
            b = np.random.default_rng(seed + 2).standard_normal(out_f).astype(np.float32)
        return cls(main=main, residual=res, bias=b)
