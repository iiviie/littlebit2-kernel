"""GPU-side layer container shared by every kernel variant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import numpy as np

from ..format import LittleBitLayer, LittleBitPath


@dataclass
class DevicePath:
    """One path as MLX arrays, ready for a kernel.

    V_bits: (r, ceil(in/32))  uint32    U_bits: (out, ceil(r/32)) uint32
    v2: (in,)   mid: (r,) = v1*u2   u1: (out,)        -- `scale_dtype`
    """

    V_bits: mx.array
    U_bits: mx.array
    v2: mx.array
    mid: mx.array
    u1: mx.array
    in_features: int
    out_features: int
    rank: int

    @classmethod
    def from_path(cls, p: LittleBitPath, scale_dtype=mx.float16) -> "DevicePath":
        u32 = lambda a: mx.array(np.ascontiguousarray(a).view(np.uint32))
        sc = lambda a: mx.array(np.asarray(a, dtype=np.float32)).astype(scale_dtype)
        return cls(
            V_bits=u32(p.V_packed), U_bits=u32(p.U_packed),
            v2=sc(p.v2), mid=sc(p.mid_scale), u1=sc(p.u1),
            in_features=p.in_features, out_features=p.out_features, rank=p.rank,
        )

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.V_bits, self.U_bits, self.v2, self.mid, self.u1))


@dataclass
class DeviceLayer:
    paths: List[DevicePath]
    bias: Optional[mx.array]
    in_features: int
    out_features: int

    @classmethod
    def from_layer(cls, layer: LittleBitLayer, scale_dtype=mx.float16) -> "DeviceLayer":
        paths = [DevicePath.from_path(p, scale_dtype) for p in layer.paths]
        bias = None
        if layer.bias is not None:
            bias = mx.array(layer.bias.astype(np.float32)).astype(scale_dtype)
        mx.eval([[p.V_bits, p.U_bits, p.v2, p.mid, p.u1] for p in paths] + ([bias] if bias is not None else []))
        return cls(paths=paths, bias=bias, in_features=layer.in_features,
                   out_features=layer.out_features)

    @property
    def residual(self) -> bool:
        return len(self.paths) > 1

    @property
    def rank(self) -> int:
        return self.paths[0].rank

    def nbytes(self) -> int:
        """Bytes of weights a forward must read at least once."""
        n = sum(p.nbytes() for p in self.paths)
        return n + (self.bias.nbytes if self.bias is not None else 0)


def round_layer_scales(layer: LittleBitLayer, dtype=np.float16) -> LittleBitLayer:
    """Copy of `layer` whose scale vectors are exactly representable in `dtype`.

    The device layer stores scales in fp16 (that is what the bit budget
    assumes).  For a fair comparison the reference must see the same rounded
    scales.  Note `mid = v1*u2` is rounded as a product, so we round u2 to
    1 and put the rounded product into v1.
    """
    def rp(p: LittleBitPath) -> LittleBitPath:
        r = lambda a: np.asarray(a, np.float32).astype(dtype).astype(np.float32)
        return LittleBitPath(U_packed=p.U_packed, V_packed=p.V_packed,
                             u1=r(p.u1), u2=np.ones_like(p.u2), v1=r(p.mid_scale), v2=r(p.v2))

    bias = None if layer.bias is None else np.asarray(layer.bias, np.float32).astype(dtype).astype(np.float32)
    return LittleBitLayer(main=rp(layer.main),
                          residual=None if layer.residual is None else rp(layer.residual),
                          bias=bias)


def as_2d(x: mx.array):
    """(..., in) -> ((M, in), restore_fn)."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    return x2, (lambda y: y.reshape(*lead, y.shape[-1]))
