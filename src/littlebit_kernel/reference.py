"""Plain reference forwards: the ground truth every kernel is tested against.

`reference_forward` is float64 NumPy on the unpacked signs -- slow, simple,
and obviously right.  `reference_forward_mlx` is the "unpack then matmul"
path in MLX (also used as a benchmark baseline).
"""

from __future__ import annotations

import numpy as np

from .format import LittleBitLayer, LittleBitPath


def path_forward_np(x: np.ndarray, p: LittleBitPath) -> np.ndarray:
    """((((x*v2) @ sign(V).T) * (v1*u2)) @ sign(U).T) * u1, in float64."""
    x = np.asarray(x, dtype=np.float64)
    V = p.V_signs().astype(np.float64)
    U = p.U_signs().astype(np.float64)
    t = (x * p.v2) @ V.T            # (.., r)
    t = t * (p.v1.astype(np.float64) * p.u2)
    return (t @ U.T) * p.u1         # (.., out)


def reference_forward(x: np.ndarray, layer: LittleBitLayer) -> np.ndarray:
    y = sum(path_forward_np(x, p) for p in layer.paths)
    if layer.bias is not None:
        y = y + layer.bias
    return y


def reference_forward_mlx(x, layer: LittleBitLayer, dtype=None):
    """Unpack-then-dense-matmul in MLX. Signs are materialised in `dtype`.

    Note this is what upstream does at inference time: the 1-bit storage is
    expanded to a full dense (but low-rank) pair of matrices.
    """
    import mlx.core as mx

    dtype = dtype or x.dtype
    y = None
    for p in layer.paths:
        V = mx.array(p.V_signs()).astype(dtype)
        U = mx.array(p.U_signs()).astype(dtype)
        t = (x * mx.array(p.v2).astype(dtype)) @ V.T
        t = t * mx.array(p.mid_scale).astype(dtype)
        out = (t @ U.T) * mx.array(p.u1).astype(dtype)
        y = out if y is None else y + out
    if layer.bias is not None:
        y = y + mx.array(layer.bias).astype(dtype)
    return y
