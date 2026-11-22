"""MLX `nn.Module` for a LittleBit linear layer, and helpers to swap it into
an mlx-lm model.

The module stores only what inference needs, already in the kernel's
layout: the transposed 1-bit sign matrices (uint32) and fp16 scales.  The
dense weight is never materialised.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .format import LittleBitLayer
from .kernels import auto


class LittleBitLinear(nn.Module):
    def __init__(self, layer: LittleBitLayer, scale_dtype=mx.float16):
        super().__init__()
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.rank = layer.main.rank
        paths = layer.paths
        u32 = lambda a: mx.array(np.ascontiguousarray(a).view(np.uint32))
        sc = lambda f: mx.array(np.stack([f(p) for p in paths]).astype(np.float32)).astype(scale_dtype)
        # word-major ("transposed") bits: see kernels/lut.py for why
        self.VT = mx.contiguous(mx.stack([u32(p.V_packed).T for p in paths]))
        self.UT = mx.contiguous(mx.stack([u32(p.U_packed).T for p in paths]))
        self.v2 = sc(lambda p: p.v2)
        self.mid = sc(lambda p: p.mid_scale)
        self.u1 = sc(lambda p: p.u1)
        if layer.bias is not None:
            self.bias = mx.array(layer.bias).astype(scale_dtype)
        else:
            self.bias = None
        mx.eval(self.parameters())
        # mx.compile traces the kernel calls once per input shape; later calls
        # replay the traced graph from C++.  That halves the per-call CPU
        # cost of building the graph (~20 -> ~8 us), which matters because
        # mlx-lm decode is partly CPU-bound for tiny layers (NOTES.md).
        self._fwd = mx.compile(partial(auto.forward_arrays, in_features=self.in_features,
                                       rank=self.rank, out_features=self.out_features))

    def __call__(self, x):
        return self._fwd(x, self.VT, self.UT, self.v2, self.mid, self.u1, self.bias)

    def nbytes(self) -> int:
        return sum(v.nbytes for _, v in nn.utils.tree_flatten(self.parameters()))


# Projections inside a transformer block that LittleBit replaces.  The
# embeddings and the LM head stay in full precision (as in the papers).
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def littlebitify(model: nn.Module, make_layer: Callable[[str, np.ndarray, Optional[np.ndarray]], LittleBitLayer],
                 targets=DEFAULT_TARGETS, verbose: bool = False) -> int:
    """Replace every nn.Linear whose name ends in one of `targets`.

    `make_layer(name, W, bias)` gets the dense weight (out, in) as float32
    NumPy and returns a LittleBitLayer (e.g. SVD init, or a converted
    checkpoint).  Returns the number of replaced layers.  Each dense weight is
    released as soon as its replacement exists, so peak memory stays low.
    """
    replaced = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.children().items()):
            if not isinstance(child, nn.Linear) or child_name not in targets:
                continue
            full = f"{name}.{child_name}" if name else child_name
            W = np.array(child.weight.astype(mx.float32))
            b = np.array(child.bias.astype(mx.float32)) if "bias" in child else None
            lb = LittleBitLinear(make_layer(full, W, b))
            setattr(mod, child_name, lb)
            del child, W
            replaced += 1
            if verbose and replaced % 20 == 0:
                print(f"  replaced {replaced} layers (last: {full})", flush=True)
    return replaced
