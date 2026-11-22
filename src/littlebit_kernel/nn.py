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
    # Scales are kept in fp32: they are tiny (in + out + 2r numbers), so it
    # costs almost nothing, and it avoids fp16 range/subnormal issues for
    # trained checkpoints with very small scale values.
    def __init__(self, layer: LittleBitLayer, scale_dtype=mx.float32):
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


# ---------------------------------------------------------------------------
# Grouping layers that read the same input (q/k/v, gate/up): one launch pair
# for the whole group instead of one per layer.  See kernels/group.py.
# ---------------------------------------------------------------------------

GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))


class LittleBitGroup:
    """Packed arrays of several same-input LittleBitLinear layers (not a Module:
    its arrays are owned by the member modules below, for parameter counting)."""

    def __init__(self, layers):
        from .kernels import group as gk
        self.in_features = layers[0].in_features
        self.outs = [l.out_features for l in layers]
        R = max(l.rank for l in layers)
        W2 = (R + 31) // 32

        def pad_last(a, n):        # zero-pad the last axis to n
            return mx.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, n - a.shape[-1])])

        def pad_rows(a, n):        # zero-pad axis 0 to n
            return mx.pad(a, [(0, n - a.shape[0])] + [(0, 0)] * (a.ndim - 1))

        self.VT = mx.contiguous(mx.concatenate([pad_last(l.VT, R) for l in layers], 0))   # (G, W1, R)
        self.v2 = mx.concatenate([l.v2 for l in layers], 0)                               # (G, in)
        self.mid = mx.concatenate([pad_last(l.mid, R) for l in layers], 0)                # (G, R)
        self.UT = mx.contiguous(mx.concatenate([pad_rows(l.UT[0], W2) for l in layers], 1))  # (W2, OUT)
        self.u1 = mx.concatenate([l.u1[0] for l in layers], 0)
        if any(l.bias is not None for l in layers):
            self.bias = mx.concatenate([l.bias if l.bias is not None else mx.zeros((l.out_features,), l.u1.dtype)
                                        for l in layers])
        else:
            self.bias = None
        offs = [0]
        for o in self.outs:
            offs.append(offs[-1] + o)
        self.offs = offs
        self.seg = mx.array(offs, dtype=mx.uint32)
        mx.eval(self.VT, self.v2, self.mid, self.UT, self.u1, self.seg)
        self._fwd = mx.compile(partial(gk.forward_group, in_features=self.in_features, rank=R,
                                       out_total=offs[-1]))
        self._last = (None, None)

    def outputs(self, x):
        last_x, outs = self._last
        if last_x is not x:
            y = self._fwd(x, self.VT, self.v2, self.mid, self.UT, self.u1, self.bias, self.seg)
            outs = [y[..., a:b] for a, b in zip(self.offs[:-1], self.offs[1:])]
            self._last = (x, outs)
        return outs

    @staticmethod
    def eligible(layers) -> bool:
        from .kernels import auto, group as gk
        return (all(isinstance(l, LittleBitLinear) for l in layers)
                and all(l.VT.shape[0] == 1 for l in layers)                  # no residual path
                and len({l.in_features for l in layers}) == 1
                and all(l.out_features % gk.TG == 0 for l in layers)
                and all(auto.pick(l.in_features, l.rank, l.out_features, 1) is auto.lut5 for l in layers))


class GroupedLittleBitLinear(nn.Module):
    """Drop-in for one member of a group: returns its slice of the group output.

    The first member called with a given input array computes all members'
    outputs in one launch pair; the others reuse them (same `x` object)."""

    def __init__(self, layer: LittleBitLinear, group: LittleBitGroup, index: int):
        super().__init__()
        self._group = lambda: group             # a function, so it isn't a child module
        self.index = index
        self.in_features, self.out_features = layer.in_features, layer.out_features
        if index == 0:
            # The group's packed arrays are this member's parameters (so they
            # are counted once); the members' original per-layer arrays are
            # not kept, so grouping costs no extra memory.
            self.VT, self.v2, self.mid, self.UT, self.u1 = group.VT, group.v2, group.mid, group.UT, group.u1
            if group.bias is not None:
                self.bias = group.bias

    def __call__(self, x):
        return self._group().outputs(x)[self.index]


def group_shared_inputs(model: nn.Module, groups=GROUPS) -> int:
    """Group q/k/v and gate/up LittleBit layers that read the same input.
    Returns the number of groups made."""
    made = 0
    for _, mod in list(model.named_modules()):
        children = mod.children()
        for names in groups:
            if not all(n in children for n in names):
                continue
            layers = [children[n] for n in names]
            if not LittleBitGroup.eligible(layers):
                continue
            g = LittleBitGroup(layers)
            for i, n in enumerate(names):
                setattr(mod, n, GroupedLittleBitLinear(layers[i], g, i))
            made += 1
    return made
