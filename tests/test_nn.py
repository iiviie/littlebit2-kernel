"""The MLX LittleBitLinear module (compiled forward, auto kernel) and the swap helper."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from littlebit_kernel.format import LittleBitLayer
from littlebit_kernel.kernels import round_layer_scales
from littlebit_kernel.nn import LittleBitLinear, littlebitify
from littlebit_kernel.reference import reference_forward


def rel_err(y, ref):
    return np.abs(y - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("in_f,out_f,bpw", [(1000, 777, 0.55), (1024, 3072, 0.1), (4096, 1024, 1.0)])
@pytest.mark.parametrize("residual,bias", [(False, False), (True, True)])
@pytest.mark.parametrize("shape", [(1,), (2, 5)])
def test_module_matches_reference(in_f, out_f, bpw, residual, bias, shape):
    L = round_layer_scales(LittleBitLayer.random(in_f, out_f, eff_bit=bpw, residual=residual, bias=bias))
    lb = LittleBitLinear(L)
    x = np.random.default_rng(0).standard_normal((*shape, in_f)).astype(np.float32)
    y = lb(mx.array(x).astype(mx.float16))
    assert y.dtype == mx.float16 and y.shape == (*shape, out_f)
    ref = reference_forward(x.reshape(-1, in_f), L).reshape(*shape, out_f)
    assert rel_err(np.array(y.astype(mx.float32)), ref) < 3e-3


def test_littlebitify_replaces_targets_only():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 96, bias=False)
            self.other = nn.Linear(64, 64, bias=False)

    m = Block()
    n = littlebitify(m, lambda name, W, b: LittleBitLayer.random(W.shape[1], W.shape[0], eff_bit=1.0))
    assert n == 1
    assert isinstance(m.q_proj, LittleBitLinear) and isinstance(m.other, nn.Linear)
    y = m.q_proj(mx.ones((1, 64), dtype=mx.float16))
    assert y.shape == (1, 96)
