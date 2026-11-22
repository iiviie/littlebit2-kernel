"""End-to-end check against the upstream implementation (the "oracle").

Pipeline under test:

    dense W --(upstream SVD [+ ITQ] init)--> upstream LittleBitLinear
            --(upstream state_dict: packed)--> layer_from_state_dict (ours)
            --> our float64 reference_forward  and  every kernel in KERNELS

and every output is compared with upstream's own `forward`.  W comes from
real Qwen3-0.6B layers (if cached locally) and from random matrices with odd
shapes and a bias.  Upstream code is imported at test time only (see
conftest.load_upstream); these tests skip if third_party/LittleBit is absent.

Tolerances (rel_err = max|y - ref| / max|ref|):
  * our float64 reference vs upstream fp32 forward: 1e-5.  Same math, upstream
    just does it in float32 (we see ~1e-6).
  * upstream fp16 forward vs our reference: 5e-3.  Upstream's fp16 path
    rounds (x*v2) to fp16, runs both binary matmuls in fp16 and rounds their
    outputs to fp16, then rounds y to fp16: ~4 roundings of 2**-11 each.
    We see ~6e-4 to 1e-3.
  * kernels with fp32 scales, fp32 x vs upstream fp32: 1e-4 (as in the gate).
  * kernels with the default fp16 scales vs upstream fp32: 3e-3.  Upstream
    keeps u1, v2, v1*u2 in fp32; we round each to fp16 (2**-12 relative on
    average, 2**-11 worst case), three such factors per output.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np
import pytest

from conftest import build_upstream_layer, load_upstream, qwen_weight, rel_err, to_np
from littlebit_kernel.convert import iter_littlebit_prefixes, layer_from_state_dict
from littlebit_kernel.format import effective_bits, sign_pm1, split_dim_for
from littlebit_kernel.kernels import KERNELS, DeviceLayer
from littlebit_kernel.reference import reference_forward

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(load_upstream() is None, reason="third_party/LittleBit not cloned"),
]

QWEN = {
    "q_proj": "model.layers.0.self_attn.q_proj.weight",      # (2048, 1024)
    "k_proj": "model.layers.0.self_attn.k_proj.weight",      # (1024, 1024)
    "down_proj": "model.layers.0.mlp.down_proj.weight",      # (1024, 3072)
}
# (eff_bit, residual, use_itq).  ITQ with 10 iterations is fast enough here
# (< 1 s per layer at rank ~400); upstream's default is 50.
CONFIGS = [
    (1.0, True, True),
    (0.55, False, True),
    (0.3, True, False),
    (0.1, False, False),
]
# Random W: (out, in, with_bias).  Odd sizes -> partial packed words.
RANDOM = [(777, 1000, True), (45, 37, True), (4097, 33, False)]
RANDOM_CONFIGS = [(0.55, True, True), (1.0, False, False)]

CASES = (
    [pytest.param(("qwen", name) + cfg, id=f"qwen-{name}-{cfg[0]}-{'res' if cfg[1] else 'single'}-{'itq' if cfg[2] else 'svd'}")
     for name in QWEN for cfg in CONFIGS]
    + [pytest.param(("random", (o, i, b)) + cfg, id=f"rand-{o}x{i}{'-bias' if b else ''}-{cfg[0]}-{'res' if cfg[1] else 'single'}-{'itq' if cfg[2] else 'svd'}")
       for (o, i, b) in RANDOM for cfg in RANDOM_CONFIGS]
)


@lru_cache(maxsize=None)
def _build(case):
    """-> (upstream LittleBitLinear, its packed state_dict, our converted layer)."""
    kind, what, eff, residual, itq = case
    bias = None
    if kind == "qwen":
        W = qwen_weight(QWEN[what])  # bf16; build_upstream_layer casts to fp32
    else:
        out_f, in_f, with_bias = what
        g = torch.Generator().manual_seed(out_f * 31 + in_f)
        W = torch.randn(out_f, in_f, generator=g) * 0.02
        if with_bias:
            bias = torch.randn(out_f, generator=g) * 0.1
    lin = build_upstream_layer(W, eff_bit=eff, residual=residual, use_itq=itq, bias=bias)
    sd = lin.state_dict()
    return lin, sd, layer_from_state_dict(sd)


def _x(in_f, shape=(4,), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, in_f, generator=g)


@torch.no_grad()
def _upstream(lin, x):
    return lin(x)


# --------------------------------------------------------------------------
# the conversion itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_conversion_structure(case):
    lin, sd, L = _build(case)
    _, _, eff, residual, _ = case
    in_f, out_f = lin.in_features, lin.out_features

    # rank: upstream's buffer, our arithmetic, and the converted arrays agree
    r = int(sd["_split_dim_final"])
    assert r == split_dim_for(in_f, out_f, eff, residual)
    assert float(sd["_eff_bit_actual"]) == pytest.approx(effective_bits(in_f, out_f, r, residual), rel=1e-6)
    assert (L.in_features, L.out_features) == (in_f, out_f)
    assert (L.residual is not None) == residual
    assert (L.bias is not None) == (lin.bias is not None)

    # packed bits == sign() of upstream's latent float factors, and u2, v1
    # are carried over as separate vectors (not pre-multiplied)
    for p, sfx in zip(L.paths, ["", "_R"]):
        assert p.rank == r
        assert p.U_packed.dtype == p.V_packed.dtype == np.int32
        np.testing.assert_array_equal(p.U_signs(), sign_pm1(getattr(lin, "U" + sfx).detach().numpy()))
        np.testing.assert_array_equal(p.V_signs(), sign_pm1(getattr(lin, "V" + sfx).detach().numpy()))
        for k in ("u1", "u2", "v1", "v2"):
            np.testing.assert_array_equal(getattr(p, k), getattr(lin, k + sfx).detach().numpy().ravel())


def test_iter_prefixes_and_nested_state_dict():
    """A full-model style state dict: nested modules, prefixed keys."""
    W = torch.randn(64, 96) * 0.02
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    model.layers[0].proj = build_upstream_layer(W, eff_bit=1.0, residual=True, use_itq=False)
    model.layers[1].proj = build_upstream_layer(W.T.contiguous(), eff_bit=0.55, residual=False,
                                                use_itq=False, bias=torch.randn(96))
    model.norm = torch.nn.LayerNorm(8)  # an ordinary module mixed in
    sd = model.state_dict()

    prefixes = iter_littlebit_prefixes(sd.keys())
    assert prefixes == ["layers.0.proj.", "layers.1.proj."]
    assert iter_littlebit_prefixes(["U_packed", "U_R_packed", "u1"]) == [""]

    x = _x(96)
    for prefix, layer in zip(prefixes, [model.layers[0].proj, model.layers[1].proj]):
        L = layer_from_state_dict(sd, prefix)
        xin = x if layer.in_features == 96 else x[:, :64]
        assert rel_err(reference_forward(xin.double().numpy(), L), _upstream(layer, xin)) <= 1e-5


def test_convert_accepts_numpy_and_bf16():
    """bf16 scales (a common on-disk dtype) and plain NumPy dicts both load."""
    _, sd, L = _build(CASES[0].values[0])
    as_bf16 = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    Lb = layer_from_state_dict(as_bf16)
    assert Lb.main.u1.dtype == np.float32
    np.testing.assert_allclose(Lb.main.u1, L.main.u1, rtol=2**-8)
    np.testing.assert_array_equal(Lb.main.U_packed, L.main.U_packed)

    Ln = layer_from_state_dict({k: v.numpy() for k, v in sd.items()})
    np.testing.assert_array_equal(Ln.residual.V_packed, L.residual.V_packed)
    np.testing.assert_array_equal(Ln.residual.v2, L.residual.v2)


# --------------------------------------------------------------------------
# forward: our reference and kernels vs upstream
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_reference_matches_upstream_fp32(case):
    lin, _, L = _build(case)
    for shape in [(1,), (4,), (2, 3)]:  # incl. a 3-D (batch, seq, in) input
        x = _x(lin.in_features, shape)
        y_up = _upstream(lin, x)
        assert y_up.dtype == torch.float32
        err = rel_err(reference_forward(x.double().numpy(), L), y_up)
        assert err <= 1e-5, f"shape {shape}: rel_err {err:.2e}"


@pytest.mark.parametrize("case", CASES)
def test_upstream_fp16_path_vs_reference(case):
    """Upstream switches to `_compute_forward_fp16` when x is fp16."""
    lin, _, L = _build(case)
    x = _x(lin.in_features).half()
    y_up = _upstream(lin, x)
    assert y_up.dtype == torch.float16
    ref = reference_forward(x.double().numpy(), L)  # same fp16 x, exact math
    err = rel_err(y_up, ref)
    assert err <= 5e-3, f"rel_err {err:.2e}"


@pytest.mark.parametrize("scale_dtype", ["float32", "float16"])
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kernel", sorted(KERNELS))
def test_kernel_matches_upstream(kernel, case, scale_dtype):
    lin, _, L = _build(case)
    x = _x(lin.in_features, (3,))
    y_up = _upstream(lin, x)
    dev = DeviceLayer.from_layer(L, scale_dtype=getattr(mx, scale_dtype))
    y = KERNELS[kernel](mx.array(x.numpy()), dev)
    mx.eval(y)
    assert y.dtype == mx.float32 and y.shape == tuple(y_up.shape)
    tol = 1e-4 if scale_dtype == "float32" else 3e-3
    err = rel_err(to_np(y), y_up)
    assert err <= tol, f"{kernel} ({scale_dtype} scales): rel_err {err:.2e} > {tol:.0e}"
