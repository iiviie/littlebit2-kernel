"""load_littlebit_checkpoint: a checkpoint in upstream's saved format runs in mlx-lm.

We write a fake trained checkpoint for Qwen3-0.6B (random LittleBit layers,
norms scaled so we can tell they were loaded), load it, and compare logits
with a model built in memory from the same layers.
"""

import glob
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from littlebit_kernel.format import LittleBitLayer, LittleBitPath
from littlebit_kernel.nn import DEFAULT_TARGETS, LittleBitLinear

BASE = "Qwen/Qwen3-0.6B"
have_base = bool(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")))
pytestmark = pytest.mark.skipif(not have_base, reason="Qwen3-0.6B not in the HF cache")


def _write_fake_checkpoint(path, n_layers_to_keep=None):
    from mlx_lm import load
    model, _ = load(BASE)
    sd, layers = {}, {}
    for name, mod in model.named_modules():
        last = name.split(".")[-1]
        if last in DEFAULT_TARGETS and isinstance(mod, nn.Linear):
            out_f, in_f = mod.weight.shape
            L = LittleBitLayer.random(in_f, out_f, eff_bit=0.3, seed=len(layers))
            layers[name] = L
            p = L.main
            sd[f"{name}.U_packed"] = mx.array(p.U_packed)
            sd[f"{name}.U_shape"] = mx.array(np.array([out_f, p.rank], dtype=np.int64))
            sd[f"{name}.V_packed"] = mx.array(p.V_packed)
            sd[f"{name}.V_shape"] = mx.array(np.array([p.rank, in_f], dtype=np.int64))
            for s in ("u1", "u2", "v1", "v2"):
                sd[f"{name}.{s}"] = mx.array(getattr(p, s)[None, :]).astype(mx.bfloat16)
            sd[f"{name}._split_dim_final"] = mx.array(np.array(p.rank, dtype=np.int64))
    for k, v in nn.utils.tree_flatten(model.parameters()):
        if not any(f".{t}." in k for t in DEFAULT_TARGETS):
            sd[k] = (v * 1.1) if "norm" in k else v          # "trained" norms
    mx.save_safetensors(os.path.join(path, "model.safetensors"), sd)
    return layers


def test_trained_checkpoint_roundtrip(tmp_path):
    from mlx_lm import load
    from littlebit_kernel.load import load_littlebit_checkpoint

    layers = _write_fake_checkpoint(str(tmp_path))
    # group=False: the reference below runs the identical ungrouped kernels, so
    # the logits must agree to fp16 noise (grouping has its own tests).
    model, tok = load_littlebit_checkpoint(str(tmp_path), base_model=BASE, group=False, verbose=False)

    def bf16(a):   # the checkpoint stores scales in bf16; so must the reference
        return np.array(mx.array(a).astype(mx.bfloat16).astype(mx.float32))

    # (A whole-model logit comparison is meaningless here: random layers make
    # the network chaotic, so bf16-vs-fp16 rounding of the norms alone flips
    # the result.  Check each piece exactly instead.)
    def get(root, dotted):
        for part in dotted.split("."):
            root = root[int(part)] if part.isdigit() else getattr(root, part)
        return root

    # every LittleBit layer == a LittleBitLinear built from the same bits and
    # bf16-rounded scales
    x = mx.random.normal((2, 1024)).astype(mx.float16)
    for name, L in layers.items():
        p = L.main
        Lb = LittleBitLayer(main=LittleBitPath(p.U_packed, p.V_packed, bf16(p.u1), bf16(p.u2), bf16(p.v1), bf16(p.v2)))
        xi = x if L.in_features == 1024 else mx.random.normal((2, L.in_features)).astype(mx.float16)
        assert mx.array_equal(get(model, name)(xi), LittleBitLinear(Lb)(xi)).item(), name

    # every other tensor == the checkpoint's value (norms were scaled by 1.1)
    ckpt = mx.load(os.path.join(str(tmp_path), "model.safetensors"))
    for k, v in nn.utils.tree_flatten(model.parameters()):
        if k in ckpt:
            assert mx.array_equal(v, ckpt[k].astype(v.dtype)).item(), k

    ids = mx.array(tok.encode("The lighthouse keeper found a bottle"))[None]
    assert mx.isfinite(model(ids)).all().item()
