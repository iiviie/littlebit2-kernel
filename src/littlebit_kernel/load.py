"""Load a *trained* LittleBit checkpoint (upstream's saved format) into mlx-lm.

Upstream's training script saves a Hugging Face style directory:

    config.json  (+ littlebit_config.json)   tokenizer files
    model.safetensors (or sharded + index)     with, per LittleBit layer:
        <prefix>.U_packed int32, <prefix>.U_shape int64, V_packed, V_shape,
        <prefix>.u1/u2/v1/v2 (bf16)  [+ _R versions if residual]
    and every other weight (embeddings, norms, lm_head) as ordinary tensors.

Steps:
1. build the model architecture + tokenizer with mlx-lm from the *base*
   model id (the checkpoint's config.json has extra LittleBit fields and no
   dense projection weights, so mlx-lm can't load it directly);
2. overwrite every non-LittleBit weight from the checkpoint (QAT trains
   the norms too, so these differ from the base model);
3. replace each projection that has packed weights with LittleBitLinear.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .convert import iter_littlebit_prefixes, layer_from_state_dict
from .nn import LittleBitLinear, group_shared_inputs

# buffers upstream keeps for bookkeeping only
_SKIP_SUFFIXES = ("_eff_bit_target", "_split_dim_final", "_eff_bit_actual")


def _read_safetensors(ckpt_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {ckpt_dir}")
    sd = {}
    for f in files:
        sd.update(mx.load(f))          # mx.load handles bf16, int32, int64
    return sd


def _to_np(a: mx.array) -> np.ndarray:
    if a.dtype in (mx.bfloat16, mx.float16):
        a = a.astype(mx.float32)
    return np.array(a)


def _get_module(root: nn.Module, dotted: str):
    mod = root
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def load_littlebit_checkpoint(ckpt_dir: str, base_model: Optional[str] = None, dtype=mx.float16,
                              group: bool = True, verbose: bool = True):
    """Returns (model, tokenizer) with LittleBit layers running on the Metal kernels."""
    from mlx_lm import load

    if base_model is None:
        cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
        base_model = cfg.get("_name_or_path") or cfg.get("name_or_path")
        if not base_model:
            raise ValueError("pass base_model= (the HF id the checkpoint was trained from)")
    model, tok = load(base_model)
    model.set_dtype(dtype)

    sd = _read_safetensors(ckpt_dir)
    prefixes = iter_littlebit_prefixes(sd.keys())       # e.g. "model.layers.0.self_attn.q_proj."
    lb_keys = {k for k in sd for p in prefixes if k.startswith(p)}

    # 2. ordinary weights (embeddings, norms, lm_head, biases of non-LB layers)
    plain = [(k, v.astype(dtype) if mx.issubdtype(v.dtype, mx.floating) else v)
             for k, v in sd.items() if k not in lb_keys and not k.endswith(_SKIP_SUFFIXES)]
    have = dict(nn.utils.tree_flatten(model.parameters()))
    unknown = [k for k, _ in plain if k not in have]
    if unknown and verbose:
        print(f"  ignoring {len(unknown)} checkpoint tensors with no slot in the model, e.g. {unknown[:3]}")
    model.load_weights([(k, v) for k, v in plain if k in have], strict=False)

    # 3. LittleBit projections
    for p in prefixes:
        sub = {k[len(p):]: _to_np(sd[k]) for k in sd if k.startswith(p)}
        layer = layer_from_state_dict(sub)
        parent, _, name = p.rstrip(".").rpartition(".")
        setattr(_get_module(model, parent), name, LittleBitLinear(layer))
    n_groups = group_shared_inputs(model) if group else 0
    mx.eval(model.parameters())
    if verbose:
        print(f"  {len(prefixes)} LittleBit layers loaded from {ckpt_dir} "
              f"({len(plain)} other tensors, {n_groups} same-input groups)")
    return model, tok
