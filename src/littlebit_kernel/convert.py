"""Read LittleBit checkpoints (the packed state_dict format) into our containers.

A packed LittleBit linear layer appears in a state dict as these keys (all
under one prefix such as ``model.layers.0.self_attn.q_proj.``):

    U_packed  int32 (out, ceil(r/32))     U_shape  int64 [out, r]
    V_packed  int32 (r, ceil(in/32))      V_shape  int64 [r, in]
    u1 (1, out)   u2 (1, r)   v1 (1, r)   v2 (1, in)      -- float scales
    U_R_packed, U_R_shape, V_R_packed, V_R_shape,
    u1_R, u2_R, v1_R, v2_R                  -- only if the layer is residual
    bias (out,)                             -- only if the layer has a bias

plus a few bookkeeping buffers (``_eff_bit_target``, ``_split_dim_final``,
``_eff_bit_actual``) that we do not need for inference.

The packed bit layout is exactly the one `format.pack_signs` produces, so the
int32 words are copied as-is (no repacking).  Note u2 and v1 are stored as two
separate vectors; we keep them separate here and the kernels multiply them
into one ``mid = v1 * u2`` scale when moving the layer to the GPU.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

import numpy as np

from .format import LittleBitLayer, LittleBitPath, words_per_row


def _to_numpy(t) -> np.ndarray:
    """torch.Tensor (any dtype, incl. bf16) or array-like -> NumPy array."""
    if hasattr(t, "detach"):  # a torch tensor; avoid importing torch here
        t = t.detach().cpu()
        if t.is_floating_point():
            t = t.float()  # NumPy has no bfloat16
        return t.numpy()
    return np.asarray(t)


def _scale(sd: Mapping, key: str) -> np.ndarray:
    """Scale vectors are stored as (1, n); we keep them as flat float32 (n,)."""
    return _to_numpy(sd[key]).astype(np.float32).reshape(-1)


def _packed(sd: Mapping, prefix: str, name: str) -> np.ndarray:
    """Load `{name}_packed` as int32 and check it against `{name}_shape`."""
    words = _to_numpy(sd[f"{prefix}{name}_packed"])
    if words.dtype not in (np.int32, np.uint32):
        raise TypeError(f"{prefix}{name}_packed has dtype {words.dtype}, expected int32")
    words = np.ascontiguousarray(words).view(np.int32)
    shape_key = f"{prefix}{name}_shape"
    if shape_key in sd:
        rows, cols = (int(v) for v in _to_numpy(sd[shape_key]).reshape(-1))
        if words.shape != (rows, words_per_row(cols)):
            raise ValueError(f"{prefix}{name}_packed has shape {words.shape}, but "
                             f"{name}_shape = {(rows, cols)} needs {(rows, words_per_row(cols))}")
    return words


def _path(sd: Mapping, prefix: str, suffix: str) -> LittleBitPath:
    """One path. suffix is "" for the main path and "_R" for the residual."""
    p = LittleBitPath(
        U_packed=_packed(sd, prefix, "U" + suffix),
        V_packed=_packed(sd, prefix, "V" + suffix),
        u1=_scale(sd, f"{prefix}u1{suffix}"),
        u2=_scale(sd, f"{prefix}u2{suffix}"),
        v1=_scale(sd, f"{prefix}v1{suffix}"),
        v2=_scale(sd, f"{prefix}v2{suffix}"),
    )
    # Cross-check the shapes implied by the scales against the packed arrays.
    out_f, r, in_f = p.out_features, p.rank, p.in_features
    if p.v1.shape != (r,):
        raise ValueError(f"{prefix}v1{suffix}: length {p.v1.shape[0]} != rank {r}")
    if p.U_packed.shape != (out_f, words_per_row(r)):
        raise ValueError(f"{prefix}U{suffix}_packed shape {p.U_packed.shape} does not match out={out_f}, r={r}")
    if p.V_packed.shape != (r, words_per_row(in_f)):
        raise ValueError(f"{prefix}V{suffix}_packed shape {p.V_packed.shape} does not match r={r}, in={in_f}")
    return p


def layer_from_state_dict(sd: Mapping, prefix: str = "") -> LittleBitLayer:
    """Build a LittleBitLayer from a packed LittleBit state dict.

    `sd` maps key -> torch.Tensor or NumPy array.  `prefix` is the layer's
    key prefix including the trailing dot (e.g. "model.layers.0.mlp.up_proj."),
    or "" for a single layer's own state_dict.
    """
    main = _path(sd, prefix, "")
    residual: Optional[LittleBitPath] = None
    if f"{prefix}U_R_packed" in sd:
        residual = _path(sd, prefix, "_R")
        if (residual.in_features, residual.out_features) != (main.in_features, main.out_features):
            raise ValueError(f"{prefix}: residual path shape differs from main path")
    bias = None
    if f"{prefix}bias" in sd and sd[f"{prefix}bias"] is not None:
        bias = _scale(sd, f"{prefix}bias")
    return LittleBitLayer(main=main, residual=residual, bias=bias)


def iter_littlebit_prefixes(keys: Iterable[str]) -> List[str]:
    """All LittleBit layer prefixes in a full-model state dict, sorted.

    A layer is recognised by its main-path ``U_packed`` key; the residual
    ``U_R_packed`` key belongs to the same layer and is not a new prefix.
    Returned prefixes keep the trailing dot ("" for a bare layer dict).
    """
    out = set()
    for k in keys:
        if k == "U_packed":
            out.add("")
        elif k.endswith(".U_packed"):
            out.add(k[: -len("U_packed")])
    return sorted(out)
