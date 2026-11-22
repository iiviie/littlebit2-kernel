"""Real per-projection linear-layer shapes, as (in_features, out_features).

A LittleBit layer replaces each nn.Linear in the transformer blocks, so the
shapes we care about are the 7 projections of one decoder block:

    attention: q_proj, k_proj, v_proj, o_proj
    MLP:       gate_proj, up_proj, down_proj

For a Llama/Qwen-style block with hidden size H, intermediate size I,
num_heads h, num_kv_heads kv and head_dim d:

    q_proj    (H      -> h*d)
    k_proj    (H      -> kv*d)
    v_proj    (H      -> kv*d)
    o_proj    (h*d    -> H)
    gate_proj (H      -> I)
    up_proj   (H      -> I)
    down_proj (I      -> H)

Note Qwen3 sets head_dim = 128 explicitly, so h*d is NOT always H: for
Qwen3-0.6B, 16 heads * 128 = 2048 != hidden 1024.

Identical shapes inside one model (k/v always, gate/up always, and q/o when
h*d == H) are timed once; their names are joined, e.g. "k_proj/v_proj".

Run `python bench/shapes.py` to print the table.
"""

from __future__ import annotations

import glob
import json
import os

# --------------------------------------------------------------------------
# model configs
# --------------------------------------------------------------------------

# Hardcoded from the public HF config.json files (no download needed):
#   Qwen/Qwen3-4B:                hidden 2560, intermediate 9728,
#                                 32 heads, 8 kv heads, head_dim 128
#   meta-llama/Meta-Llama-3-8B:   hidden 4096, intermediate 14336,
#                                 32 heads, 8 kv heads, head_dim 128 (= 4096/32)
# Qwen3-0.6B is read from the local HF cache when present (it is the model
# we actually run); the fallback values below are copied from that file.
_QWEN3_06B_FALLBACK = dict(hidden=1024, intermediate=3072, heads=16, kv_heads=8, head_dim=128)

_HARDCODED = {
    "qwen3-4b": dict(hidden=2560, intermediate=9728, heads=32, kv_heads=8, head_dim=128),
    "llama3-8b": dict(hidden=4096, intermediate=14336, heads=32, kv_heads=8, head_dim=128),
}


def _qwen3_06b_config() -> dict:
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/config.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            c = json.load(f)
        return dict(
            hidden=c["hidden_size"],
            intermediate=c["intermediate_size"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            # Qwen3 has an explicit head_dim; older configs derive it.
            head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
        )
    return dict(_QWEN3_06B_FALLBACK)


MODEL_CONFIGS = {"qwen3-0.6b": _qwen3_06b_config(), **_HARDCODED}

# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


def projection_shapes(cfg: dict) -> dict:
    """name -> (in, out) for the 7 projections of one block."""
    H, I = cfg["hidden"], cfg["intermediate"]
    q_out = cfg["heads"] * cfg["head_dim"]
    kv_out = cfg["kv_heads"] * cfg["head_dim"]
    return {
        "q_proj": (H, q_out),
        "k_proj": (H, kv_out),
        "v_proj": (H, kv_out),
        "o_proj": (q_out, H),
        "gate_proj": (H, I),
        "up_proj": (H, I),
        "down_proj": (I, H),
    }


def dedup(shapes: dict) -> list:
    """[(joined_names, in, out)] with identical (in, out) merged, order kept."""
    merged = {}  # (in, out) -> [names]
    for name, shape in shapes.items():
        merged.setdefault(shape, []).append(name)
    return [("/".join(names), i, o) for (i, o), names in merged.items()]


def model_shapes(model: str) -> list:
    """[(proj_name, in, out)] for one model, deduplicated."""
    return dedup(projection_shapes(MODEL_CONFIGS[model]))


MODELS = list(MODEL_CONFIGS)

# The "quick" subset: small, mid and large, for fast iteration.
#   qwen3-0.6b k_proj/v_proj   (1024 -> 1024)    tiny; overhead dominated
#   qwen3-4b   gate/up         (2560 -> 9728)    mid
#   llama3-8b  down_proj       (14336 -> 4096)   large; bandwidth dominated
QUICK = [
    ("qwen3-0.6b", "k_proj/v_proj"),
    ("qwen3-4b", "gate_proj/up_proj"),
    ("llama3-8b", "down_proj"),
]


def get_shapes(models=None, quick: bool = False) -> list:
    """[(model, proj, in, out)] for the requested models (default: all)."""
    models = models or MODELS
    out = []
    for m in models:
        if m not in MODEL_CONFIGS:
            raise ValueError(f"unknown model {m!r}; choose from {MODELS}")
        for proj, i, o in model_shapes(m):
            if quick and (m, proj) not in QUICK:
                continue
            out.append((m, proj, i, o))
    return out


if __name__ == "__main__":
    for m in MODELS:
        print(f"{m}: {MODEL_CONFIGS[m]}")
        for proj, i, o in model_shapes(m):
            tag = "  (quick)" if (m, proj) in QUICK else ""
            print(f"    {proj:20s} in={i:6d} out={o:6d}{tag}")
