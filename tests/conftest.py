"""Shared helpers and fixtures for the test suite.

Two external things some tests need, both optional (tests skip without them):

* the upstream LittleBit repo, cloned read-only at third_party/LittleBit.
  It is CC BY-NC 4.0, so we never copy its code into this repo; we only
  *import* it at test time and use it as an oracle.
* the Qwen3-0.6B checkpoint in the Hugging Face cache (real weights).

Helpers that test files import directly (`from conftest import ...`):
`rel_err`, `to_np`, `load_upstream`, `qwen_weight`, `build_upstream_layer`.
"""

from __future__ import annotations

import glob
import importlib
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Override with LITTLEBIT_UPSTREAM_DIR=/some/path (or a bogus path to check
# that everything upstream-dependent skips cleanly).
UPSTREAM_DIR = Path(os.environ.get("LITTLEBIT_UPSTREAM_DIR",
                                   REPO_ROOT / "third_party" / "LittleBit")).resolve()
QWEN_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------


def to_np(a) -> np.ndarray:
    """mx.array / torch.Tensor / array-like -> float64 NumPy array."""
    try:
        import mlx.core as mx
        if isinstance(a, mx.array):
            return np.array(a.astype(mx.float32)).astype(np.float64)
    except ImportError:
        pass
    if hasattr(a, "detach"):  # torch
        return a.detach().cpu().double().numpy()
    return np.asarray(a, dtype=np.float64)


def rel_err(y, ref) -> float:
    """max|y - ref| / max|ref|.

    Normalising by the largest reference value (not element-wise) means
    outputs that happen to be near zero don't blow the metric up; the error
    is measured on the scale of the layer's output.
    """
    y, ref = to_np(y), to_np(ref)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    denom = np.abs(ref).max()
    return float(np.abs(y - ref).max() / (denom if denom > 0 else 1.0))


# --------------------------------------------------------------------------
# upstream oracle
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_upstream():
    """Import the upstream modules we need, or return None if not cloned.

    The upstream package __init__ files pull in `transformers` and have a
    circular import (quantization.utils -> quant_util -> quantization.modules
    -> littlebit -> quantization.utils).  We sidestep both by registering empty
    package objects for `quantization`, `quantization.utils`, ... (with the
    right __path__ so submodules are found) and importing only the three leaf
    modules, which need nothing but torch.
    """
    if not (UPSTREAM_DIR / "quantization" / "modules" / "littlebit.py").is_file():
        return None
    pytest.importorskip("torch")
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    for pkg in ("quantization", "quantization.utils", "quantization.modules",
                "quantization.functions"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(UPSTREAM_DIR / pkg.replace(".", "/"))]
            sys.modules[pkg] = m
    packer = importlib.import_module("quantization.utils.binary_packer")
    binary = importlib.import_module("quantization.functions.binary")
    littlebit = importlib.import_module("quantization.modules.littlebit")
    return types.SimpleNamespace(
        LittleBitLinear=littlebit.LittleBitLinear,
        binary_packer=packer.binary_packer,
        binary_unpacker=packer.binary_unpacker,
        STEBinary=binary.STEBinary,
    )


@pytest.fixture(scope="session")
def upstream():
    up = load_upstream()
    if up is None:
        pytest.skip(f"upstream LittleBit not cloned at {UPSTREAM_DIR}")
    return up


def build_upstream_layer(W, *, eff_bit, residual, use_itq, bias=None, seed=0,
                         itq_n_iter=10):
    """Run upstream's own initialisation (SVD [+ ITQ] + rank-one scales) on W.

    W: (out, in) torch tensor.  Returns the upstream LittleBitLinear in eval
    mode on CPU.  This is the "standalone" recipe: make a plain nn.Linear,
    swap its class, and call __quant_convert__ as upstream's patcher would.
    """
    import torch

    up = load_upstream()
    torch.manual_seed(seed)  # svd_lowrank and ITQ's random init are random
    out_f, in_f = W.shape
    lin = torch.nn.Linear(in_f, out_f, bias=bias is not None)
    lin.weight.data = W.detach().float().clone()
    if bias is not None:
        lin.bias.data = torch.as_tensor(bias).float().clone()
    lin.__class__ = up.LittleBitLinear
    lin.__quant_convert__(do_train=True, quant_func=up.STEBinary, eff_bit=eff_bit,
                          residual=residual, use_itq=use_itq, itq_n_iter=itq_n_iter,
                          split_dim=1024, min_split_dim=8)
    return lin.eval()


# --------------------------------------------------------------------------
# real weights
# --------------------------------------------------------------------------


def qwen_path():
    hits = glob.glob(QWEN_GLOB)
    return hits[0] if hits else None


@lru_cache(maxsize=None)
def qwen_weight(name: str):
    """One bf16 tensor from Qwen3-0.6B as a torch tensor (skips if missing)."""
    path = qwen_path()
    if path is None:
        pytest.skip("Qwen3-0.6B safetensors not in the HF cache")
    safetensors = pytest.importorskip("safetensors")
    with safetensors.safe_open(path, framework="pt") as f:
        return f.get_tensor(name)
