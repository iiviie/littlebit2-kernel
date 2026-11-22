"""`auto`: the kernel to actually use -- picks the best variant by layer size.

Interleaved A/B on M1 Pro (bench/results/*_ab_lut5.json, NOTES.md):

* small layers (< ~1 MB of packed weights: every Qwen3-0.6B projection,
  4B at 0.1 bpw): time is dominated by per-dispatch latency, and `lut5`
  (2 dispatches, no atomics) is 1.2-1.5x faster than `lut4` (5 dispatches).
* large layers (r >~ 1000): `lut4` (split-K with float atomics) is 1.2-1.6x
  faster than `lut5`, whose stage 2 has to re-sum many stage-1 partials in
  every threadgroup.

Crossover measured around 0.7-1.6 MB; we switch at 1 MB.
"""

from __future__ import annotations

from . import lut4, lut5
from .common import DeviceLayer
from .lut import _prepare

SWITCH_BYTES = 1 << 20


def weight_bytes(in_features, rank, out_features, n_paths) -> int:
    words = rank * ((in_features + 31) // 32) + out_features * ((rank + 31) // 32)
    return n_paths * 4 * words


def pick(in_features, rank, out_features, n_paths):
    big = weight_bytes(in_features, rank, out_features, n_paths) >= SWITCH_BYTES
    return lut4 if big else lut5


def forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features):
    k = pick(in_features, rank, out_features, VT.shape[0])
    return k.forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features)


def forward(x, layer: DeviceLayer):
    c = _prepare(layer)
    return forward_arrays(x, c["VT"], c["UT"], c["v2"], c["mid"], c["u1"], layer.bias,
                          layer.in_features, layer.rank, layer.out_features)
