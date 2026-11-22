"""`group`: several small LittleBit layers that read the SAME input in one launch pair.

In a transformer block, q/k/v_proj all read the same normalised hidden
state, and so do gate/up_proj.  Each layer separately costs 2 dispatches
(lut5), so a block costs 14.  Grouping them costs 2 per group -> 8 per
block.  For small models, where each layer is dominated by fixed dispatch
latency (NOTES.md), that is the main lever.

How it maps onto the existing kernels:

* Stage 1 is exactly lut5's stage 1 with the layers as "paths": every path
  shares x and has its own V bits, v2 and mid.  Paths must have the same
  rank, so smaller ranks are zero-padded (padded rows get mid = 0 -> t = 0,
  and padded columns of U are bit 0 -> they add +0).
* Stage 2 sees ONE tall matrix: the U bits of all layers concatenated along
  the output dimension (word-major, so concatenating = stacking columns).
  Each threadgroup's row tile lies inside one layer's segment (we require
  every `out` to be a multiple of the tile size), so it builds its table
  from that layer's t and writes its slice of the concatenated output.
  The caller splits the output back into q / k / v (cheap slices).

Residual layers (main + residual path, both reading x) group too: each
layer then has PPL = 2 paths.  Stage 1 sees NL * PPL paths (layer-major);
stage 2 loops over the PPL paths of its tile's layer and sums them, exactly
like lut5's stage 2 does for one residual layer.  UT is (PPL, W2, OUT) and
u1 is (PPL, OUT) -- path p of every layer, concatenated along the outputs.
"""

from __future__ import annotations

import mlx.core as mx

from .common import as_2d
from .lut5 import _BUILD, _HEADER2, _LOOKUP, _PREFETCH, _ceil, _k1, _kr, _split

_STAGE2G = """
    const uint tid   = thread_position_in_threadgroup.x;
    const uint row   = threadgroup_position_in_grid.x * TG + tid;
    const uint chunk = threadgroup_position_in_grid.y;
    const uint m     = threadgroup_position_in_grid.z;
    const uint kbase = chunk * KC;
    const uint w0    = kbase / 32;
    threadgroup float T[(KC / 4) * 16];

    // which layer owns this row tile
    const uint tile_row0 = threadgroup_position_in_grid.x * TG;
    uint layer = 0;
    while (layer + 1 < NP / PPL && tile_row0 >= seg[layer + 1]) ++layer;

    float y = 0.0f;
    for (uint p = 0; p < PPL; ++p) {
        const device uint* B = UT + p * W * N;
""" + _PREFETCH + """
        const device float* tp = tpart + (m * NP + layer * PPL + p) * NCH1 * K;
        #define LOAD(k) partial_sum(tp, k)
""" + _BUILD + """
        #undef LOAD
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc;
""" + _LOOKUP + """
        if (row < N) y += acc * float(u1[p * N + row]);
        if (PPL > 1) threadgroup_barrier(mem_flags::mem_threadgroup);   // T is rebuilt next path
    }
    if (row >= N) return;
    if (FINAL) {
        if (HAS_BIAS) y += float(bias[row]);
        out[m * N + row] = static_cast<TO>(y);
    } else {
        out[(m * NCH + chunk) * N + row] = static_cast<TO>(y);
    }
"""

_k2g = mx.fast.metal_kernel(name="lb_group_s2", input_names=["tpart", "UT", "u1", "bias", "seg"],
                            output_names=["out"], source=_STAGE2G, header=_HEADER2)

TG = 128
CFG = dict(KC1=512, KC2=512, MIN_TGS=64)


def forward_group(x, VT, v2, mid, UT, u1, bias, seg, in_features, rank, out_total, cfg=None):
    """NP = layers * PPL paths, layer-major.
    VT (NP, ceil(in/32), r) | v2 (NP, in) | mid (NP, r) | UT (PPL, ceil(r/32), OUT) |
    u1 (PPL, OUT) | bias (OUT,) or None | seg: uint32 (layers+1,) row offsets.
    Returns (..., OUT) -- the layers' outputs side by side."""
    c = {**CFG, **(cfg or {})}
    x2, restore = as_2d(x)
    M, NP, PPL = x2.shape[0], VT.shape[0], UT.shape[0]
    K, R, OUT = in_features, rank, out_total

    tiles1 = _ceil(R, TG)
    kc1, nch1 = _split(K, tiles1, c["KC1"], c["MIN_TGS"], NP)
    (tpart,) = _k1(
        inputs=[x2, VT, v2, mid],
        template=[("TX", x2.dtype), ("TS", v2.dtype), ("K", K), ("N", R), ("W", _ceil(K, 32)),
                  ("KC", kc1), ("NCH", nch1), ("NP", NP), ("TG", TG)],
        grid=(tiles1 * TG, NP * nch1, M), threadgroup=(TG, 1, 1),
        output_shapes=[(M, NP, nch1, R)], output_dtypes=[mx.float32])

    tiles2 = _ceil(OUT, TG)
    kc2 = min(c["KC2"], max(32, 1 << (R - 1).bit_length()))
    kc2, nch2 = _split(R, tiles2, kc2, c["MIN_TGS"])
    final = nch2 == 1
    b = bias if bias is not None else u1
    (y,) = _k2g(
        inputs=[tpart, UT, u1, b, seg],
        template=[("TO", x2.dtype if final else mx.float32), ("K", R), ("N", OUT), ("W", _ceil(R, 32)),
                  ("KC", kc2), ("NCH", nch2), ("NCH1", nch1), ("NP", NP), ("PPL", PPL), ("TG", TG),
                  ("FINAL", final), ("HAS_BIAS", bias is not None)],
        grid=(tiles2 * TG, nch2, M), threadgroup=(TG, 1, 1),
        output_shapes=[(M, OUT) if final else (M, nch2, OUT)],
        output_dtypes=[x2.dtype if final else mx.float32])
    if not final:
        (y,) = _kr(
            inputs=[y, b],
            template=[("TO", x2.dtype), ("N", OUT), ("NCH", nch2), ("HAS_BIAS", bias is not None)],
            grid=(OUT, M, 1), threadgroup=(min(256, OUT), 1, 1),
            output_shapes=[(M, OUT)], output_dtypes=[x2.dtype])
    return restore(y)
