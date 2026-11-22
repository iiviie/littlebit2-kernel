"""Naive kernel: the simplest correct GPU version. One thread per output row.

The forward of one path is two binary matrix-vector products:

    stage 1 (r rows):   t[j] = mid[j] * sum_k sign(V[j,k]) * x[k] * v2[k]
    stage 2 (out rows): y[i] = u1[i]  * sum_j sign(U[i,j]) * t[j]

Each thread owns one row. It walks the row's packed 32-bit words and, for
every bit, adds or subtracts the matching activation: bit 0 means +1, bit 1
means -1. No multiplies by weights happen at all -- the weight *is* a sign.

This is deliberately slow: each thread reads its row one word at a time
(uncoalesced across the threadgroup) and recomputes x*v2 for every row.
It exists so the faster variants have something obviously-correct to beat.
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d

# Template parameters (substituted at JIT time, so each shape gets its own
# specialised kernel):  K = inner length, N = rows, W = words per row.

_STAGE1_SRC = """
    uint row = thread_position_in_grid.x;   // which latent dim j
    uint m   = thread_position_in_grid.y;   // which activation vector
    if (row >= N) return;

    const device uint* bits = V_bits + row * W;
    const device T*    xm   = x + m * K;

    float acc = 0.0f;
    for (uint w = 0; w < W; ++w) {
        uint word = bits[w];
        uint base = w * 32;
        uint n = min(32u, K - base);          // last word may be partial
        for (uint b = 0; b < n; ++b) {
            float a = float(xm[base + b]) * float(v2[base + b]);
            // bit set -> weight is -1 -> subtract
            acc += ((word >> b) & 1u) ? -a : a;
        }
    }
    t[m * N + row] = acc * float(mid[row]);
"""

_STAGE2_SRC = """
    uint row = thread_position_in_grid.x;   // which output feature i
    uint m   = thread_position_in_grid.y;
    if (row >= N) return;

    const device uint*  bits = U_bits + row * W;
    const device float* tm   = t + m * K;

    float acc = 0.0f;
    for (uint w = 0; w < W; ++w) {
        uint word = bits[w];
        uint base = w * 32;
        uint n = min(32u, K - base);
        for (uint b = 0; b < n; ++b) {
            float a = tm[base + b];
            acc += ((word >> b) & 1u) ? -a : a;
        }
    }
    y[m * N + row] = acc * float(u1[row]);
"""

_stage1 = mx.fast.metal_kernel(
    name="lb_naive_stage1", input_names=["x", "V_bits", "v2", "mid"],
    output_names=["t"], source=_STAGE1_SRC)
_stage2 = mx.fast.metal_kernel(
    name="lb_naive_stage2", input_names=["t", "U_bits", "u1"],
    output_names=["y"], source=_STAGE2_SRC)


def _tg(n: int) -> int:
    return max(1, min(256, n))


def forward(x: mx.array, layer: DeviceLayer) -> mx.array:
    x2, restore = as_2d(x)
    M = x2.shape[0]
    y = None
    for p in layer.paths:
        K, R, OUT = p.in_features, p.rank, p.out_features
        (t,) = _stage1(
            inputs=[x2, p.V_bits, p.v2, p.mid],
            template=[("T", x2.dtype), ("K", K), ("N", R), ("W", (K + 31) // 32)],
            grid=(R, M, 1), threadgroup=(_tg(R), 1, 1),
            output_shapes=[(M, R)], output_dtypes=[mx.float32])
        (yp,) = _stage2(
            inputs=[t, p.U_bits, p.u1],
            template=[("K", R), ("N", OUT), ("W", (R + 31) // 32)],
            grid=(OUT, M, 1), threadgroup=(_tg(OUT), 1, 1),
            output_shapes=[(M, OUT)], output_dtypes=[mx.float32])
        y = yp if y is None else y + yp
    if layer.bias is not None:
        y = y + layer.bias.astype(mx.float32)
    return restore(y.astype(x.dtype))
