"""`lut5`: `lut4`'s latency hiding with `lut2`'s low dispatch count.

What we learned so far (NOTES.md):

* big layers: splitting BOTH stages over the inner dimension (for
  occupancy) and prefetching weight words into registers wins (`lut4`);
* small layers: the time is mostly fixed per-dispatch latency (~6-7 us per
  dependent dispatch on M1 Pro + MLX), so `lut4`'s 5 dispatches (2 zero
  fills for the atomic outputs, 2 kernels, 1 cast) hurt.

`lut5` gets the split-K parallelism without atomics, hence without fills:

    stage 1:  tpart[m, path, c1, :]  = per-chunk partial of t        (plain stores)
    stage 2:  each threadgroup takes (row tile, r-chunk c2); its table is built
              from  t = sum_c1 tpart  (the reduction is free: the values are
              read once to build the table anyway).
              NCH2 == 1 -> writes final y (+bias, cast)  -> 2 dispatches total
              NCH2  > 1 -> writes ypart[m, c2, :]; a tiny reduce kernel sums
                           the NCH2 partials, adds bias, casts -> 3 dispatches

NCH2 is only > 1 when there are too few output-row tiles to fill the GPU
(big r with modest `out`).  Both stages prefetch all their weight words
into registers before building the table (see lut4.py).
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d
from .lut import _prepare

# Shared pieces --------------------------------------------------------------
# PREFETCH: words [w0, w0+KC/32) of this thread's row (column `row` of the
# word-major matrix `B`) into registers, 0 past the end.
_PREFETCH = """
    uint wb[KC / 32];
    {
        const device uint* bp = B + row;
        #pragma unroll
        for (uint w = 0; w < KC / 32; ++w)
            wb[w] = (row < N && w0 + w < W) ? bp[(w0 + w) * N] : 0u;
    }
"""
# BUILD: 16-entry signed-sum table per group of 4 inputs, inputs via LOAD(k).
_BUILD = """
    for (uint g = tid; g < KC / 4; g += TG) {
        float a[4];
        for (uint i = 0; i < 4; ++i) {
            uint k = kbase + 4 * g + i;
            a[i] = (k < K) ? LOAD(k) : 0.0f;
        }
        // built in registers, then stored (see lut4.py)
        float e[16];
        e[0] = a[0] + a[1] + a[2] + a[3];
        #pragma unroll
        for (uint p = 1; p < 16; ++p) e[p] = e[p & (p - 1)] - 2.0f * a[ctz(p)];
        threadgroup float4* t4 = (threadgroup float4*)(T + g * 16);
        #pragma unroll
        for (uint q = 0; q < 4; ++q) t4[q] = float4(e[4 * q], e[4 * q + 1], e[4 * q + 2], e[4 * q + 3]);
    }
"""
# LOOKUP: acc += table[nibble] for all prefetched words.
_LOOKUP = """
    {
        float a0 = 0.0f, a1 = 0.0f;
        #pragma unroll
        for (uint w = 0; w < KC / 32; ++w) {
            uint word = wb[w];
            threadgroup const float* t = T + w * 128;
            #pragma unroll
            for (uint n = 0; n < 8; n += 2) {
                a0 += t[n * 16 + ((word >> (4 * n)) & 15u)];
                a1 += t[(n + 1) * 16 + ((word >> (4 * n + 4)) & 15u)];
            }
        }
        acc = a0 + a1;
    }
"""

_STAGE1 = """
    const uint tid   = thread_position_in_threadgroup.x;
    const uint row   = threadgroup_position_in_grid.x * TG + tid;
    const uint path  = threadgroup_position_in_grid.y / NCH;
    const uint chunk = threadgroup_position_in_grid.y % NCH;
    const uint m     = threadgroup_position_in_grid.z;
    const uint kbase = chunk * KC;
    const uint w0    = kbase / 32;
    const device uint* B = VT + path * W * N;
""" + _PREFETCH + """
    const device TX* xm = x + m * K;
    const device TS* sc = v2 + path * K;
    threadgroup float T[(KC / 4) * 16];
    #define LOAD(k) (float(xm[k]) * float(sc[k]))
""" + _BUILD + """
    #undef LOAD
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= N) return;
    float acc;
""" + _LOOKUP + """
    tpart[((m * NP + path) * NCH + chunk) * N + row] = acc * float(mid[path * N + row]);
"""

# Stage 2: K = r, N = out.  Grid y = r-chunk c2.  Loops over paths inside.
_STAGE2 = """
    const uint tid   = thread_position_in_threadgroup.x;
    const uint row   = threadgroup_position_in_grid.x * TG + tid;
    const uint chunk = threadgroup_position_in_grid.y;
    const uint m     = threadgroup_position_in_grid.z;
    const uint kbase = chunk * KC;
    const uint w0    = kbase / 32;
    threadgroup float T[(KC / 4) * 16];

    float y = 0.0f;
    for (uint path = 0; path < NP; ++path) {
        const device uint* B = UT + path * W * N;
""" + _PREFETCH + """
        const device float* tp = tpart + (m * NP + path) * NCH1 * K;
        #define LOAD(k) partial_sum(tp, k)
""" + _BUILD + """
        #undef LOAD
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc;
""" + _LOOKUP + """
        if (row < N) y += acc * float(u1[path * N + row]);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // T is rebuilt next path
    }
    if (row >= N) return;
    if (FINAL) {
        if (HAS_BIAS) y += float(bias[row]);
        out[m * N + row] = static_cast<TO>(y);
    } else {
        out[(m * NCH + chunk) * N + row] = static_cast<TO>(y);
    }
"""

_HEADER2 = """
template <uint NCH1, uint K>
inline float partial_sum_impl(const device float* tp, uint k) {
    float s = 0.0f;
    for (uint c = 0; c < NCH1; ++c) s += tp[c * K + k];
    return s;
}
#define partial_sum(tp, k) partial_sum_impl<NCH1, K>(tp, k)
"""

# Reduce NCH2 partial outputs, add bias, cast.
_REDUCE = """
    const uint row = thread_position_in_grid.x;
    const uint m   = thread_position_in_grid.y;
    if (row >= N) return;
    float s = 0.0f;
    for (uint c = 0; c < NCH; ++c) s += ypart[(m * NCH + c) * N + row];
    if (HAS_BIAS) s += float(bias[row]);
    out[m * N + row] = static_cast<TO>(s);
"""

_k1 = mx.fast.metal_kernel(name="lb_lut5_s1", input_names=["x", "VT", "v2", "mid"],
                           output_names=["tpart"], source=_STAGE1)
_k2 = mx.fast.metal_kernel(name="lb_lut5_s2", input_names=["tpart", "UT", "u1", "bias"],
                           output_names=["out"], source=_STAGE2, header=_HEADER2)
_kr = mx.fast.metal_kernel(name="lb_lut5_red", input_names=["ypart", "bias"],
                           output_names=["out"], source=_REDUCE)

CFG = dict(TG=128, KC1=512, KC2=512, MIN_TGS=64)


def _ceil(a, b):
    return (a + b - 1) // b


def _split(K, tiles, kc, min_tgs, mult=1):
    """Halve the chunk until tiles * chunks >= min_tgs (or chunk hits 128)."""
    while kc > 128 and tiles * mult * _ceil(K, kc) < min_tgs:
        kc //= 2
    return kc, _ceil(K, kc)


def forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features, cfg=None):
    c = {**CFG, **(cfg or {})}
    x2, restore = as_2d(x)
    M, NP = x2.shape[0], VT.shape[0]
    K, R, OUT = in_features, rank, out_features
    tg = c["TG"]

    tiles1 = _ceil(R, tg)
    kc1, nch1 = _split(K, tiles1, c["KC1"], c["MIN_TGS"], NP)
    (tpart,) = _k1(
        inputs=[x2, VT, v2, mid],
        template=[("TX", x2.dtype), ("TS", v2.dtype), ("K", K), ("N", R), ("W", _ceil(K, 32)),
                  ("KC", kc1), ("NCH", nch1), ("NP", NP), ("TG", tg)],
        grid=(tiles1 * tg, NP * nch1, M), threadgroup=(tg, 1, 1),
        output_shapes=[(M, NP, nch1, R)], output_dtypes=[mx.float32])

    tiles2 = _ceil(OUT, tg)
    # chunk must stay a multiple of 32 when halved: use a power of two >= 32
    kc2 = min(c["KC2"], max(32, 1 << (R - 1).bit_length()))
    kc2, nch2 = _split(R, tiles2, kc2, c["MIN_TGS"])
    final = nch2 == 1
    b = bias if bias is not None else u1
    (y,) = _k2(
        inputs=[tpart, UT, u1, b],
        template=[("TO", x2.dtype if final else mx.float32), ("K", R), ("N", OUT), ("W", _ceil(R, 32)),
                  ("KC", kc2), ("NCH", nch2), ("NCH1", nch1), ("NP", NP), ("TG", tg),
                  ("FINAL", final), ("HAS_BIAS", bias is not None)],
        grid=(tiles2 * tg, nch2, M), threadgroup=(tg, 1, 1),
        output_shapes=[(M, OUT) if final else (M, nch2, OUT)],
        output_dtypes=[x2.dtype if final else mx.float32])
    if not final:
        (y,) = _kr(
            inputs=[y, b],
            template=[("TO", x2.dtype), ("N", OUT), ("NCH", nch2), ("HAS_BIAS", bias is not None)],
            grid=(OUT, M, 1), threadgroup=(min(256, OUT), 1, 1),
            output_shapes=[(M, OUT)], output_dtypes=[x2.dtype])
    return restore(y)


def forward(x: mx.array, layer: DeviceLayer, cfg=None) -> mx.array:
    c = _prepare(layer)
    return forward_arrays(x, c["VT"], c["UT"], c["v2"], c["mid"], c["u1"], layer.bias,
                          layer.in_features, layer.rank, layer.out_features, cfg)
