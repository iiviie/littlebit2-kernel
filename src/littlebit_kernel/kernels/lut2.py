"""`lut2`: the `lut` kernel with 2 GPU dispatches per layer instead of 5.

At batch 1, sub-1-bit layers are tiny (tens to hundreds of KB), so the
fixed cost per GPU dispatch is comparable to the useful work.  `lut` used:

    zero-fill t  ->  stage 1 (atomics)  ->  zero-fill y  ->  stage 2 (atomics)  ->  cast to fp16

(MLX implements a kernel's `init_value` as a separate fill dispatch, and the
float32 atomic output needs a cast.)  `lut2` removes three of them:

* Stage 1 still splits the inner dimension into chunks for parallelism, but
  instead of atomically adding into t it writes one partial vector per chunk
  (`tpart[m, path, chunk, :]`, plain stores, no zero-fill needed).
* Stage 2 sums those partials *while it builds its lookup table* -- it has
  to read t anyway, so the reduction is free.  It does not split its inner
  dimension (r), it loops over r in table-sized chunks instead, so every
  output element is produced by exactly one thread: no atomics, no fill.
  That thread also adds the residual path and the bias and writes the final
  dtype directly, so there is no cast.

The table trick itself is unchanged; see lut.py.
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d
from .lut import _prepare

# Builds the signed-sum table for groups [g0, g0+G) of the vector `a`
# (TG threads cooperate).  LOAD(k) must yield a[k] (0 past the end).
_BUILD = """
    for (uint g = tid; g < G; g += TG) {
        float a[4];
        for (uint i = 0; i < 4; ++i) {
            uint k = kbase + 4 * g + i;
            a[i] = (k < K) ? LOAD(k) : 0.0f;
        }
        threadgroup float* t = T + g * 16;
        t[0] = a[0] + a[1] + a[2] + a[3];
        for (uint p = 1; p < 16; ++p)
            t[p] = t[p & (p - 1)] - 2.0f * a[ctz(p)];
    }
"""

_LOOKUP = """
    for (uint w = w0; w < w1; ++w) {
        uint word = bits[w * N];
        threadgroup const float* t = T + (w - w0) * 128;
        #pragma unroll
        for (uint n = 0; n < 8; ++n)
            acc += t[n * 16 + ((word >> (4 * n)) & 15u)];
    }
"""

_STAGE1 = """
    const uint tid   = thread_position_in_threadgroup.x;
    const uint row   = threadgroup_position_in_grid.x * TG + tid;
    const uint path  = threadgroup_position_in_grid.y / NCH;
    const uint chunk = threadgroup_position_in_grid.y % NCH;
    const uint m     = threadgroup_position_in_grid.z;
    const uint kbase = chunk * KC;
    const uint G     = KC / 4;
    const device TX* xm = x + m * K;
    const device TS* sc = v2 + path * K;

    threadgroup float T[(KC / 4) * 16];
    #define LOAD(k) (float(xm[k]) * float(sc[k]))
""" + _BUILD + """
    #undef LOAD
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= N) return;

    const uint w0 = kbase / 32, w1 = min(w0 + KC / 32, (uint)W);
    const device uint* bits = VT + path * W * N + row;
    float acc = 0.0f;
""" + _LOOKUP + """
    tpart[((m * NP + path) * NCH + chunk) * N + row] = acc * float(mid[path * N + row]);
"""

# Stage 2: K = r (inner), N = out rows.  NCH1 = stage-1 chunk count.
_STAGE2 = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x * TG + tid;
    const uint m   = threadgroup_position_in_grid.z;
    const uint G   = KC / 4;
    threadgroup float T[(KC / 4) * 16];

    float y = 0.0f;
    for (uint path = 0; path < NP; ++path) {
        const device float* tp = tpart + (m * NP + path) * NCH1 * K;
        const device uint* bits = UT + path * W * N + row;
        float acc = 0.0f;
        for (uint kbase = 0; kbase < K; kbase += KC) {
            // t[k] = sum over stage-1 chunks of the partials
            #define LOAD(k) partial_sum(tp, k)
""" + _BUILD + """
            #undef LOAD
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row < N) {
                const uint w0 = kbase / 32, w1 = min(w0 + KC / 32, (uint)W);
""" + _LOOKUP + """
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);   // before T is rebuilt
        }
        if (row < N) y += acc * float(u1[path * N + row]);
    }
    if (row >= N) return;
    if (HAS_BIAS) y += float(bias[row]);
    out[m * N + row] = static_cast<TO>(y);
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

_k1 = mx.fast.metal_kernel(name="lb_lut2_s1", input_names=["x", "VT", "v2", "mid"],
                           output_names=["tpart"], source=_STAGE1)
_k2 = mx.fast.metal_kernel(name="lb_lut2_s2", input_names=["tpart", "UT", "u1", "bias"],
                           output_names=["out"], source=_STAGE2, header=_HEADER2)

TG1 = 128
TG2 = 64
KC_MAX = 1024


def _choose_kc(K, n_tiles, NP):
    kc = KC_MAX
    while kc > 128 and n_tiles * NP * ((K + kc - 1) // kc) < 64:
        kc //= 2
    return kc


def forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features):
    x2, restore = as_2d(x)
    M, NP = x2.shape[0], VT.shape[0]
    K, R, OUT = in_features, rank, out_features

    n1 = (R + TG1 - 1) // TG1
    kc1 = _choose_kc(K, n1, NP)
    nch1 = (K + kc1 - 1) // kc1
    (tpart,) = _k1(
        inputs=[x2, VT, v2, mid],
        template=[("TX", x2.dtype), ("TS", v2.dtype), ("K", K), ("N", R), ("W", (K + 31) // 32),
                  ("KC", kc1), ("NCH", nch1), ("NP", NP), ("TG", TG1)],
        grid=(n1 * TG1, NP * nch1, M), threadgroup=(TG1, 1, 1),
        output_shapes=[(M, NP, nch1, R)], output_dtypes=[mx.float32])

    n2 = (OUT + TG2 - 1) // TG2
    kc2 = min(KC_MAX, ((R + 31) // 32) * 32)
    (y,) = _k2(
        inputs=[tpart, UT, u1, bias if bias is not None else u1],
        template=[("TO", x2.dtype), ("K", R), ("N", OUT), ("W", (R + 31) // 32), ("KC", kc2),
                  ("NCH1", nch1), ("NP", NP), ("TG", TG2), ("HAS_BIAS", bias is not None)],
        grid=(n2 * TG2, 1, M), threadgroup=(TG2, 1, 1),
        output_shapes=[(M, OUT)], output_dtypes=[x2.dtype])
    return restore(y)


def forward(x: mx.array, layer: DeviceLayer) -> mx.array:
    c = _prepare(layer)
    return forward_arrays(x, c["VT"], c["UT"], c["v2"], c["mid"], c["u1"], layer.bias,
                          layer.in_features, layer.rank, layer.out_features)
