"""`lut3`: `lut2` tuned for occupancy -- several rows per thread, smaller tables.

Profiling lut/lut2 showed the big layers running at only ~40 GB/s even when
the weights sat in cache, i.e. the kernel itself was the bottleneck, not
DRAM.  Two reasons:

* Occupancy.  A GPU core hides memory latency by switching between many
  resident SIMD-groups.  Each threadgroup reserved a 16 KB table in
  threadgroup memory, so only ~2 threadgroups (8 SIMD-groups) fit on a
  core at once -- too few to keep enough loads in flight.
* Too little independent work per thread.  Each thread had one
  accumulator, so every lookup-add waited on the previous one.

`lut3` makes three knobs template parameters (and `CFG` lets the benchmark
sweep them):

* KC  -- inner-dim chunk per table.  Smaller chunk -> smaller table ->
        more threadgroups resident per core.
* RPT -- rows per thread.  A thread handles RPT rows (row, row+TG, ...)
        with RPT independent accumulators that share the *same* table
        lookups' address math; the table build is amortised over TG*RPT
        rows.  Weight loads stay coalesced (consecutive lanes = consecutive
        rows for each of the RPT rows).
* TG  -- threads per threadgroup.

Everything else (2 dispatches, chunk partials summed in stage 2's table
build, residual + bias + output cast fused) is as in lut2.
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d
from .lut import _prepare

_BUILD = """
    for (uint g = tid; g < KC / 4; g += TG) {
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

# RPT rows per thread; acc[j] belongs to row0 + j*TG.
_LOOKUP = """
    for (uint w = w0; w < w1; ++w) {
        threadgroup const float* t = T + (w - w0) * 128;
        uint word[RPT];
        #pragma unroll
        for (uint j = 0; j < RPT; ++j)
            word[j] = (row0 + j * TG < N) ? bits[w * N + j * TG] : 0u;
        #pragma unroll
        for (uint n = 0; n < 8; ++n) {
            #pragma unroll
            for (uint j = 0; j < RPT; ++j)
                acc[j] += t[n * 16 + ((word[j] >> (4 * n)) & 15u)];
        }
    }
"""

_STAGE1 = """
    const uint tid   = thread_position_in_threadgroup.x;
    const uint row0  = threadgroup_position_in_grid.x * TG * RPT + tid;
    const uint path  = threadgroup_position_in_grid.y / NCH;
    const uint chunk = threadgroup_position_in_grid.y % NCH;
    const uint m     = threadgroup_position_in_grid.z;
    const uint kbase = chunk * KC;
    const device TX* xm = x + m * K;
    const device TS* sc = v2 + path * K;

    threadgroup float T[(KC / 4) * 16];
    #define LOAD(k) (float(xm[k]) * float(sc[k]))
""" + _BUILD + """
    #undef LOAD
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row0 >= N) return;

    const uint w0 = kbase / 32, w1 = min(w0 + KC / 32, (uint)W);
    const device uint* bits = VT + path * W * N + row0;
    float acc[RPT];
    for (uint j = 0; j < RPT; ++j) acc[j] = 0.0f;
""" + _LOOKUP + """
    device float* o = tpart + ((m * NP + path) * NCH + chunk) * N;
    for (uint j = 0; j < RPT; ++j) {
        uint row = row0 + j * TG;
        if (row < N) o[row] = acc[j] * float(mid[path * N + row]);
    }
"""

_STAGE2 = """
    const uint tid  = thread_position_in_threadgroup.x;
    const uint row0 = threadgroup_position_in_grid.x * TG * RPT + tid;
    const uint m    = threadgroup_position_in_grid.z;
    threadgroup float T[(KC / 4) * 16];

    float y[RPT];
    for (uint j = 0; j < RPT; ++j) y[j] = 0.0f;
    for (uint path = 0; path < NP; ++path) {
        const device float* tp = tpart + (m * NP + path) * NCH1 * K;
        const device uint* bits = UT + path * W * N + row0;
        float acc[RPT];
        for (uint j = 0; j < RPT; ++j) acc[j] = 0.0f;
        for (uint kbase = 0; kbase < K; kbase += KC) {
            #define LOAD(k) partial_sum(tp, k)
""" + _BUILD + """
            #undef LOAD
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (row0 < N) {
                const uint w0 = kbase / 32, w1 = min(w0 + KC / 32, (uint)W);
""" + _LOOKUP + """
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint j = 0; j < RPT; ++j) {
            uint row = row0 + j * TG;
            if (row < N) y[j] += acc[j] * float(u1[path * N + row]);
        }
    }
    for (uint j = 0; j < RPT; ++j) {
        uint row = row0 + j * TG;
        if (row >= N) break;
        float v = y[j];
        if (HAS_BIAS) v += float(bias[row]);
        out[m * N + row] = static_cast<TO>(v);
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

_k1 = mx.fast.metal_kernel(name="lb_lut3_s1", input_names=["x", "VT", "v2", "mid"],
                           output_names=["tpart"], source=_STAGE1)
_k2 = mx.fast.metal_kernel(name="lb_lut3_s2", input_names=["tpart", "UT", "u1", "bias"],
                           output_names=["out"], source=_STAGE2, header=_HEADER2)

# Tunables (the benchmark sweep edits these).  MIN_TGS: keep splitting the
# stage-1 inner dim until at least this many threadgroups exist.
CFG = dict(TG1=128, RPT1=2, KC1=512, TG2=64, RPT2=2, KC2=512, MIN_TGS=128)


def _ceil(a, b):
    return (a + b - 1) // b


def forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features, cfg=None):
    c = {**CFG, **(cfg or {})}
    x2, restore = as_2d(x)
    M, NP = x2.shape[0], VT.shape[0]
    K, R, OUT = in_features, rank, out_features

    tg1, rpt1 = c["TG1"], c["RPT1"]
    n1 = _ceil(R, tg1 * rpt1)
    kc1 = c["KC1"]
    while kc1 > 128 and n1 * NP * _ceil(K, kc1) < c["MIN_TGS"]:
        kc1 //= 2
    nch1 = _ceil(K, kc1)
    (tpart,) = _k1(
        inputs=[x2, VT, v2, mid],
        template=[("TX", x2.dtype), ("TS", v2.dtype), ("K", K), ("N", R), ("W", _ceil(K, 32)),
                  ("KC", kc1), ("NCH", nch1), ("NP", NP), ("TG", tg1), ("RPT", rpt1)],
        grid=(n1 * tg1, NP * nch1, M), threadgroup=(tg1, 1, 1),
        output_shapes=[(M, NP, nch1, R)], output_dtypes=[mx.float32])

    tg2, rpt2 = c["TG2"], c["RPT2"]
    n2 = _ceil(OUT, tg2 * rpt2)
    kc2 = min(c["KC2"], _ceil(R, 32) * 32)
    (y,) = _k2(
        inputs=[tpart, UT, u1, bias if bias is not None else u1],
        template=[("TO", x2.dtype), ("K", R), ("N", OUT), ("W", _ceil(R, 32)), ("KC", kc2),
                  ("NCH1", nch1), ("NP", NP), ("TG", tg2), ("RPT", rpt2),
                  ("HAS_BIAS", bias is not None)],
        grid=(n2 * tg2, 1, M), threadgroup=(tg2, 1, 1),
        output_shapes=[(M, OUT)], output_dtypes=[x2.dtype])
    return restore(y)


def forward(x: mx.array, layer: DeviceLayer, cfg=None) -> mx.array:
    c = _prepare(layer)
    return forward_arrays(x, c["VT"], c["UT"], c["v2"], c["mid"], c["u1"], layer.bias,
                          layer.in_features, layer.rank, layer.out_features, cfg)
