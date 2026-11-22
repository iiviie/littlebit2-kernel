"""`fused`: the whole layer in ONE GPU dispatch (for small layers).

Measured on M1 Pro with MLX: every *dependent* dispatch costs ~6-7 us of
fixed latency (an empty kernel in a dependency chain).  A 0.1-bpw layer of
a small model is only a few KB-tens of KB of bits, so a two-stage kernel
spends most of its time waiting on that latency twice.

Stage 2 needs *all* of t = stage-1 output before any output row can be
finished, and Metal has no grid-wide barrier.  The trick here: every
threadgroup computes the whole of stage 1 itself (redundantly), keeps t in
threadgroup memory, and then computes only its own slice of output rows.

    phase A (every TG):  t = mid * signV @ (x * v2)        -> threadgroup memory
    phase B (TG slice):  y[rows] = u1 * signU[rows] @ t    (+ residual, + bias)

The redundant work is (#threadgroups) x r x in weight bits -- it comes from
cache after the first threadgroup, but it costs ALU time, so this only wins
when r*in is small.  `kernels.auto` picks between `fused` and `lut3` with a
size threshold measured by the benchmark.

Both phases use the 4-bit lookup tables from lut.py; phase A loops over the
inner dimension in KC-sized chunks so the table stays small.
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d
from .lut import _prepare

_SRC = """
    const uint tid  = thread_position_in_threadgroup.x;
    const uint m    = threadgroup_position_in_grid.z;
    const device TX* xm = x + m * K;

    threadgroup float T[(KC / 4) * 16];   // lookup table for one chunk
    threadgroup float tv[NP * R];         // stage-1 result t, all paths

    // ---------------- phase A: t = mid * signV @ (x * v2) ----------------
    for (uint path = 0; path < NP; ++path) {
        float acc[Q1];
        for (uint q = 0; q < Q1; ++q) acc[q] = 0.0f;
        const device TS* sc = v2 + path * K;
        const device uint* vb = VT + path * WK * R;
        for (uint kbase = 0; kbase < K; kbase += KC) {
            // build table for inputs [kbase, kbase+KC)
            for (uint g = tid; g < KC / 4; g += TG) {
                float a[4];
                for (uint i = 0; i < 4; ++i) {
                    uint k = kbase + 4 * g + i;
                    a[i] = (k < K) ? float(xm[k]) * float(sc[k]) : 0.0f;
                }
                threadgroup float* t = T + g * 16;
                t[0] = a[0] + a[1] + a[2] + a[3];
                for (uint p = 1; p < 16; ++p) t[p] = t[p & (p - 1)] - 2.0f * a[ctz(p)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint w0 = kbase / 32, w1 = min(w0 + KC / 32, (uint)WK);
            for (uint w = w0; w < w1; ++w) {
                threadgroup const float* t = T + (w - w0) * 128;
                #pragma unroll
                for (uint q = 0; q < Q1; ++q) {
                    uint row = tid + q * TG;
                    if (row < R) {
                        uint word = vb[w * R + row];
                        #pragma unroll
                        for (uint n = 0; n < 8; ++n)
                            acc[q] += t[n * 16 + ((word >> (4 * n)) & 15u)];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint q = 0; q < Q1; ++q) {
            uint row = tid + q * TG;
            if (row < R) tv[path * R + row] = acc[q] * float(mid[path * R + row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------------- phase B: y = sum_p u1_p * signU_p @ t_p ----------------
    const uint row0 = threadgroup_position_in_grid.x * TG * RPT + tid;
    float y[RPT];
    for (uint j = 0; j < RPT; ++j) y[j] = 0.0f;
    for (uint path = 0; path < NP; ++path) {
        float acc[RPT];
        for (uint j = 0; j < RPT; ++j) acc[j] = 0.0f;
        const device uint* ub = UT + path * WR * N + row0;
        for (uint kbase = 0; kbase < R; kbase += KC) {
            for (uint g = tid; g < KC / 4; g += TG) {
                float a[4];
                for (uint i = 0; i < 4; ++i) {
                    uint k = kbase + 4 * g + i;
                    a[i] = (k < R) ? tv[path * R + k] : 0.0f;
                }
                threadgroup float* t = T + g * 16;
                t[0] = a[0] + a[1] + a[2] + a[3];
                for (uint p = 1; p < 16; ++p) t[p] = t[p & (p - 1)] - 2.0f * a[ctz(p)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint w0 = kbase / 32, w1 = min(w0 + KC / 32, (uint)WR);
            for (uint w = w0; w < w1; ++w) {
                threadgroup const float* t = T + (w - w0) * 128;
                #pragma unroll
                for (uint j = 0; j < RPT; ++j) {
                    if (row0 + j * TG < N) {
                        uint word = ub[w * N + j * TG];
                        #pragma unroll
                        for (uint n = 0; n < 8; ++n)
                            acc[j] += t[n * 16 + ((word >> (4 * n)) & 15u)];
                    }
                }
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

_kernel = mx.fast.metal_kernel(name="lb_fused", input_names=["x", "VT", "v2", "mid", "UT", "u1", "bias"],
                               output_names=["out"], source=_SRC)

CFG = dict(TG=256, RPT=1, KC=512)
# threadgroup memory budget (bytes) for the t vector; beyond this, use lut3
MAX_T_BYTES = 12 * 1024


def _ceil(a, b):
    return (a + b - 1) // b


def supported(in_features, rank, n_paths) -> bool:
    return n_paths * rank * 4 <= MAX_T_BYTES


def forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features, cfg=None):
    c = {**CFG, **(cfg or {})}
    x2, restore = as_2d(x)
    M, NP = x2.shape[0], VT.shape[0]
    K, R, OUT = in_features, rank, out_features
    assert supported(K, R, NP), "layer too large for the fused kernel"
    tg, rpt, kc = c["TG"], c["RPT"], c["KC"]
    (y,) = _kernel(
        inputs=[x2, VT, v2, mid, UT, u1, bias if bias is not None else u1],
        template=[("TX", x2.dtype), ("TS", v2.dtype), ("TO", x2.dtype), ("K", K), ("R", R), ("N", OUT),
                  ("WK", _ceil(K, 32)), ("WR", _ceil(R, 32)), ("NP", NP), ("TG", tg),
                  ("RPT", rpt), ("KC", kc), ("Q1", _ceil(R, tg)), ("HAS_BIAS", bias is not None)],
        grid=(_ceil(OUT, tg * rpt) * tg, 1, M), threadgroup=(tg, 1, 1),
        output_shapes=[(M, OUT)], output_dtypes=[x2.dtype])
    return restore(y)


def forward(x: mx.array, layer: DeviceLayer, cfg=None) -> mx.array:
    c = _prepare(layer)
    if not supported(layer.in_features, layer.rank, len(layer.paths)):
        # fall back so the correctness gate can still run every shape
        from . import lut3
        return lut3.forward(x, layer)
    return forward_arrays(x, c["VT"], c["UT"], c["v2"], c["mid"], c["u1"], layer.bias,
                          layer.in_features, layer.rank, layer.out_features, cfg)
