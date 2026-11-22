"""`lut4`: `lut` + weight prefetch into registers (latency hiding).

Diagnosis of `lut` on big layers: stage 1 of Llama-3-8B down_proj at 1 bpw
did ~4 table lookups per clock per GPU core -- far below what the ALUs or
threadgroup memory can do.  That number matches a *latency*-bound loop:
each thread loaded one weight word, waited for DRAM (~hundreds of cycles),
did its 8 lookups, then loaded the next word; and only ~8 SIMD-groups fit
on a core (16 KB of table per threadgroup), too few to cover the wait.

Fix: each thread issues the loads for *all* KC/32 words of its chunk into
registers at the very start, before building the table.  The memory system
then has 32 loads per thread in flight, and their latency overlaps the
table build and the barrier.  The lookup loop then runs out of registers.
Everything else (split-K with float atomics, fused residual/bias) is `lut`.
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d
from .lut import _prepare

_SRC = """
    // ---- who am I -------------------------------------------------------
    const uint tid   = thread_position_in_threadgroup.x;   // 0 .. TG-1
    const uint tile  = threadgroup_position_in_grid.x;     // which TG rows
    const uint pc    = threadgroup_position_in_grid.y;     // path * NCH + chunk
    const uint m     = threadgroup_position_in_grid.z;     // activation row
    const uint path  = pc / NCH;
    const uint chunk = pc % NCH;
    const uint k0    = chunk * KC;                         // first element
    const uint row   = tile * TG + tid;

    // Input vector for this path: stage 1 shares x across paths, stage 2 has
    // one t per path.
    const device TX* xm = x + (X_PER_PATH ? (m * NP + path) : m) * K;
    const device TS* sc = pre + path * K;

    // ---- 0. issue this thread's weight loads FIRST ------------------------
    // They don't depend on the table, so their DRAM latency (hundreds of
    // cycles) overlaps the table build and the barrier below.  All KC/32
    // loads are in flight at once instead of one per loop iteration.
    const uint w0 = k0 / 32;
    const device uint* bits = bitsT + path * W * N + row;   // column `row`
    uint wb[KC / 32];
    #pragma unroll
    for (uint w = 0; w < KC / 32; ++w)
        wb[w] = (row < N && w0 + w < W) ? bits[(w0 + w) * N] : 0u;

    // ---- 1. build the 16-entry signed-sum table for every group of 4 ----
    threadgroup float T[(KC / 4) * 16];
    for (uint g = tid; g < KC / 4; g += TG) {
        float a[4];
        for (uint i = 0; i < 4; ++i) {
            uint k = k0 + 4 * g + i;
            float v = 0.0f;                    // past the end -> 0
            if (k < K) {
                v = float(xm[k]);
                if (PRE) v *= float(sc[k]);
            }
            a[i] = v;
        }
        // T[p] = sum of (+a_i if bit i of p is 0, -a_i if 1).  Built in
        // registers (fully unrolled -> constant indices) and then stored: the
        // obvious version that reads back T[p & (p-1)] from threadgroup
        // memory makes 15 dependent memory round trips and was the single
        // biggest cost of the kernel (NOTES.md, "table build").
        float e[16];
        e[0] = a[0] + a[1] + a[2] + a[3];
        #pragma unroll
        for (uint p = 1; p < 16; ++p) e[p] = e[p & (p - 1)] - 2.0f * a[ctz(p)];
        threadgroup float4* t4 = (threadgroup float4*)(T + g * 16);
        #pragma unroll
        for (uint q = 0; q < 4; ++q) t4[q] = float4(e[4 * q], e[4 * q + 1], e[4 * q + 2], e[4 * q + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= N) return;

    // ---- 2. lookups from the prefetched words ---------------------------
    // Words past the end of the row are 0 -> they look up entry 0 of an
    // all-zero table (groups past K were built from a = 0), adding nothing.
    // Two accumulators halve the dependent-add chain.
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint w = 0; w < KC / 32; ++w) {
        uint word = wb[w];
        threadgroup const float* t = T + w * 128;
        #pragma unroll
        for (uint n = 0; n < 8; n += 2) {
            acc0 += t[n * 16 + ((word >> (4 * n)) & 15u)];
            acc1 += t[(n + 1) * 16 + ((word >> (4 * n + 4)) & 15u)];
        }
    }
    float acc = acc0 + acc1;

    // ---- 3. scale and accumulate into the output -------------------------
    float v = acc * float(post[path * N + row]);
    if (HAS_BIAS && path == 0 && chunk == 0) v += float(bias[row]);
    atomic_fetch_add_explicit(&out[(m * OUT_NP + (OUT_PER_PATH ? path : 0)) * N + row],
                              v, memory_order_relaxed);
"""

_kernel = mx.fast.metal_kernel(
    name="lb_lut4_rowdot",
    input_names=["x", "bitsT", "pre", "post", "bias"],
    output_names=["out"], source=_SRC, atomic_outputs=True)

# KC1/KC2: inner-dim chunk for stage 1 / stage 2 (A/B-tuned, see NOTES.md)
CFG = dict(TG=128, KC1=512, KC2=512, MIN_TGS=64)


def _rowdot(x, bitsT, pre, post, bias, K, N, NP, M, prescale, x_per_path, out_per_path, cfg, kc):
    tg = cfg["TG"]
    n_tiles = (N + tg - 1) // tg
    while kc > 128 and n_tiles * NP * ((K + kc - 1) // kc) < cfg["MIN_TGS"]:
        kc //= 2
    nch = (K + kc - 1) // kc
    out_np = NP if out_per_path else 1
    (out,) = _kernel(
        inputs=[x, bitsT, pre, post, bias if bias is not None else post],
        template=[("TX", x.dtype), ("TS", pre.dtype), ("K", K), ("N", N), ("W", (K + 31) // 32),
                  ("KC", kc), ("NCH", nch), ("NP", NP), ("TG", tg), ("PRE", prescale),
                  ("X_PER_PATH", x_per_path), ("OUT_PER_PATH", out_per_path),
                  ("OUT_NP", out_np), ("HAS_BIAS", bias is not None)],
        grid=(n_tiles * tg, NP * nch, M), threadgroup=(tg, 1, 1),
        output_shapes=[(M, out_np, N)], output_dtypes=[mx.float32], init_value=0.0)
    return out


def forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features, cfg=None):
    c = {**CFG, **(cfg or {})}
    x2, restore = as_2d(x)
    M, NP = x2.shape[0], VT.shape[0]
    K, R, OUT = in_features, rank, out_features
    t = _rowdot(x2, VT, v2, mid, None, K, R, NP, M, True, False, True, c, c["KC1"])
    y = _rowdot(t, UT, u1, u1, bias, R, OUT, NP, M, False, True, False, c, c["KC2"])
    return restore(y.reshape(M, OUT).astype(x.dtype))


def forward(x: mx.array, layer: DeviceLayer, cfg=None) -> mx.array:
    c = _prepare(layer)
    return forward_arrays(x, c["VT"], c["UT"], c["v2"], c["mid"], c["u1"], layer.bias,
                          layer.in_features, layer.rank, layer.out_features, cfg)
