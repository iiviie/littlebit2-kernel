"""`lut`: table lookup instead of per-bit add/subtract (T-MAC style).

Why: the bit-by-bit kernels spend ~4 GPU instructions per weight bit (test
bit, select +a/-a, add, plus loading `a`).  A 1-bit weight is only 1/8 of a
byte, so at ~200 GB/s the GPU would need to retire ~1.6 trillion weight
bits per second -- more than its ALUs manage at 4 instructions per bit.
The bit kernels are compute-bound, not memory-bound.

Idea: group the activations in fours.  For one group (a0, a1, a2, a3) there
are only 16 possible sign patterns, so precompute all 16 signed sums once:

    T[p] = sum_i (bit i of p ? -a_i : +a_i)          p = 0..15

Then a row's dot product over that group is just T[nibble of the row's
weight word] -- one lookup + one add for 4 weights.  The table is built once
per threadgroup and reused by every row in it, so its cost is amortised.

Layout: tables are read by many rows at once, so we want the 32 lanes of a
SIMD-group to process 32 *different rows* at the *same* group (they then
all read the same 16-entry table: no bank conflicts in threadgroup memory).
With the upstream row-major bit layout those 32 rows' words are far apart in
memory, so at load time we transpose the packed matrix to word-major
`bitsT[w, row]`: the 32 lanes then read 32 consecutive uint32 = one 128-byte
line.  (Same bytes, different order -- a one-time repack.)

Parallelism: one lane per row gives too few threads for short matrices
(stage 1 has only r rows, e.g. r=96), so the inner dimension is also split
into chunks of KC elements ("split-K").  Each (row tile, chunk) threadgroup
adds its partial result into the output with a float atomic add.  Because
the identity is linear, the scales and the residual path can be added the
same way: path p, chunk c just atomically adds post_p[row] * partial into
the same output.  The bias is added by (path 0, chunk 0).  So residual and
bias cost no extra launches.
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d

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
        // T[p] = sum of (+a_i if bit i of p is 0, -a_i if 1).
        // Build it incrementally: start from all-plus and flip signs.
        float s = a[0] + a[1] + a[2] + a[3];
        threadgroup float* t = T + g * 16;
        t[0] = s;
        for (uint p = 1; p < 16; ++p) {
            uint low = ctz(p);                 // lowest set bit of p
            t[p] = t[p & (p - 1)] - 2.0f * a[low];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= N) return;

    // ---- 2. each lane walks its row over this chunk ----------------------
    // Words of this chunk: [k0/32, k0/32 + KC/32), clipped to W.
    const uint w0 = k0 / 32;
    const uint w1 = min(w0 + KC / 32, (uint)W);
    const device uint* bits = bitsT + path * W * N + row;   // column `row`
    float acc = 0.0f;
    for (uint w = w0; w < w1; ++w) {
        uint word = bits[w * N];                          // coalesced over lanes
        threadgroup const float* t = T + (w - w0) * 8 * 16;
        #pragma unroll
        for (uint n = 0; n < 8; ++n)
            acc += t[n * 16 + ((word >> (4 * n)) & 15u)];
    }

    // ---- 3. scale and accumulate into the output -------------------------
    float v = acc * float(post[path * N + row]);
    if (HAS_BIAS && path == 0 && chunk == 0) v += float(bias[row]);
    atomic_fetch_add_explicit(&out[(m * OUT_NP + (OUT_PER_PATH ? path : 0)) * N + row],
                              v, memory_order_relaxed);
"""

_kernel = mx.fast.metal_kernel(
    name="lb_lut_rowdot",
    input_names=["x", "bitsT", "pre", "post", "bias"],
    output_names=["out"], source=_SRC, atomic_outputs=True)

TG = 128          # threads (= rows) per threadgroup
KC_MAX = 1024     # chunk length; table = KC/4*16 floats = 16 KB at 1024


def _choose_kc(K: int, n_tiles: int, NP: int) -> int:
    """Split K so there are enough threadgroups to fill the GPU (~64+)."""
    kc = KC_MAX
    while kc > 128 and n_tiles * NP * ((K + kc - 1) // kc) < 64:
        kc //= 2
    return kc


def _prepare(layer: DeviceLayer):
    """One-time repack: stack paths and transpose bits to word-major."""
    cache = getattr(layer, "_lut_cache", None)
    if cache is not None:
        return cache
    ps = layer.paths
    st = lambda f: mx.stack([f(p) for p in ps])
    cache = dict(
        VT=st(lambda p: p.V_bits.T),     # (NP, W1, r)
        UT=st(lambda p: p.U_bits.T),     # (NP, W2, out)
        v2=st(lambda p: p.v2), mid=st(lambda p: p.mid), u1=st(lambda p: p.u1),
    )
    cache = {k: mx.contiguous(v) for k, v in cache.items()}
    mx.eval(list(cache.values()))
    layer._lut_cache = cache
    return cache


def _rowdot(x, bitsT, pre, post, bias, K, N, NP, M, prescale, x_per_path, out_per_path):
    n_tiles = (N + TG - 1) // TG
    kc = _choose_kc(K, n_tiles, NP)
    nch = (K + kc - 1) // kc
    out_np = NP if out_per_path else 1
    (out,) = _kernel(
        inputs=[x, bitsT, pre, post, bias if bias is not None else post],
        template=[("TX", x.dtype), ("TS", pre.dtype), ("K", K), ("N", N), ("W", (K + 31) // 32),
                  ("KC", kc), ("NCH", nch), ("NP", NP), ("TG", TG), ("PRE", prescale),
                  ("X_PER_PATH", x_per_path), ("OUT_PER_PATH", out_per_path),
                  ("OUT_NP", out_np), ("HAS_BIAS", bias is not None)],
        grid=(n_tiles * TG, NP * nch, M), threadgroup=(TG, 1, 1),
        output_shapes=[(M, out_np, N)], output_dtypes=[mx.float32], init_value=0.0)
    return out


def forward_arrays(x, VT, UT, v2, mid, u1, bias, in_features, rank, out_features):
    """Forward from the repacked arrays (used by `littlebit_kernel.nn`).

    VT: (NP, ceil(in/32), r) uint32    UT: (NP, ceil(r/32), out) uint32
    v2: (NP, in)  mid: (NP, r)  u1: (NP, out)  bias: (out,) or None
    """
    x2, restore = as_2d(x)
    M, NP = x2.shape[0], VT.shape[0]
    K, R, OUT = in_features, rank, out_features
    # stage 1: t[m, p, :] = mid_p * signV_p @ (x * v2_p)
    t = _rowdot(x2, VT, v2, mid, None, K, R, NP, M,
                prescale=True, x_per_path=False, out_per_path=True)
    # stage 2: y[m, :] = sum_p u1_p * signU_p @ t[m, p] (+ bias)
    y = _rowdot(t, UT, u1, u1, bias, R, OUT, NP, M,
                prescale=False, x_per_path=True, out_per_path=False)
    return restore(y.reshape(M, OUT).astype(x.dtype))


def forward(x: mx.array, layer: DeviceLayer) -> mx.array:
    c = _prepare(layer)
    return forward_arrays(x, c["VT"], c["UT"], c["v2"], c["mid"], c["u1"], layer.bias,
                          layer.in_features, layer.rank, layer.out_features)
