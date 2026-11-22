"""`simd`: a SIMD-group (32 threads) cooperates on each row, reduced with simd_sum.

What changes vs `naive`:

1. Coalesced weight reads.  In `naive` one thread walks one row word by
   word, so the 32 threads of a SIMD-group touch 32 different rows at once
   (32 scattered 4-byte reads).  Here the lanes of a SIMD-group read
   *consecutive* words of the same row, so one load instruction fetches a
   contiguous 128-byte line.  That is what DRAM wants.

2. More parallelism for long rows.  A row of V has in/32 words (448 for
   in=14336); 32 lanes now share that loop, and `simd_sum` adds the 32
   partial sums in registers -- no threadgroup memory, no barriers.

3. Short rows don't waste lanes.  Stage 2 rows have only ceil(r/32) words
   (e.g. 5 words for r=136): a whole SIMD-group per row would leave 27 lanes
   idle.  So the lanes-per-row (LPR) is a template parameter: the smallest
   power of two >= words per row, capped at 32.  A SIMD-group then handles
   32/LPR rows, and the partial sums are combined with `simd_shuffle_xor`
   inside each LPR-lane slice.

The inner loop over the 32 bits of a word is the same add/subtract as the
naive kernel.
"""

from __future__ import annotations

import mlx.core as mx

from .common import DeviceLayer, as_2d

# Shared device function: signed sum of 32 activations selected by one word.
# `a` points at the 32 activations covered by `word`; `n` is how many are
# real (< 32 only for the last word of a row whose length isn't a multiple
# of 32).  PRE=true multiplies each activation by v2 on the fly (stage 1).
_HEADER = """
template <bool PRE, typename TX, typename TS>
inline float signed_sum32(uint word, const device TX* a, const device TS* s, uint n) {
    float acc = 0.0f;
    if (n == 32) {
        // Full word: fixed trip count so the compiler fully unrolls.
        #pragma unroll
        for (uint b = 0; b < 32; ++b) {
            float v = float(a[b]);
            if (PRE) v *= float(s[b]);
            acc += (word & (1u << b)) ? -v : v;
        }
    } else {
        for (uint b = 0; b < n; ++b) {
            float v = float(a[b]);
            if (PRE) v *= float(s[b]);
            acc += (word & (1u << b)) ? -v : v;
        }
    }
    return acc;
}
"""

# Template params: K inner length, N rows, W words/row, LPR lanes per row,
# NSG simdgroups per threadgroup, PRE (stage-1 prescale by v2).
_SRC = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint m    = threadgroup_position_in_grid.y;

    // Which row this lane works on, and its index within that row's slice.
    const uint rows_per_sg = 32 / LPR;
    const uint sub  = lane / LPR;            // row slot inside the simdgroup
    const uint li   = lane % LPR;            // lane index inside the row slice
    const uint row  = (threadgroup_position_in_grid.x * NSG + sg) * rows_per_sg + sub;

    float acc = 0.0f;
    if (row < N) {
        const device uint* bits = W_bits + row * W;
        const device TX*   xm   = x + m * K;
        for (uint w = li; w < W; w += LPR) {
            uint base = w * 32;
            acc += signed_sum32<PRE>(bits[w], xm + base, pre + base, min(32u, K - base));
        }
    }
    // Reduce over the LPR lanes of each slice (xor butterfly stays in-slice).
    for (uint off = LPR / 2; off > 0; off /= 2)
        acc += simd_shuffle_xor(acc, off);

    if (row < N && li == 0)
        out[m * N + row] = acc * float(post[row]);
"""

_kernel = mx.fast.metal_kernel(
    name="lb_simd_rowdot", input_names=["x", "W_bits", "pre", "post"],
    output_names=["out"], header=_HEADER, source=_SRC)

NSG = 4  # simdgroups per threadgroup (128 threads)


def _lpr(words: int) -> int:
    p = 1
    while p < words and p < 32:
        p *= 2
    return p


def rowdot(x2, bits, pre, post, K, N, prescale: bool):
    """out[m, n] = post[n] * sum_k sign(bits[n, k]) * x2[m, k] (* pre[k])."""
    M = x2.shape[0]
    W = (K + 31) // 32
    lpr = _lpr(W)
    rows_per_tg = NSG * (32 // lpr)
    n_tg = (N + rows_per_tg - 1) // rows_per_tg
    (out,) = _kernel(
        inputs=[x2, bits, pre, post],
        template=[("TX", x2.dtype), ("K", K), ("N", N), ("W", W), ("LPR", lpr),
                  ("NSG", NSG), ("PRE", prescale)],
        grid=(n_tg * NSG * 32, M, 1), threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(M, N)], output_dtypes=[mx.float32])
    return out


def forward(x: mx.array, layer: DeviceLayer) -> mx.array:
    x2, restore = as_2d(x)
    y = None
    for p in layer.paths:
        t = rowdot(x2, p.V_bits, p.v2, p.mid, p.in_features, p.rank, True)
        yp = rowdot(t, p.U_bits, p.u1, p.u1, p.rank, p.out_features, False)
        y = yp if y is None else y + yp
    if layer.bias is not None:
        y = y + layer.bias.astype(mx.float32)
    return restore(y.astype(x.dtype))
