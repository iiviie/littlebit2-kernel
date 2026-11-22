"""Turn a dense weight into an (untrained) LittleBit layer.

This is the *initialisation* LittleBit starts QAT from, written from the
papers' description:

1. Truncated SVD  W ~= U_r S_r V_r^T  and split S evenly:
       U = U_r sqrt(S_r)   (out x r),     V = sqrt(S_r) V_r^T   (r x in)
2. (LittleBit-2, optional) Joint-ITQ: find an orthogonal R (r x r) that
   rotates the latent rows of [U; V^T] towards the corners of the binary
   hypercube, then use U R and R^T V.  UV is unchanged, but sign(U R) loses
   less information than sign(U).
3. Binarise: keep sign(U), sign(V).  Recover magnitudes with rank-one
   approximations of |U| ~= u1^T u2 and |V| ~= v1^T v2 (top singular pair).

Without the QAT phase the result is a poor approximation of W (model output
is gibberish) -- but the *shapes* and the inference cost are exactly those
of a trained checkpoint, which is all a speed benchmark needs.
"""

from __future__ import annotations

import numpy as np

from .format import LittleBitLayer, LittleBitPath, split_dim_for


def _top_singular_pair(A: np.ndarray, iters: int = 30):
    """Rank-one approx A ~= a b^T with a, b >= 0 (A is non-negative)."""
    # power iteration on A A^T; A >= 0 so the top pair is non-negative
    b = np.ones(A.shape[1]) / np.sqrt(A.shape[1])
    for _ in range(iters):
        a = A @ b
        a /= np.linalg.norm(a) + 1e-30
        b = A.T @ a
        s = np.linalg.norm(b) + 1e-30
        b /= s
    rs = np.sqrt(s)
    return np.abs(a) * rs, np.abs(b) * rs


def _itq_rotation(X: np.ndarray, n_iter: int, rng) -> np.ndarray:
    """Orthogonal R minimising ||sign(X R) - X R||_F (Gong & Lazebnik ITQ)."""
    r = X.shape[1]
    R, _ = np.linalg.qr(rng.standard_normal((r, r)))
    for _ in range(n_iter):
        B = np.where(X @ R < 0, -1.0, 1.0)
        # Procrustes: R = argmax tr(R^T X^T B)
        Uo, _, Vto = np.linalg.svd(X.T @ B)
        R = Uo @ Vto
    return R


def _decompose(W: np.ndarray, r: int, use_itq: bool, itq_iters: int, rng) -> LittleBitPath:
    Us, S, Vt = np.linalg.svd(W, full_matrices=False)
    sq = np.sqrt(S[:r])
    U = Us[:, :r] * sq             # (out, r)
    V = sq[:, None] * Vt[:r]       # (r, in)
    if use_itq:
        R = _itq_rotation(np.concatenate([U, V.T], 0), itq_iters, rng)
        U, V = U @ R, R.T @ V
    u1, u2 = _top_singular_pair(np.abs(U))
    v1, v2 = _top_singular_pair(np.abs(V))
    return LittleBitPath.from_float(U, V, u1=u1, u2=u2, v1=v1, v2=v2)


def littlebit_from_dense(W, eff_bit: float, residual: bool = False, use_itq: bool = False,
                         itq_iters: int = 20, ratio_factor: float = 1.0, seed: int = 0,
                         bias=None) -> LittleBitLayer:
    """Dense (out, in) weight -> LittleBitLayer at roughly `eff_bit` bits/weight."""
    W = np.asarray(W, dtype=np.float64)
    out_f, in_f = W.shape
    r = min(split_dim_for(in_f, out_f, eff_bit, residual, ratio_factor), min(in_f, out_f))
    rng = np.random.default_rng(seed)
    main = _decompose(W, r, use_itq, itq_iters, rng)
    res = None
    if residual:
        # the residual path factorises what the main path missed
        approx = (main.U_signs() * main.u1[:, None] * main.u2[None, :]) @ \
                 (main.V_signs() * main.v1[:, None] * main.v2[None, :])
        res = _decompose(W - approx, r, use_itq, itq_iters, rng)
    b = None if bias is None else np.asarray(bias, np.float32)
    return LittleBitLayer(main=main, residual=res, bias=b)
