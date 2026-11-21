"""Explanation metrics, implemented exactly as specified in Stage 2.6 section 10.

NDF (normalised deletion faithfulness), primary:
    s(x)   = +1 if f(x) >= t else -1             (t = 0.5 for probabilities)
    c_j    = s(x) * e_j(x)
    F(S|x) = mean_b f(x_{S<-b})                   (marginal operator, b in B_del)
    G(x)   = s(x) * (f(x) - F(all|x))
    T_k    = k features with largest positive c_j, K = min(K_max, #positive c_j)
    D_k    = s(x) * (f(x) - F(T_k|x)) / G(x)
    NDF(x) = (1/K) * sum_k ( D_k^expl - mean_r D_k^rand ),  20 random orderings
Inclusion: G(x) >= min_gap (applied per model pair by the caller).

Implementation decisions where the report is silent (logged in docs/IMPLEMENTATION_STATUS.md):
  * ties in c_j are broken by the lower feature index (stable sort), deterministic;
  * D_k^rand uses the first k features of each random ordering, for the same k = 1..K;
  * NDF is undefined (NaN, reason recorded) if K = 0 or G(x) <= 1e-12 (numerical tolerance;
    found necessary by test K5, where a constant model gave G = -2.2e-16 by rounding).
"""
from __future__ import annotations

import numpy as np


def _replace(x: np.ndarray, cols, background: np.ndarray) -> np.ndarray:
    Z = np.repeat(x[None, :], len(background), axis=0)
    if len(cols):
        Z[:, cols] = background[:, cols]
    return Z


def knn_background(x: np.ndarray, X_ref: np.ndarray, keep_cols, n: int, scale: np.ndarray):
    """Sensitivity operator: the n reference rows nearest to x on the kept features."""
    keep = np.asarray(keep_cols, int)
    if len(keep) == 0:
        idx = np.arange(min(n, len(X_ref)))
    else:
        diff = (X_ref[:, keep] - x[keep]) / scale[keep]
        dist = np.sqrt(np.nansum(diff ** 2, axis=1))
        idx = np.argsort(dist, kind="stable")[:n]
    return X_ref[idx]


def ndf(f, x, attribution, background, K_max: int = 5, n_random: int = 20, seed: int = 0,
        threshold: float = 0.5, operator: str = "marginal", X_ref=None, knn_n: int = 20) -> dict:
    """Return NDF and its components for one patient and one model."""
    x = np.asarray(x, float)
    e = np.asarray(attribution, float)
    background = np.asarray(background, float)
    d = len(x)
    fx = float(f(x[None])[0])
    s = 1.0 if fx >= threshold else -1.0
    F_all = float(np.mean(f(background)))
    G = s * (fx - F_all)
    c = s * e
    order = np.argsort(-c, kind="stable")          # ties -> lower index first
    positive = [j for j in order if c[j] > 0]
    K = min(K_max, len(positive))
    res = {"f_x": fx, "s": s, "G": G, "K": K, "ndf": np.nan, "reason": None,
           "D_expl": [], "D_rand_mean": []}
    if K == 0:
        res["reason"] = "no positive contributors"
        return res
    if not G > 1e-12:          # numerical tolerance; the 0.05 inclusion rule is applied by callers
        res["reason"] = "non-positive gap"
        return res

    if operator == "knn":
        if X_ref is None:
            raise ValueError("knn operator needs X_ref (outer-training rows)")
        scale = np.nanstd(X_ref, axis=0)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(d) for _ in range(n_random)]
    # All deletion sets: explanation order first, then each random ordering (same k = 1..K).
    sets = [list(positive[:k]) for k in range(1, K + 1)]
    sets += [list(p[:k]) for p in perms for k in range(1, K + 1)]
    blocks = []
    for cols in sets:
        if operator == "marginal":
            bg = background
        else:
            keep = [j for j in range(d) if j not in set(cols)]
            bg = knn_background(x, X_ref, keep, knn_n, scale)
        blocks.append(_replace(x, cols, bg))
    sizes = [len(b) for b in blocks]
    preds = f(np.vstack(blocks))                       # one batched model call
    means = np.array([m.mean() for m in np.split(preds, np.cumsum(sizes)[:-1])])
    D_all = s * (fx - means) / G
    D_expl = D_all[:K]
    D_rand = D_all[K:].reshape(n_random, K)
    res.update(D_expl=D_expl.tolist(), D_rand_mean=D_rand.mean(0).tolist(),
               ndf=float(np.mean(D_expl - D_rand.mean(0))))
    return res


# ------------------------------------------------------------------ robustness (secondary)
def perturb(x, feature_types: list[str], sd: np.ndarray, lo: np.ndarray, hi: np.ndarray,
            sigma_frac: float, rng) -> np.ndarray:
    """Type-aware perturbation (Stage 2.6 sec. 10): continuous Gaussian; counts Gaussian then
    rounded; binary and nominal unchanged; all clipped to the outer-training range.
    Missing values stay missing."""
    x2 = np.array(x, float, copy=True)
    for j, t in enumerate(feature_types):
        if np.isnan(x2[j]) or t in ("binary", "nominal"):
            continue
        v = x2[j] + rng.normal(0.0, sigma_frac * sd[j])
        if t == "count":
            v = np.round(v)
        x2[j] = np.clip(v, lo[j], hi[j])
    return x2


def topk(e, k=5) -> set:
    return set(np.argsort(-np.abs(np.asarray(e, float)), kind="stable")[:k].tolist())


def topk_overlap(e1, e2, k=5) -> float:
    return len(topk(e1, k) & topk(e2, k)) / k


def robustness(e_x, e_neighbours, same_class_mask, k=5) -> dict:
    """RB(x): mean top-5 overlap over neighbours whose predicted class is unchanged."""
    keep = [e for e, m in zip(e_neighbours, same_class_mask) if m]
    flip_rate = 1.0 - (len(keep) / len(e_neighbours)) if len(e_neighbours) else np.nan
    rb = float(np.mean([topk_overlap(e_x, e, k) for e in keep])) if keep else np.nan
    return {"rb": rb, "flip_rate": flip_rate, "n_used": len(keep)}


# ------------------------------------------------------------------ diagnostics
def agreement(e_a, e_b, k=5) -> dict:
    a, b = np.asarray(e_a, float), np.asarray(e_b, float)
    shared = sorted(topk(a, k) & topk(b, k))
    sign = float(np.mean(np.sign(a[shared]) == np.sign(b[shared]))) if shared else np.nan
    return {"top5_overlap": len(shared) / k, "sign_agreement": sign}
