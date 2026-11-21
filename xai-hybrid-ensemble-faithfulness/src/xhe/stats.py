"""Inference procedures (Stage 2.6 section 11, with the H2 amendment proposed in
docs/STATISTICAL_AUDIT.md).

H1 (AUC) and the PROPOSED primary H2 (fold-level NDF) both use the corrected repeated
k-fold t-test on per-fold differences (Nadeau & Bengio 2003; Bouckaert & Frank 2004).
The patient-level Wilcoxon from the report is kept as a conditional, secondary analysis.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def corrected_repeated_kfold_ttest(diffs, k: int, r: int, test_train_ratio: float,
                                   alpha: float = 0.05) -> dict:
    """t = mean(d) / sqrt((1/(k r) + n2/n1) * var(d)),  df = k r - 1."""
    d = np.asarray(diffs, float)
    if d.size != k * r:
        raise ValueError(f"expected {k * r} fold differences, got {d.size}")
    mean, var = float(d.mean()), float(d.var(ddof=1))
    se = np.sqrt((1.0 / (k * r) + test_train_ratio) * var)
    df = k * r - 1
    if se == 0:
        t, p = (0.0, 1.0) if mean == 0 else (np.inf * np.sign(mean), 0.0)
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), df))
    q = stats.t.ppf(1 - alpha / 2, df)
    return {"mean_diff": mean, "t": float(t), "df": df, "p": float(p),
            "ci": (mean - q * se, mean + q * se), "se_corrected": float(se)}


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[key] = running
    return {k: {"p": pvalues[k], "p_holm": adj[k], "reject": adj[k] <= alpha} for k in pvalues}


def wilcoxon_paired(diffs) -> dict:
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"n": int(d.size), "median_diff": float(np.median(d)), "statistic": float(res.statistic),
            "p": float(res.pvalue), "share_positive": float(np.mean(d > 0))}


def stratified_bootstrap_median_ci(diffs, strata, n_boot: int = 2000, seed: int = 0,
                                   alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap of the median paired difference, resampling patients within outer folds."""
    d, s = np.asarray(diffs, float), np.asarray(strata)
    ok = np.isfinite(d)
    d, s = d[ok], s[ok]
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(s == g) for g in np.unique(s)]
    meds = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        meds[b] = np.median(d[idx])
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))


def per_fold_means(diffs, folds) -> dict:
    d, f = np.asarray(diffs, float), np.asarray(folds)
    return {str(g): float(np.nanmean(d[f == g])) for g in np.unique(f)}
