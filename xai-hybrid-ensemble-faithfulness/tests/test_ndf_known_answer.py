"""Known-answer validation of NDF (Stage 2.6 section 10, tests K1-K5) plus checks requested
in the implementation brief: ties, misleading rankings, different feature counts.

Pass criteria are the ones written in the design report BEFORE any real-data run.
"""
import numpy as np
import pytest

from xhe.explain import permutation_shap
from xhe.metrics_xai import ndf

RNG = np.random.default_rng(12345)


def linear_problem(d=10, n_bin=2, n=500, seed=0):
    rng = np.random.default_rng(seed)
    beta = rng.normal(0, 1, d)
    def draw(m):
        Z = rng.normal(0, 1, (m, d))
        if n_bin:
            Z[:, -n_bin:] = rng.integers(0, 2, (m, n_bin))
        return Z
    X, B = draw(n), draw(50)
    f = lambda Z: np.asarray(Z, float) @ beta
    return f, beta, X, B


def analytic_D(c, positive, K):
    G = c.sum()
    return np.array([c[positive[:k]].sum() / G for k in range(1, K + 1)])


# ---------------------------------------------------------------- K1 additive, independent
@pytest.mark.parametrize("d", [5, 10, 13, 30])
def test_k1_additive_known_answer(d):
    f, beta, X, B = linear_problem(d=d, n_bin=2 if d >= 5 else 0, n=200 if d < 30 else 80, seed=d)
    sh = permutation_shap(f, X, B, max_evals=500, seed=1)
    truth = beta * (X - B.mean(0))
    assert np.max(np.abs(sh["values"] - truth)) < 1e-6           # SHAP = beta_j (x_j - xbar_j)
    n_incl = 0
    for i, x in enumerate(X):
        e = sh["values"][i]
        r_true = ndf(f, x, e, B, threshold=0.0, seed=i)
        r_rev = ndf(f, x, -e, B, threshold=0.0, seed=i)
        if np.isnan(r_true["ndf"]) or r_true["G"] < 1e-9:
            continue
        n_incl += 1
        s = r_true["s"]
        c = s * truth[i]
        order = list(np.argsort(-c, kind="stable"))
        positive = [j for j in order if c[j] > 0]
        np.testing.assert_allclose(r_true["D_expl"], analytic_D(c, positive, r_true["K"]),
                                   atol=1e-8)                          # analytic value
        assert r_true["ndf"] > 0                                       # better than random
        if not np.isnan(r_rev["ndf"]):
            assert r_rev["ndf"] < 0                                    # misleading -> negative
    assert n_incl > 0.3 * len(X)


def test_k1_random_attribution_is_null():
    f, beta, X, B = linear_problem(d=10, n=500, seed=7)
    rng = np.random.default_rng(99)
    vals = []
    for i, x in enumerate(X):
        r = ndf(f, x, rng.normal(0, 1, 10), B, threshold=0.0, seed=1000 + i)
        if not np.isnan(r["ndf"]):
            vals.append(r["ndf"])
    assert len(vals) > 200
    assert abs(np.mean(vals)) <= 0.02, np.mean(vals)


def test_misleading_ranking_scores_below_true():
    """Ranking by the SMALLEST positive contributions must score below the true ranking."""
    f, beta, X, B = linear_problem(d=10, n=150, seed=3)
    truth = beta * (X - B.mean(0))
    worse = 0
    total = 0
    for i, x in enumerate(X):
        r_true = ndf(f, x, truth[i], B, threshold=0.0, seed=i)
        if np.isnan(r_true["ndf"]):
            continue
        s = r_true["s"]
        c = s * truth[i]
        mis = np.where(c > 0, 1.0 / (c + 1e-9), 0.0) * s          # same sign set, inverted size order
        r_mis = ndf(f, x, mis, B, threshold=0.0, seed=i)
        total += 1
        worse += r_mis["ndf"] <= r_true["ndf"] + 1e-12
    assert worse == total


def test_ties_are_deterministic_and_order_invariant():
    beta = np.array([1.0, 1.0, 0.5, -0.3, 0.2])
    f = lambda Z: np.asarray(Z, float) @ beta
    B = np.zeros((50, 5))
    x = np.array([2.0, 2.0, 1.0, 1.0, 1.0])            # c_0 == c_1 exactly
    e = beta * x
    r1 = ndf(f, x, e, B, threshold=0.0, seed=0)
    r2 = ndf(f, x, e, B, threshold=0.0, seed=0)
    assert r1["ndf"] == r2["ndf"]
    e_alt = e.copy(); e_alt[1] += 1e-12                 # flip the tie order
    r3 = ndf(f, x, e_alt, B, threshold=0.0, seed=0)
    np.testing.assert_allclose(r1["D_expl"], r3["D_expl"], atol=1e-9)


# ---------------------------------------------------------------- K3 correlated redundancy
def test_k3_correlated_redundant_feature():
    rng = np.random.default_rng(5)
    n, d = 200, 5
    X = rng.normal(0, 1, (n, d)); X[:, 1] = X[:, 0] + rng.normal(0, 0.1, n)
    B = rng.normal(0, 1, (50, d)); B[:, 1] = B[:, 0] + rng.normal(0, 0.1, 50)
    f = lambda Z: 2.0 * np.asarray(Z, float)[:, 0]
    sh = permutation_shap(f, X, B, max_evals=200, seed=1)
    assert np.max(np.abs(sh["values"][:, 1])) < 1e-10       # interventional SHAP ignores x2
    ranked_first = 0; included = 0
    knn_vals, marg_vals = [], []
    Xref = rng.normal(0, 1, (300, d)); Xref[:, 1] = Xref[:, 0] + rng.normal(0, 0.1, 300)
    for i in range(40):
        r = ndf(f, X[i], sh["values"][i], B, threshold=0.0, seed=i)
        if np.isnan(r["ndf"]):
            continue
        included += 1
        ranked_first += r["K"] == 1                          # only x1 contributes
        rk = ndf(f, X[i], sh["values"][i], B, threshold=0.0, seed=i, operator="knn", X_ref=Xref)
        assert np.isfinite(rk["ndf"]) or rk["reason"]
        marg_vals.append(r["D_expl"][0]); knn_vals.append(rk["D_expl"][0] if rk["D_expl"] else np.nan)
    assert included > 0 and ranked_first == included
    # Recorded behaviour (not a pass criterion): conditional removal keeps x1's information
    # through x2, so deletion removes less of the gap than the marginal operator.
    print("K3 mean D_1 marginal", np.nanmean(marg_vals), "knn", np.nanmean(knn_vals))


# ---------------------------------------------------------------- K4 interaction
def test_k4_interaction_shap_beats_random():
    rng = np.random.default_rng(11)
    d = 6
    X = rng.normal(0, 1, (150, d)); B = rng.normal(0, 1, (50, d))
    f = lambda Z: np.asarray(Z)[:, 0] * np.asarray(Z)[:, 1] + 0.5 * np.asarray(Z)[:, 2]
    sh = permutation_shap(f, X, B, max_evals=500, seed=2)
    a, r = [], []
    for i, x in enumerate(X):
        rs = ndf(f, x, sh["values"][i], B, threshold=0.0, seed=i)
        rr = ndf(f, x, rng.normal(0, 1, d), B, threshold=0.0, seed=i)
        if not (np.isnan(rs["ndf"]) or np.isnan(rr["ndf"])):
            assert np.isfinite(rs["ndf"]) and np.isfinite(rr["ndf"])
            a.append(rs["ndf"]); r.append(rr["ndf"])
    assert np.mean(a) >= np.mean(r)


# ---------------------------------------------------------------- K5 degenerate
def test_k5_constant_model_excluded():
    f = lambda Z: np.full(len(np.atleast_2d(Z)), 0.7)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4); B = rng.normal(0, 1, (50, 4))
    res = ndf(f, x, np.array([0.1, 0.0, -0.2, 0.3]), B)
    assert np.isnan(res["ndf"]) and abs(res["G"]) < 1e-12 and res["reason"] == "non-positive gap"
    res0 = ndf(f, x, np.zeros(4), B)
    assert np.isnan(res0["ndf"]) and res0["reason"] == "no positive contributors"


def test_probability_model_shap_beats_random():
    """Non-additive probability scale (logistic link), threshold 0.5."""
    rng = np.random.default_rng(21)
    d = 8
    beta = rng.normal(0, 1.2, d)
    f = lambda Z: 1 / (1 + np.exp(-(np.asarray(Z, float) @ beta)))
    X = rng.normal(0, 1, (150, d)); B = rng.normal(0, 1, (50, d))
    sh = permutation_shap(f, X, B, max_evals=500, seed=3)
    a, r = [], []
    for i, x in enumerate(X):
        rs = ndf(f, x, sh["values"][i], B, seed=i)
        rr = ndf(f, x, rng.normal(0, 1, d), B, seed=i)
        if not (np.isnan(rs["ndf"]) or np.isnan(rr["ndf"])):
            a.append(rs["ndf"]); r.append(rr["ndf"])
    assert np.mean(a) > np.mean(r) + 0.05
