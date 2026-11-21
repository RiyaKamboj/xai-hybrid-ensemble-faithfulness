"""SHAP output handling, decomposition gates, LIME conversion (K2) and leakage tests.

All data here is synthetic. Gates follow Stage 2.6 section 10; tolerances as written there.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from scipy.stats import spearmanr

from xhe import models
from xhe.explain import additivity_error, exact_shap, lime_contributions, permutation_shap

FT = {"c1": "continuous", "c2": "continuous", "c3": "continuous", "k1": "count",
      "b1": "binary", "n1": "nominal"}
COLS = list(FT)


def synthetic(n=300, seed=0, missing=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "c1": rng.normal(0, 1, n), "c2": rng.normal(5, 2, n), "c3": rng.normal(0, 1, n),
        "k1": rng.poisson(3, n).astype(float), "b1": rng.integers(0, 2, n).astype(float),
        "n1": rng.integers(1, 4, n).astype(float)})
    z = 1.2 * X.c1 - 0.5 * (X.c2 - 5) + 0.8 * X.b1 + 0.6 * (X.n1 == 3) + 0.2 * X.k1 - 0.6
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-z))).astype(int), name="target")
    if missing:
        X.loc[rng.choice(n, 6, replace=False), "n1"] = np.nan
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = synthetic()
    tr, te = np.arange(240), np.arange(240, 300)
    out = {}
    for name in ["e1", "e2", "e3"]:
        m = models.make_model(name, FT, COLS, seed=1).fit(X.iloc[tr], y.iloc[tr])
        out[name] = m
    return X, y, tr, te, out


def test_identity_and_class_order(fitted):
    X, y, tr, te, ms = fitted
    for name, m in ms.items():
        f = models.positive_proba_fn(m, COLS)
        assert np.array_equal(f(X.iloc[te].values), m.predict_proba(X.iloc[te])[:, 1])
        assert list(m.classes_) == [0, 1]
        p = f(X.iloc[te].values)
        assert np.all((p >= 0) & (p <= 1))


def test_permutation_additivity_and_reproducibility(fitted):
    X, y, tr, te, ms = fitted
    f = models.positive_proba_fn(ms["e3"], COLS)
    bg, xs = X.iloc[tr[:30]].values, X.iloc[te[:4]].values
    a = permutation_shap(f, xs, bg, max_evals=200, seed=7)
    b = permutation_shap(f, xs, bg, max_evals=200, seed=7)
    assert np.max(additivity_error(f, xs, a)) < 1e-6
    assert np.array_equal(a["values"], b["values"])            # same seed -> identical


def _component_fns(stack_or_vote):
    return [models.positive_proba_fn(est, COLS) for est in stack_or_vote.estimators_]


def test_voting_linearity_exact(fitted):
    X, y, tr, te, ms = fitted
    e1 = ms["e1"]
    bg, xs = X.iloc[tr[:20]].values, X.iloc[te[:3]].values
    phi_e1 = exact_shap(models.positive_proba_fn(e1, COLS), xs, bg)["values"]
    phis = [exact_shap(fc, xs, bg)["values"] for fc in _component_fns(e1)]
    assert np.max(np.abs(phi_e1 - np.mean(phis, axis=0))) < 1e-8


@pytest.mark.parametrize("name", ["e2", "e3"])
def test_stacking_decomposition_exact(fitted, name):
    X, y, tr, te, ms = fitted
    st = ms[name]
    bg, xs = X.iloc[tr[:20]].values, X.iloc[te[:3]].values
    f = models.positive_proba_fn(st, COLS)
    flogit = lambda Z: logit(np.clip(f(Z), 1e-15, 1 - 1e-15))
    phi = exact_shap(flogit, xs, bg)["values"]
    w = st.final_estimator_.coef_.ravel()
    assert len(w) == len(st.estimators_)                      # binary: positive column only
    parts = [exact_shap(fc, xs, bg)["values"] for fc in _component_fns(st)]
    assert np.max(np.abs(phi - np.tensordot(w, np.array(parts), 1))) < 1e-8


def test_k2_lime_conversion_recovers_true_contributions():
    rng = np.random.default_rng(4)
    n, d = 600, 6
    X = rng.normal(0, 1, (n, d)); X[:, 4] = rng.integers(0, 2, n); X[:, 5] = rng.integers(0, 2, n)
    beta = np.array([1.5, -1.0, 0.6, 0.3, 0.9, -0.7])
    f = lambda Z: np.atleast_2d(np.asarray(Z, float)) @ beta
    types = ["continuous"] * 4 + ["binary"] * 2
    rhos = []
    for i in range(25):
        x = X[i]
        out = lime_contributions(f, x, X, types, num_samples=3000, seed=i)
        truth = beta * (x - X.mean(0))
        rhos.append(spearmanr(out["contributions"], truth).statistic)
    assert np.median(rhos) >= 0.9, np.median(rhos)


def test_no_outer_test_rows_reach_any_fit():
    X, y = synthetic(seed=3, missing=True)
    tr, te = np.arange(240), np.arange(240, 300)
    for name in ["e2", "e3", "svm"]:
        models.FIT_REGISTRY.clear()
        m = models.make_model(name, FT, COLS, seed=2, record_fits=True)
        m.fit(X.iloc[tr], y.iloc[tr])
        test_rows = {tuple(np.round(r, 10)) for r in np.nan_to_num(X.iloc[te].values, nan=-999.0)}
        seen = set().union(*models.FIT_REGISTRY)
        assert len(models.FIT_REGISTRY) > 0
        assert not (seen & test_rows), f"{name}: outer-test rows reached a fit"


def test_inner_fits_use_strict_subsets_for_stacking():
    """Every inner fit inside stacking must see fewer rows than the outer training fold."""
    X, y = synthetic(seed=5)
    models.FIT_REGISTRY.clear()
    m = models.make_model("e2", FT, COLS, seed=2, record_fits=True)
    m.fit(X.iloc[:240], y.iloc[:240])
    sizes = sorted(len(s) for s in models.FIT_REGISTRY)
    assert sizes[0] < 240 and sizes[-1] == 240          # inner folds + final refit


def test_s_star_selection_uses_training_data_only_and_tie_rule():
    X, y = synthetic(seed=8)
    sel = models.select_s_star(X.iloc[:240], y.iloc[:240], FT, seed=3)
    assert sel["selected"] in models.COMPONENTS
    best = max(sel["inner_oof_auc"].values())
    assert best - sel["inner_oof_auc"][sel["selected"]] <= 0.001


def test_e3_technical_check_runs_on_training_data():
    X, y = synthetic(seed=9)
    chk = models.e3_technical_check(X.iloc[:240], y.iloc[:240], FT, seed=3)
    assert set(chk) >= {"trigger_fallback_to_e2", "reasons", "meta_coef"}
    assert len(chk["meta_coef"]) == 2
