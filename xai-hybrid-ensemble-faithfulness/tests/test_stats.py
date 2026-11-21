import numpy as np
import pytest
from scipy import stats as sps

from xhe import stats


def test_corrected_ttest_matches_hand_calculation():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 25)
    out = stats.corrected_repeated_kfold_ttest(d, k=5, r=5, test_train_ratio=0.25)
    se = np.sqrt((1 / 25 + 0.25) * d.var(ddof=1))
    assert out["t"] == pytest.approx(d.mean() / se)
    assert out["p"] == pytest.approx(2 * sps.t.sf(abs(d.mean() / se), 24))
    # the correction always widens the interval relative to a naive paired t-test
    naive_se = d.std(ddof=1) / np.sqrt(25)
    assert out["se_corrected"] > naive_se


def test_corrected_ttest_requires_k_times_r_values():
    with pytest.raises(ValueError):
        stats.corrected_repeated_kfold_ttest(np.zeros(24), 5, 5, 0.25)


def test_holm_known_example():
    res = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert res["a"]["p_holm"] == pytest.approx(0.03)
    assert res["c"]["p_holm"] == pytest.approx(0.06)
    assert res["b"]["p_holm"] == pytest.approx(0.06)       # monotone step-down
    assert res["a"]["reject"] and not res["b"]["reject"]


def test_bootstrap_ci_contains_median():
    rng = np.random.default_rng(1)
    d = rng.normal(0.1, 0.2, 150)
    folds = np.repeat(np.arange(5), 30)
    lo, hi = stats.stratified_bootstrap_median_ci(d, folds, n_boot=500)
    assert lo <= np.median(d) <= hi
