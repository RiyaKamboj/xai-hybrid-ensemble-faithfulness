"""SHAP and LIME wrappers (Stage 2.6 section 10).

The function explained is always f(X) = model.predict_proba(X)[:, 1] of the full fitted
pipeline on raw features (see models.positive_proba_fn). Never a component or a surrogate.
"""
from __future__ import annotations

import numpy as np
import shap


def permutation_shap(f, X_explain: np.ndarray, background: np.ndarray, max_evals: int,
                     seed: int) -> dict:
    """Permutation SHAP with an Independent masker on raw background rows.

    The same seed is used for every model on a given patient (common random numbers)."""
    masker = shap.maskers.Independent(np.asarray(background, float),
                                      max_samples=len(background))
    expl = shap.PermutationExplainer(f, masker, seed=seed)
    d = X_explain.shape[1]
    me = max(max_evals, 2 * d + 1)
    out = expl(np.asarray(X_explain, float), max_evals=me, silent=True)
    vals = np.asarray(out.values, float)
    base = np.asarray(out.base_values, float).reshape(-1)
    if vals.shape != X_explain.shape:
        raise ValueError(f"unexpected SHAP shape {vals.shape} for input {X_explain.shape}")
    return {"values": vals, "base_values": base, "max_evals": me}


def exact_shap(f, X_explain: np.ndarray, background: np.ndarray) -> dict:
    """Exact interventional Shapley values (2^d coalitions); for gates on small d only."""
    masker = shap.maskers.Independent(np.asarray(background, float),
                                      max_samples=len(background))
    out = shap.ExactExplainer(f, masker)(np.asarray(X_explain, float))
    return {"values": np.asarray(out.values, float),
            "base_values": np.asarray(out.base_values, float).reshape(-1)}


def additivity_error(f, X_explain, shap_out) -> np.ndarray:
    fx = f(np.asarray(X_explain, float))
    return np.abs(fx - shap_out["base_values"] - shap_out["values"].sum(axis=1))


def lime_contributions(f, x: np.ndarray, X_train: np.ndarray, feature_types: list[str],
                       num_samples: int, seed: int) -> dict:
    """LIME (discretize_continuous=False) with coefficients converted to contributions.

    numeric:  e_j = w_j * (x_j - mu_j) / s_j     (LIME works on standardised features)
    categorical (binary, nominal): e_j = w_j * (1 - pi_j(x_j)), pi_j = training frequency
    of the patient's category. Stage 2.6 section 10; validated by test K2.
    Rows with missing values cannot be explained by LIME (explanation set = complete cases).
    """
    from lime.lime_tabular import LimeTabularExplainer
    X_train = np.asarray(X_train, float)
    if np.isnan(x).any():
        raise ValueError("LIME cannot explain a row with missing values")
    cat_idx = [j for j, t in enumerate(feature_types) if t in ("binary", "nominal")]
    train_cc = X_train[~np.isnan(X_train).any(axis=1)]
    explainer = LimeTabularExplainer(train_cc, mode="classification",
                                     categorical_features=cat_idx or None,
                                     discretize_continuous=False, random_state=seed)

    def proba(Z):
        p1 = f(Z)
        return np.column_stack([1 - p1, p1])

    d = len(x)
    exp = explainer.explain_instance(np.asarray(x, float), proba, labels=(1,),
                                     num_features=d, num_samples=num_samples)
    w = np.zeros(d)
    for j, coef in exp.local_exp[1]:
        w[j] = coef
    mu = explainer.scaler.mean_
    sd = explainer.scaler.scale_
    e = np.zeros(d)
    for j in range(d):
        if j in cat_idx:
            freq = float(np.mean(train_cc[:, j] == x[j]))
            e[j] = w[j] * (1.0 - freq)
        else:
            e[j] = w[j] * (x[j] - mu[j]) / sd[j]
    local_pred = float(np.ravel(exp.local_pred)[0])
    return {"contributions": e, "coefficients": w, "local_pred": local_pred,
            "local_error": abs(float(f(x[None])[0]) - local_pred), "score": float(exp.score)}
