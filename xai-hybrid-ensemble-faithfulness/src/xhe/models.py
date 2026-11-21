"""Leakage-safe model construction (Stage 2.6 section 8).

Components: LR, SVM (RBF), KNN, RF, XGB; each a full Pipeline whose preprocessing is
fitted only on the data passed to `fit` (the outer training fold, or an inner fold inside
stacking).
E1 soft voting; E2 stacking with an L2-LR meta-learner on out-of-fold probabilities;
E3 stacking over two soft-voting groups (A = LR, SVM, KNN; B = RF, XGB), i.e. stacking
with weights tied within families. S* = component with the highest inner out-of-fold AUC.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data import CONFIG

COMPONENTS = ["lr", "svm", "knn", "rf", "xgb"]
SCALED = {"lr", "svm", "knn"}

# ---------------------------------------------------------------- leakage instrumentation
FIT_REGISTRY: list[set] = []   # filled only when record_fits=True (tests)


class FitRecorder(BaseEstimator, TransformerMixin):
    """Identity transformer that records the rows it is fitted on (leakage tests only)."""

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        FIT_REGISTRY.append({tuple(np.round(r, 10)) for r in np.nan_to_num(arr, nan=-999.0)})
        return self

    def transform(self, X):
        return X


# ---------------------------------------------------------------- preprocessing
def make_preprocessor(feature_types: dict, columns: list[str], scale: bool) -> ColumnTransformer:
    numeric = [c for c in columns if feature_types[c] in ("continuous", "count", "binary")]
    nominal = [c for c in columns if feature_types[c] == "nominal"]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), numeric)]
    if nominal:
        transformers.append(("nom", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), nominal))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_component(name: str, feature_types: dict, columns: list[str], seed: int,
                   record_fits: bool = False) -> Pipeline:
    p = CONFIG["models"]
    est = {
        "lr": LogisticRegression(C=p["lr"]["C"], max_iter=p["lr"]["max_iter"]),
        "svm": SVC(C=p["svm"]["C"], gamma=p["svm"]["gamma"], kernel="rbf",
                   probability=True, random_state=seed),
        "knn": KNeighborsClassifier(n_neighbors=p["knn"]["n_neighbors"]),
        "rf": RandomForestClassifier(n_estimators=p["rf"]["n_estimators"], random_state=seed,
                                     n_jobs=1),
        "xgb": XGBClassifier(random_state=seed, n_jobs=1, **p["xgb"]),
    }[name]
    steps = [("rec", FitRecorder())] if record_fits else []
    steps += [("prep", make_preprocessor(feature_types, columns, scale=name in SCALED)),
              ("clf", est)]
    return Pipeline(steps)


def inner_cv(seed: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=CONFIG["cv"]["inner_splits"], shuffle=True, random_state=seed)


def meta_learner() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=5000)   # L2, as in Stage 2.6 section 8


def make_model(name: str, feature_types: dict, columns: list[str], seed: int,
               record_fits: bool = False):
    """name in COMPONENTS or {'e1', 'e2', 'e3'}."""
    comps = {c: make_component(c, feature_types, columns, seed, record_fits) for c in COMPONENTS}
    if name in comps:
        return comps[name]
    if name == "e1":
        return VotingClassifier([(c, comps[c]) for c in COMPONENTS], voting="soft")
    if name == "e2":
        return StackingClassifier([(c, comps[c]) for c in COMPONENTS],
                                  final_estimator=meta_learner(), cv=inner_cv(seed),
                                  stack_method="predict_proba", passthrough=False)
    if name == "e3":
        g = CONFIG["models"]["e3_groups"]
        group_a = VotingClassifier([(c, comps[c]) for c in g["A"]], voting="soft")
        group_b = VotingClassifier([(c, comps[c]) for c in g["B"]], voting="soft")
        return StackingClassifier([("A", group_a), ("B", group_b)],
                                  final_estimator=meta_learner(), cv=inner_cv(seed),
                                  stack_method="predict_proba", passthrough=False)
    raise ValueError(name)


# ---------------------------------------------------------------- S* selection
def select_s_star(X_train, y_train, feature_types: dict, seed: int) -> dict:
    """Choose the component with the highest inner out-of-fold AUC on the training fold only.

    Uses the same inner StratifiedKFold (same seed) as stacking. Ties within the configured
    tolerance are broken by the fixed order in configs/study.yaml."""
    cols = list(X_train.columns)
    aucs = {}
    for c in COMPONENTS:
        oof = cross_val_predict(make_component(c, feature_types, cols, seed), X_train, y_train,
                                cv=inner_cv(seed), method="predict_proba")[:, 1]
        aucs[c] = float(roc_auc_score(y_train, oof))
    best = max(aucs.values())
    tol = CONFIG["models"]["s_star_tie_tolerance"]
    tied = [c for c in COMPONENTS if best - aucs[c] <= tol]
    order = CONFIG["models"]["s_star_tie_order"]
    chosen = sorted(tied, key=order.index)[0]
    return {"selected": chosen, "inner_oof_auc": aucs, "tied": tied}


# ---------------------------------------------------------------- E3 technical-failure rule
def e3_technical_check(X_train, y_train, feature_types: dict, seed: int) -> dict:
    """Stage 2.6 section 8 fallback rule, evaluated on TRAINING data only.

    Trigger if the meta-learner raises a convergence warning, produces non-finite outputs,
    or gives a constant prediction on the inner out-of-fold group scores."""
    cols = list(X_train.columns)
    g = CONFIG["models"]["e3_groups"]
    oof = []
    for grp in ("A", "B"):
        vc = VotingClassifier([(c, make_component(c, feature_types, cols, seed)) for c in g[grp]],
                              voting="soft")
        oof.append(cross_val_predict(vc, X_train, y_train, cv=inner_cv(seed),
                                     method="predict_proba")[:, 1])
    Z = np.column_stack(oof)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        meta = meta_learner().fit(Z, y_train)
    p = meta.predict_proba(Z)[:, 1]
    reasons = []
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        reasons.append("meta-learner convergence warning")
    if not np.all(np.isfinite(p)):
        reasons.append("non-finite meta outputs")
    if np.std(p) < 1e-12:
        reasons.append("constant meta prediction")
    return {"trigger_fallback_to_e2": bool(reasons), "reasons": reasons,
            "meta_coef": meta.coef_.ravel().tolist(), "meta_intercept": float(meta.intercept_[0]),
            "corr_groupA_groupB_oof": float(np.corrcoef(Z.T)[0, 1])}


def positive_proba_fn(model, columns: list[str]):
    """The function explained everywhere: raw features -> P(y = 1). Checks class order."""
    import pandas as pd
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(f"unexpected class order {classes}")

    def f(Z):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        return model.predict_proba(pd.DataFrame(Z, columns=columns))[:, 1]
    return f
