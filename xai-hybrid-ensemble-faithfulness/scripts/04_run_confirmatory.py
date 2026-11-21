"""Step 4: CONFIRMATORY EXPERIMENTS (protocol frozen 11 Nov 2026, docs/PROTOCOL_FREEZE.md).

Stage A  per outer fold (25 x 3 datasets): fit components once, inner out-of-fold predictions,
         S* selection, E3 technical rule, E1/E2/E3 assembly, test-fold predictive metrics.
Stage B  per outer fold: primary H2 sample (20 complete-case test patients), permutation SHAP
         for E3 and S* (common random numbers), NDF (marginal; k-NN sensitivity).
Stage C  repeat-1 folds: 150 patients per dataset; NDF for all 8 models; robustness for
         E1, E2, E3, S*; sigma sensitivity for E3, S*; SHAP-LIME diagnostics.

Every task writes its own JSON file and is skipped if that file exists (resumable).
Usage: python scripts/04_run_confirmatory.py [--stage A|C|all] [--quick] [--jobs 2]
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xhe import cv, data, explain, metrics_xai, models  # noqa: E402

CFG = data.CONFIG
SEED = CFG["master_seed"]
EXP = CFG["explanation"]
COMP = models.COMPONENTS
ALL_MODELS = COMP + ["e1", "e2", "e3"]


def GROUPS(name, idx):
    """Subject groups for grouped inner CV (Study 2 Parkinsons); None otherwise."""
    gp = ROOT / "data" / "processed" / f"{name}_groups.csv"
    if not gp.exists():
        return None
    return pd.read_csv(gp)["group"].values[np.asarray(idx)]


def load_processed(name):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    y = df.pop("target").astype(int)
    folds = cv.load_folds(ROOT / "data" / "processed" / f"{name}_outer_folds.json")
    ft = data.FEATURE_TYPES.get(name)
    if ft is None:
        ft = json.loads((ROOT / "data" / "processed" / "feature_types_study2.json").read_text())[name]
    return df, y, folds, ft


# ---------------------------------------------------------------- fold model bundle
class FoldModels:
    """Fitted components plus the E2/E3 meta-learners and S*, for one outer training fold.

    Equivalent to sklearn VotingClassifier / StackingClassifier built by models.make_model
    (checked numerically by check_equivalence)."""

    def __init__(self, Xtr, ytr, ft, seed, groups=None):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict
        self.cols = list(Xtr.columns)
        if groups is None:
            icv, gkw = models.inner_cv(seed), {}
        else:   # Study 2 (Parkinsons): inner folds grouped by subject
            from sklearn.model_selection import StratifiedGroupKFold
            icv, gkw = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed), {"groups": np.asarray(groups)}
        oof = {}
        for c in COMP:
            oof[c] = cross_val_predict(models.make_component(c, ft, self.cols, seed), Xtr, ytr,
                                       cv=icv, method="predict_proba", **gkw)[:, 1]
        # S*
        aucs = {c: float(roc_auc_score(ytr, oof[c])) for c in COMP}
        best = max(aucs.values())
        tied = [c for c in COMP if best - aucs[c] <= CFG["models"]["s_star_tie_tolerance"]]
        self.s_star = sorted(tied, key=CFG["models"]["s_star_tie_order"].index)[0]
        self.inner_auc = aucs
        # meta-learners
        g = CFG["models"]["e3_groups"]
        Z2 = np.column_stack([oof[c] for c in COMP])
        Z3 = np.column_stack([np.mean([oof[c] for c in g["A"]], 0), np.mean([oof[c] for c in g["B"]], 0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.meta3 = models.meta_learner().fit(Z3, ytr)
        p3 = self.meta3.predict_proba(Z3)[:, 1]
        reasons = []
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            reasons.append("meta-learner convergence warning")
        if not np.all(np.isfinite(p3)):
            reasons.append("non-finite meta outputs")
        if np.std(p3) < 1e-12:
            reasons.append("constant meta prediction")
        self.e3_fallback = reasons
        self.meta2 = models.meta_learner().fit(Z2, ytr)
        self.groupA, self.groupB = g["A"], g["B"]
        self.corr_qA_qB = float(np.corrcoef(Z3.T)[0, 1])
        # full refit of the components
        self.comp = {c: models.make_component(c, ft, self.cols, seed).fit(Xtr, ytr) for c in COMP}

    def _df(self, Z):
        Z = np.asarray(Z, float)
        if Z.ndim == 1:
            Z = Z[None, :]
        return pd.DataFrame(Z, columns=self.cols)

    def proba(self, name, Z):
        df = self._df(Z)
        if name in COMP:
            return self.comp[name].predict_proba(df)[:, 1]
        P = {c: self.comp[c].predict_proba(df)[:, 1] for c in COMP}
        if name == "e1":
            return np.mean([P[c] for c in COMP], 0)
        if name == "e2":
            return self.meta2.predict_proba(np.column_stack([P[c] for c in COMP]))[:, 1]
        if name == "e3":
            qa = np.mean([P[c] for c in self.groupA], 0)
            qb = np.mean([P[c] for c in self.groupB], 0)
            return self.meta3.predict_proba(np.column_stack([qa, qb]))[:, 1]
        if name == "s_star":
            return self.proba(self.s_star, Z)
        raise ValueError(name)

    def f(self, name):
        return lambda Z: self.proba(name, Z)


def check_equivalence(Xtr, ytr, Xte, ft, fm):
    out = {}
    for name in ["e1", "e2", "e3"]:
        m = models.make_model(name, ft, list(Xtr.columns), SEED).fit(Xtr, ytr)
        out[name] = float(np.max(np.abs(m.predict_proba(Xte)[:, 1] - fm.proba(name, Xte.values))))
    return out


# ---------------------------------------------------------------- helpers
def pred_metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "f1": 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else np.nan}


def stratified_sample(idx, y, n, seed):
    """Class-proportional sample of n positions from idx (labels y[idx])."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx); yy = np.asarray(y)[idx]
    out = []
    classes = np.unique(yy)
    alloc = {c: int(round(n * np.mean(yy == c))) for c in classes}
    diff = n - sum(alloc.values())
    alloc[classes[np.argmax([np.mean(yy == c) for c in classes])]] += diff
    for c in classes:
        pool = idx[yy == c]
        out.extend(rng.choice(pool, size=min(alloc[c], len(pool)), replace=False).tolist())
    return sorted(out)


def backgrounds(Xtr, seed):
    cc = Xtr.dropna()
    perm = cc.sample(len(cc), random_state=seed).index
    nb, nd = EXP["shap_background"], EXP["ndf_background"]
    b_shap = Xtr.loc[perm[:nb]].values
    b_del = Xtr.loc[perm[nb:nb + nd]].values
    return b_shap, b_del, cc.values


def ndf_record(f, x, e, b_del, seed, operator="marginal", X_ref=None):
    r = metrics_xai.ndf(f, x, e, b_del, K_max=EXP["ndf_K"], n_random=EXP["ndf_random_orderings"],
                        seed=seed, operator=operator, X_ref=X_ref, knn_n=EXP["knn_operator_neighbours"])
    return {"ndf": r["ndf"], "G": r["G"], "K": r["K"], "f_x": r["f_x"], "reason": r["reason"]}


# ---------------------------------------------------------------- Stage A + B task
def task_fold(name, fold, quick, outdir):
    path = outdir / f"{name}_fold{fold['fold_id']:02d}.json"
    if path.exists():
        return str(path)
    t0 = time.time()
    X, y, folds, ft = load_processed(name)
    Xtr, ytr = X.iloc[fold["train"]], y.iloc[fold["train"]]
    Xte, yte = X.iloc[fold["test"]], y.iloc[fold["test"]].values
    fm = FoldModels(Xtr, ytr, ft, SEED, groups=GROUPS(name, fold["train"]))
    res = {"dataset": name, "fold_id": fold["fold_id"], "repeat": fold["repeat"],
           "n_train": len(Xtr), "n_test": len(Xte), "s_star": fm.s_star, "inner_auc": fm.inner_auc,
           "e3_fallback_reasons": fm.e3_fallback, "corr_qA_qB": fm.corr_qA_qB,
           "meta3_coef": fm.meta3.coef_.ravel().tolist(), "meta2_coef": fm.meta2.coef_.ravel().tolist()}
    if fold["fold_id"] == 0:
        res["equivalence_max_abs_diff"] = check_equivalence(Xtr, ytr, Xte, ft, fm)
    # predictive
    pm = {}
    for m in ALL_MODELS + ["s_star"]:
        t1 = time.time(); p = fm.proba(m, Xte.values); dt = time.time() - t1
        pm[m] = {**pred_metrics(yte, p), "predict_ms_per_1000": 1000 * dt / len(Xte) * 1000}
    res["predictive"] = pm
    # Stage B: primary H2 sample
    test_cc = [i for i in fold["test"] if not X.iloc[i].isna().any()]
    m_h2 = 6 if quick else 20
    sample = stratified_sample(test_cc, y.values, m_h2, SEED + fold["fold_id"])
    b_shap, b_del, cc_tr = backgrounds(Xtr, SEED + fold["fold_id"])
    xs = X.iloc[sample].values
    h2 = []
    shap_out = {}
    for m in ["e3", "s_star"]:
        shap_out[m] = []
        for k, i in enumerate(sample):
            sh = explain.permutation_shap(fm.f(m), xs[k:k + 1], b_shap, EXP["shap_max_evals"], seed=10_000 + i)
            shap_out[m].append(sh["values"][0])
    for k, i in enumerate(sample):
        row = {"row": int(i), "y": int(y.iloc[i])}
        for m in ["e3", "s_star"]:
            e = shap_out[m][k]
            row[m] = ndf_record(fm.f(m), xs[k], e, b_del, seed=20_000 + i)
            row[m + "_knn"] = ndf_record(fm.f(m), xs[k], e, b_del, seed=20_000 + i, operator="knn", X_ref=cc_tr)
            row[m + "_shap"] = e.tolist()
        h2.append(row)
    res["h2_patients"] = h2
    res["seconds"] = round(time.time() - t0, 1)
    path.write_text(json.dumps(res, default=float))
    return str(path)


# ---------------------------------------------------------------- Stage C task (repeat-1 fold)
def task_secondary(name, fold, patients, quick, outdir):
    path = outdir / f"{name}_r1fold{fold['fold_id']:02d}.json"
    if path.exists():
        return str(path)
    t0 = time.time()
    X, y, folds, ft = load_processed(name)
    cols = list(X.columns); types = [ft[c] for c in cols]
    Xtr, ytr = X.iloc[fold["train"]], y.iloc[fold["train"]]
    fm = FoldModels(Xtr, ytr, ft, SEED)
    b_shap, b_del, cc_tr = backgrounds(Xtr, SEED + fold["fold_id"])
    sd = Xtr.std().values; lo = Xtr.min().values; hi = Xtr.max().values
    rows = []
    # Pre-registered reduction order (Stage 2.6 sec. 12) applied 11 Nov 2026 for runtime:
    # step 2 neighbours 5 -> 3; step 3 robustness restricted to E3 and S*; step 4 patients 150 -> 100.
    rob_models = ["e3", "s_star"]
    n_nb = 2 if quick else 3
    for i in patients:
        x = X.iloc[i].values
        rec = {"row": int(i), "y": int(y.iloc[i]), "fold_id": fold["fold_id"], "s_star": fm.s_star}
        seed_i = 10_000 + i
        # neighbours (shared across models: common random numbers)
        rng = np.random.default_rng(30_000 + i)
        nb = {s: [metrics_xai.perturb(x, types, sd, lo, hi, s, rng) for _ in range(n_nb)]
              for s in [EXP["robustness_sigma_frac"]] + list(EXP["robustness_sigma_sensitivity"])}
        for m in ALL_MODELS + ["s_star"]:
            if m == "s_star":
                f = fm.f(fm.s_star)
            else:
                f = fm.f(m)
            if m == "s_star":
                e = np.array(rec[fm.s_star]["shap"])
            else:
                e = explain.permutation_shap(f, x[None], b_shap, EXP["shap_max_evals"], seed=seed_i)["values"][0]
            r = {"shap": e.tolist(), **ndf_record(f, x, e, b_del, seed=20_000 + i)}
            if m in rob_models:
                fx = float(f(x[None])[0]); cls = fx >= 0.5
                e_seed = explain.permutation_shap(f, x[None], b_shap, EXP["shap_max_evals"], seed=seed_i + 7)["values"][0]
                r["noise_floor"] = metrics_xai.topk_overlap(e, e_seed)
                sigmas = [EXP["robustness_sigma_frac"]] + list(EXP["robustness_sigma_sensitivity"])
                for s in sigmas:
                    Xn = np.array(nb[s])
                    fn = f(Xn)
                    En = explain.permutation_shap(f, Xn, b_shap, EXP["shap_max_evals"], seed=seed_i)["values"]
                    rb = metrics_xai.robustness(e, list(En), list((fn >= 0.5) == cls))
                    r[f"rb_{s}"] = rb["rb"]; r[f"flip_{s}"] = rb["flip_rate"]
                    r[f"dfx_{s}"] = float(np.mean(np.abs(fn - fx)))
                # LIME diagnostic
                try:
                    lc = explain.lime_contributions(f, x, Xtr.values, types, 1000 if quick else EXP["lime_samples"], seed_i)
                    ag = metrics_xai.agreement(e, lc["contributions"])
                    r["lime_top5_overlap"] = ag["top5_overlap"]; r["lime_sign_agree"] = ag["sign_agreement"]
                    r["lime_local_error"] = lc["local_error"]; r["lime_score"] = lc["score"]
                except Exception as exc:  # noqa: BLE001
                    r["lime_error"] = repr(exc)
            rec[m] = r
        rows.append(rec)
    out = {"dataset": name, "fold_id": fold["fold_id"], "s_star": fm.s_star, "patients": rows,
           "seconds": round(time.time() - t0, 1)}
    path.write_text(json.dumps(out, default=float))
    return str(path)


def secondary_plan(name, quick):
    X, y, folds, ft = load_processed(name)
    r1 = [f for f in folds if f["repeat"] == 0]
    pool = [i for f in r1 for i in f["test"] if not X.iloc[i].isna().any()]
    n = 12 if quick else 100   # reduction step 4 (150 -> 100)
    chosen = set(stratified_sample(pool, y.values, n, SEED + 999))
    return [(f, sorted(i for i in f["test"] if i in chosen)) for f in r1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["A", "C", "all"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--datasets", default="wdbc,dr_debrecen,cleveland")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap2 = ROOT / "results" / ("study5" if args.datasets == "cdc_diabetes" else "study2" if args.datasets in ("ilpd,parkinsons,mammographic,ctg",) or set(args.datasets.split(",")) <= {"ilpd", "parkinsons", "mammographic", "ctg"} else "confirmatory")
    out = (ROOT / "results" / "confirmatory_quick" / "raw") if args.quick else ap2 / "raw"
    out.mkdir(parents=True, exist_ok=True)
    names = args.datasets.split(",")
    if args.stage in ("A", "all"):
        jobs = []
        for n in names:
            X, y, folds, ft = load_processed(n)
            sel = folds[:2] if args.quick else folds
            jobs += [(n, f) for f in sel]
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(task_fold)(n, f, args.quick, out) for n, f in jobs)
        print(f"Stage A+B done: {len(res)} tasks in {time.time() - t0:.0f}s", flush=True)
    if args.stage in ("C", "all"):
        jobs = []
        for n in names:
            plan = secondary_plan(n, args.quick)
            if args.quick:
                plan = plan[:1]
            jobs += [(n, f, p) for f, p in plan]
        t0 = time.time()
        res = Parallel(n_jobs=args.jobs, verbose=5)(delayed(task_secondary)(n, f, p, args.quick, out) for n, f, p in jobs)
        print(f"Stage C done: {len(res)} tasks in {time.time() - t0:.0f}s", flush=True)
