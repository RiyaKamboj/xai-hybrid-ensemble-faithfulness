"""Study 3: controlled simulation of the mechanism (docs/PROTOCOL_STUDY2_STUDY3.md).

Conditions: rho in {0, 0.5, 0.9} (equicorrelated block x1..x4) x outcome model {L, N},
3 replicates each; 5-fold stratified CV; 10 test patients per fold explained with E3, LR and S*.
Outputs: results/simulation/raw/<cond>_rep<k>.json
"""
import importlib.util
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xhe import explain  # noqa: E402

spec = importlib.util.spec_from_file_location("runner", ROOT / "scripts" / "04_run_confirmatory.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
EXP = runner.EXP
OUT = ROOT / "results" / "simulation" / "raw"
D, N, NPAT = 10, 600, 10


def simulate(rho, truth, seed):
    rng = np.random.default_rng(seed)
    cov = np.eye(D)
    cov[:4, :4] = rho
    np.fill_diagonal(cov, 1.0)
    X = rng.multivariate_normal(np.zeros(D), cov, size=N)
    x = lambda j: X[:, j - 1]
    if truth == "L":
        eta = 1.2 * x(1) + 0.8 * x(2) + 0.6 * x(5) - 0.6 * x(6) + 0.4 * x(7)
    else:
        eta = 1.2 * x(1) + 0.8 * (x(2) > 0) + 0.8 * x(5) * x(6) - 0.6 * x(6) + 0.4 * x(7) ** 2
    eta = eta - np.median(eta)
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    cols = [f"x{j}" for j in range(1, D + 1)]
    return pd.DataFrame(X, columns=cols), pd.Series(y)


def task(rho, truth, rep):
    path = OUT / f"{truth}_rho{rho}_rep{rep}.json"
    if path.exists():
        return str(path)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    seed = 1000 * rep + int(rho * 10) + (0 if truth == "L" else 500)
    X, y = simulate(rho, truth, seed)
    ft = {c: "continuous" for c in X.columns}
    rows, folds = [], []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)):
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te].values
        fm = runner.FoldModels(Xtr, ytr, ft, seed)
        folds.append({"fold": k, "s_star": fm.s_star,
                      **{f"auc_{m}": float(roc_auc_score(yte, fm.proba(m, Xte.values))) for m in ["e3", "lr", "s_star"]}})
        b_shap, b_del, cc = runner.backgrounds(Xtr, seed + k)
        pats = runner.stratified_sample(list(te), y.values, NPAT, seed + 100 + k)
        for i in pats:
            xi = X.iloc[i].values
            rec = {"fold": k, "row": int(i), "y": int(y.iloc[i]), "s_star_name": fm.s_star}
            for m in ["e3", "lr", "s_star"]:
                if m == "s_star" and fm.s_star in ("lr",):
                    rec["s_star"] = {**rec["lr"]}; continue
                f = fm.f(m)
                e = explain.permutation_shap(f, xi[None], b_shap, EXP["shap_max_evals"], seed=10_000 + i)["values"][0]
                rec[m] = {"marg": runner.ndf_record(f, xi, e, b_del, seed=20_000 + i),
                          "knn": runner.ndf_record(f, xi, e, b_del, seed=20_000 + i, operator="knn", X_ref=cc)}
            rec.setdefault("s_star_name", fm.s_star)
            rows.append(rec)
    out = {"truth": truth, "rho": rho, "rep": rep, "seed": seed, "folds": folds, "patients": rows,
           "prevalence": float(y.mean()), "seconds": round(time.time() - t0, 1)}
    path.write_text(json.dumps(out, default=float))
    return str(path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(rho, t, r) for t in ["L", "N"] for rho in [0.0, 0.5, 0.9] for r in [1, 2, 3]]
    if "--quick" in sys.argv:
        jobs = jobs[:1]
    t0 = time.time()
    Parallel(n_jobs=2, verbose=5)(delayed(task)(*j) for j in jobs)
    print(f"simulation done: {len(jobs)} tasks in {time.time() - t0:.0f}s", flush=True)
