"""Revision analyses R-A, R-B, R-C (docs/PROTOCOL_STUDY2_STUDY3.md, 'Revision analyses').
R-A: permutation SHAP + NDF for every component and for E1/E2 on the primary patients of repetition-1 folds.
R-B: exact interventional TreeSHAP for RF/XGB (datasets without nominal features).
R-C: Monte-Carlo error of SHAP and NDF for E3 and S* (fold 0).
Usage: python scripts/14_component_faithfulness.py [--datasets ...] [--jobs 2]
"""
import argparse, importlib.util, json, os, time, warnings
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
ALL = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
S1 = {"wdbc", "dr_debrecen", "cleveland"}
MODELS = ["lr", "svm", "knn", "rf", "xgb", "e1", "e2"]


def task(name, fid, outdir):
    path = Path(outdir) / f"{name}_fold{fid:02d}.json"
    if path.exists():
        return str(path)
    warnings.filterwarnings("ignore"); t0 = time.time()
    s = importlib.util.spec_from_file_location("run", ROOT / "scripts" / "04_run_confirmatory.py")
    run = importlib.util.module_from_spec(s); s.loader.exec_module(run)
    from xhe import explain, metrics_xai
    EXP, SEED = run.EXP, run.SEED
    X, y, folds, ft = run.load_processed(name)
    fold = [f for f in folds if f["fold_id"] == fid][0]
    sub = "confirmatory" if name in S1 else "study5" if name == "cdc_diabetes" else "study2"
    prev = json.loads((ROOT / "results" / sub / "raw" / f"{name}_fold{fid:02d}.json").read_text())
    Xtr, ytr = X.iloc[fold["train"]], y.iloc[fold["train"]]
    fm = run.FoldModels(Xtr, ytr, ft, SEED, groups=run.GROUPS(name, fold["train"]))
    assert fm.s_star == prev["s_star"]
    b_shap, b_del, cc_tr = run.backgrounds(Xtr, SEED + fid)

    def ndf_full(f, x, e, seed, op="marginal"):
        r = metrics_xai.ndf(f, x, e, b_del, K_max=EXP["ndf_K"], n_random=EXP["ndf_random_orderings"], seed=seed,
                            operator=op, X_ref=cc_tr if op == "knn" else None, knn_n=EXP["knn_operator_neighbours"])
        return {"ndf": r["ndf"], "G": r["G"], "K": r["K"], "f_x": r["f_x"], "D_expl": r["D_expl"], "D_rand": r["D_rand_mean"]}

    nominal = any(t == "nominal" for t in ft.values())
    tree = {}
    if not nominal:
        import shap
        for c in ["rf", "xgb"]:
            pipe = fm.comp[c]; prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
            bg_t = prep.transform(fm._df(b_shap))
            tree[c] = (prep, shap.TreeExplainer(clf, data=bg_t, feature_perturbation="interventional", model_output="probability"))
    rows = []
    for r in prev["h2_patients"]:
        i = r["row"]; x = X.iloc[i].values.astype(float); out = {"row": i, "y": r["y"]}
        for m in MODELS + ["e3", "s_star"]:
            f = fm.f(m)
            if m in ("e3", "s_star"):
                e = np.asarray(r[m + "_shap"], float)
            else:
                e = explain.permutation_shap(f, x[None], b_shap, EXP["shap_max_evals"], seed=10_000 + i)["values"][0]
            q = {"marg": ndf_full(f, x, e, 20_000 + i), "knn": ndf_full(f, x, e, 20_000 + i, "knn"), "shap": np.asarray(e).tolist()}
            if m in tree:
              try:
                prep, te = tree[m]
                sv = te.shap_values(prep.transform(fm._df(x)))
                sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
                sv = sv[0, :, 1] if sv.ndim == 3 else sv.reshape(-1)
                q["tree_marg"] = ndf_full(f, x, sv, 20_000 + i)
                from scipy.stats import spearmanr
                q["tree_rho"] = float(spearmanr(sv, e).statistic)
              except Exception as ex:
                q["tree_error"] = repr(ex)[:200]
            out[m] = q
        if fid == 0:   # R-C Monte-Carlo error
            mc = {}
            for m in ["e3", "s_star"]:
                f = fm.f(m); e0 = np.asarray(r[m + "_shap"], float)
                shap_rep = [e0] + [explain.permutation_shap(f, x[None], b_shap, EXP["shap_max_evals"], seed=10_000 + i + k * 1_000_000)["values"][0] for k in (1, 2)]
                mc[m] = {"ndf_by_shap_seed": [ndf_full(f, x, e, 20_000 + i)["ndf"] for e in shap_rep],
                         "ndf_by_order_seed": [ndf_full(f, x, e0, 20_000 + i + k * 1_000_000)["ndf"] for k in (0, 1, 2)]}
            out["mc"] = mc
        rows.append(out)
    res = {"dataset": name, "fold_id": fid, "s_star": fm.s_star, "tree_check": bool(tree), "patients": rows, "seconds": round(time.time() - t0, 1)}
    path.write_text(json.dumps(res, default=float))
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="raw")
    a = ap.parse_args()
    od = ROOT / "results" / "components" / a.out; od.mkdir(parents=True, exist_ok=True)
    jobs = [(d, k) for k in a.folds for d in a.datasets]
    for p in Parallel(n_jobs=a.jobs, verbose=5)(delayed(task)(d, k, str(od)) for d, k in jobs):
        print(p, flush=True)
