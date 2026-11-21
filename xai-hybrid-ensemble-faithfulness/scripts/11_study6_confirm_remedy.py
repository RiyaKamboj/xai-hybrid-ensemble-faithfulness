"""Study 6 (pre-specified 21 Nov 2026): confirmatory test of the group remedy on held-out repetitions 2-5,
random-partition placebo, and the non-additivity mechanism test. See docs/PROTOCOL_STUDY2_STUDY3.md.
Usage: python scripts/11_study6_confirm_remedy.py [--datasets ...] [--jobs 2] [--quick]
"""
import argparse, importlib.util, json, os, sys, time, warnings
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
ALL = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
S1 = {"wdbc", "dr_debrecen", "cleveland"}


def _mods():
    def load(n, f):
        s = importlib.util.spec_from_file_location(n, ROOT / "scripts" / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    return load("run", "04_run_confirmatory.py"), load("s4", "09_study4_remedy_metrics.py")


def non_additivity(f, x, bg, G, s, seed, n_pairs=20):
    d = len(x); rng = np.random.default_rng(seed)
    pairs = [tuple(rng.choice(d, 2, replace=False)) for _ in range(n_pairs)]
    feats = sorted({j for p in pairs for j in p})
    sets = [[j] for j in feats] + [list(p) for p in pairs]
    fx = float(f(x[None])[0])
    Z = np.vstack([np.where(np.isin(np.arange(d), S)[None, :], bg, x[None, :]) for S in sets])
    F = f(Z).reshape(len(sets), len(bg)).mean(1)
    D = {tuple(S): s * (fx - F[i]) for i, S in enumerate(sets)}
    return float(np.mean([abs(D[(j, l)] - D[(j,)] - D[(l,)]) for j, l in pairs]) / G)


def random_partitions(groups, d, seed, n=5):
    sizes = [len(g) for g in groups]; rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        p = rng.permutation(d); k = 0; part = []
        for sz in sizes:
            part.append(sorted(p[k:k + sz].tolist())); k += sz
        out.append(part)
    return out


def task(name, fid, outdir):
    path = Path(outdir) / f"{name}_fold{fid:02d}.json"
    if path.exists():
        return str(path)
    warnings.filterwarnings("ignore"); t0 = time.time()
    run, s4 = _mods(); EXP, SEED = run.EXP, run.SEED
    X, y, folds, ft = run.load_processed(name)
    fold = [f for f in folds if f["fold_id"] == fid][0]
    sub = "confirmatory" if name in S1 else "study5" if name == "cdc_diabetes" else "study2"
    prev = json.loads((ROOT / "results" / sub / "raw" / f"{name}_fold{fid:02d}.json").read_text())
    Xtr, ytr = X.iloc[fold["train"]], y.iloc[fold["train"]]
    fm = run.FoldModels(Xtr, ytr, ft, SEED, groups=run.GROUPS(name, fold["train"]))
    assert fm.s_star == prev["s_star"], (name, fid)
    b_shap, b_del, cc_tr = run.backgrounds(Xtr, SEED + fid)
    groups = s4.corr_groups(Xtr); d = X.shape[1]; merged = any(len(g) > 1 for g in groups)
    rows = []
    for r in prev["h2_patients"]:
        i = r["row"]; x = X.iloc[i].values.astype(float); out = {"row": i}
        rparts = random_partitions(groups, d, 70_000 + i)
        for m in ["e3", "s_star"]:
            f = fm.f(m); ea = np.asarray(r[m + "_shap"], float); G = r[m]["G"]
            out[m + "_fx_diff"] = abs(float(f(x[None])[0]) - r[m]["f_x"])
            esum = np.array([ea[g].sum() for g in groups])
            q = {"G": G, "feat": r[m]["ndf"], "feat_knn": r[m + "_knn"]["ndf"],
                 "grp": s4.ndf_group(f, x, esum, groups, b_del, 60_000 + i)["ndf"],
                 "grp_knn": s4.ndf_group(f, x, esum, groups, b_del, 60_000 + i, "knn", cc_tr)["ndf"]}
            vals = []
            for k, part in enumerate(rparts):
                vals.append(s4.ndf_group(f, x, np.array([ea[g].sum() for g in part]), part, b_del, 60_000 + i)["ndf"])
            q["rand"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            if merged and len(groups) <= 8:
                eg, _ = s4.group_shapley(f, x, groups, b_shap, 40_000 + i)
                q["gshap"] = s4.ndf_group(f, x, eg, groups, b_del, 60_000 + i)["ndf"]
            s = 1.0 if r[m]["f_x"] >= 0.5 else -1.0
            q["na"] = non_additivity(f, x, b_del, G, s, 80_000 + i) if G > 1e-12 else np.nan
            out[m] = q
        rows.append(out)
    res = {"dataset": name, "fold_id": fid, "s_star": fm.s_star, "groups": groups, "n_groups": len(groups), "d": d,
           "merged": merged, "patients": rows, "seconds": round(time.time() - t0, 1)}
    path.write_text(json.dumps(res, default=float))
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    od = ROOT / "results" / "study6" / ("quick" if a.quick else "raw"); od.mkdir(parents=True, exist_ok=True)
    fids = [5] if a.quick else list(range(5, 25))
    jobs = [(d, k) for k in fids for d in a.datasets]       # interleave datasets
    for p in Parallel(n_jobs=a.jobs, verbose=5)(delayed(task)(d, k, str(od)) for d, k in jobs):
        print(p, flush=True)
