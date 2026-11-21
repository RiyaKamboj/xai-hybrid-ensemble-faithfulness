"""Study 4 (exploratory; docs/PROTOCOL_STUDY2_STUDY3.md, section 'Study 4').

For each dataset (7) and repeat-1 outer fold (5), refit the fold models (deterministic, identical to
Studies 1-2), take the same 20 H2 patients and, for E3 and S*:
  (a) stored permutation SHAP (feature level)      -> Bhatt faithfulness correlation
  (b) KernelSHAP (feature level, 500 samples)      -> NDF marginal + kNN, Bhatt correlation
  (c) group Shapley values on correlation clusters -> group-level NDF marginal + kNN
      and group-level NDF of (a) summed within groups (same deletion units, different attribution).
Usage: python scripts/09_study4_remedy_metrics.py [--datasets ...] [--jobs 2] [--quick]
"""
import argparse, importlib.util, itertools, json, math, os, sys, time, warnings
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run", ROOT / "scripts" / "04_run_confirmatory.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
EXP, SEED = run.EXP, run.SEED
S1 = {"wdbc", "dr_debrecen", "cleveland"}
ALL = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg"]
OUT = ROOT / "results" / "study4" / "raw"
CUT = 0.3            # 1 - |Spearman rho| linkage cut  ->  |rho| >= 0.7 merged
EXACT_MAX = 8


def corr_groups(Xtr):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    R = Xtr.corr(method="spearman").abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(R, 1.0)
    d = Xtr.shape[1]
    if d == 1:
        return [[0]]
    D = np.clip(1 - R, 0, None); D = (D + D.T) / 2
    lab = fcluster(linkage(squareform(D, checks=False), "average"), t=CUT, criterion="distance")
    groups = {}
    for j, l in enumerate(lab):
        groups.setdefault(l, []).append(j)
    return sorted(groups.values(), key=lambda g: g[0])


def _rep(x, cols, bg):
    Z = np.repeat(x[None, :], len(bg), 0)
    if len(cols):
        Z[:, cols] = bg[:, cols]
    return Z


def group_shapley(f, x, groups, bg, seed):
    """Interventional group Shapley values; v(S) = mean_b f(x_S, b_-S)."""
    g = len(groups)
    cols_of = lambda S: [c for k in range(g) if k not in S for c in groups[k]]   # columns replaced by bg
    if g <= EXACT_MAX:
        coal = [frozenset(c) for r in range(g + 1) for c in itertools.combinations(range(g), r)]
        preds = f(np.vstack([_rep(x, cols_of(S), bg) for S in coal]))
        v = dict(zip(coal, preds.reshape(len(coal), len(bg)).mean(1)))
        phi = np.zeros(g)
        for k in range(g):
            for S in coal:
                if k in S:
                    continue
                w = math.factorial(len(S)) * math.factorial(g - len(S) - 1) / math.factorial(g)
                phi[k] += w * (v[S | {k}] - v[S])
        return phi, "exact"
    rng = np.random.default_rng(seed); phi = np.zeros(g); n = 0
    for _ in range(25):
        p = rng.permutation(g)
        for perm in (p, p[::-1]):
            chain = [frozenset(perm[:i]) for i in range(g + 1)]
            preds = f(np.vstack([_rep(x, cols_of(S), bg) for S in chain])).reshape(g + 1, len(bg)).mean(1)
            for i, k in enumerate(perm):
                phi[k] += preds[i + 1] - preds[i]
            n += 1
    return phi / n, "permutation_50"


def ndf_group(f, x, eg, groups, bg, seed, operator="marginal", X_ref=None, K_max=5, n_random=20):
    """NDF with whole groups as deletion units (same definition as metrics_xai.ndf)."""
    from xhe.metrics_xai import knn_background
    fx = float(f(x[None])[0]); s = 1.0 if fx >= 0.5 else -1.0
    G = s * (fx - float(np.mean(f(bg)))); c = s * np.asarray(eg, float); g = len(groups)
    order = np.argsort(-c, kind="stable"); pos = [k for k in order if c[k] > 0]; K = min(K_max, len(pos))
    res = {"ndf": np.nan, "G": G, "K": K, "f_x": fx}
    if K == 0 or not G > 1e-12:
        return res
    if operator == "knn":
        scale = np.nanstd(X_ref, 0); scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    rng = np.random.default_rng(seed); perms = [rng.permutation(g) for _ in range(n_random)]
    kmax = min(K, g)
    sets = [pos[:k] for k in range(1, K + 1)] + [list(p[:k]) for p in perms for k in range(1, K + 1)]
    blocks = []
    for S in sets:
        cols = [cc for k in S for cc in groups[k]]
        b = bg if operator == "marginal" else knn_background(x, X_ref, [j for j in range(len(x)) if j not in set(cols)], EXP["knn_operator_neighbours"], scale)
        blocks.append(_rep(x, cols, b))
    sizes = [len(b) for b in blocks]; preds = f(np.vstack(blocks))
    means = np.array([m.mean() for m in np.split(preds, np.cumsum(sizes)[:-1])])
    D = s * (fx - means) / G
    res["ndf"] = float(np.mean(D[:K] - D[K:].reshape(n_random, K).mean(0)))
    return res


def bhatt(f, x, e, bg, seed, n_sub=50):
    """Faithfulness correlation (Bhatt et al., 2020), marginal removal."""
    d = len(x); k = max(1, math.ceil(d / 4)); rng = np.random.default_rng(seed)
    subs = [rng.choice(d, k, replace=False) for _ in range(n_sub)]
    fx = float(f(x[None])[0])
    drops = fx - f(np.vstack([_rep(x, list(S), bg) for S in subs])).reshape(n_sub, len(bg)).mean(1)
    attr = np.array([np.sum(np.asarray(e)[S]) for S in subs])
    if np.std(attr) < 1e-12 or np.std(drops) < 1e-12:
        return np.nan
    return float(np.corrcoef(attr, drops)[0, 1])


def kernel_shap(f, x, bg, seed):
    import shap
    np.random.seed(seed)
    ex = shap.KernelExplainer(lambda Z: f(Z), bg)
    return np.asarray(ex.shap_values(x[None], nsamples=500, silent=True), float).reshape(-1)


def task(name, fid, quick, outdir):
    path = Path(outdir) / f"{name}_fold{fid:02d}.json"
    if path.exists():
        return str(path)
    t0 = time.time(); warnings.filterwarnings("ignore")
    X, y, folds, ft = run.load_processed(name)
    fold = [f for f in folds if f["fold_id"] == fid][0]
    src = ROOT / "results" / ("confirmatory" if name in S1 else "study5" if name == "cdc_diabetes" else "study2") / "raw" / f"{name}_fold{fid:02d}.json"
    prev = json.loads(src.read_text())
    Xtr, ytr = X.iloc[fold["train"]], y.iloc[fold["train"]]
    fm = run.FoldModels(Xtr, ytr, ft, SEED, groups=run.GROUPS(name, fold["train"]))
    assert fm.s_star == prev["s_star"], (name, fid, fm.s_star, prev["s_star"])
    b_shap, b_del, cc_tr = run.backgrounds(Xtr, SEED + fid)
    groups = corr_groups(Xtr)
    pats = prev["h2_patients"][:3] if quick else prev["h2_patients"]
    rows = []
    for r in pats:
        i = r["row"]; x = X.iloc[i].values.astype(float); out = {"row": i, "y": r["y"]}
        for m in ["e3", "s_star"]:
            f = fm.f(m); ea = np.asarray(r[m + "_shap"], float)
            # reproducibility check of the refit: same f(x) as stored
            out[m + "_fx_diff"] = abs(float(f(x[None])[0]) - r[m]["f_x"])
            eb = kernel_shap(f, x, b_shap, 30_000 + i)
            eg, how = group_shapley(f, x, groups, b_shap, 40_000 + i)
            ea_g = np.array([ea[g].sum() for g in groups])
            out[m] = {
                "a_ndf": r[m]["ndf"], "a_G": r[m]["G"], "a_ndf_knn": r[m + "_knn"]["ndf"],
                "a_bhatt": bhatt(f, x, ea, b_del, 50_000 + i),
                "b_ndf": run.ndf_record(f, x, eb, b_del, seed=20_000 + i)["ndf"],
                "b_ndf_knn": run.ndf_record(f, x, eb, b_del, seed=20_000 + i, operator="knn", X_ref=cc_tr)["ndf"],
                "b_bhatt": bhatt(f, x, eb, b_del, 50_000 + i),
                "c_ndf": ndf_group(f, x, eg, groups, b_del, 60_000 + i)["ndf"],
                "c_ndf_knn": ndf_group(f, x, eg, groups, b_del, 60_000 + i, "knn", cc_tr)["ndf"],
                "asum_ndf_g": ndf_group(f, x, ea_g, groups, b_del, 60_000 + i)["ndf"],
                "c_method": how, "rank_corr_ab": float(pd.Series(ea).corr(pd.Series(eb), method="spearman")),
                "kernel": eb.tolist(), "group": eg.tolist()}
        rows.append(out)
    res = {"dataset": name, "fold_id": fid, "s_star": fm.s_star, "groups": groups, "n_groups": len(groups),
           "d": X.shape[1], "patients": rows, "seconds": round(time.time() - t0, 1)}
    path.write_text(json.dumps(res, default=float))
    return str(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    od = ROOT / "results" / "study4" / ("quick" if a.quick else "raw"); od.mkdir(parents=True, exist_ok=True)
    jobs = [(d, k) for d in a.datasets for k in range(5)]
    for p in Parallel(n_jobs=a.jobs, verbose=5)(delayed(task)(d, k, a.quick, str(od)) for d, k in jobs):
        print(p, flush=True)
