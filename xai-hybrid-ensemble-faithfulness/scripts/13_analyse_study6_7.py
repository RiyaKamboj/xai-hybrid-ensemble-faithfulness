"""Analysis of Study 6 (confirmatory remedy, placebo, mechanism) and Study 7 (tuned models)."""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "scripts" / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ana = _load("ana", "05_analyse.py"); s23 = _load("s23", "08_analyse_study2_3.py")
ALL = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
OUT = ROOT / "results" / "study6_7"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
pool = lambda R: s23.dersimonian_laird([r["mean_diff"] for r in R], [s23.se_of(r) for r in R])
short = lambda r: {k: round(float(r[k]), 4) for k in ["pooled" if "pooled" in r else "mean_diff", "ci_low", "ci_high", "p"] + (["I2"] if "I2" in r else [])}


def study6():
    rows = []
    for p in sorted((ROOT / "results" / "study6" / "raw").glob("*.json")):
        d = json.loads(p.read_text())
        for r in d["patients"]:
            a, b = r["e3"], r["s_star"]
            if not (a["G"] >= 0.05 and b["G"] >= 0.05):
                continue
            rec = {"dataset": d["dataset"], "fold": d["fold_id"], "merged": d["merged"], "cl": f'{d["dataset"]}_{d["fold_id"]}'}
            for k in ["feat", "grp", "rand", "grp_knn", "feat_knn", "gshap", "na"]:
                rec["d_" + k] = (a[k] - b[k]) if (k in a and np.isfinite(a.get(k, np.nan)) and np.isfinite(b.get(k, np.nan))) else np.nan
            rec["na_e3"], rec["na_s"] = a["na"], b["na"]
            rows.append(rec)
    P = pd.DataFrame(rows); P.to_csv(OUT / "study6_patients.csv", index=False)
    P["R"] = P.d_grp - P.d_feat; P["PL"] = P.d_grp - P.d_rand; P["Rk"] = P.d_grp_knn - P.d_feat_knn
    res = {"per_dataset": {}, "pooled": {}}
    for d in ALL:
        g = P[P.dataset == d]
        if g.empty:
            continue
        fm = g.groupby("fold").mean(numeric_only=True)
        res["per_dataset"][d] = {"n_folds": int(g.fold.nunique()), "n_patients": len(g), "merged_share": float(g.groupby("fold").merged.first().mean()),
                                 **{k: ana.corrected_t(fm[k].values) for k in ["d_feat", "d_grp", "d_rand", "R", "PL", "d_feat_knn", "d_grp_knn", "Rk", "d_na"]},
                                 **({"d_gshap": ana.corrected_t(fm["d_gshap"].dropna().values)} if fm["d_gshap"].notna().sum() >= 3 else {})}
    names = list(res["per_dataset"])
    merged = [d for d in names if res["per_dataset"][d]["merged_share"] >= 0.5]
    for k in ["d_feat", "d_grp", "d_rand", "R", "PL", "d_feat_knn", "d_grp_knn", "Rk", "d_na"]:
        res["pooled"][k] = pool([res["per_dataset"][d][k] for d in names])
        res["pooled"][k + "_merged"] = pool([res["per_dataset"][d][k] for d in merged])
    gs = [d for d in names if "d_gshap" in res["per_dataset"][d]]
    if gs:
        res["pooled"]["d_gshap_sub"] = pool([res["per_dataset"][d]["d_gshap"] for d in gs])
        res["pooled"]["d_grp_sub"] = pool([res["per_dataset"][d]["d_grp"] for d in gs])
    res["merged_datasets"] = merged; res["gshap_datasets"] = gs
    # mechanism: patient-level dNDF ~ dNA + dataset FE, clustered by dataset x fold
    import statsmodels.formula.api as smf
    Q = P.dropna(subset=["d_feat", "d_na"])
    m = smf.ols("d_feat ~ d_na + C(dataset)", data=Q).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(Q.cl)[0]})
    res["mechanism_ols"] = {"coef": float(m.params["d_na"]), "se": float(m.bse["d_na"]), "p": float(m.pvalues["d_na"]), "n": int(len(Q))}
    from scipy.stats import spearmanr
    res["mechanism_spearman_sensitivity"] = {"rho": float(spearmanr(Q.d_feat, Q.d_na).statistic), "p": float(spearmanr(Q.d_feat, Q.d_na).pvalue)}
    # fold-level version (less sensitive to heavy tails): within-dataset demeaned fold means
    F = Q.groupby(["dataset", "fold"])[["d_feat", "d_na"]].mean().reset_index()
    mf = smf.ols("d_feat ~ d_na + C(dataset)", data=F).fit(cov_type="HC3")
    res["mechanism_fold_ols"] = {"coef": float(mf.params["d_na"]), "se": float(mf.bse["d_na"]), "p": float(mf.pvalues["d_na"]), "n": int(len(F))}
    return res, P


def study7():
    tuned = s23.fold_effects(ROOT / "results" / "study7" / "raw", ALL)
    base = {}
    for sub, ds in [("confirmatory", ["wdbc", "dr_debrecen", "cleveland"]), ("study2", ["ilpd", "parkinsons", "mammographic", "ctg"]), ("study5", ["cdc_diabetes"])]:
        tmp = ROOT / "results" / "study6_7" / f"_untuned_r1_{sub}"; tmp.mkdir(exist_ok=True)
        for d in ds:
            for k in range(5):
                src = ROOT / "results" / sub / "raw" / f"{d}_fold{k:02d}.json"
                (tmp / src.name).write_text(src.read_text())
        base.update(s23.fold_effects(tmp, ds))
    res = {"tuned": tuned, "untuned_same_folds": base, "pooled": {}}
    for lab, src in [("tuned", tuned), ("untuned", base)]:
        names = [d for d in ALL if d in src]
        for k in ["H1", "H2", "H2_knn"]:
            res["pooled"][f"{lab}_{k}"] = pool([src[d][k] for d in names])
    return res


if __name__ == "__main__":
    out = {}
    if (ROOT / "results" / "study6" / "raw").exists():
        r6, P = study6(); out["study6"] = r6
        print("STUDY 6 pooled:"); [print(f"  {k:18s}", short(v)) for k, v in r6["pooled"].items()]
        for d, v in r6["per_dataset"].items():
            print(f"  {d:13s} n={v['n_patients']:4d} merged={v['merged_share']:.1f}", " ".join(f"{k}:{v[k]['mean_diff']:+.3f}(p{v[k]['p']:.3f})" for k in ["d_feat", "d_grp", "R", "PL", "d_na"]))
        print("  mechanism", r6["mechanism_ols"], r6["mechanism_fold_ols"], r6["mechanism_spearman_sensitivity"])
    if (ROOT / "results" / "study7" / "raw").exists() and len(list((ROOT / "results" / "study7" / "raw").glob("*.json"))) >= 8:
        r7 = study7(); out["study7"] = r7
        print("STUDY 7 pooled:"); [print(f"  {k:18s}", short(v)) for k, v in r7["pooled"].items()]
        for d in r7["tuned"]:
            t, u = r7["tuned"][d], r7["untuned_same_folds"][d]
            print(f"  {d:13s} S* tuned {t['s_star_counts']} | dAUC {t['H1']['mean_diff']:+.4f} vs {u['H1']['mean_diff']:+.4f} | dNDF {t['H2']['mean_diff']:+.3f}(p{t['H2']['p']:.3f}) vs {u['H2']['mean_diff']:+.3f}")
    (OUT / "summary_study6_7.json").write_text(json.dumps(out, indent=1, default=float))
