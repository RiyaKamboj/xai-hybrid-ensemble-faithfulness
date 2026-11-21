"""Revision analyses R-D (re-analysis of existing outputs) and summaries of R-A/R-B/R-C.
Writes results/revision/summary_revision.json."""
import importlib.util, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
np.seterr(all="ignore")
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "scripts" / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ana = _load("ana", "05_analyse.py"); s23 = _load("s23", "08_analyse_study2_3.py")
OUT = ROOT / "results" / "revision"; OUT.mkdir(parents=True, exist_ok=True)
DS = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
SUB = {d: "confirmatory" for d in DS[:3]} | {d: "study2" for d in DS[3:7]} | {"cdc_diabetes": "study5"}
G0 = 0.05


def meta_full(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float); k = len(est)
    dl = s23.dersimonian_laird(est, se); tau2 = dl["tau2"]; w = 1 / (se ** 2 + tau2); mu = dl["pooled"]
    q = np.sum(w * (est - mu) ** 2) / (k - 1); se_hk = np.sqrt(q / np.sum(w)); t = st.t.ppf(.975, k - 1)
    se_dl = np.sqrt(1 / np.sum(w)); tp = st.t.ppf(.975, k - 2)
    return {**dl, "hksj_ci": [mu - t * se_hk, mu + t * se_hk], "hksj_p": float(2 * st.t.sf(abs(mu / se_hk), k - 1)),
            "pred_int": [mu - tp * np.sqrt(tau2 + se_dl ** 2), mu + tp * np.sqrt(tau2 + se_dl ** 2)]}


def primary_patients():
    rows = []
    for d in DS:
        for p in sorted((ROOT / "results" / SUB[d] / "raw").glob(f"{d}_fold*.json")):
            f = json.loads(p.read_text())
            for r in f["h2_patients"]:
                rows.append({"dataset": d, "fold": f["fold_id"], "s_star": f["s_star"], "G_e3": r["e3"]["G"], "G_s": r["s_star"]["G"],
                             "ndf_e3": r["e3"]["ndf"], "ndf_s": r["s_star"]["ndf"], "knn_e3": r["e3_knn"]["ndf"], "knn_s": r["s_star_knn"]["ndf"],
                             "fx_e3": r["e3"]["f_x"], "fx_s": r["s_star"]["f_x"], "K_e3": r["e3"]["K"], "K_s": r["s_star"]["K"]})
    P = pd.DataFrame(rows)
    P["inc"] = (P.G_e3 >= G0) & (P.G_s >= G0) & np.isfinite(P.ndf_e3) & np.isfinite(P.ndf_s)
    P["d"] = np.where(P.inc, P.ndf_e3 - P.ndf_s, np.nan)
    return P


def per_dataset_effect(P, col="d"):
    return {d: ana.corrected_t(P[P.dataset == d].groupby("fold")[col].mean().dropna().values) for d in DS}


def main():
    res = {}
    P = primary_patients(); P.to_csv(OUT / "primary_patients.csv", index=False)
    E = json.load(open(ROOT / "results/study2_3/summary_study2_3.json"))
    allds = {**E["study1_recomputed"], **E["study2"]}
    # 1. meta-analysis extras
    res["meta"] = {}
    for key in ["H1", "H2", "H2_knn"]:
        est = [allds[d][key]["mean_diff"] for d in DS]; se = [s23.se_of(allds[d][key]) for d in DS]
        m = meta_full(est, se)
        m["loo"] = {d: {k: v for k, v in s23.dersimonian_laird([e for e, dd in zip(est, DS) if dd != d], [s for s, dd in zip(se, DS) if dd != d]).items() if k in ("pooled", "ci_low", "ci_high", "p", "I2")} for d in DS}
        res["meta"][key] = m
    # 2. exclusions and G
    X = {}
    for d in DS:
        g = P[P.dataset == d]
        X[d] = {"sampled": len(g), "excl_e3": int((g.G_e3 < G0).sum()), "excl_s": int((g.G_s < G0).sum()), "included": int(g.inc.sum()),
                "median_G_e3": float(g.G_e3.median()), "median_G_s": float(g.G_s.median()),
                "median_ndf_e3": float(g[g.inc].ndf_e3.median()), "median_ndf_s": float(g[g.inc].ndf_s.median()),
                "share_e3_lower": float((g[g.inc].d < 0).mean()), "share_K_lt5_e3": float((g.K_e3 < 5).mean()), "share_K_lt5_s": float((g.K_s < 5).mean())}
    res["exclusions"] = X
    res["exclusions_total"] = {"sampled": int(len(P)), "excl_e3": int((P.G_e3 < G0).sum()), "excl_s": int((P.G_s < G0).sum()), "included": int(P.inc.sum()),
                               "median_G_e3": float(P.G_e3.median()), "median_G_s": float(P.G_s.median()),
                               "wilcoxon_G_p": float(st.wilcoxon(P.G_e3, P.G_s).pvalue),
                               "median_ndf_e3": float(P[P.inc].ndf_e3.median()), "median_ndf_s": float(P[P.inc].ndf_s.median()),
                               "share_e3_lower": float((P[P.inc].d < 0).mean())}
    # complete-case exclusions in test folds
    cc = {}
    for d in DS:
        df = pd.read_csv(ROOT / "data/processed" / f"{d}.csv").drop(columns="target")
        cc[d] = {"rows_with_missing": int(df.isna().any(axis=1).sum()), "rows": len(df)}
    res["complete_case"] = cc
    # 3. shrinkage check: similar G
    P["simG"] = P.inc & (np.maximum(P.G_e3, P.G_s) / np.minimum(P.G_e3, P.G_s) < 1.5)
    P["d_sim"] = np.where(P.simG, P.d, np.nan)
    pe = per_dataset_effect(P, "d_sim"); ok = [d for d in DS if np.isfinite(pe[d]["mean_diff"]) and pe[d]["n_folds"] >= 3]
    res["shrinkage_similarG"] = {"per_dataset": pe, "n_patients": int(P.simG.sum()),
                                 "pooled": meta_full([pe[d]["mean_diff"] for d in ok], [s23.se_of(pe[d]) for d in ok]), "datasets": ok}
    # 4. tuned results by S* type
    T = []
    for p in sorted((ROOT / "results/study7/raw").glob("*.json")):
        f = json.loads(p.read_text())
        for r in f["h2_patients"]:
            ok_ = r["e3"]["G"] >= G0 and r["s_star"]["G"] >= G0 and np.isfinite(r["e3"]["ndf"]) and np.isfinite(r["s_star"]["ndf"])
            if ok_:
                T.append({"dataset": f["dataset"], "fold": f["fold_id"], "s_star": f["s_star"], "nonlin": f["s_star"] != "lr",
                          "d": r["e3"]["ndf"] - r["s_star"]["ndf"], "dk": (r["e3_knn"]["ndf"] - r["s_star_knn"]["ndf"])})
    T = pd.DataFrame(T); T.to_csv(OUT / "tuned_patients.csv", index=False)
    Fm = T.groupby(["dataset", "fold", "nonlin", "s_star"]).agg(d=("d", "mean"), dk=("dk", "mean"), n=("d", "size")).reset_index()
    sub = {}
    for nl, lab in [(False, "linear_S*"), (True, "nonlinear_S*")]:
        g = Fm[Fm.nonlin == nl]
        for c in ["d", "dk"]:
            m, sd, n = g[c].mean(), g[c].std(ddof=1), len(g); se = sd / np.sqrt(n); t = st.t.ppf(.975, n - 1)
            sub[f"{lab}_{c}"] = {"mean": m, "ci": [m - t * se, m + t * se], "p": float(2 * st.t.sf(abs(m / se), n - 1)), "n_folds": n,
                                "datasets": sorted(g.dataset.unique().tolist())}
    import statsmodels.formula.api as smf
    mm = smf.ols("d ~ nonlin + C(dataset)", data=Fm).fit(cov_type="HC3")
    sub["model_nonlin_coef"] = {"coef": float(mm.params["nonlin[T.True]"]), "se": float(mm.bse["nonlin[T.True]"]), "p": float(mm.pvalues["nonlin[T.True]"])}
    sub["by_sstar_model"] = {k: {"mean_d": float(g.d.mean()), "n_folds": len(g)} for k, g in Fm.groupby("s_star")}
    res["tuned_by_sstar_type"] = sub
    # 5. components (R-A, R-B, R-C) if available
    C = sorted((ROOT / "results/components/raw").glob("*.json"))
    if C:
        res["components"] = components(C)
    (OUT / "summary_revision.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def components(files):
    M = ["lr", "svm", "knn", "rf", "xgb", "e1", "e2", "e3", "s_star"]
    rows, mc, tr = [], [], []
    for p in files:
        f = json.loads(p.read_text())
        for r in f["patients"]:
            for m in M:
                q = r[m]["marg"]
                rows.append({"dataset": f["dataset"], "fold": f["fold_id"], "row": r["row"], "model": m, "G": q["G"], "ndf": q["ndf"],
                             "knn": r[m]["knn"]["ndf"], "K": q["K"],
                             "Dexpl_K": q["D_expl"][-1] if q["D_expl"] else np.nan, "Drand_K": q["D_rand"][-1] if q["D_rand"] else np.nan,
                             "Dexpl_1": q["D_expl"][0] if q["D_expl"] else np.nan})
                if "tree_marg" in r[m]:
                    tr.append({"dataset": f["dataset"], "model": m, "perm": q["ndf"], "tree": r[m]["tree_marg"]["ndf"], "rho": r[m]["tree_rho"], "G": q["G"]})
            if "mc" in r:
                for m in ["e3", "s_star"]:
                    a, b = np.array(r["mc"][m]["ndf_by_shap_seed"], float), np.array(r["mc"][m]["ndf_by_order_seed"], float)
                    mc.append({"dataset": f["dataset"], "model": m, "G": r[m]["marg"]["G"], "sd_shap": np.nanstd(a, ddof=1), "sd_order": np.nanstd(b, ddof=1)})
    A = pd.DataFrame(rows); A.to_csv(OUT / "component_patients.csv", index=False)
    out = {"per_model_median_ndf": {}, "pairs": {}, "pooled_pairs": {}}
    # pairwise differences vs LR and E3 vs RF/XGB, per dataset fold means, gated on both G
    W = A.pivot_table(index=["dataset", "fold", "row"], columns="model", values=["ndf", "G", "knn"])
    def pair(a, b, col="ndf"):
        ok = (W["G"][a] >= G0) & (W["G"][b] >= G0) & np.isfinite(W[col][a]) & np.isfinite(W[col][b])
        dd = (W[col][a] - W[col][b])[ok].reset_index()
        per = {d: ana.corrected_t(dd[dd.dataset == d].groupby("fold")[0].mean().values) for d in DS if (dd.dataset == d).sum() > 0}
        good = [d for d in per if per[d]["n_folds"] >= 3 and np.isfinite(per[d]["mean_diff"])]
        return per, meta_full([per[d]["mean_diff"] for d in good], [s23.se_of(per[d]) for d in good])
    for a, b in [("e3", "rf"), ("e3", "xgb"), ("e3", "lr"), ("e3", "s_star"), ("rf", "lr"), ("xgb", "lr"), ("svm", "lr"), ("knn", "lr"),
                 ("e1", "lr"), ("e2", "lr"), ("e1", "s_star"), ("e2", "s_star")]:
        per, pooled = pair(a, b)
        out["pairs"][f"{a}-{b}"] = {d: per[d]["mean_diff"] for d in per}; out["pooled_pairs"][f"{a}-{b}"] = pooled
        _, pk = pair(a, b, "knn"); out["pooled_pairs"][f"{a}-{b}_knn"] = pk
    inc = A[A.G >= G0]
    out["per_model_median_ndf"] = inc.groupby("model").ndf.median().to_dict()
    out["per_model_mean_DexplK"] = inc.groupby("model").Dexpl_K.mean().to_dict()
    out["per_model_mean_Dexpl1"] = inc.groupby("model").Dexpl_1.mean().to_dict()
    out["per_model_mean_DrandK"] = inc.groupby("model").Drand_K.mean().to_dict()
    out["per_dataset_model_median"] = {d: inc[inc.dataset == d].groupby("model").ndf.median().to_dict() for d in DS}
    if tr:
        Tq = pd.DataFrame(tr); Tq = Tq[(Tq.G >= G0) & np.isfinite(Tq.perm) & np.isfinite(Tq.tree)]
        out["treeshap"] = {m: {"median_rho": float(g.rho.median()), "mean_abs_ndf_diff": float((g.perm - g.tree).abs().mean()),
                               "mean_perm": float(g.perm.mean()), "mean_tree": float(g.tree.mean()), "n": len(g)} for m, g in Tq.groupby("model")}
    if mc:
        Mq = pd.DataFrame(mc); Mq = Mq[Mq.G >= G0]
        out["mc_error"] = {m: {"median_sd_shap_seed": float(g.sd_shap.median()), "median_sd_order_seed": float(g.sd_order.median()), "n": len(g)} for m, g in Mq.groupby("model")}
    return out


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ["exclusions_total", "tuned_by_sstar_type"]}, indent=1, default=float))
    for k in ["H1", "H2", "H2_knn"]:
        m = r["meta"][k]; print(k, round(m["pooled"], 4), "DL", [round(m["ci_low"], 3), round(m["ci_high"], 3)], "HKSJ", np.round(m["hksj_ci"], 3), round(m["hksj_p"], 4), "PI", np.round(m["pred_int"], 3))
        print("   LOO", {d: round(v["pooled"], 3) for d, v in m["loo"].items()}, "max p", round(max(v["p"] for v in m["loo"].values()), 3))
    sh = r["shrinkage_similarG"]; print("shrinkage", sh["n_patients"], {k: round(sh["pooled"][k], 4) for k in ["pooled", "p"]}, np.round(sh["pooled"]["hksj_ci"], 3))
    print("complete-case", r["complete_case"])
