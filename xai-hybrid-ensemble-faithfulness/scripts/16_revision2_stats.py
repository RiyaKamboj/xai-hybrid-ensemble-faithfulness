"""Second-round revision analyses (post hoc; docs/PROTOCOL_STUDY2_STUDY3.md 'Revision round 2').
(a) G-threshold sensitivity (0.02, 0.10) of the primary H2 and the conditional variant;
(b) tuned results by S* type pooled with a dataset random intercept (MixedLM) instead of treating folds as independent;
(c) top-1 vs random-1 and top-K vs random-K deletion shares from the component run;
(d) bookkeeping of tuned folds with no included patient.
Writes results/revision/summary_revision2.json"""
import importlib.util, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
np.seterr(all="ignore")
ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("rv", ROOT / "scripts" / "15_revision_stats.py"); rv = importlib.util.module_from_spec(s); s.loader.exec_module(rv)
DS = rv.DS
out = {}
# (a) thresholds
P = rv.primary_patients()
out["threshold"] = {}
for thr in [0.02, 0.05, 0.10]:
    inc = (P.G_e3 >= thr) & (P.G_s >= thr) & np.isfinite(P.ndf_e3) & np.isfinite(P.ndf_s)
    P["dt"] = np.where(inc, P.ndf_e3 - P.ndf_s, np.nan)
    inck = inc & np.isfinite(P.knn_e3) & np.isfinite(P.knn_s)
    P["dk"] = np.where(inck, P.knn_e3 - P.knn_s, np.nan)
    res = {"included": int(inc.sum()), "included_by_dataset": {d: int(inc[P.dataset == d].sum()) for d in DS}}
    for col, lab in [("dt", "marg"), ("dk", "knn")]:
        per = rv.per_dataset_effect(P, col)
        m = rv.meta_full([per[d]["mean_diff"] for d in DS], [rv.s23.se_of(per[d]) for d in DS])
        res[lab] = {"per_dataset": {d: {k: per[d][k] for k in ["mean_diff", "p"]} for d in DS},
                    "pooled": {k: m[k] for k in ["pooled", "ci_low", "ci_high", "p", "I2"]}, "hksj_ci": m["hksj_ci"], "hksj_p": m["hksj_p"]}
    out["threshold"][str(thr)] = res
# (b) tuned by S* type, mixed model
import statsmodels.formula.api as smf
T = pd.read_csv(ROOT / "results/revision/tuned_patients.csv")
Fm = T.groupby(["dataset", "fold", "nonlin"]).agg(d=("d", "mean"), dk=("dk", "mean")).reset_index()
mix = {}
for nl, lab in [(False, "linear"), (True, "nonlinear")]:
    g = Fm[Fm.nonlin == nl]
    for c in ["d", "dk"]:
        if g.dataset.nunique() > 1:
            m = smf.mixedlm(f"{c} ~ 1", g, groups=g["dataset"]).fit(reml=True)
            est, se, p = float(m.params["Intercept"]), float(m.bse["Intercept"]), float(m.pvalues["Intercept"])
        else:
            est, se, p = float(g[c].mean()), float("nan"), float("nan")
        mix[f"{lab}_{c}"] = {"est": est, "ci": [est - 1.96 * se, est + 1.96 * se], "p": p, "n_folds": len(g), "n_datasets": int(g.dataset.nunique())}
all_tuned = []
for p in sorted((ROOT / "results/study7/raw").glob("*.json")):
    f = json.loads(p.read_text())
    inc = sum(1 for r in f["h2_patients"] if r["e3"]["G"] >= 0.05 and r["s_star"]["G"] >= 0.05 and np.isfinite(r["e3"]["ndf"]) and np.isfinite(r["s_star"]["ndf"]))
    all_tuned.append({"dataset": f["dataset"], "fold": f["fold_id"], "s_star": f["s_star"], "included": inc})
A = pd.DataFrame(all_tuned)
mix["folds_total"] = len(A); mix["folds_nonlin_total"] = int((A.s_star != "lr").sum())
mix["folds_without_included_patients"] = A[A.included == 0][["dataset", "fold", "s_star"]].to_dict("records")
out["tuned_mixed"] = mix
# (c) deletion shares
C = pd.read_csv(ROOT / "results/revision/component_patients.csv")
rows = []
for p in sorted((ROOT / "results/components/raw").glob("*.json")):
    f = json.loads(p.read_text())
    for r in f["patients"]:
        for m in ["e3", "s_star", "lr", "xgb"]:
            q = r[m]["marg"]
            if q["G"] >= 0.05 and q["D_expl"]:
                rows.append({"model": m, "top1": q["D_expl"][0], "rand1": q["D_rand"][0], "topK": q["D_expl"][-1], "randK": q["D_rand"][-1], "K": q["K"],
                             "over100_topK": q["D_expl"][-1] > 1.0})
D = pd.DataFrame(rows)
out["deletion_shares"] = {m: {k: float(g[k].mean()) for k in ["top1", "rand1", "topK", "randK", "over100_topK"]} | {"n": len(g)} for m, g in D.groupby("model")}
(ROOT / "results/revision/summary_revision2.json").write_text(json.dumps(out, indent=1, default=float))
for t, r in out["threshold"].items():
    print("G>=", t, r["included"], {k: round(v, 4) for k, v in r["marg"]["pooled"].items()}, "HKSJ", np.round(r["marg"]["hksj_ci"], 3), "| knn", round(r["knn"]["pooled"]["pooled"], 3), round(r["knn"]["pooled"]["p"], 3))
    print("    per-dataset p<.05:", [d for d in DS if r["marg"]["per_dataset"][d]["p"] < 0.05], {d: r["included_by_dataset"][d] for d in ["ilpd", "cdc_diabetes", "parkinsons"]})
print(json.dumps(out["tuned_mixed"], indent=0, default=float)); print(json.dumps(out["deletion_shares"], indent=0))

# (e) round 3: all tuned folds with the same random-intercept model, subgroup difference with random intercept
g = Fm.copy()
allmix = {}
for c in ["d", "dk"]:
    m = smf.mixedlm(f"{c} ~ 1", g, groups=g["dataset"]).fit(reml=True)
    est, se = float(m.params["Intercept"]), float(m.bse["Intercept"])
    allmix[c] = {"est": est, "ci": [est - 1.96 * se, est + 1.96 * se], "p": float(m.pvalues["Intercept"]), "n_folds": len(g)}
m = smf.mixedlm("d ~ nonlin", g.assign(nonlin=g.nonlin.astype(int)), groups=g["dataset"]).fit(reml=True)
allmix["diff_nonlin"] = {"coef": float(m.params["nonlin"]), "se": float(m.bse["nonlin"]), "p": float(m.pvalues["nonlin"])}
out["tuned_mixed_all"] = allmix
(ROOT / "results/revision/summary_revision2.json").write_text(json.dumps(out, indent=1, default=float))
print("ALL", json.dumps(allmix, default=float))
