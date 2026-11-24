"""Supervisor-review analyses (24 Nov 2026; post hoc re-analysis of saved outputs, no model refitting).
(a) Equivalence of AUC with the pre-specified margin +/-0.02 (two one-sided tests, TOST), per dataset and pooled.
(b) Exclusion-free check: absolute deletion faithfulness ADF = NDF x G (probability units), which needs no gap
    threshold; computed for every sampled patient whose NDF is defined for both models.
(c) Per-dataset exclusion table.
Writes results/revision/summary_supervisor.json"""
import importlib.util, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st
np.seterr(all="ignore")
ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("rv", ROOT / "scripts" / "15_revision_stats.py"); rv = importlib.util.module_from_spec(s); s.loader.exec_module(rv)
DS, s23, ana = rv.DS, rv.s23, rv.ana
E = json.load(open(ROOT / "results/study2_3/summary_study2_3.json"))
allds = {**E["study1_recomputed"], **E["study2"]}
MARGIN = 0.02
out = {"margin": MARGIN, "auc_tost": {}}
# (a) TOST per dataset: corrected-t SE recovered from the 95% CI
for d in DS:
    r = allds[d]["H1"]; se = s23.se_of(r); df = r["df"]; m = r["mean_diff"]
    t90 = st.t.ppf(0.95, df)
    p_low = st.t.sf((m + MARGIN) / se, df)     # H0: diff <= -margin
    p_up = st.t.cdf((m - MARGIN) / se, df)     # H0: diff >= +margin
    out["auc_tost"][d] = {"mean": m, "ci90": [m - t90 * se, m + t90 * se], "p_tost": float(max(p_low, p_up)),
                          "equivalent": bool(max(p_low, p_up) < 0.05)}
V = json.load(open(ROOT / "results/revision/summary_revision.json"))
M = V["meta"]["H1"]; est = [allds[d]["H1"]["mean_diff"] for d in DS]; se = np.array([s23.se_of(allds[d]["H1"]) for d in DS])
w = 1 / (se ** 2 + M["tau2"]); mu = M["pooled"]; sdl = np.sqrt(1 / w.sum())
q = np.sum(w * (np.array(est) - mu) ** 2) / (len(DS) - 1); shk = np.sqrt(q / w.sum())
z90 = st.norm.ppf(0.95); t90 = st.t.ppf(0.95, len(DS) - 1)
out["auc_tost_pooled"] = {"mean": mu, "dl_ci90": [mu - z90 * sdl, mu + z90 * sdl], "hksj_ci90": [mu - t90 * shk, mu + t90 * shk],
                          "dl_p_tost": float(max(st.norm.sf((mu + MARGIN) / sdl), st.norm.cdf((mu - MARGIN) / sdl))),
                          "hksj_p_tost": float(max(st.t.sf((mu + MARGIN) / shk, len(DS) - 1), st.t.cdf((mu - MARGIN) / shk, len(DS) - 1)))}
out["auc_equivalent_datasets"] = [d for d in DS if out["auc_tost"][d]["equivalent"]]
# (b) ADF without exclusions
P = rv.primary_patients()
P["defined"] = np.isfinite(P.ndf_e3) & np.isfinite(P.ndf_s)
P["adf_d"] = np.where(P.defined, P.ndf_e3 * P.G_e3 - P.ndf_s * P.G_s, np.nan)
P["adfk_d"] = np.where(P.defined & np.isfinite(P.knn_e3) & np.isfinite(P.knn_s), P.knn_e3 * P.G_e3 - P.knn_s * P.G_s, np.nan)
res = {}
for col, lab in [("adf_d", "marg"), ("adfk_d", "knn")]:
    per = rv.per_dataset_effect(P, col)
    m = rv.meta_full([per[d]["mean_diff"] for d in DS], [s23.se_of(per[d]) for d in DS])
    res[lab] = {"per_dataset": {d: {k: per[d][k] for k in ["mean_diff", "ci_low", "ci_high", "p"]} for d in DS},
                "pooled": {k: m[k] for k in ["pooled", "ci_low", "ci_high", "p", "I2"]}, "hksj_ci": m["hksj_ci"], "hksj_p": m["hksj_p"]}
res["n_defined"] = int(P.defined.sum()); res["n_sampled"] = len(P)
res["undefined_by_dataset"] = {d: int((~P.defined[P.dataset == d]).sum()) for d in DS}
# context: ADF of each model (median, probability units)
res["median_adf_e3"] = float(np.nanmedian(np.where(P.defined, P.ndf_e3 * P.G_e3, np.nan)))
res["median_adf_s"] = float(np.nanmedian(np.where(P.defined, P.ndf_s * P.G_s, np.nan)))
out["adf_all_patients"] = res
# (c) exclusion table
X = {}
for d in DS:
    g = P[P.dataset == d]
    X[d] = {"sampled": len(g), "excl_he": int((g.G_e3 < 0.05).sum()), "excl_s": int((g.G_s < 0.05).sum()),
            "excl_either": int(((g.G_e3 < 0.05) | (g.G_s < 0.05)).sum()), "included": int(g.inc.sum()),
            "pct_included": float(100 * g.inc.mean())}
out["exclusions"] = X
(ROOT / "results/revision/summary_supervisor.json").write_text(json.dumps(out, indent=1, default=float))
for d in DS:
    a = out["auc_tost"][d]; e = X[d]; b = res["marg"]["per_dataset"][d]
    print(f"{d:13s} dAUC {a['mean']:+.4f} 90%CI [{a['ci90'][0]:+.4f},{a['ci90'][1]:+.4f}] TOST p {a['p_tost']:.3f} eq={a['equivalent']} | incl {e['included']}/{e['sampled']} | ADF {b['mean_diff']:+.4f} p {b['p']:.3f}")
print("pooled AUC", out["auc_tost_pooled"])
print("ADF pooled", res["marg"]["pooled"], res["hksj_ci"] if False else res["marg"]["hksj_ci"], "| knn", res["knn"]["pooled"], res["knn"]["hksj_ci"])
print("defined", res["n_defined"], "/", res["n_sampled"], res["undefined_by_dataset"], res["median_adf_e3"], res["median_adf_s"])
