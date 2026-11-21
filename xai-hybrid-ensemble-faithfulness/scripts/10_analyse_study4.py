"""Analysis of Study 4 (exploratory): remedy (group attribution), KernelSHAP, Bhatt faithfulness correlation."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, ROOT / "scripts" / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ana = _load("ana", "05_analyse.py"); s23 = _load("s23", "08_analyse_study2_3.py")
RAW = ROOT / "results" / "study4" / "raw"; OUT = ROOT / "results" / "study4"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
ALL = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
METRICS = {"a_ndf": "Permutation SHAP, feature NDF (marginal)", "b_ndf": "KernelSHAP, feature NDF (marginal)",
           "c_ndf": "Group Shapley, group NDF (marginal)", "asum_ndf_g": "Permutation SHAP summed, group NDF (marginal)",
           "a_ndf_knn": "Permutation SHAP, feature NDF (k-NN)", "b_ndf_knn": "KernelSHAP, feature NDF (k-NN)",
           "c_ndf_knn": "Group Shapley, group NDF (k-NN)", "a_bhatt": "Permutation SHAP, faithfulness correlation",
           "b_bhatt": "KernelSHAP, faithfulness correlation"}


def rows():
    out = []
    for p in sorted(RAW.glob("*.json")):
        d = json.loads(p.read_text())
        for r in d["patients"]:
            ok = r["e3"]["a_G"] >= 0.05 and r["s_star"]["a_G"] >= 0.05
            rec = {"dataset": d["dataset"], "fold": d["fold_id"], "row": r["row"], "included": ok, "n_groups": d["n_groups"], "d": d["d"],
                   "s_star": d["s_star"], "rank_corr_ab_e3": r["e3"]["rank_corr_ab"], "rank_corr_ab_s": r["s_star"]["rank_corr_ab"]}
            for k in METRICS:
                a, b = r["e3"][k], r["s_star"][k]
                gate = ok if "bhatt" not in k else True       # Bhatt needs no gap gate
                rec[k + "_e3"], rec[k + "_s"] = a, b
                rec["d_" + k] = a - b if gate and np.isfinite(a) and np.isfinite(b) else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def main():
    P = rows(); P.to_csv(OUT / "study4_patients.csv", index=False)
    res = {"per_dataset": {}, "pooled": {}, "contrast": {}}
    for d in ALL:
        g = P[P.dataset == d]
        if g.empty:
            continue
        fm = g.groupby("fold").mean(numeric_only=True)
        res["per_dataset"][d] = {"n_groups": int(g.n_groups.iloc[0]), "d": int(g.d.iloc[0]), "n_folds": int(g.fold.nunique()),
                                 "included": int(g.included.sum()), "sampled": len(g),
                                 **{k: ana.corrected_t(fm["d_" + k].values) for k in METRICS},
                                 "fold_median_c_minus_a": ana.corrected_t((g.groupby("fold")["d_c_ndf"].median() - g.groupby("fold")["d_a_ndf"].median()).values),
                                 "median_rank_corr_kernel_vs_perm": float(np.nanmedian(np.r_[g.rank_corr_ab_e3, g.rank_corr_ab_s]))}
        # remedy contrast: does group attribution shrink the deficit?  (ΔNDF_c - ΔNDF_a), and attribution-only part
        res["contrast"][d] = {"c_minus_a": ana.corrected_t((fm["d_c_ndf"] - fm["d_a_ndf"]).values),
                              "c_minus_asum": ana.corrected_t((fm["d_c_ndf"] - fm["d_asum_ndf_g"]).values),
                              "asum_minus_a": ana.corrected_t((fm["d_asum_ndf_g"] - fm["d_a_ndf"]).values)}
    names = list(res["per_dataset"])
    for k in METRICS:
        R = [res["per_dataset"][d][k] for d in names]
        if all(np.isfinite(r["mean_diff"]) and r["ci_high"] > r["mean_diff"] for r in R):
            res["pooled"][k] = s23.dersimonian_laird([r["mean_diff"] for r in R], [s23.se_of(r) for r in R])
    for c in ["c_minus_a", "c_minus_asum", "asum_minus_a"]:
        R = [res["contrast"][d][c] for d in names]
        res["pooled"]["contrast_" + c] = s23.dersimonian_laird([r["mean_diff"] for r in R], [s23.se_of(r) for r in R])
    (OUT / "summary_study4.json").write_text(json.dumps(res, indent=1, default=float))
    figure(res, names)
    for k in ["a_ndf", "b_ndf", "c_ndf", "asum_ndf_g", "a_ndf_knn", "c_ndf_knn", "a_bhatt", "b_bhatt"]:
        print(f"{k:12s}", " ".join(f"{d[:6]}:{res['per_dataset'][d][k]['mean_diff']:+.3f}(p{res['per_dataset'][d][k]['p']:.2f})" for d in names),
              "| pooled", {x: round(res['pooled'][k][x], 4) for x in ['pooled', 'ci_low', 'ci_high', 'p', 'I2']} if k in res["pooled"] else "")
    for c in ["c_minus_a", "c_minus_asum", "asum_minus_a"]:
        print(c, {x: round(res['pooled']['contrast_' + c][x], 4) for x in ['pooled', 'ci_low', 'ci_high', 'p', 'I2']})


def figure(res, names):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    show = [("a_ndf", "Feature-level permutation SHAP", "#eb6834"), ("b_ndf", "Feature-level KernelSHAP", "#b04a8f"),
            ("c_ndf", "Correlation-group Shapley", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(8, 4.8)); off = {"a_ndf": -0.22, "b_ndf": 0.0, "c_ndf": 0.22}
    for k, lab, col in show:
        for i, d in enumerate(names):
            r = res["per_dataset"][d][k]
            ax.errorbar(r["mean_diff"], i + off[k], xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]], fmt="o", color=col, ms=5, capsize=2, lw=1.4, label=lab if i == 0 else None)
        m = res["pooled"].get(k)
        if m:
            y = len(names) + 0.5 + off[k]
            ax.errorbar(m["pooled"], y, xerr=[[m["pooled"] - m["ci_low"]], [m["ci_high"] - m["pooled"]]], fmt="D", color=col, ms=7, capsize=3, lw=2)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(list(range(len(names))) + [len(names) + 0.5], [s23.LAB[d] for d in names] + ["Pooled (random effects)"]); ax.invert_yaxis()
    ax.set_xlabel("ΔNDF (E3 − S*), marginal removal, 95% CI"); ax.legend(frameon=False, fontsize=9, loc="lower left"); ax.grid(axis="x", color="#eee")
    fig.tight_layout(); fig.savefig(FIG / "fig_study4_remedy.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
