"""Step 5: statistical analysis of the confirmatory run (docs/PROTOCOL_FREEZE.md).

Reads results/confirmatory/raw/*.json and writes tables, figures and summary.json to
results/confirmatory/. Primary family: H1 and H2 x 3 datasets (Holm). Everything else is
labelled secondary/exploratory or descriptive.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xhe import stats  # noqa: E402

RAW = ROOT / "results" / "confirmatory" / "raw"
OUT = ROOT / "results" / "confirmatory"
DS = ["wdbc", "dr_debrecen", "cleveland"]
DS_LABEL = {"wdbc": "WDBC", "dr_debrecen": "DR Debrecen", "cleveland": "Cleveland"}
MODELS = ["lr", "svm", "knn", "rf", "xgb", "e1", "e2", "e3", "s_star"]
MLABEL = {"lr": "LR", "svm": "SVM", "knn": "KNN", "rf": "RF", "xgb": "XGB", "e1": "E1", "e2": "E2",
          "e3": "E3", "s_star": "S*"}
GMIN = 0.05
RATIO = 0.25  # n_test / n_train for 5-fold CV


def corrected_t(d):
    """Nadeau-Bengio corrected t on per-fold differences (n folds; n may be < 25 if a fold is dropped)."""
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    n = len(d); mean = float(d.mean()); var = float(d.var(ddof=1))
    se = np.sqrt((1 / n + RATIO) * var); df = n - 1
    t = mean / se if se > 0 else 0.0
    p = float(2 * st.t.sf(abs(t), df)); q = st.t.ppf(0.975, df)
    return {"n_folds": n, "mean_diff": mean, "ci_low": mean - q * se, "ci_high": mean + q * se,
            "t": float(t), "df": df, "p": p}


def load():
    folds = [json.loads(p.read_text()) for p in sorted(RAW.glob("*_fold*.json"))]
    sec = [json.loads(p.read_text()) for p in sorted(RAW.glob("*_r1fold*.json"))]
    return folds, sec


def main():
    folds, sec = load()
    summary = {"n_fold_files": len(folds), "n_secondary_files": len(sec)}
    # ---------------------------------------------------------- predictive table
    rows = []
    for f in folds:
        for m, v in f["predictive"].items():
            rows.append({"dataset": f["dataset"], "fold": f["fold_id"], "model": m, **v})
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictive_per_fold.csv", index=False)
    desc = pred.groupby(["dataset", "model"]).agg(["mean", "std"]).drop(columns=["fold"], level=0)
    desc.to_csv(OUT / "predictive_summary.csv")

    # S* selection, fallback, equivalence
    sel = pd.DataFrame([{"dataset": f["dataset"], "fold": f["fold_id"], "s_star": f["s_star"],
                         "fallback": bool(f["e3_fallback_reasons"]), "corr_qA_qB": f["corr_qA_qB"],
                         "vA": f["meta3_coef"][0], "vB": f["meta3_coef"][1]} for f in folds])
    summary["s_star_counts"] = {d: sel[sel.dataset == d].s_star.value_counts().to_dict() for d in DS}
    summary["e3_fallback_triggers"] = {d: int(sel[sel.dataset == d].fallback.sum()) for d in DS}
    summary["meta3_weights_mean"] = {d: {"vA": float(sel[sel.dataset == d].vA.mean()),
                                         "vB": float(sel[sel.dataset == d].vB.mean()),
                                         "corr_qA_qB": float(sel[sel.dataset == d].corr_qA_qB.mean())} for d in DS}
    summary["equivalence_check"] = {f["dataset"]: f["equivalence_max_abs_diff"] for f in folds if "equivalence_max_abs_diff" in f}

    # ---------------------------------------------------------- H1
    prim = {}
    h1 = {}
    for d in DS:
        p = pred[pred.dataset == d].pivot(index="fold", columns="model", values="auc")
        h1[d] = corrected_t(p["e3"] - p["s_star"])
        h1[d].update({"auc_e3_mean": float(p["e3"].mean()), "auc_s_star_mean": float(p["s_star"].mean())})
        prim[f"H1_{d}"] = h1[d]["p"]

    # ---------------------------------------------------------- H2 (fold level, primary)
    h2rows = []
    for f in folds:
        for r in f["h2_patients"]:
            h2rows.append({"dataset": f["dataset"], "fold": f["fold_id"], "row": r["row"], "y": r["y"],
                           **{f"{m}_{k}": r[m][k] for m in ["e3", "s_star", "e3_knn", "s_star_knn"] for k in ["ndf", "G", "K"]}})
    h2p = pd.DataFrame(h2rows)
    for op, suffix in [("marginal", ""), ("knn", "_knn")]:
        ok = (h2p["e3_G"] >= GMIN) & (h2p["s_star_G"] >= GMIN) & h2p[f"e3{suffix}_ndf"].notna() & h2p[f"s_star{suffix}_ndf"].notna()
        h2p[f"incl{suffix}"] = ok
        h2p[f"dndf{suffix}"] = np.where(ok, h2p[f"e3{suffix}_ndf"] - h2p[f"s_star{suffix}_ndf"], np.nan)
    h2p.to_csv(OUT / "h2_patients_fold_level.csv", index=False)
    h2 = {}
    h2_sens = {}
    for d in DS:
        sub = h2p[h2p.dataset == d]
        fm = sub.groupby("fold")["dndf"].mean().dropna()
        h2[d] = corrected_t(fm.values)
        h2[d].update({"patients_included": int(sub.incl.sum()), "patients_sampled": int(len(sub)),
                      "folds_dropped": int(25 - len(fm)),
                      "median_ndf_e3": float(sub.loc[sub.incl, "e3_ndf"].median()),
                      "median_ndf_s_star": float(sub.loc[sub.incl, "s_star_ndf"].median())})
        prim[f"H2_{d}"] = h2[d]["p"]
        # exploratory sensitivity: fold medians; k-NN operator
        fmed = sub.groupby("fold")["dndf"].median().dropna()
        fknn = sub.groupby("fold")["dndf_knn"].mean().dropna()
        h2_sens[d] = {"fold_median": corrected_t(fmed.values), "knn_operator": corrected_t(fknn.values)}
    hol = stats.holm(prim)
    summary["primary"] = {"H1": h1, "H2": h2, "holm": hol}
    summary["h2_sensitivity_exploratory"] = h2_sens

    # ---------------------------------------------------------- secondary predictive (Holm within dataset)
    secpred = {}
    for d in DS:
        p = pred[pred.dataset == d].pivot(index="fold", columns="model", values="auc")
        b = pred[pred.dataset == d].pivot(index="fold", columns="model", values="brier")
        tests = {"E1-S*": corrected_t(p["e1"] - p["s_star"]), "E2-S*": corrected_t(p["e2"] - p["s_star"]),
                 "E3-E2": corrected_t(p["e3"] - p["e2"])}
        ho = stats.holm({k: v["p"] for k, v in tests.items()})
        for k in tests:
            tests[k]["p_holm"] = ho[k]["p_holm"]
        tests["Brier E3-S* (descriptive)"] = corrected_t(b["e3"] - b["s_star"])
        secpred[d] = tests
    summary["secondary_predictive"] = secpred

    # ---------------------------------------------------------- Stage C (repeat-1, 150 patients)
    crow = []
    for s in sec:
        for p in s["patients"]:
            base = {"dataset": s["dataset"], "fold": s["fold_id"], "row": p["row"], "y": p["y"], "s_star_name": p["s_star"]}
            for m in MODELS:
                r = p[m]
                crow.append({**base, "model": m, **{k: v for k, v in r.items() if k != "shap"}})
    C = pd.DataFrame(crow)
    C.to_csv(OUT / "secondary_patients.csv", index=False)
    secx = {}
    for d in DS:
        sub = C[C.dataset == d]
        out = {"n_patients": int(sub.row.nunique())}
        # NDF per model (included G>=0.05)
        ndfm = {}
        for m in MODELS:
            q = sub[(sub.model == m) & (sub.G >= GMIN) & sub.ndf.notna()]
            ndfm[m] = {"n": int(len(q)), "median": float(q.ndf.median()), "q1": float(q.ndf.quantile(.25)),
                       "q3": float(q.ndf.quantile(.75)), "share_pos": float((q.ndf > 0).mean())}
        out["ndf_by_model"] = ndfm
        piv = sub.pivot(index="row", columns="model", values="ndf")
        gpiv = sub.pivot(index="row", columns="model", values="G")
        foldof = sub.groupby("row").fold.first()
        tests = {}
        for a in ["e3", "e1", "e2"]:
            ok = (gpiv[a] >= GMIN) & (gpiv["s_star"] >= GMIN) & piv[a].notna() & piv["s_star"].notna()
            dd = (piv[a] - piv["s_star"])[ok]
            w = stats.wilcoxon_paired(dd.values)
            w["ci_median"] = stats.stratified_bootstrap_median_ci(dd.values, foldof[ok].values, seed=1)
            w["per_fold_means"] = stats.per_fold_means(dd.values, foldof[ok].values)
            tests[f"{a.upper()}-S*"] = w
        ho = stats.holm({k: v["p"] for k, v in tests.items() if k != "E3-S*"})
        for k in ho:
            tests[k]["p_holm"] = ho[k]["p_holm"]
        out["patient_level_ndf_tests"] = tests
        # robustness
        rob = {}
        for m in ["e3", "s_star"]:
            q = sub[sub.model == m]
            rob[m] = {"rb_0.05": float(q["rb_0.05"].mean()), "flip_0.05": float(q["flip_0.05"].mean()),
                      "dfx_0.05": float(q["dfx_0.05"].mean()), "noise_floor": float(q["noise_floor"].mean())}
            if m in ("e3", "s_star"):
                for s in ["0.02", "0.1"]:
                    rob[m][f"rb_{s}"] = float(q[f"rb_{s}"].mean())
        rpiv = sub.pivot(index="row", columns="model", values="rb_0.05")
        rt = {}
        for a in ["e3"]:
            dd = (rpiv[a] - rpiv["s_star"]).dropna()
            rt[f"{a.upper()}-S*"] = stats.wilcoxon_paired(dd.values) if (dd != 0).any() else {"n": int(len(dd)), "median_diff": 0.0, "p": 1.0}
        ho = stats.holm({k: v["p"] for k, v in rt.items()})
        for k in rt:
            rt[k]["p_holm"] = ho[k]["p_holm"]
            rt[k]["mean_diff"] = float((rpiv[k.split("-")[0].lower()] - rpiv["s_star"]).mean())
        out["robustness"] = rob
        out["robustness_tests"] = rt
        # LIME diagnostics
        out["lime"] = {m: {"top5_overlap": float(sub[sub.model == m].lime_top5_overlap.mean()),
                           "sign_agree": float(sub[sub.model == m].lime_sign_agree.mean()),
                           "local_error": float(sub[sub.model == m].lime_local_error.mean())}
                       for m in ["e3", "s_star"] if "lime_top5_overlap" in sub}
        secx[d] = out
    summary["secondary_explanation"] = secx
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    figures(pred, h2p, C, summary)
    tables_md(pred, summary)
    print(json.dumps(summary["primary"], indent=1, default=float))


def figures(pred, h2p, C, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
    figd = OUT / "figures"; figd.mkdir(exist_ok=True)
    # Fig A: AUC per model per dataset
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, d in zip(axes, DS):
        p = pred[pred.dataset == d]
        data = [p[p.model == m].auc.values for m in MODELS]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="#222"))
        for k, b in enumerate(bp["boxes"]):
            col = C2 if MODELS[k] == "e3" else (C3 if MODELS[k] == "s_star" else "#c9d3d6")
            b.set_facecolor(col); b.set_alpha(0.7); b.set_edgecolor("#555")
        ax.set_xticks(range(1, len(MODELS) + 1), [MLABEL[m] for m in MODELS], rotation=0, fontsize=9)
        ax.set_title(DS_LABEL[d]); ax.grid(axis="y", color="#e5e5e5")
    axes[0].set_ylabel("ROC-AUC (25 outer folds)")
    fig.tight_layout(); fig.savefig(figd / "fig_auc_by_model.png", dpi=220); plt.close(fig)
    # Fig B: primary effects forest plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, key, lab, thr in [(axes[0], "H1", "ΔAUC (E3 − S*)", 0.02), (axes[1], "H2", "ΔNDF (E3 − S*), fold means", 0.05)]:
        for k, d in enumerate(DS):
            r = summary["primary"][key][d]
            ax.errorbar(r["mean_diff"], k, xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
                        fmt="o", color=C1, ms=7, capsize=4, lw=2)
        ax.axvline(0, color="#444", lw=1)
        for s in (-thr, thr):
            ax.axvline(s, color="#999", lw=1, ls=":")
        ax.set_yticks(range(3), [DS_LABEL[d] for d in DS]); ax.invert_yaxis()
        ax.set_xlabel(lab + " with 95% CI"); ax.grid(axis="x", color="#eee")
    fig.tight_layout(); fig.savefig(figd / "fig_primary_effects.png", dpi=220); plt.close(fig)
    # Fig C: NDF by model (stage C)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, d in zip(axes, DS):
        sub = C[(C.dataset == d) & (C.G >= GMIN)]
        data = [sub[sub.model == m].ndf.dropna().values for m in MODELS]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="#222"))
        for k, b in enumerate(bp["boxes"]):
            col = C2 if MODELS[k] == "e3" else (C3 if MODELS[k] == "s_star" else "#c9d3d6")
            b.set_facecolor(col); b.set_alpha(0.7); b.set_edgecolor("#555")
        ax.axhline(0, color="#666", ls=":", lw=1)
        ax.set_xticks(range(1, len(MODELS) + 1), [MLABEL[m] for m in MODELS], fontsize=9)
        ax.set_title(DS_LABEL[d]); ax.grid(axis="y", color="#e5e5e5")
    axes[0].set_ylabel("NDF (patients with G ≥ 0.05)")
    fig.tight_layout(); fig.savefig(figd / "fig_ndf_by_model.png", dpi=220); plt.close(fig)
    # Fig D: robustness
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.2
    rm = ["e3", "s_star"]
    for k, m in enumerate(rm):
        vals = [summary["secondary_explanation"][d]["robustness"][m]["rb_0.05"] for d in DS]
        nf = [summary["secondary_explanation"][d]["robustness"][m]["noise_floor"] for d in DS]
        x = np.arange(3) + (k - 0.5) * w
        ax.bar(x, vals, w * 0.9, color=[C2, C3][k], label=MLABEL[m])
        ax.scatter(x, nf, marker="_", s=120, color="#222", zorder=3)
    ax.set_xticks(range(3), [DS_LABEL[d] for d in DS]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean top-5 overlap (RB, 0.05×SD)"); ax.legend(frameon=False, ncol=4, fontsize=9, loc="lower center")
    ax.grid(axis="y", color="#eee")
    fig.tight_layout(); fig.savefig(figd / "fig_robustness.png", dpi=220); plt.close(fig)


def tables_md(pred, summary):
    L = ["# Confirmatory results", ""]
    L += ["## Predictive performance (mean ± SD over 25 outer folds)", ""]
    for d in DS:
        L += [f"### {DS_LABEL[d]}", "", "| Model | AUC | Brier | Accuracy | F1 | Recall | Specificity |", "|---|---|---|---|---|---|---|"]
        p = pred[pred.dataset == d]
        for m in MODELS:
            q = p[p.model == m]
            L.append(f"| {MLABEL[m]} | " + " | ".join(f"{q[c].mean():.3f} ± {q[c].std():.3f}" for c in ["auc", "brier", "accuracy", "f1", "recall", "specificity"]) + " |")
        L.append("")
    L += ["## Primary tests", "", "| Test | Dataset | Mean diff | 95% CI | p | p (Holm) |", "|---|---|---|---|---|---|"]
    for key in ["H1", "H2"]:
        for d in DS:
            r = summary["primary"][key][d]; h = summary["primary"]["holm"][f"{key}_{d}"]
            L.append(f"| {key} | {DS_LABEL[d]} | {r['mean_diff']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['p']:.4f} | {h['p_holm']:.4f} |")
    (OUT / "results_tables.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
