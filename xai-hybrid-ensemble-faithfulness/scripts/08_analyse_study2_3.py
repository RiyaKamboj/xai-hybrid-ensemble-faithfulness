"""Analysis of Study 2 (replication) and Study 3 (simulation); docs/PROTOCOL_STUDY2_STUDY3.md."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xhe import stats  # noqa: E402
spec = importlib.util.spec_from_file_location("ana", ROOT / "scripts" / "05_analyse.py")
ana = importlib.util.module_from_spec(spec); spec.loader.exec_module(ana)
OUT = ROOT / "results" / "study2_3"; OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
DS2 = ["ilpd", "parkinsons", "mammographic", "ctg"]
LAB = {"wdbc": "WDBC", "dr_debrecen": "DR Debrecen", "cleveland": "Cleveland", "ilpd": "ILPD (liver)",
       "parkinsons": "Parkinsons", "mammographic": "Mammographic mass", "ctg": "Cardiotocography", "cdc_diabetes": "CDC Diabetes (n = 5,000)"}
GMIN = 0.05


def fold_effects(raw_dir, names):
    F = [json.loads(p.read_text()) for p in sorted(Path(raw_dir).glob("*_fold*.json"))]
    res = {}
    for d in names:
        Fd = [f for f in F if f["dataset"] == d]
        if not Fd:
            continue
        auc = pd.DataFrame([{"fold": f["fold_id"], **{m: f["predictive"][m]["auc"] for m in f["predictive"]}} for f in Fd]).set_index("fold")
        rows = []
        for f in Fd:
            for r in f["h2_patients"]:
                ok = r["e3"]["G"] >= GMIN and r["s_star"]["G"] >= GMIN and np.isfinite(r["e3"]["ndf"]) and np.isfinite(r["s_star"]["ndf"])
                okk = ok and np.isfinite(r["e3_knn"]["ndf"]) and np.isfinite(r["s_star_knn"]["ndf"])
                rows.append({"fold": f["fold_id"], "dm": r["e3"]["ndf"] - r["s_star"]["ndf"] if ok else np.nan,
                             "dk": r["e3_knn"]["ndf"] - r["s_star_knn"]["ndf"] if okk else np.nan,
                             "e3": r["e3"]["ndf"] if ok else np.nan, "s": r["s_star"]["ndf"] if ok else np.nan})
        H = pd.DataFrame(rows)
        res[d] = {"n_folds": len(Fd), "s_star_counts": pd.Series([f["s_star"] for f in Fd]).value_counts().to_dict(),
                  "fallback": int(sum(bool(f["e3_fallback_reasons"]) for f in Fd)),
                  "auc_mean": auc.mean().round(4).to_dict(),
                  "H1": ana.corrected_t((auc["e3"] - auc["s_star"]).values),
                  "H2": ana.corrected_t(H.groupby("fold").dm.mean().dropna().values),
                  "H2_fold_median": ana.corrected_t(H.groupby("fold").dm.median().dropna().values),
                  "H2_knn": ana.corrected_t(H.groupby("fold").dk.mean().dropna().values),
                  "included": int(H.dm.notna().sum()), "sampled": int(len(H)),
                  "median_ndf_e3": float(H.e3.median()), "median_ndf_s_star": float(H.s.median())}
    return res


def se_of(r):
    q = st.t.ppf(0.975, r["df"])
    return (r["ci_high"] - r["mean_diff"]) / q


def dersimonian_laird(est, se):
    est, se = np.asarray(est), np.asarray(se); w = 1 / se ** 2
    fe = np.sum(w * est) / np.sum(w); Q = np.sum(w * (est - fe) ** 2); k = len(est)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (se ** 2 + tau2); mu = np.sum(ws * est) / np.sum(ws); sm = np.sqrt(1 / np.sum(ws))
    z = mu / sm
    return {"pooled": float(mu), "ci_low": float(mu - 1.96 * sm), "ci_high": float(mu + 1.96 * sm), "p": float(2 * st.norm.sf(abs(z))),
            "tau2": float(tau2), "Q": float(Q), "I2": float(max(0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, "k": k}


def study2():
    s2 = fold_effects(ROOT / "results" / "study2" / "raw", DS2)
    s1 = fold_effects(ROOT / "results" / "confirmatory" / "raw", ["wdbc", "dr_debrecen", "cleveland"])
    pv = {f"{h}_{d}": s2[d][h]["p"] for d in s2 for h in ["H1", "H2"]}
    hol = stats.holm(pv)
    for d in s2:
        for h in ["H1", "H2"]:
            s2[d][h]["p_holm"] = hol[f"{h}_{d}"]["p_holm"]
    allds = {**s1, **s2}
    meta = {}
    for key in ["H1", "H2", "H2_knn"]:
        names = list(allds)
        meta[key] = dersimonian_laird([allds[d][key]["mean_diff"] for d in names], [se_of(allds[d][key]) for d in names])
    # Study 5 (CDC, large-sample replication): pooled over 8 datasets, reported next to the 7-dataset estimates
    s5 = fold_effects(ROOT / "results" / "study5" / "raw", ["cdc_diabetes"]) if (ROOT / "results" / "study5" / "raw").exists() else {}
    if s5:
        s2.update(s5); all8 = {**allds, **s5}
        for key in ["H1", "H2", "H2_knn"]:
            meta[key + "_8"] = dersimonian_laird([all8[d][key]["mean_diff"] for d in all8], [se_of(all8[d][key]) for d in all8])
    return s1, s2, meta


def study3():
    rows, aucs = [], []
    for p in sorted((ROOT / "results" / "simulation" / "raw").glob("*.json")):
        d = json.loads(p.read_text())
        for f in d["folds"]:
            aucs.append({"truth": d["truth"], "rho": d["rho"], "rep": d["rep"], "fold": f["fold"], "d_auc_lr": f["auc_e3"] - f["auc_lr"],
                         "d_auc_sstar": f["auc_e3"] - f["auc_s_star"]})
        for r in d["patients"]:
            for op in ["marg", "knn"]:
                a, b = r["e3"][op], r["lr"][op]
                if a["G"] >= GMIN and b["G"] >= GMIN and np.isfinite(a["ndf"]) and np.isfinite(b["ndf"]):
                    rows.append({"truth": d["truth"], "rho": d["rho"], "rep": d["rep"], "fold": r["fold"], "op": op,
                                 "dndf": a["ndf"] - b["ndf"], "cl": f'{d["truth"]}{d["rho"]}_{d["rep"]}_{r["fold"]}'})
    P, A = pd.DataFrame(rows), pd.DataFrame(aucs)
    P.to_csv(OUT / "simulation_patients.csv", index=False); A.to_csv(OUT / "simulation_folds.csv", index=False)
    summ = []
    for (t, rho, op), g in P.groupby(["truth", "rho", "op"]):
        cm = g.groupby("cl").dndf.mean()       # cluster (replicate x fold) means
        m, se = cm.mean(), cm.std(ddof=1) / np.sqrt(len(cm))
        summ.append({"truth": t, "rho": rho, "op": op, "n_patients": len(g), "n_clusters": len(cm), "mean_dndf": m,
                     "ci_low": m - st.t.ppf(.975, len(cm) - 1) * se, "ci_high": m + st.t.ppf(.975, len(cm) - 1) * se,
                     "median_dndf": float(g.dndf.median())})
    S = pd.DataFrame(summ)
    Aa = A.groupby(["truth", "rho"]).agg(d_auc=("d_auc_lr", "mean"), sd=("d_auc_lr", "std")).reset_index()
    import statsmodels.formula.api as smf
    P["marg"] = (P.op == "marg").astype(int); P["N"] = (P.truth == "N").astype(int)
    m1 = smf.ols("dndf ~ rho * marg + N", data=P).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(P.cl)[0]})
    reg = {k: {"coef": float(m1.params[k]), "se": float(m1.bse[k]), "p": float(m1.pvalues[k])} for k in m1.params.index}
    return S, Aa, reg


def figures(s1, s2, meta, S):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    allds = {**s1, **s2}; names = list(allds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, lab, thr in [(axes[0], "H1", "ΔAUC (E3 − S*)", 0.02), (axes[1], "H2", "ΔNDF (E3 − S*), marginal operator", 0.05)]:
        for i, d in enumerate(names):
            r = allds[d][key]; col = "#2a78d6" if d in s1 else "#7a4fc4" if d == "cdc_diabetes" else "#eb6834"
            ax.errorbar(r["mean_diff"], i, xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]], fmt="o", color=col, ms=6, capsize=3, lw=1.6)
        m = meta.get(key + "_8", meta[key]); y = len(names) + 0.6
        ax.fill([m["ci_low"], m["pooled"], m["ci_high"], m["pooled"]], [y, y + .3, y, y - .3], color="#222")
        ax.axvline(0, color="#444", lw=1)
        for s in (-thr, thr): ax.axvline(s, color="#999", ls=":", lw=1)
        ax.set_yticks(list(range(len(names))) + [y], [LAB[d] for d in names] + ["Pooled (random effects)"]); ax.invert_yaxis()
        ax.set_xlabel(lab + " with 95% CI"); ax.grid(axis="x", color="#eee")
    fig.tight_layout(); fig.savefig(FIG / "fig_forest_7_datasets.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, t, ttl in [(axes[0], "L", "Linear-logistic outcome"), (axes[1], "N", "Non-linear outcome")]:
        for op, col, mk, lab in [("marg", "#eb6834", "o", "Marginal removal"), ("knn", "#1baf7a", "s", "Conditional (k-NN) removal")]:
            g = S[(S.truth == t) & (S.op == op)].sort_values("rho")
            ax.errorbar(g.rho + (0.01 if op == "knn" else -0.01), g.mean_dndf, yerr=[g.mean_dndf - g.ci_low, g.ci_high - g.mean_dndf], fmt=mk + "-", color=col, capsize=3, lw=1.8, label=lab)
        ax.axhline(0, color="#444", lw=1); ax.set_title(ttl); ax.set_xlabel("Correlation ρ within feature block"); ax.set_xticks([0, .5, .9]); ax.grid(axis="y", color="#eee")
    axes[0].set_ylabel("ΔNDF (E3 − LR), 95% CI"); axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_simulation.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    out = {}
    s1, s2, meta = study2(); out["study1_recomputed"] = s1; out["study2"] = s2; out["meta"] = meta
    if (ROOT / "results" / "simulation" / "raw").exists():
        S, Aa, reg = study3(); out["study3_summary"] = S.to_dict("records"); out["study3_auc"] = Aa.to_dict("records"); out["study3_regression"] = reg
        figures(s1, s2, meta, S)
    (OUT / "summary_study2_3.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ["meta"]}, indent=1, default=float))
