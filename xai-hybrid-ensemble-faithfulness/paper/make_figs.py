"""Print-ready figures for the IEEE paper (fonts >= 8 pt at final size; colour-blind-safe; markers + colours)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pass
R = Path(__file__).resolve().parents[1] / "results"
E = json.load(open(R / "study2_3/summary_study2_3.json"))
V = json.load(open(R / "revision/summary_revision.json"))
DS = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
LAB = {"wdbc": "WDBC", "dr_debrecen": "DR Debrecen", "cleveland": "Cleveland", "ilpd": "ILPD", "parkinsons": "Parkinsons",
       "mammographic": "Mammographic", "ctg": "CTG", "cdc_diabetes": "CDC Diabetes"}
GROUP = {d: ("Primary", "#1f77b4", "o") for d in DS[:3]} | {d: ("Replication", "#e66100", "s") for d in DS[3:7]} | {"cdc_diabetes": ("Large-sample", "#5d3a9b", "^")}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
allds = {**E["study1_recomputed"], **E["study2"]}

# Fig. 1: forest, two panels, full text width (7.16 in)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"wspace": 0.08}, sharey=True)
for ax, key, title, xl, margin, xlim in [(axes[0], "H1", "ΔAUC (HE − S*)", "Difference in ROC-AUC", 0.02, (-0.06, 0.06)),
                                          (axes[1], "H2", "ΔNDF (HE − S*), marginal removal", "Difference in NDF", 0.05, (-1.15, 0.35))]:
    for i, d in enumerate(DS):
        r = allds[d][key]; g, col, mk = GROUP[d]
        lo, hi = r["ci_low"], r["ci_high"]
        ax.plot([max(lo, xlim[0]), min(hi, xlim[1])], [i, i], color=col, lw=1.3)
        ax.plot(r["mean_diff"], i, mk, color=col, ms=5, mec="white", mew=0.5)
        if hi > xlim[1]: ax.annotate("", xy=(xlim[1], i), xytext=(xlim[1] - 0.001, i), arrowprops=dict(arrowstyle="->", color=col, lw=1))
        if lo < xlim[0]: ax.annotate("", xy=(xlim[0], i), xytext=(xlim[0] + 0.001, i), arrowprops=dict(arrowstyle="->", color=col, lw=1))
    m = V["meta"][key]; y = len(DS) + 0.7
    ax.fill([m["ci_low"], m["pooled"], m["ci_high"], m["pooled"]], [y, y + .28, y, y - .28], color="black")
    ax.plot([max(m["pred_int"][0], xlim[0]), min(m["pred_int"][1], xlim[1])], [y + 0.55, y + 0.55], color="black", lw=1, ls="--")
    ax.axvline(0, color="#444", lw=0.8)
    for s in (-margin, margin): ax.axvline(s, color="#999", ls=":", lw=0.8)
    ax.set_xlim(*xlim); ax.set_title(title, fontsize=9, fontweight="bold"); ax.set_xlabel(xl); ax.grid(axis="x", color="#eee", lw=0.6)
axes[0].set_yticks(list(range(len(DS))) + [len(DS) + 0.7, len(DS) + 1.25], [LAB[d] for d in DS] + ["Pooled (95% CI)", "Prediction interval"])
axes[0].invert_yaxis()
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=mk, color=c, ls="", ms=5, label=g) for g, c, mk in [GROUP["wdbc"], GROUP["ilpd"], GROUP["cdc_diabetes"]]]
axes[1].legend(handles=h, loc="lower left", frameon=False)
fig.savefig("fig_forest.png", dpi=400, bbox_inches="tight"); fig.savefig("fig_forest.pdf", bbox_inches="tight"); plt.close(fig)

# Fig. 2: simulation, one column (3.5 in), single panel per outcome stacked side by side
S = {(r["truth"], r["rho"], r["op"]): r for r in E["study3_summary"]}
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=True, gridspec_kw={"wspace": 0.1})
for ax, t, ttl in [(axes[0], "L", "Linear outcome"), (axes[1], "N", "Non-linear outcome")]:
    for op, col, mk, ls, lab, dx in [("marg", "#e66100", "o", "-", "Marginal", -0.015), ("knn", "#1f77b4", "s", "--", "Conditional", 0.015)]:
        x = np.array([0, .5, .9]); yv = np.array([S[(t, r, op)]["mean_dndf"] for r in x])
        lo = np.array([S[(t, r, op)]["ci_low"] for r in x]); hi = np.array([S[(t, r, op)]["ci_high"] for r in x])
        ax.errorbar(x + dx, yv, yerr=[yv - lo, hi - yv], fmt=mk, ls=ls, color=col, ms=4, capsize=2, lw=1.1, label=lab)
    ax.axhline(0, color="#444", lw=0.7); ax.set_title(ttl, fontsize=8.5); ax.set_xticks([0, .5, .9]); ax.set_xlabel("Correlation ρ")
    ax.grid(axis="y", color="#eee", lw=0.6)
axes[0].set_ylabel("ΔNDF (HE − LR)"); axes[0].legend(frameon=False, loc="lower left", handlelength=1.8)
fig.savefig("fig_simulation.png", dpi=400, bbox_inches="tight"); fig.savefig("fig_simulation.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written")
