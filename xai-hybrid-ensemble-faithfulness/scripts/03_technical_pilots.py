"""Step 3: TECHNICAL PILOTS (Stage 2.6 section 12). NOT CONFIRMATORY.

What this script does NOT do: it never computes test-fold AUC, never reports model-wise
NDF on real data, and never compares E3 with S*. Real-data NDF in P4 is logged only pooled
across all models (blinded), so the pilot cannot inform the confirmatory comparison.

Pilots:
  P0  processed data reproduce the audited counts (after explicit deduplication)
  P1  gates on Cleveland (repeat 1, fold 1): identity, additivity, voting linearity,
      E2/E3 stacking decomposition (exact SHAP, logit scale), reproducibility
  P3  runtime and explainer seed-noise floor on WDBC (E2 and KNN, 10 patients)
  P4  blinded check that NDF separates SHAP ordering from random ordering on real data
  P5  mixed-type handling on Cleveland (missing values, perturbation/replacement validity)
  P6  E3 technical-failure rule on repeat-1 training folds of all datasets
P2 (synthetic known-answer tests) is run with pytest: tests/test_ndf_known_answer.py.

Usage: python scripts/03_technical_pilots.py [--quick]
Output: results/pilot/pilot_log.json and results/pilot/pilot_log.md
"""
import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xhe import cv, data, explain, metrics_xai, models  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="fewer patients (smoke run)")
args = ap.parse_args()

CFG = data.CONFIG
SEED = CFG["master_seed"]
EXP = CFG["explanation"]
OUT = ROOT / "results" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)
log = {"label": "TECHNICAL PILOT - NOT CONFIRMATORY - NO PERFORMANCE RESULTS",
       "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
       "python": platform.python_version(), "machine": platform.platform(), "pilots": {}}


def load_processed(name):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    y = df.pop("target")
    folds = cv.load_folds(ROOT / "data" / "processed" / f"{name}_outer_folds.json")
    return df, y, folds, data.FEATURE_TYPES[name]


def save():
    log["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "pilot_log.json").write_text(json.dumps(log, indent=2, default=float))


# ------------------------------------------------------------------ P0
p0 = {}
for name, spec in CFG["datasets"].items():
    X, y, folds, ft = load_processed(name)
    exp = spec["expected"]
    exp_rows = exp["rows"] - exp["exact_duplicate_rows"]
    p0[name] = {"rows_after_dedup": len(X), "expected_rows_after_dedup": exp_rows,
                "missing_cells": int(X.isna().sum().sum()), "n_outer_folds": len(folds),
                "pass": len(X) == exp_rows and int(X.isna().sum().sum()) == exp["missing_cells"]}
log["pilots"]["P0_data"] = {"pass": all(v["pass"] for v in p0.values()), "detail": p0}
save()
print("P0", log["pilots"]["P0_data"]["pass"])

# ------------------------------------------------------------------ P1 + P5 (Cleveland)
X, y, folds, ft = load_processed("cleveland")
cols = list(X.columns)
f0 = folds[0]
Xtr, ytr, Xte = X.iloc[f0["train"]], y.iloc[f0["train"]], X.iloc[f0["test"]]
fitted = {}
for name in ["e1", "e2", "e3"]:
    fitted[name] = models.make_model(name, ft, cols, SEED).fit(Xtr, ytr)
rng = np.random.default_rng(SEED)
bg_rows = Xtr.dropna().sample(10, random_state=SEED).values          # small for exact SHAP
xs = Xte.dropna().iloc[:2].values
p1 = {}
for name, m in fitted.items():
    f = models.positive_proba_fn(m, cols)
    p1[f"{name}_identity"] = bool(np.array_equal(f(Xte.values), m.predict_proba(Xte)[:, 1]))
    sh = explain.permutation_shap(f, xs, bg_rows, max_evals=500, seed=1)
    p1[f"{name}_additivity_max_err"] = float(explain.additivity_error(f, xs, sh).max())
    sh2 = explain.permutation_shap(f, xs, bg_rows, max_evals=500, seed=1)
    p1[f"{name}_reproducible"] = bool(np.array_equal(sh["values"], sh2["values"]))
t0 = time.time()
e1 = fitted["e1"]
phi_e1 = explain.exact_shap(models.positive_proba_fn(e1, cols), xs, bg_rows)["values"]
phis = [explain.exact_shap(models.positive_proba_fn(c, cols), xs, bg_rows)["values"]
        for c in e1.estimators_]
p1["voting_linearity_max_err"] = float(np.abs(phi_e1 - np.mean(phis, 0)).max())
for name in ["e2", "e3"]:
    st = fitted[name]
    fp = models.positive_proba_fn(st, cols)
    flog = lambda Z, fp=fp: logit(np.clip(fp(Z), 1e-15, 1 - 1e-15))
    phi = explain.exact_shap(flog, xs, bg_rows)["values"]
    parts = [explain.exact_shap(models.positive_proba_fn(c, cols), xs, bg_rows)["values"]
             for c in st.estimators_]
    w = st.final_estimator_.coef_.ravel()
    p1[f"{name}_decomposition_max_err"] = float(np.abs(phi - np.tensordot(w, np.array(parts), 1)).max())
p1["exact_shap_seconds"] = round(time.time() - t0, 1)
tol_ok = (all(p1[k] for k in p1 if k.endswith(("_identity", "_reproducible")))
          and all(p1[k] < 1e-6 for k in p1 if k.endswith("additivity_max_err"))
          and all(p1[k] < 1e-8 for k in p1 if k.endswith(("linearity_max_err", "decomposition_max_err"))))
log["pilots"]["P1_gates_cleveland"] = {"pass": bool(tol_ok), "detail": p1}
save()
print("P1", tol_ok, p1)

# P5: mixed types
sd = Xtr.std().values; lo = Xtr.min().values; hi = Xtr.max().values
types = [ft[c] for c in cols]
invalid = 0
for i in range(50):
    x = Xte.values[i % len(Xte)]
    xp = metrics_xai.perturb(x, types, sd, lo, hi, EXP["robustness_sigma_frac"], rng)
    for j, t in enumerate(types):
        if np.isnan(x[j]):
            invalid += not np.isnan(xp[j]); continue
        if t in ("binary", "nominal"):
            invalid += xp[j] != x[j]
        if t == "count":
            invalid += xp[j] != np.round(xp[j])
        invalid += not (lo[j] <= xp[j] <= hi[j])
nominal_idx = [j for j, t in enumerate(types) if t == "nominal"]
bgr = Xtr.sample(50, random_state=SEED).values
repl = metrics_xai._replace(Xte.dropna().values[0], nominal_idx, bgr)
train_levels = {j: set(Xtr.iloc[:, j].dropna().unique()) for j in nominal_idx}
bad_repl = sum(int(not (np.isnan(v) or v in train_levels[j])) for j in nominal_idx for v in repl[:, j])
pipe_ok = bool(np.isfinite(fitted["e3"].predict_proba(Xte)[:, 1]).all())
lime_ok = True
try:
    lc = explain.lime_contributions(models.positive_proba_fn(fitted["e3"], cols),
                                    Xte.dropna().values[0], Xtr.values, types, 2000, SEED)
    lime_ok = bool(np.isfinite(lc["contributions"]).all())
except Exception as exc:  # noqa: BLE001
    lime_ok = False; log.setdefault("errors", []).append(f"P5 LIME: {exc!r}")
log["pilots"]["P5_mixed_types_cleveland"] = {
    "pass": invalid == 0 and bad_repl == 0 and pipe_ok and lime_ok,
    "detail": {"invalid_perturbed_values": int(invalid), "invalid_replacement_values": int(bad_repl),
               "pipeline_handles_missing": pipe_ok, "lime_on_complete_case_ok": lime_ok,
               "note": "replacement values with NaN come from background rows with missing ca/thal; "
                       "they are imputed inside the pipeline, as for any patient"}}
save()
print("P5", log["pilots"]["P5_mixed_types_cleveland"])

# ------------------------------------------------------------------ P3 runtime and noise (WDBC)
X, y, folds, ft = load_processed("wdbc")
cols = list(X.columns)
f0 = folds[0]
Xtr, ytr, Xte = X.iloc[f0["train"]], y.iloc[f0["train"]], X.iloc[f0["test"]]
n_p3 = 4 if args.quick else 10
bg = Xtr.sample(EXP["shap_background"], random_state=SEED).values
xs = Xte.values[:n_p3]
p3 = {}
for name in ["e2", "knn"]:
    m = models.make_model(name, ft, cols, SEED).fit(Xtr, ytr)
    f = models.positive_proba_fn(m, cols)
    for me in [500, 1000]:
        t0 = time.time()
        a = explain.permutation_shap(f, xs, bg, max_evals=me, seed=11)
        secs = (time.time() - t0) / len(xs)
        b = explain.permutation_shap(f, xs, bg, max_evals=me, seed=12)
        overlaps = [metrics_xai.topk_overlap(a["values"][i], b["values"][i]) for i in range(len(xs))]
        p3[f"{name}_maxevals{me}"] = {"seconds_per_explanation": round(secs, 2),
                                      "median_seed_top5_overlap": float(np.median(overlaps)),
                                      "effective_max_evals": a["max_evals"]}
# projected runtime per dataset: 1,200 + 3,600 SHAP runs (Stage 2.6 sec. 12) at the slower model
runs500 = {k: v for k, v in p3.items() if isinstance(v, dict) and k.endswith("maxevals500")}
worst = max(v["seconds_per_explanation"] for v in runs500.values())
p3["projected_hours_per_dataset_at_max_evals_500"] = round(4800 * worst / 3600, 1)
p3["noise_floor_criterion_met_at_500"] = all(v["median_seed_top5_overlap"] >= 0.8
                                              for v in runs500.values())
log["pilots"]["P3_runtime_noise_wdbc"] = {
    "pass": bool(p3["projected_hours_per_dataset_at_max_evals_500"] <= 8
                 and p3["noise_floor_criterion_met_at_500"]),
    "detail": p3,
    "caveat": "Measured in the cloud workspace (Linux, Python 3.13); laptop timing will differ."}
save()
print("P3", log["pilots"]["P3_runtime_noise_wdbc"])

# ------------------------------------------------------------------ P4 blinded NDF separation
X, y, folds, ft = load_processed("cleveland")
cols = list(X.columns)
f0 = folds[0]
Xtr, ytr, Xte = X.iloc[f0["train"]], y.iloc[f0["train"]], X.iloc[f0["test"]]
cc = Xtr.dropna()
idx = cc.sample(len(cc), random_state=SEED).index
bg_shap = Xtr.loc[idx[:EXP["shap_background"]]].values
bg_del = Xtr.loc[idx[EXP["shap_background"]:EXP["shap_background"] + EXP["ndf_background"]]].values
assert not set(idx[:50]) & set(idx[50:100])
xs = Xte.dropna().values[: (8 if args.quick else 20)]
pooled_true, pooled_rand, excluded = [], [], 0
for name in models.COMPONENTS + ["e1", "e2", "e3"]:
    m = models.make_model(name, ft, cols, SEED).fit(Xtr, ytr)
    f = models.positive_proba_fn(m, cols)
    sh = explain.permutation_shap(f, xs, bg_shap, max_evals=EXP["shap_max_evals"], seed=21)
    for i, x in enumerate(xs):
        r = metrics_xai.ndf(f, x, sh["values"][i], bg_del, K_max=EXP["ndf_K"],
                            n_random=EXP["ndf_random_orderings"], seed=1000 + i)
        rr = metrics_xai.ndf(f, x, rng.normal(0, 1, len(cols)), bg_del, K_max=EXP["ndf_K"],
                             n_random=EXP["ndf_random_orderings"], seed=1000 + i)
        if np.isnan(r["ndf"]) or r["G"] < EXP["ndf_min_gap"]:
            excluded += 1
            continue
        pooled_true.append(r["ndf"])
        if not np.isnan(rr["ndf"]):
            pooled_rand.append(rr["ndf"])
# Only pooled summaries are kept (blinding). Per-model values are discarded on purpose.
p4 = {"n_patient_model_pairs_included": len(pooled_true), "n_excluded_gap_or_K": excluded,
      "pooled_median_ndf_shap_order": float(np.median(pooled_true)),
      "pooled_median_ndf_random_attribution": float(np.median(pooled_rand)),
      "share_shap_ndf_above_0.02": float(np.mean(np.array(pooled_true) > 0.02))}
log["pilots"]["P4_ndf_separation_blinded"] = {
    "pass": p4["pooled_median_ndf_shap_order"] > 0.02, "detail": p4,
    "blinding": "pooled over 8 models; no per-model or E3-vs-S* values computed or stored"}
save()
print("P4", log["pilots"]["P4_ndf_separation_blinded"])

# ------------------------------------------------------------------ P6 E3 technical rule
p6 = {}
for name in CFG["datasets"]:
    X, y, folds, ft = load_processed(name)
    for f in folds[:5]:                                   # repeat 1
        Xtr, ytr = X.iloc[f["train"]], y.iloc[f["train"]]
        chk = models.e3_technical_check(Xtr, ytr, ft, SEED)
        p6[f"{name}_fold{f['fold_id']}"] = {"trigger": chk["trigger_fallback_to_e2"],
                                            "reasons": chk["reasons"]}
log["pilots"]["P6_e3_technical_rule"] = {"pass": not any(v["trigger"] for v in p6.values()),
                                         "detail": p6}
save()
print("P6", log["pilots"]["P6_e3_technical_rule"]["pass"])

# ------------------------------------------------------------------ markdown summary
lines = [f"# Technical pilot log", "", f"**{log['label']}**", "",
         f"Started {log['started_utc']}, finished {log['finished_utc']}; Python {log['python']}; "
         f"{log['machine']}", ""]
for k, v in log["pilots"].items():
    lines += [f"## {k}: {'PASS' if v['pass'] else 'FAIL / ACTION NEEDED'}", "",
              "```", json.dumps(v["detail"], indent=2, default=float), "```", ""]
(OUT / "pilot_log.md").write_text("\n".join(lines))
print("done")
