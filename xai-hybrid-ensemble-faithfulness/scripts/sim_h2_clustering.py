"""Synthetic simulation supporting docs/STATISTICAL_AUDIT.md (no real data).

Question: if patient-level NDF differences share a fold-level component (because all
patients in a test fold are explained by the same two fitted models), does a patient-level
Wilcoxon test keep its nominal 5% false-positive rate when the true mean difference is 0?

Model: delta_{i,f} = u_f + e_i,  u_f ~ N(0, tau^2) per fold, e_i ~ N(0, 1) per patient.
Design A (report): 5 folds x 30 patients, Wilcoxon on the 150 deltas.
Design B (proposed): 25 folds x 20 patients, per-fold means, corrected repeated k-fold t-test.
Note: Design B's correction targets training-set overlap; here folds are independent by
construction, so B is conservative in this simulation. The point is A's inflation.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xhe.stats import corrected_repeated_kfold_ttest  # noqa: E402

rng = np.random.default_rng(2026)
n_sim = 4000
print("tau   A_wilcoxon_patients  B_fold_corrected_t  C_fold_naive_t  power_B(delta=0.3)")
for tau in [0.0, 0.1, 0.2, 0.3, 0.5]:
    ra = rb = rc = pw = 0
    for _ in range(n_sim):
        u = rng.normal(0, tau, 5)
        dA = (u[:, None] + rng.normal(0, 1, (5, 30))).ravel()
        ra += sps.wilcoxon(dA).pvalue < 0.05
        u25 = rng.normal(0, tau, 25)
        dB = u25 + rng.normal(0, 1, (25, 20)).mean(1)
        rb += corrected_repeated_kfold_ttest(dB, 5, 5, 0.25)["p"] < 0.05
        rc += sps.ttest_1samp(dB, 0).pvalue < 0.05
        pw += corrected_repeated_kfold_ttest(dB + 0.3, 5, 5, 0.25)["p"] < 0.05
    print(f"{tau:<5} {ra / n_sim:<20.3f} {rb / n_sim:<19.3f} {rc / n_sim:<15.3f} {pw / n_sim:.3f}")
print("\nInterpretation: A ignores the shared fold component and over-rejects as tau grows.\n"
      "C (fold-level, independent folds) is calibrated here; B adds the Nadeau-Bengio\n"
      "correction, which is needed in real repeated CV (overlapping training sets) but is\n"
      "conservative when folds are truly independent, as in this simulation.")
