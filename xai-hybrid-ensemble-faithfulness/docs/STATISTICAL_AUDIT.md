# Statistical analysis audit (pre-freeze)

Status: recommendation for supervisor approval. Nothing here has been applied to real
outcomes; no confirmatory analysis has been run.

## 1. H1 (AUC, E3 vs S*): keep as designed

Per-fold AUC difference over 25 outer folds (5 x 5 repeated stratified CV), corrected
repeated k-fold t-test (Nadeau and Bengio 2003; Bouckaert and Frank 2004), k = 5, r = 5,
n2/n1 = 1/4, df = 24. The unit of inference is the learning procedure on that dataset.
Implemented in `src/xhe/stats.py::corrected_repeated_kfold_ttest` (unit-tested against a hand
calculation). Assumption: the correction approximates the dependence from overlapping
training sets. It is approximate and known to be conservative in some settings.

## 2. H2 (NDF, E3 vs S*): the planned test is not defensible as primary

**Stage 2.6 section 11 plan:** Wilcoxon signed-rank test on 150 patient-level differences,
using repeat-1 models only.

**Problem.** All patients in one test fold are explained by the same two fitted models, so
their differences share a fold-level component. The Wilcoxon test assumes independent
differences. Simulation (`scripts/sim_h2_clustering.py`, output in
`results/audit/sim_h2_clustering.txt`; synthetic data, true mean difference 0):

| Fold-level SD (tau) | Patient Wilcoxon (5 folds x 30) | Fold-level naive t (25 folds) | Fold-level corrected t (25 folds) |
|---|---|---|---|
| 0.0 | 0.049 | 0.049 | 0.000 |
| 0.1 | 0.081 | 0.052 | 0.000 |
| 0.2 | 0.169 | 0.045 | 0.000 |
| 0.3 | 0.280 | 0.049 | 0.000 |
| 0.5 | 0.434 | 0.047 | 0.000 |

False-positive rates at nominal 0.05, from 4,000 simulations per row.

**What the simulation shows.** The patient-level test's false-positive rate grows with the
fold component, reaching 43% at tau = 0.5. Aggregating to the fold level removes this
inflation. In the simulation folds are independent, so the naive fold-level t-test is
calibrated and the corrected test is very conservative. In real repeated cross-validation
the folds share training data, which is what the correction is for.

**Recommended amendment (needs approval before the protocol freeze).**

- **Primary H2:** for each of the 25 outer folds, explain a stratified random subsample of
  m = 20 complete-case test patients with E3 and with that fold's S\*, using the same SHAP
  seed and the same deletion background. Compute the fold mean of
  ΔNDF = NDF(E3) - NDF(S\*), counting only patients that meet the inclusion rule for both
  models. Test the 25 fold means with the same corrected repeated k-fold t-test as H1.
- **Unit of inference:** the procedure (training plus explanation), the same as H1, so the
  two primary hypotheses describe the same kind of population.
- **Cost:** 25 x 20 x 2 = 1,000 permutation-SHAP runs per dataset. That is fewer than the
  1,200 in the original plan.
- **Power caveat:** with a large fold component, power is low. In the simulation, power for
  a true difference of 0.3 SD fell from 0.84 at tau = 0 to 0.12 at tau = 0.3. A null result
  must therefore be reported as inconclusive, not as "no difference".
- **Secondary H2 (conditional):** keep the 150-patient repeat-1 Wilcoxon from Stage 2.6.
  Label it as conditional on the fitted repeat-1 models. Report it with the stratified
  bootstrap CI and the per-fold means. Do not treat it as confirmatory.
- **If a fold has fewer than 5 included patients** (G < 0.05 or K = 0 for either model),
  its mean is still used if it has at least 1 patient. A fold with 0 included patients is
  dropped, which changes df. Report this. (This rule is an implementation choice and needs
  approval.)

## 3. Multiplicity

Holm (1979) across the 6 primary tests (H1 and H2 x 3 datasets): `stats.holm`, unit-tested.
Secondary families use Holm within dataset x metric and are labelled exploratory.

## 4. Assumptions to state in the paper

1. The three datasets are separate case studies, so no pooled test is run.
2. The corrected t-test is an approximation for repeated CV.
3. NDF depends on the removal operator. A marginal operator is primary and the k-NN
   conditional operator is a sensitivity analysis. Synthetic test K3 shows the two can
   differ sharply with redundant features: mean D1 was 1.00 (marginal) vs. 0.24 (k-NN).
4. Thresholds of interest (ΔAUC 0.02, ΔNDF 0.05) are interpretive, not tested.
