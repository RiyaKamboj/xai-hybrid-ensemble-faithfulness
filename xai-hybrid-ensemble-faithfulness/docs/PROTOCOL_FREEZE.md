# Protocol freeze — confirmatory experiments

**Frozen:** 11 November 2026, before any confirmatory model was fitted on real data.
**Frozen by:** Riya Kamboj (instruction to start the full experiments, 11 Nov 2026).
**Basis:** Stage 2.6 design report (26 Oct 2026), `configs/study.yaml`, and the two amendments below.

Supervisor approval of the H2 amendment (A1) and implementation decisions D1–D6 had not been
recorded at the time of freezing. They are reported as protocol decisions made before the data
were analysed. Any later change is a logged deviation, and the results are reported under both
the frozen and the changed analysis.

## Data
- WDBC (UCI 17), DR Debrecen (UCI 329) and Cleveland (UCI 45) are used, verified by SHA-256 checksums.
- For DR Debrecen, 5 exact duplicates are removed before splitting, leaving 1,146 rows.
- Cleveland's 6 missing values are imputed inside each pipeline.
- Validation uses 5×5 RepeatedStratifiedKFold outer folds, which were saved on 26 Oct 2026. The seed is 20261026.

## Models (fixed settings, no tuning)
- **Components:** LR, SVM-RBF, KNN (k = 15), RF (500 trees) and XGB.
- **Ensembles:**
  - E1 = soft vote of all five components.
  - E2 = L2-LR stacking on inner 5-fold out-of-fold probabilities.
  - E3 = L2-LR stacking on the two group means q_A = mean(LR, SVM, KNN) and q_B = mean(RF, XGB).
- **S\*:** the component with the best inner out-of-fold AUC. Ties within 0.001 are broken in the order LR, RF, XGB, SVM, KNN.
- **E3 → E2 fallback:** only on technical failure on training data, meaning a convergence warning, non-finite output or constant output.
- **Implementation note:** ensembles are assembled from the same fitted components and out-of-fold predictions that sklearn's `StackingClassifier` and `VotingClassifier` would use. Their predictions are checked numerically against the sklearn objects on the first fold of every dataset.

## Primary hypotheses (6 tests; Holm, family-wise α = 0.05)

**H1 (per dataset).** Outer-fold ROC-AUC of E3 minus S\* over 25 folds.
- Test: corrected repeated k-fold t-test (k = 5, r = 5, n2/n1 = 1/4).

**H2 (per dataset) — amendment A1, from docs/STATISTICAL_AUDIT.md.**
- **Patients:** in each of the 25 outer folds, a class-stratified random sample of 20 complete-case test patients (seed = master seed + fold id).
- **Explanations:** each patient is explained with E3 and with that fold's S\*. Permutation SHAP is used with max_evals = 500 and B_shap = 50 training rows. The SHAP seed is the same for both models (common random numbers).
- **Faithfulness:** NDF uses the marginal operator, B_del = 50 training rows disjoint from B_shap, K = min(5, number of positive contributors) and 20 random orderings.
- **Inclusion:** a patient is included only if G ≥ 0.05 under both models.
- **Statistic:** the fold mean of ΔNDF = NDF(E3) − NDF(S\*), tested with the corrected t-test.
- **Missing folds:** a fold with 0 included patients is dropped, and n becomes the number of remaining folds.

Smallest differences of interest are ΔAUC 0.02 and ΔNDF 0.05. These are for interpretation only.

## Secondary and exploratory analyses (Holm within each dataset × metric; labelled exploratory)
- **Predictive comparisons:** E1 vs S\*, E2 vs S\* and E3 vs E2 on AUC. Brier score is also reported.
- **Descriptive metrics:** accuracy, precision, recall, F1 and specificity at a threshold of 0.5, plus prediction time.
- **Patient-level H2 (from the Stage 2.6 plan):**
  - Sample: 150 class-stratified complete-case patients from the repeat-1 test folds, each explained by the fold model that held it out.
  - Test: Wilcoxon on ΔNDF, with a fold-stratified bootstrap CI.
  - This is labelled conditional on the repeat-1 models.
- **NDF of all 8 models** on the same 150 patients, and NDF(E1) and NDF(E2) vs NDF(S\*).
- **NDF sensitivity:** the conditional (k-NN, 20 neighbours) removal operator for E3 and S\*, on the H2 fold samples.
- **Local robustness (RB):**
  - Models: E1, E2, E3 and S\* on the 150 patients.
  - Neighbours: 5 type-aware neighbours at 0.05×SD.
  - Also reported: the class-flip rate, the mean change in the model's prediction, and the seed noise floor.
  - Sensitivity at 0.02×SD and 0.10×SD with 3 neighbours, for E3 and S\* only. This is reduction step 2 of the pre-registered reduction order, recorded here.
- **Diagnostics (descriptive only):** SHAP–LIME top-5 agreement and LIME local error, using LIME with 5,000 samples.

## Implementation choices (D1–D7)
- **D1:** NDF ties are broken by the lower feature index.
- **D2:** NDF is undefined when G ≤ 1e-12.
- **D3:** the random baseline uses the first k features of each random ordering.
- **D4:** binary and count features are scaled in LR, SVM and KNN.
- **D5:** the seed is 20261026.
- **D6:** SVC(probability=True) is used.
- **D7 (new):** SHAP and deletion backgrounds are drawn from complete-case training rows.

## Reporting
- Effect sizes and intervals are reported first, then raw and Holm-adjusted p-values.
- Every pre-specified analysis is reported, including null results.
- Results are written to `results/confirmatory/`.

## Deviation log
- 11 Nov 2026, 14:25. Stage A+B (all primary analyses) had finished. The secondary Stage C run
  was projected to take about 2.5 h on the 2-CPU workspace, so it was stopped before any Stage C
  output was written. It was restarted with the remaining steps of the pre-registered reduction
  order (Stage 2.6 §12):
  - step 2: robustness neighbours reduced from 5 to 3 at every σ;
  - step 3: robustness and LIME diagnostics restricted to E3 and S\*;
  - step 4: the patient-level explanation set reduced from 150 to 100 patients per dataset.

  Primary analyses H1 and H2 are unaffected.
- 11 Nov 2026, 15:30. Four WDBC Stage C files (repeat-1 folds 0–3) had been written by worker
  processes of the stopped Stage C run that kept running under the original 150-patient
  settings. For consistency they were moved to `results/confirmatory/superseded/` and re-run
  under the reduced settings. No primary analysis was affected.
