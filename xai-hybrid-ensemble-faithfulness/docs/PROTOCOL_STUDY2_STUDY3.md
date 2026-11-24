# Pre-specified extension studies (written 20 Nov 2026, before any data for these studies were generated or downloaded)

Study 1, the confirmatory comparison on WDBC, DR Debrecen and Cleveland, is complete and frozen (PROTOCOL_FREEZE.md). Its results are not altered. Two extensions are added to test whether the Study 1 findings generalise and to explain them. Both are reported as separate, clearly labelled studies.

## Study 2 — External replication on additional public healthcare datasets
**Purpose:** test whether the Study 1 findings hold on datasets not used to develop the protocol.

**Datasets:** UCI Indian Liver Patient Dataset (ILPD), Parkinsons, Mammographic Mass and Cardiotocography.
- Cardiotocography is binary: normal vs suspect/pathologic, using the NSP column.
- Cardiotocography features: the 21 CTG features only. The class-code columns, which leak the target, are excluded.
- Records with missing values are kept and imputed inside the pipelines, as in Study 1.

**Procedure:** identical to Study 1 Stage A+B.
- Models: same models, settings and S\* rule.
- Validation: 5×5 repeated stratified cross-validation with a new saved seed, 20261120.
- Explanations: 20 patients per fold, same SHAP and NDF settings.
- Faithfulness: NDF with both the marginal and the k-NN operator.

**Hypotheses (two-sided, per dataset):**
- **H1:** ΔAUC (E3 − S\*) ≠ 0.
- **H2:** ΔNDF (E3 − S\*) ≠ 0, using the fold-level corrected t-test.

**Multiplicity:** Holm correction across the 8 tests of Study 2.

**Pooled summary (pre-specified, descriptive plus one test):**
- Per-dataset effects are combined across all 7 datasets (Studies 1 + 2) with a random-effects meta-analysis (DerSimonian–Laird) of ΔAUC and of ΔNDF (marginal).
- These pooled effects are reported with 95% CIs as an overall summary.
- They are labelled as a summary across case studies, not as a population estimate.

## Study 3 — Controlled simulation of the mechanism
**Purpose:** test the explanation proposed in the Study 1 Discussion. The marginal-operator faithfulness disadvantage of E3 relative to a linear single model should grow with feature correlation, and should shrink under the conditional operator.

**Design:**
- **Data:** n = 600 per replicate, d = 10 Gaussian features. Features 1–4 form an equicorrelated block with correlation ρ ∈ {0, 0.5, 0.9}; features 5–10 are independent.
- **Outcome model T:**
  - linear-logistic (L): logit = 1.2x1 + 0.8x2 + 0.6x5 − 0.6x6 + 0.4x7;
  - non-linear (N): logit = 1.2x1 + 0.8·1[x2 > 0] + 0.8·x5·x6 − 0.6x6 + 0.4x7².
  - In both cases the outcome is centred so that prevalence is about 0.5.
- **Grid and replicates:** 3 values of ρ × 2 outcome models = 6 conditions, with 3 independent replicates each (seeds 1–3).
- **Validation and explanations:** 5-fold stratified cross-validation per replicate, with 10 test patients explained per fold.
- **Models:** E3 vs LR (the S\* chosen in most Study 1 folds), plus E3 vs the S\* rule.
- **Outcomes:** ΔNDF (E3 − LR) with the marginal and with the k-NN operator, and ΔAUC.

**Pre-specified expectations:**
- **(a)** Under L, the marginal-operator ΔNDF becomes more negative as ρ increases.
- **(b)** Under the k-NN operator, ΔNDF is closer to 0 than under the marginal operator.
- **(c)** Under N, E3's AUC advantage over LR is larger than under L.

**Analysis:** ΔNDF is analysed by linear regression on ρ, outcome model and operator (patient-level, cluster-robust standard errors by replicate × fold). Results are reported whatever their direction.

## Study 2 dataset-specific decisions

These were recorded on 20 Nov 2026, after the ILPD file was received but before any other file was opened and before any model was fitted.

**ILPD**
- Positive class: liver patient (selector = 1).
- Feature types: gender is binary; the other 9 features are continuous.
- Exact duplicate rows are removed before splitting, and the removal is logged.

**Parkinsons**
- Positive class: status = 1.
- The `name` column identifies the recording and is not used as a feature. The subject ID is the `S##` part of the name.
- There are several recordings per subject, so outer and inner folds are grouped by subject (StratifiedGroupKFold, 5 folds × 5 repeats with different seeds). This keeps the same person out of both training and test.

**Mammographic Mass**
- Positive class: severity = 1 (malignant).
- The BI-RADS assessment is excluded. The UCI documentation marks it as "non-predictive", and it is a radiologist's summary judgement.
- Features: age (continuous); shape and margin (nominal); density (count).
- Missing values (`?`) are imputed in the pipeline.

**Cardiotocography**
- Positive class: NSP ∈ {2, 3} (suspect or pathologic).
- Features: the 21 measured CTG features only.
- Excluded columns: the A–SUSP class indicators, CLASS and NSP.

**Mammographic Mass correction (20 Nov 2026, during data preparation, before any model was fitted)**
- With only four coarse features (age, shape, margin, density), 353 of 961 rows are identical to another row.
- These rows are distinct patients with the same coarse values, not duplicated records, so they are kept.
- The duplicate-removal rule is applied only where features are fine-grained: ILPD (13 rows removed) and Cardiotocography (12 rows removed).

## Study 4 — exploratory: remedy, additional metrics and explainer (planned 20 Nov 2026, before running)
**Purpose:**
- **(i)** Test whether attributing to groups of correlated features closes E3's faithfulness gap.
- **(ii)** Check the main finding with established faithfulness metrics and a second explainer.

All of Study 4 is labelled exploratory. Every outcome will be reported.

**Data:**
- **Datasets:** all 7.
- **Folds:** the repeat-1 outer folds (5 per dataset).
- **Patients:** the same 20 complete-case test patients per fold as in the primary H2 sample.
- **Models:** E3 and that fold's S*. They are refitted deterministically, which gives the same models as before.

**Feature groups:**
- Groups are formed on the outer-training fold only.
- Method: average-linkage hierarchical clustering on 1 − |Spearman ρ|, cut at 0.3. Features with |ρ| ≥ 0.7 are merged; any other feature stays on its own.

**Attributions for each patient and model:**
- **(a)** Permutation SHAP per feature. This is the Study 1 setting.
- **(b)** KernelSHAP per feature, with 500 samples and the same background.
- **(c)** Group Shapley values, using the marginal value function on the same background:
  - computed exactly when there are 12 groups or fewer;
  - otherwise estimated by permutation sampling.

**Faithfulness metrics:**
- **NDF at feature level:** with the marginal operator and with the k-NN operator, for (a) and (b).
- **NDF at group level:** whole groups are deleted, with the marginal operator, for (c). For comparison, the group-level NDF of (a) is also computed by summing (a) within each group.
- **Faithfulness correlation (Bhatt et al.):** the Pearson correlation, over 50 random feature subsets of size ⌈d/4⌉, between the summed attribution of the subset and the prediction drop when that subset is removed (marginal operator). Computed for (a) and (b).

**Main exploratory question:** is ΔNDF (E3 − S*) closer to 0 for group-level attributions (c) than for feature-level attributions (a)?

**Analysis:**
- Per dataset: mean fold differences with corrected t-intervals (n = 5 folds).
- Across datasets: random-effects pooling.

**Pre-run implementation note, 20 Nov 2026 (made before any Study 4 output existed):**
- **Exact group Shapley values:** computed when there are 8 groups or fewer (2^8 coalitions × 50 background rows), not the 12 groups planned above.
- **More than 8 groups:** group Shapley values are estimated with 25 antithetic permutation pairs. The cut-off was lowered because of compute cost.
- **Group-level NDF with the k-NN operator:** also computed, as an extra sensitivity analysis.
- **Feature-level SHAP (a):** not re-explained. The values stored in the Study 1 and Study 2 fold files are reused, so they are identical to the earlier runs.

## Study 5: large-sample replication on CDC Diabetes Health Indicators (planned 20 Nov 2026, before fitting)
**Data:**
- **Source:** BRFSS 2015, binary file (Kaggle release; the same data is published as UCI dataset 891).
- **SHA-256:** ecbd6572…
- **Sample:** a stratified subsample of 5,000 respondents (697 positive). Exact duplicates are kept. Decisions are listed in scripts/07b_prepare_cdc.py.

**Runs:**
- Stage A+B of the confirmatory runner, with 5×5 cross-validation and seed 20261120.
- The H1/H2 tests use the same definitions as before.
- Study 4 metrics are also computed on the repeat-1 folds.

**Prediction, written before running:**
- In the training data, the largest absolute Spearman correlation between any two features is 0.45.
- So the clustering forms no feature groups.
- The correlation mechanism therefore predicts a small marginal ΔNDF (|Δ| < 0.1) and a ΔNDF under k-NN removal of about 0.

**Pooling:** Study 5 is added to the random-effects pooled analysis as an eighth dataset. The 7-dataset pooled estimates are reported unchanged next to it.

**Diabetes 130-US Hospitals:** downloaded (SHA-256 f82ac129…). It is not analysed yet.

---
## Study 6: confirmatory test of the group remedy, a placebo control and a mechanism test
**Written 21 Nov 2026 before any of these quantities were computed on the folds involved.**

**Why this study exists.** Study 4 was exploratory and used only repetition 1. Study 6 tests its main claim on held-out data: the outer folds of repetitions 2–5, which are 20 folds per dataset and were never used in Study 4. The same 8 datasets are used, with the same 20 primary patients per fold. The fold models are refitted deterministically, and the stored feature-level permutation SHAP values are reused.

### Remedy
**Correlation groups.** Built exactly as in Study 4: Spearman correlation on the outer-training fold, average linkage, cut at 0.3.

**R1, primary.** Remedy = sum the feature-level SHAP values within each group, then compute group-level NDF (marginal operator).
- Reduction R = ΔNDF_group − ΔNDF_feature, where Δ = E3 − S*.
- H6a: the pooled R across the 8 datasets is > 0. Test: per-dataset corrected t on 20 folds, then DerSimonian–Laird pooling; two-sided α = 0.05.
- Also reported:
  - the pooled ΔNDF_group itself, with the expectation that it is no longer significantly negative;
  - the same analysis restricted to datasets where at least one group merges features.

**R2, placebo control (H6b).** Random partitions with the same group sizes as the correlation partition. Features are shuffled with seed 70,000 + row, and 5 random partitions are averaged per patient.
- H6b: the correlation partition reduces the deficit more than the random partitions do. Statistic: R_corr − R_rand > 0, pooled.
- If H6b fails, the remedy is only a coarsening effect and will be reported as such.

**R3, secondary.** Exact group Shapley values, computed when there are 8 groups or fewer and at least one group merges features; group NDF as above.

**R4, secondary.** Group NDF under the k-NN conditional operator.

### Mechanism (H6c)
**Non-additivity index.** For model m at patient x:
- NA_m(x) = mean over 20 random feature pairs (j, l) of |Δ({j,l}) − Δ({j}) − Δ({l})| / G_m(x).
- Δ(S) = s·(f(x) − F(S)), using marginal deletion with B_del.

**Hypotheses.**
- H6c: E3 has higher NA than S*. Tested by pooling the fold-level ΔNA.
- H6c': patients with a larger ΔNA have a more negative feature-level ΔNDF. Test: linear regression of patient-level ΔNDF on ΔNA with dataset fixed effects; standard errors clustered by dataset × fold.

**Specific prediction for CDC:** ΔNA > 0 there. This would account for the deficit that correlation cannot explain.

### Inclusion
- Same as before: G ≥ 0.05 for both models.
- Correction factor for the corrected t: n_test/n_train = 0.25. The pooled estimates use DerSimonian–Laird.

---
## Study 7: robustness to hyperparameter tuning
**Written 21 Nov 2026, before running.**

**Tuning.** Every component is tuned inside each training set. Grid search uses 3-fold stratified inner cross-validation on ROC-AUC; it is nested inside the out-of-fold loop used for stacking and S* selection, so test data never touch tuning.

| Model | Grid |
|---|---|
| LR | C ∈ {0.01, 0.1, 1, 10} |
| SVM | C ∈ {0.3, 1, 3}, gamma = scale |
| KNN | k ∈ {5, 15, 31} |
| RF (500 trees) | min_samples_leaf ∈ {1, 5} × max_features ∈ {sqrt, 0.5} |
| XGB | max_depth ∈ {2, 4, 6} × learning_rate ∈ {0.05, 0.3} |

**Scope.** Repetition-1 folds (5 per dataset), 8 datasets. H1 (ΔAUC) and H2 (ΔNDF, marginal and k-NN) are computed exactly as in Stage A+B.

**Prediction.** Both conclusions hold:
- pooled ΔAUC within ±0.02;
- pooled ΔNDF < 0.

**Comparison.** The tuned results are compared with the untuned results on the same folds.
**Implementation note (21 Nov 2026, before the full Study 7 run):**
- **Parkinsons tuning folds are not grouped by subject.** The 3-fold tuning splits inside GridSearchCV ignore subjects. This affects only the choice of hyperparameters. The outer and inner stacking folds are still grouped by subject, as in Studies 1–2.
- **Smoke test, not used for decisions.** One smoke-test fold (Mammographic fold 0) was run first to check that the code works. It is kept as part of the full run.

**Run log, 21 Nov 2026 (Studies 6 and 7):**
- **Study 6, container restart:** the cloud container restarted part-way through. Folds already written were kept, and the run resumed from the remaining folds. Every output file was checked and parsed without errors.
- **Study 7, first-run crash:** the first full run crashed because of a code bug. Loky workers are reused between tasks, so the tuning wrapper was applied twice to the same model. That produced an error; no results were written. The bug was fixed and the run restarted. The two files written before the crash (WDBC fold 0, Mammographic fold 0) came from the first task in each worker, before the double patch, and are valid.
- **Pre-specified outcomes:**
  - H6a (pooled remedy): NOT supported.
  - H6b (placebo, pooled): NOT supported. On DR Debrecen alone, the correlation-group remedy did better than the random-group placebo.
  - H6c (ΔNA > 0 pooled): NOT supported. The CDC-specific prediction (ΔNA > 0) was supported.
  - H6c′: NOT supported, and the estimate went in the opposite direction.
  - Study 7 predictions: both supported.

---
## Revision analyses (written 21 Nov 2026, before running; prompted by an internal review)
**Question.** Is the faithfulness deficit "ensemble versus single model", or just "non-additive model versus near-additive LR"?

### R-A. Component analysis (new computation)
- **Scope:** repetition-1 outer folds, 8 datasets, the same 20 primary patients per fold.
- **Explanations:** permutation SHAP (same settings and seeds as the primary analysis) for LR, SVM, KNN, RF, XGB, E1 (soft vote of all five) and E2 (plain stacking of all five). E3 values are taken from the stored files.
- **NDF:** marginal and k-NN operators, with D_k^expl and mean D_k^rand stored, so absolute deletion curves can be reported.
- **Comparisons:** E3 − RF and E3 − XGB on every dataset, regardless of which model internal validation selected. Inclusion requires G ≥ 0.05 for both models in each pair.
- **Reading the results:**
  - If RF and XGB alone are also less faithful than LR by a similar amount, the effect is attributed to non-additive learners, not to ensembling, and the paper is reframed accordingly.
  - E1 and E2 test whether the deficit holds for standard ensembles.

### R-B. Exact TreeSHAP check
- **Method:** for RF and XGB, interventional TreeSHAP on the same 50-row background, applied after the (identity) imputer for complete-case rows.
- **Comparison:** NDF of the permutation-SHAP and TreeSHAP attributions, and the Spearman correlation between the two attributions.
- **Exclusion:** Mammographic is skipped, because one-hot encoding changes the input space.

### R-C. Monte-Carlo error
- **Scope:** fold 0 of each dataset, E3 and S*.
- **Method:** permutation SHAP repeated with 2 more seeds, and NDF repeated with 2 more seeds for the random orderings.
- **Reported:** within-patient SD of NDF and of ΔNDF.

### R-D. Re-analysis of existing outputs (no new fitting)
- **Tuned results by S* type:** Study 7 folds split by whether S* was LR or a non-linear model (SVM, RF, XGB).
  - Pooled ΔNDF per subgroup: inverse-variance pooling of fold-level means.
  - Mixed model: patient ΔNDF ~ S*-type + dataset fixed effects, with standard errors clustered by fold.
- **Meta-analysis extras (8 datasets):**
  - leave-one-out pooled estimates;
  - Hartung–Knapp–Sidik–Jonkman confidence intervals;
  - 95% prediction intervals.
- **Exclusions:** per dataset and model, the number excluded by G < 0.05, the distributions of G, and the number of complete-case exclusions.
- **Shrinkage check:** H2 recomputed on patients whose G values for E3 and S* differ by less than a factor of 1.5.
- **NDF margin:** the margin of practical relevance, 0.05, was pre-specified in PROTOCOL_FREEZE.md (line 41) and will now be stated in the paper.

### Status
All revision analyses are secondary or exploratory. The confirmatory set is unchanged: H1 and H2 of Studies 1, 2 and 5, and R of Study 6.

---
## Revision round 2 (21 Nov 2026; post hoc, after a second internal review)
**Status.** These analyses were added after the primary results were known. They are reported as post hoc, as are all "Revision analyses" above. The revision addendum was written before those analyses were run, but it was prompted by the review and came after the primary results.

**(a) Sensitivity to the G threshold.** H2 was recomputed with G ≥ 0.02 and G ≥ 0.10, alongside the protocol value of 0.05.

**(b) Tuned results split by type of S*.** These are pooled with a dataset random intercept (MixedLM on fold means), replacing the earlier analysis that treated folds as independent. The earlier fold-level t-intervals are superseded for reporting.

**(c) Deletion shares.** Top-1 is compared with random-1, and top-K with random-K.

**(d) Fold bookkeeping.** Tuned S* was non-linear in 21 of 40 folds. One of these, Parkinsons fold 2 (RF), had no patient with G ≥ 0.05 for both models, so 20 folds enter the subgroup analysis.

**Script.** scripts/16_revision2_stats.py, with output in results/revision/summary_revision2.json.

---
## Supervisor review (24 Nov 2026; post hoc re-analysis of saved outputs, no refitting)
**Script.** scripts/17_supervisor_revision.py, with output in results/revision/summary_supervisor.json.

**(a) Formal AUC equivalence test.** The margin of ±0.02 was fixed in PROTOCOL_FREEZE.md. Equivalence is tested with TOST: the 90% CI must lie within ±0.02, using the corrected repeated-CV variance. This is done per dataset and for the pooled estimate (DerSimonian–Laird and HKSJ).

**(b) Exclusion-free faithfulness.** ADF = NDF × G, the prediction drop beyond random deletion, in probability units. It needs no G threshold and is defined whenever G > 0.

**(c) Per-dataset exclusion table and dataset audit table.** These are now in the paper.

**Moved to the supplement.** The remedy table and the robustness table moved to the supplement (S11, S13) to keep the paper at 6 pages.

**Results.**
- **AUC equivalence:** shown for WDBC, Cleveland, Mammographic, CTG, CDC and pooled; not shown for DR Debrecen, ILPD and Parkinsons.
- **ADF, marginal operator:** pooled −0.033 (DL p = 0.004; HKSJ CI [−0.069, 0.002]).
  - 6 datasets significantly negative.
  - CTG significantly positive (+0.038).
