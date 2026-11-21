# Implementation status (26 Oct 2026)

**State:** implementation complete for the data, model, explanation and statistics layers;
technical pilots passed. **No confirmatory experiment has been run and no performance
result exists.** Confirmatory runs wait for supervisor approval (G1) and the protocol freeze
(G2) from the Stage 2.6 report.

## Completed

| Item | Evidence |
|---|---|
| Official data verified | All 3 zips match the SHA-256 fingerprints taken from archive.ics.uci.edu; every count matches the Stage 2.6 report (`results/audit/dataset_audit.md`) |
| Explicit deduplication | DR Debrecen: 5 exact duplicates removed (original rows 171, 627, 680, 708, 782), logged in `results/audit/dedup_log.json`; WDBC and Cleveland: 0 |
| Outer folds | 5 x 5 repeated stratified folds saved per dataset (`data/processed/*_outer_folds.json`) |
| Leakage-safe models | LR, SVM, KNN, RF, XGB pipelines; E1, E2, E3 (groups A = LR/SVM/KNN, B = RF/XGB); S\* by inner out-of-fold AUC with tie rule; E3 technical-failure rule. Tests show no outer-test row reaches any fit |
| NDF + SHAP | NDF exactly as in Stage 2.6 section 10. Known-answer tests K1-K5, ties, misleading rankings, d = 5/10/13/30 and a probability-scale model all pass. SHAP gates hold to about 1e-14 |
| LIME conversion (K2) | Passes (median Spearman >= 0.9 against true contributions) |
| Statistics | Corrected repeated k-fold t-test, Holm, Wilcoxon, stratified bootstrap; unit-tested |
| Test suite | 25 tests pass (`python -m pytest tests -q`) |
| Technical pilots | P0, P1, P3, P4, P5, P6 pass (`results/pilot/pilot_log.md`), labelled non-confirmatory |

## Pilot findings that matter for the protocol

1. **P1 gates (Cleveland):** additivity error <= 1.4e-16. Voting linearity 1.5e-15. E2 and E3
   decompositions 1.1e-14 and 6.2e-15. All within tolerance.
2. **P3 runtime (WDBC, cloud CPU):** 0.8 s per stacking explanation at `max_evals` = 500,
   so the original plan is projected at about 1.2 h per dataset. **The noise floor is exactly
   0.80 at both 500 and 1,000 evaluations**, right at the threshold, and doubling the budget
   did not improve it. Ties among collinear WDBC features probably limit top-5 stability.
   Consequence: local-robustness results on WDBC will be bounded by explainer noise and must
   be read against that floor. Recommendation: keep `max_evals` = 500 and report the floor.
3. **P4 (blinded, Cleveland, 20 patients x 8 models):** NDF separates SHAP ordering from
   random. Pooled median 0.82 vs. -0.04 for random attributions; 12 of 160 pairs excluded
   by the gap or K rule. No per-model values were computed or stored.
4. **P5:** missing values, one-hot encoding and type-aware perturbation produce no invalid
   values. LIME runs on complete cases.
5. **P6:** the E3 technical-failure rule did not trigger in any repeat-1 training fold of the
   three datasets.

## Implementation decisions where the report was silent (need confirmation)

| # | Decision | Reason |
|---|---|---|
| D1 | NDF tie-break: lower feature index first (stable sort) | Deterministic; NDF is unchanged when tied contributions are equal (tested) |
| D2 | NDF undefined if G <= 1e-12 (numerical tolerance), in addition to the 0.05 inclusion rule | Test K5 found G = -2.2e-16 for a constant model; without a tolerance a tiny positive G would explode the ratio |
| D3 | Random baseline uses the first k features of each of 20 orderings, for the same k = 1..K | The report says "20 seeded uniform random orderings" without specifying the prefix rule |
| D4 | Preprocessing: median imputation + scaling for numeric/binary/count features in LR/SVM/KNN; no scaling for trees; nominal = most-frequent imputation + one-hot | Follows report section 8; the report did not say binary features are scaled |
| D5 | Master seed 20261026 | Any fixed value; recorded |
| D6 | `SVC(probability=True)` kept | Specified by the report; deprecated in scikit-learn 1.9 (removal in 1.11). Works with the pinned version |

## Unresolved issues

1. **H2 statistical procedure.** The patient-level Wilcoxon over-rejects under fold
   clustering (simulation: up to 43% at nominal 5%). A fold-level H2 is proposed in
   `docs/STATISTICAL_AUDIT.md`. **Needs supervisor approval before the freeze.**
2. **WDBC noise floor** at the threshold (see pilot finding 2).
3. **Literature gate G0** (Arif 2024 and other paywalled texts) is handled in the separate
   literature chat; it affects wording only, not code.
4. Timing was measured in the cloud workspace; your laptop timing will differ (rerun
   `scripts/03_technical_pilots.py` there).

## Next action

Take the Stage 2.6 report plus `docs/STATISTICAL_AUDIT.md` to your supervisors (G1). Ask
them to approve the fold-level H2 and decisions D1-D6, then file the pre-registration (G2).
After that, the next code to write is `scripts/04_run_confirmatory.py`, which reuses the
modules unchanged.
