# Technical pilot log

**TECHNICAL PILOT - NOT CONFIRMATORY - NO PERFORMANCE RESULTS**

Started 2026-10-26T14:14:41+00:00, finished 2026-10-26T14:20:32+00:00; Python 3.13.17; Linux-6.18.44-fc-v153-x86_64-with-glibc2.39

## P0_data: PASS

```
{
  "wdbc": {
    "rows_after_dedup": 569,
    "expected_rows_after_dedup": 569,
    "missing_cells": 0,
    "n_outer_folds": 25,
    "pass": true
  },
  "dr_debrecen": {
    "rows_after_dedup": 1146,
    "expected_rows_after_dedup": 1146,
    "missing_cells": 0,
    "n_outer_folds": 25,
    "pass": true
  },
  "cleveland": {
    "rows_after_dedup": 303,
    "expected_rows_after_dedup": 303,
    "missing_cells": 6,
    "n_outer_folds": 25,
    "pass": true
  }
}
```

## P1_gates_cleveland: PASS

```
{
  "e1_identity": true,
  "e1_additivity_max_err": 1.1102230246251565e-16,
  "e1_reproducible": true,
  "e2_identity": true,
  "e2_additivity_max_err": 1.3183898417423734e-16,
  "e2_reproducible": true,
  "e3_identity": true,
  "e3_additivity_max_err": 1.3877787807814457e-16,
  "e3_reproducible": true,
  "voting_linearity_max_err": 1.4675760606763788e-15,
  "e2_decomposition_max_err": 1.124100812432971e-14,
  "e3_decomposition_max_err": 6.2103100439969694e-15,
  "exact_shap_seconds": 16.3
}
```

## P5_mixed_types_cleveland: PASS

```
{
  "invalid_perturbed_values": 0,
  "invalid_replacement_values": 0,
  "pipeline_handles_missing": true,
  "lime_on_complete_case_ok": true,
  "note": "replacement values with NaN come from background rows with missing ca/thal; they are imputed inside the pipeline, as for any patient"
}
```

## P3_runtime_noise_wdbc: PASS

```
{
  "e2_maxevals500": {
    "seconds_per_explanation": 0.83,
    "median_seed_top5_overlap": 0.8,
    "effective_max_evals": 500
  },
  "e2_maxevals1000": {
    "seconds_per_explanation": 1.73,
    "median_seed_top5_overlap": 0.8,
    "effective_max_evals": 1000
  },
  "knn_maxevals500": {
    "seconds_per_explanation": 0.11,
    "median_seed_top5_overlap": 0.8,
    "effective_max_evals": 500
  },
  "knn_maxevals1000": {
    "seconds_per_explanation": 0.23,
    "median_seed_top5_overlap": 0.8,
    "effective_max_evals": 1000
  },
  "projected_hours_per_dataset_at_max_evals_500": 1.1,
  "noise_floor_criterion_met_at_500": true
}
```

## P4_ndf_separation_blinded: PASS

```
{
  "n_patient_model_pairs_included": 148,
  "n_excluded_gap_or_K": 12,
  "pooled_median_ndf_shap_order": 0.81518064557934,
  "pooled_median_ndf_random_attribution": -0.03596845067264697,
  "share_shap_ndf_above_0.02": 1.0
}
```

## P6_e3_technical_rule: PASS

```
{
  "wdbc_fold0": {
    "trigger": false,
    "reasons": []
  },
  "wdbc_fold1": {
    "trigger": false,
    "reasons": []
  },
  "wdbc_fold2": {
    "trigger": false,
    "reasons": []
  },
  "wdbc_fold3": {
    "trigger": false,
    "reasons": []
  },
  "wdbc_fold4": {
    "trigger": false,
    "reasons": []
  },
  "dr_debrecen_fold0": {
    "trigger": false,
    "reasons": []
  },
  "dr_debrecen_fold1": {
    "trigger": false,
    "reasons": []
  },
  "dr_debrecen_fold2": {
    "trigger": false,
    "reasons": []
  },
  "dr_debrecen_fold3": {
    "trigger": false,
    "reasons": []
  },
  "dr_debrecen_fold4": {
    "trigger": false,
    "reasons": []
  },
  "cleveland_fold0": {
    "trigger": false,
    "reasons": []
  },
  "cleveland_fold1": {
    "trigger": false,
    "reasons": []
  },
  "cleveland_fold2": {
    "trigger": false,
    "reasons": []
  },
  "cleveland_fold3": {
    "trigger": false,
    "reasons": []
  },
  "cleveland_fold4": {
    "trigger": false,
    "reasons": []
  }
}
```
