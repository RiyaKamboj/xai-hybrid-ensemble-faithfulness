# Are hybrid ensembles explained as faithfully as the single model they replace?

Code, protocols, cross-validation folds and per-fold results for the paper:

> R. Kamboj, K. Kaur, A. S. Bhuyan, "Are Hybrid Ensembles Explained as Faithfully as the Single Model They Replace? Evidence from Eight Healthcare Datasets" (under review).

The study compares a hybrid stacked-voting ensemble (E3) with the single model selected by internal cross-validation (S*) on eight public healthcare datasets. It looks at two outcomes:

- **Discrimination:** ROC-AUC.
- **Faithfulness of SHAP explanations:** normalised deletion faithfulness (NDF).

Every study was specified in a dated protocol (`docs/`) before it was run.

## Main results (all numbers are generated from `results/`)

| Outcome (E3 − S*, pooled over 8 datasets) | Estimate | p |
|---|---|---|
| ΔAUC | −0.0007 [−0.0035, 0.0021] | 0.65 |
| ΔNDF, marginal removal (DerSimonian–Laird) | −0.125 [−0.22, −0.03] | 0.008 |
| same, Hartung–Knapp–Sidik–Jonkman interval | [−0.27, 0.02] | 0.089 |
| ΔNDF, held-out repetitions 2–5 | −0.127 | 0.009 |
| ΔNDF, KernelSHAP | −0.130 | 0.009 |
| ΔNDF, tuned components | −0.118 | 0.005 |

Pre-specified hypotheses that were **not** supported are also reported:

- the correlation-group remedy as a general fix;
- the random-group placebo;
- non-additivity as the mechanism.

Each is recorded in `docs/PROTOCOL_STUDY2_STUDY3.md`.

## Repository layout

```
configs/study.yaml        design constants and SHA-256 fingerprints of the Study 1 files
src/xhe/                  library: data loading, leakage-safe models, CV, SHAP, NDF, statistics
scripts/                  numbered pipeline (see table below)
tests/                    pytest suite: NDF known-answer tests, SHAP checks, leakage, statistics
notebooks/                walkthrough.ipynb (one fold end to end); results_overview.ipynb
data/raw/                 put the downloaded zips here (see data/raw/README.md)
data/processed/           processed CSVs, saved outer folds, subject groups, CDC sample row indices
results/                  raw per-fold JSON outputs and summary JSONs/CSVs/figures for every study
docs/                     dated protocols, deviation logs, statistical audit
paper/                    LaTeX source + supplement; build.sh regenerates figures, every number and the supplement from results/
logs/                     run logs
```

## Studies and scripts

| Study | Protocol | Scripts | Output |
|---|---|---|---|
| Data check and folds (Study 1 data) | `docs/PROTOCOL_FREEZE.md` | `01_verify_data.py`, `02_prepare_data_and_folds.py` | `data/processed/`, `results/audit/` |
| Technical pilots (not confirmatory) | – | `03_technical_pilots.py` | `results/pilot/` |
| 1 – Primary comparison (WDBC, DR Debrecen, Cleveland) | `PROTOCOL_FREEZE.md` | `04_run_confirmatory.py`, `05_analyse.py` | `results/confirmatory/` |
| 2 – Replication (ILPD, Parkinsons, Mammographic, CTG) | `PROTOCOL_STUDY2_STUDY3.md` | `07a_prepare_study2.py`, `04_run_confirmatory.py --datasets ...`, `08_analyse_study2_3.py` | `results/study2/`, `results/study2_3/` |
| 3 – Simulation | `PROTOCOL_STUDY2_STUDY3.md` | `06_simulation.py`, `08_analyse_study2_3.py` | `results/simulation/` |
| 4 – KernelSHAP, faithfulness correlation, group remedy (exploratory) | `PROTOCOL_STUDY2_STUDY3.md` | `09_study4_remedy_metrics.py`, `10_analyse_study4.py` | `results/study4/` |
| 5 – Large-sample replication (CDC Diabetes, n = 5,000) | `PROTOCOL_STUDY2_STUDY3.md` | `07b_prepare_cdc.py`, `04_run_confirmatory.py --datasets cdc_diabetes` | `results/study5/` |
| 6 – Held-out confirmation of the remedy, placebo, mechanism | `PROTOCOL_STUDY2_STUDY3.md` | `11_study6_confirm_remedy.py`, `13_analyse_study6_7.py` | `results/study6/`, `results/study6_7/` |
| 7 – Tuned components | `PROTOCOL_STUDY2_STUDY3.md` | `12_study7_tuned.py`, `13_analyse_study6_7.py` | `results/study7/`, `results/study6_7/` |
| Revision round 2 – gap-threshold sensitivity, tuned results with dataset random intercept, deletion shares | `PROTOCOL_STUDY2_STUDY3.md` (Revision round 2) | `16_revision2_stats.py` | `results/revision/` |
| Supervisor review – AUC equivalence (TOST), exclusion-free ADF, exclusion table | `PROTOCOL_STUDY2_STUDY3.md` (Supervisor review) | `17_supervisor_revision.py` | `results/revision/summary_supervisor.json` |
| Revision – component NDF, HE vs RF/XGB, SV/ST, TreeSHAP, Monte-Carlo error, HKSJ/prediction intervals, exclusions | `PROTOCOL_STUDY2_STUDY3.md` (Revision analyses) | `14_component_faithfulness.py`, `15_revision_stats.py` | `results/components/`, `results/revision/` |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions: requirements.lock (Python 3.13)
pytest -q                                # 25 tests

# 1) download the raw zips into data/raw/ (data/raw/README.md); fingerprints are checked
python scripts/01_verify_data.py && python scripts/02_prepare_data_and_folds.py
python scripts/07a_prepare_study2.py && python scripts/07b_prepare_cdc.py

# 2) experiments (resumable: each fold writes its own JSON and is skipped if present)
python scripts/04_run_confirmatory.py --stage all --jobs 2                       # Study 1
python scripts/04_run_confirmatory.py --stage A --datasets ilpd,parkinsons,mammographic,ctg   # Study 2
python scripts/04_run_confirmatory.py --stage A --datasets cdc_diabetes          # Study 5
python scripts/06_simulation.py                                                  # Study 3
python scripts/09_study4_remedy_metrics.py && python scripts/11_study6_confirm_remedy.py && python scripts/12_study7_tuned.py
python scripts/14_component_faithfulness.py

# 3) analyses and paper
python scripts/05_analyse.py && python scripts/08_analyse_study2_3.py
python scripts/10_analyse_study4.py && python scripts/13_analyse_study6_7.py && python scripts/15_revision_stats.py
bash paper/build.sh
```

**Processed data and folds.** These are included, so the analyses and the paper can be regenerated from `results/` without re-running the experiments. Re-running everything takes roughly 15–20 hours on two CPU cores.

**Reproducibility.** Random seeds are fixed in `configs/study.yaml` and in the scripts. Results were produced on Linux with the pinned versions. A different BLAS or library version can change the last digits.

## Data

All datasets are public. See `data/raw/README.md` for the sources, licences and SHA-256 fingerprints. Raw files are not redistributed here.

## Licence

- **Code:** MIT (see `LICENSE`).
- **Datasets:** they keep their original licences.

## Citation

See `CITATION.cff`.
