"""Step 2: explicit, logged preparation and fold generation (no modelling).

* Exact duplicate rows are removed (Stage 2.6 sec. 8; DR Debrecen expects 5). Logged.
* Missing values are NOT imputed here (imputation happens inside model pipelines).
* Outer folds (5 x 5 repeated stratified) are generated once and saved.

Usage: python scripts/02_prepare_data_and_folds.py
Outputs: data/processed/<name>.csv, results/audit/dedup_log.json,
         data/processed/<name>_outer_folds.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xhe import cv, data  # noqa: E402

seed = data.CONFIG["master_seed"]
logs = {}
for name, spec in data.CONFIG["datasets"].items():
    ds = data.load(name, ROOT / "data" / "raw" / spec["zip_name"], verify=True)
    ds, log = data.drop_exact_duplicates(ds)
    logs[name] = log
    out = pd.concat([ds.X, ds.y], axis=1)
    out.to_csv(ROOT / "data" / "processed" / f"{name}.csv", index=False)
    folds = cv.make_outer_folds(ds.y.values, seed)
    cv.save_folds(folds, ROOT / "data" / "processed" / f"{name}_outer_folds.json")
    print(f"{name}: {log['rows_before']} -> {log['rows_after']} rows "
          f"(removed {log['rows_removed']}: original indices {log['removed_original_row_indices']}); "
          f"{len(folds)} outer folds saved")
(ROOT / "results" / "audit" / "dedup_log.json").write_text(json.dumps(logs, indent=2))
