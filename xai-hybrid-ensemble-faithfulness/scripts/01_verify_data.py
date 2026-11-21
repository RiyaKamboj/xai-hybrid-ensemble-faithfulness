"""Step 1: verify the official UCI files and write a reproducible dataset audit.

Usage:  python scripts/01_verify_data.py
Inputs: data/raw/<official zip names> (see configs/study.yaml)
Output: results/audit/dataset_audit.json and results/audit/dataset_audit.md

This script does not modify data. It reports duplicates and missing values; removal and
imputation happen later, explicitly (duplicates) or inside model pipelines (imputation).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from xhe import data  # noqa: E402

OUT = ROOT / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

report = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
          "datasets": {}}
lines = ["# Dataset audit", "",
         f"Generated {report['generated_utc']} by `scripts/01_verify_data.py`. "
         "Fingerprints were recorded from archive.ics.uci.edu on 26 Oct 2026.", ""]
all_ok = True
for name, spec in data.CONFIG["datasets"].items():
    zpath = ROOT / "data" / "raw" / spec["zip_name"]
    if not zpath.exists():
        report["datasets"][name] = {"status": "MISSING FILE", "expected_path": str(zpath)}
        lines += [f"## {name}", "", f"**Missing file**: {zpath}", ""]
        all_ok = False
        continue
    ds = data.load(name, zpath, verify=False)
    aud = data.audit(ds)
    issues = data.reconcile(name, aud)
    hash_ok = ds.provenance["member_hash_match"] and ds.provenance["zip_hash_match"]
    status = "OK" if (hash_ok and not issues and not aud["type_checks_failed"]) else "DISCREPANCY"
    all_ok &= status == "OK"
    report["datasets"][name] = {"status": status, "provenance": ds.provenance, "audit": aud,
                                "discrepancies_vs_design_report": issues,
                                "feature_types": ds.feature_types}
    lines += [f"## {name} (UCI {spec['uci_id']}): {status}", "",
              f"- Zip SHA-256 match: {ds.provenance['zip_hash_match']}; member "
              f"`{spec['member']}` SHA-256 match: {ds.provenance['member_hash_match']}",
              f"- Rows {aud['rows']}, features {aud['features']}, positives {aud['positives']}, "
              f"negatives {aud['negatives']}",
              f"- Missing cells {aud['missing_cells']} {aud['missing_by_column'] or ''} "
              f"(rows with missing: {aud['rows_with_missing']})",
              f"- Exact duplicate rows {aud['exact_duplicate_rows']}; feature-only duplicates "
              f"{aud['feature_only_duplicates']}; duplicate groups with conflicting labels "
              f"{aud['duplicate_groups_with_conflicting_labels']}",
              f"- Feature types: {aud['feature_type_counts']}; type checks failed: "
              f"{aud['type_checks_failed'] or 'none'}; constant columns: "
              f"{aud['constant_columns'] or 'none'}",
              f"- Discrepancies vs. design report: {issues or 'none'}", ""]

report["all_ok"] = bool(all_ok)
(OUT / "dataset_audit.json").write_text(json.dumps(report, indent=2))
(OUT / "dataset_audit.md").write_text("\n".join(lines + [f"Overall: {'OK' if all_ok else 'CHECK'}"]))
print("\n".join(lines))
print("Overall:", "OK" if all_ok else "CHECK")
sys.exit(0 if all_ok else 1)
