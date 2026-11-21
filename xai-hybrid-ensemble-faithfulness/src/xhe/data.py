"""Loading, fingerprint verification and auditing of the three official UCI datasets.

Rules (Stage 2.6 section 9 and the implementation brief):
* Files are read from the official UCI zip archives and checked against SHA-256
  fingerprints recorded from archive.ics.uci.edu on 26 Oct 2026.
* Nothing is silently changed. Missing values are kept as NaN (imputation happens
  only inside model pipelines). Duplicate removal is a separate, explicit, logged step
  (`drop_exact_duplicates`) applied by analysis scripts, never by the loader.
* Target definitions follow the design report: WDBC M=1; DR class 1=1; Cleveland num>=1 -> 1.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs" / "study.yaml").read_text())

WDBC_TRAITS = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
               "concavity", "concave_points", "symmetry", "fractal_dimension"]
WDBC_COLUMNS = (["id", "diagnosis"]
                + [f"{t}_mean" for t in WDBC_TRAITS]
                + [f"{t}_se" for t in WDBC_TRAITS]
                + [f"{t}_worst" for t in WDBC_TRAITS])

# Names follow the UCI variable descriptions for dataset 329 (indices 0-18).
DR_COLUMNS = (["quality", "pre_screening"]
              + [f"ma_{i}" for i in range(1, 7)]
              + [f"exudate_{i}" for i in range(1, 9)]
              + ["macula_opticdisc_distance", "opticdisc_diameter", "am_fm", "label"])

CLEVELAND_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                     "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Feature types drive preprocessing and the type-aware perturbation (Stage 2.6 sec. 8, 10).
#   continuous: scaled; Gaussian perturbation
#   count:      treated as numeric in models; perturbation is rounded
#   binary:     numeric 0/1 in models; never perturbed
#   nominal:    one-hot in models; never perturbed
FEATURE_TYPES = {
    "wdbc": {c: "continuous" for c in WDBC_COLUMNS[2:]},
    "dr_debrecen": {
        "quality": "binary", "pre_screening": "binary", "am_fm": "binary",
        **{f"ma_{i}": "count" for i in range(1, 7)},
        **{f"exudate_{i}": "continuous" for i in range(1, 9)},
        "macula_opticdisc_distance": "continuous", "opticdisc_diameter": "continuous",
    },
    "cleveland": {
        "age": "continuous", "trestbps": "continuous", "chol": "continuous",
        "thalach": "continuous", "oldpeak": "continuous",
        "sex": "binary", "fbs": "binary", "exang": "binary",
        "cp": "nominal", "restecg": "nominal", "slope": "nominal", "thal": "nominal",
        "ca": "count",
    },
}


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    feature_types: dict
    provenance: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_member(zip_path: Path, name: str, verify: bool = True) -> tuple[bytes, dict]:
    """Return the member bytes and a provenance record; raise if fingerprints differ."""
    spec = CONFIG["datasets"][name]
    zip_bytes = Path(zip_path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        member = zf.read(spec["member"])
    prov = {
        "dataset": name,
        "uci_id": spec["uci_id"],
        "zip_file": str(zip_path),
        "zip_sha256": sha256_bytes(zip_bytes),
        "member": spec["member"],
        "member_sha256": sha256_bytes(member),
        "expected_member_sha256": spec["member_sha256"],
        "expected_zip_sha256": spec["zip_sha256"],
    }
    prov["member_hash_match"] = prov["member_sha256"] == spec["member_sha256"]
    prov["zip_hash_match"] = prov["zip_sha256"] == spec["zip_sha256"]
    if verify and not prov["member_hash_match"]:
        raise ValueError(f"{name}: member SHA-256 {prov['member_sha256']} does not match the "
                         f"official fingerprint {spec['member_sha256']}. Do not use this file.")
    return member, prov


def parse_wdbc(raw: bytes) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(io.BytesIO(raw), header=None, names=WDBC_COLUMNS)
    y = df["diagnosis"].map({"M": 1, "B": 0})
    if y.isna().any():
        raise ValueError("WDBC: unexpected diagnosis codes")
    X = df.drop(columns=["id", "diagnosis"]).astype(float)
    return X, y.astype(int).rename("target")


def parse_dr(raw: bytes) -> tuple[pd.DataFrame, pd.Series]:
    text = raw.decode("utf-8")
    data_part = text.split("@data", 1)[1] if "@data" in text else text.split("@DATA", 1)[1]
    rows = [ln.strip() for ln in data_part.splitlines() if ln.strip() and not ln.startswith("%")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=DR_COLUMNS)
    y = df.pop("label").astype(int).rename("target")
    X = df.astype(float)
    return X, y


def parse_cleveland(raw: bytes) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(io.BytesIO(raw), header=None, names=CLEVELAND_COLUMNS, na_values=["?"])
    y = (df.pop("num") >= 1).astype(int).rename("target")   # UCI page: presence = num 1-4
    X = df.astype(float)
    return X, y


PARSERS = {"wdbc": parse_wdbc, "dr_debrecen": parse_dr, "cleveland": parse_cleveland}


def load(name: str, zip_path: Path, verify: bool = True) -> Dataset:
    raw, prov = read_member(Path(zip_path), name, verify=verify)
    X, y = PARSERS[name](raw)
    ds = Dataset(name=name, X=X, y=y, feature_types=FEATURE_TYPES[name], provenance=prov)
    missing_types = set(X.columns) - set(ds.feature_types)
    if missing_types:
        raise ValueError(f"{name}: no declared type for {sorted(missing_types)}")
    return ds


def audit(ds: Dataset) -> dict:
    """Facts about the file as parsed. No modification of the data."""
    X, y = ds.X, ds.y
    full = pd.concat([X, y], axis=1)
    exact_dups = int(full.duplicated().sum())
    feat_dups = int(X.duplicated().sum())
    conflicting = 0
    if feat_dups:
        grp = full.groupby(list(X.columns), dropna=False)["target"].nunique()
        conflicting = int((grp > 1).sum())
    type_checks = {}
    for col, t in ds.feature_types.items():
        vals = X[col].dropna()
        if t == "binary":
            type_checks[col] = bool(set(vals.unique()) <= {0.0, 1.0})
        elif t == "count":
            type_checks[col] = bool(np.allclose(vals, np.round(vals)) and (vals >= 0).all())
        elif t == "nominal":
            type_checks[col] = bool(np.allclose(vals, np.round(vals)))
        else:
            type_checks[col] = bool(np.isfinite(vals).all())
    return {
        "dataset": ds.name,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "missing_cells": int(X.isna().sum().sum()),
        "missing_by_column": {c: int(v) for c, v in X.isna().sum().items() if v},
        "rows_with_missing": int(X.isna().any(axis=1).sum()),
        "exact_duplicate_rows": exact_dups,
        "feature_only_duplicates": feat_dups,
        "duplicate_groups_with_conflicting_labels": conflicting,
        "feature_type_counts": pd.Series(ds.feature_types).value_counts().to_dict(),
        "type_checks_failed": [c for c, ok in type_checks.items() if not ok],
        "constant_columns": [c for c in X.columns if X[c].nunique(dropna=True) <= 1],
    }


def reconcile(name: str, observed: dict) -> list[str]:
    """Compare an audit with the counts recorded in the design report. Returns discrepancies."""
    exp = CONFIG["datasets"][name]["expected"]
    issues = []
    for key, val in exp.items():
        obs = observed["features" if key == "features" else key]
        if obs != val:
            issues.append(f"{key}: expected {val}, observed {obs}")
    return issues


def drop_exact_duplicates(ds: Dataset) -> tuple[Dataset, dict]:
    """Explicit, logged removal of exact duplicate rows (features + target), keeping the first.

    Stage 2.6 section 8 specifies this step before splitting (DR Debrecen: 5 rows)."""
    full = pd.concat([ds.X, ds.y], axis=1)
    mask = full.duplicated(keep="first")
    removed_index = [int(i) for i in full.index[mask]]
    new = Dataset(ds.name, ds.X.loc[~mask].reset_index(drop=True),
                  ds.y.loc[~mask].reset_index(drop=True), ds.feature_types,
                  dict(ds.provenance), list(ds.notes))
    log = {"dataset": ds.name, "rows_before": int(len(full)), "rows_removed": int(mask.sum()),
           "removed_original_row_indices": removed_index, "rows_after": int(len(new.X))}
    new.notes.append(f"exact duplicates removed: {log['rows_removed']}")
    return new, log


def sklearn_wdbc_for_technical_pilots() -> Dataset:
    """scikit-learn's bundled WDBC copy, for technical pilots ONLY when the official
    zip is unavailable. Not hash-verifiable; never used for confirmatory analyses.
    Note: scikit-learn codes malignant = 0, so the target is flipped to malignant = 1."""
    from sklearn.datasets import load_breast_cancer
    b = load_breast_cancer(as_frame=True)
    X = b.data.copy()
    X.columns = WDBC_COLUMNS[2:]
    y = (1 - b.target).astype(int).rename("target")
    return Dataset("wdbc", X, y, FEATURE_TYPES["wdbc"],
                   {"source": "sklearn.datasets.load_breast_cancer", "verified": False},
                   ["TECHNICAL-PILOT COPY: not the hash-verified UCI file"])
