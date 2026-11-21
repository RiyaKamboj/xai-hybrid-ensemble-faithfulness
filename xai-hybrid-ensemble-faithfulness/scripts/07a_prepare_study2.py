"""Study 2 data preparation (docs/PROTOCOL_STUDY2_STUDY3.md): parse the four UCI zips,
record SHA-256 fingerprints, apply the pre-specified decisions, save processed data and folds."""
import hashlib, io, json, re, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 20261120
audit, ftypes = {}, {}


def member(zipname, pattern):
    z = zipfile.ZipFile(RAW / zipname)
    name = [n for n in z.namelist() if re.search(pattern, n)][0]
    return z.read(name), name, hashlib.sha256((RAW / zipname).read_bytes()).hexdigest()


def finish(name, X, y, types, groups=None, notes=None, zip_sha=None, file=None, dedup=True):
    dup = X.assign(_y=y.values).duplicated()
    n_dup = int(dup.sum())
    if groups is None and n_dup and dedup:
        X, y = X[~dup].reset_index(drop=True), y[~dup].reset_index(drop=True)
    df = X.copy(); df["target"] = y.values
    df.to_csv(PROC / f"{name}.csv", index=False)
    folds = []
    if groups is None:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
        for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
            folds.append({"fold_id": i, "repeat": i // 5, "split": i % 5, "train": tr.tolist(), "test": te.tolist()})
    else:
        g = np.asarray(groups)
        for r in range(5):
            sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
            for s, (tr, te) in enumerate(sgk.split(np.zeros(len(y)), y, g)):
                assert not set(g[tr]) & set(g[te])
                folds.append({"fold_id": r * 5 + s, "repeat": r, "split": s, "train": tr.tolist(), "test": te.tolist()})
        pd.Series(g).to_csv(PROC / f"{name}_groups.csv", index=False, header=["group"])
    (PROC / f"{name}_outer_folds.json").write_text(json.dumps(folds))
    ftypes[name] = types
    audit[name] = {"file": file, "zip_sha256": zip_sha, "rows": int(len(X)), "features": int(X.shape[1]),
                   "positives": int(y.sum()), "negatives": int((1 - y).sum()), "missing_cells": int(X.isna().sum().sum()),
                   "exact_duplicates_found": n_dup, "duplicates_removed": n_dup if (groups is None and dedup) else 0,
                   "grouped_folds": groups is not None, "n_groups": int(len(set(groups))) if groups is not None else None,
                   "notes": notes or []}


# ILPD
b, f, h = member("ilpd+indian+liver+patient+dataset.zip", r"\.csv$")
cols = ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp", "alb", "ag_ratio", "selector"]
d = pd.read_csv(io.BytesIO(b), header=None, names=cols)
d["gender"] = (d["gender"].str.strip().str.lower() == "male").astype(float)
y = (d.pop("selector") == 1).astype(int)
finish("ilpd", d.astype(float), y, {c: ("binary" if c == "gender" else "continuous") for c in d.columns}, zip_sha=h, file=f)

# Parkinsons
b, f, h = member("parkinsons.zip", r"parkinsons\.data$")
d = pd.read_csv(io.BytesIO(b))
groups = d["name"].str.extract(r"_(S\d+)_")[0]
y = d.pop("status").astype(int); d = d.drop(columns=["name"])
d.columns = [re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_") for c in d.columns]
finish("parkinsons", d.astype(float), y, {c: "continuous" for c in d.columns}, groups=groups.values, zip_sha=h, file=f,
       notes=["recording-level rows; folds grouped by subject"])

# Mammographic Mass
b, f, h = member("mammographic+mass.zip", r"\.data$")
d = pd.read_csv(io.BytesIO(b), header=None, names=["birads", "age", "shape", "margin", "density", "severity"], na_values="?")
y = d.pop("severity").astype(int); d = d.drop(columns=["birads"])
finish("mammographic", d.astype(float), y, {"age": "continuous", "shape": "nominal", "margin": "nominal", "density": "count"},
       zip_sha=h, file=f, dedup=False, notes=["BI-RADS excluded (non-predictive per UCI documentation)",
       "identical rows kept: with 4 coarse features distinct patients share values (decision logged before model fitting)"])

# Cardiotocography
z = zipfile.ZipFile(RAW / "cardiotocography.zip"); xls = [n for n in z.namelist() if n.endswith(".xls")][0]
h = hashlib.sha256((RAW / "cardiotocography.zip").read_bytes()).hexdigest()
d = pd.read_excel(io.BytesIO(z.read(xls)), sheet_name="Data", header=1)
d = d[d["NSP"].notna()].copy()
feat = ["LB", "AC.1", "FM.1", "UC.1", "DL.1", "DS.1", "DP.1", "ASTV", "MSTV", "ALTV", "MLTV", "Width", "Min", "Max",
        "Nmax", "Nzeros", "Mode", "Mean", "Median", "Variance", "Tendency"]
X = d[feat].astype(float).reset_index(drop=True); X.columns = [c.replace(".1", "") for c in feat]
y = d["NSP"].isin([2, 3]).astype(int).reset_index(drop=True)
types = {c: "continuous" for c in X.columns}; types.update({"Nmax": "count", "Nzeros": "count", "Tendency": "count"})
finish("ctg", X, y, types, zip_sha=h, file=xls, notes=["21 measured features (per-second rates); NSP 2/3 positive"])

(PROC / "feature_types_study2.json").write_text(json.dumps(ftypes, indent=1))
(ROOT / "results" / "audit" / "study2_dataset_audit.json").write_text(json.dumps(audit, indent=1))
print(json.dumps(audit, indent=1))
