"""Study 5 (large-sample replication): CDC Diabetes Health Indicators (BRFSS 2015, binary file; Kaggle release
of the dataset also published as UCI id 891). Decisions fixed 20 Nov 2026 before any model was fitted:
  * file diabetes_binary_health_indicators_BRFSS2015.csv; target Diabetes_binary (1 = prediabetes or diabetes);
  * exact duplicate rows KEPT (survey data with coarse items: identical rows are distinct respondents);
  * a stratified random subsample of n = 5,000 respondents (seed 20261120) is analysed with the same 5x5 CV,
    because SVM/KernelSHAP costs make the full 253,680 rows infeasible on the available compute; the subsample
    is the analysis population and is saved with its source row indices;
  * 0/1 items binary; BMI, MentHlth, PhysHlth continuous; GenHlth, Age, Education, Income count (ordinal codes).
"""
import hashlib, importlib.util, io, json, zipfile, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p2", ROOT / "scripts" / "07a_prepare_study2.py")
SEED, N = 20261120, 5000
zp = ROOT / "data" / "raw" / "cdc_diabetes_health_indicators.zip"
sha = hashlib.sha256(zp.read_bytes()).hexdigest()
d = pd.read_csv(io.BytesIO(zipfile.ZipFile(zp).read("diabetes_binary_health_indicators_BRFSS2015.csv")))
full_rows, full_pos = len(d), int(d.Diabetes_binary.sum())
from sklearn.model_selection import train_test_split
idx, _ = train_test_split(np.arange(len(d)), train_size=N, stratify=d.Diabetes_binary, random_state=SEED)
idx = np.sort(idx); s = d.iloc[idx].reset_index(drop=True)
y = s.pop("Diabetes_binary").astype(int)
cont, cnt = {"BMI", "MentHlth", "PhysHlth"}, {"GenHlth", "Age", "Education", "Income"}
types = {c: "continuous" if c in cont else "count" if c in cnt else "binary" for c in s.columns}
PROC = ROOT / "data" / "processed"
pd.Series(idx).to_csv(PROC / "cdc_diabetes_source_rows.csv", index=False, header=["source_row"])
# reuse finish() from 07a without re-running its module body: exec only the definitions
src = (ROOT / "scripts" / "07a_prepare_study2.py").read_text()
head = src.split("# ILPD")[0]
g = {"__file__": str(ROOT / "scripts" / "07a_prepare_study2.py")}; exec(head, g)
g["ftypes"].update(json.loads((PROC / "feature_types_study2.json").read_text()))
aud_p = ROOT / "results" / "audit" / "study2_dataset_audit.json"
g["audit"].update(json.loads(aud_p.read_text()))
g["finish"]("cdc_diabetes", s.astype(float), y, types, zip_sha=sha, file="diabetes_binary_health_indicators_BRFSS2015.csv", dedup=False,
            notes=[f"full file {full_rows} rows, {full_pos} positive", f"stratified subsample n={N}, seed {SEED}", "duplicates kept"])
(PROC / "feature_types_study2.json").write_text(json.dumps(g["ftypes"], indent=1))
aud_p.write_text(json.dumps(g["audit"], indent=1))
print(json.dumps(g["audit"]["cdc_diabetes"], indent=1)); print(s.corr(method="spearman").abs().where(lambda m: m < 1).max().sort_values().tail(5))
