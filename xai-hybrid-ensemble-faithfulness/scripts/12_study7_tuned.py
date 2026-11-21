"""Study 7 (pre-specified 21 Nov 2026): robustness of H1/H2 to hyperparameter tuning.
Each component is a GridSearchCV (3-fold stratified, ROC-AUC) fitted only on the data it receives, so tuning is
nested inside the inner out-of-fold loop and inside the outer training fold. Repetition-1 folds, 8 datasets.
Usage: python scripts/12_study7_tuned.py [--datasets ...] [--jobs 2]
"""
import argparse, importlib.util, json, os
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
ALL = ["wdbc", "dr_debrecen", "cleveland", "ilpd", "parkinsons", "mammographic", "ctg", "cdc_diabetes"]
GRIDS = {"lr": {"clf__C": [0.01, 0.1, 1, 10]}, "svm": {"clf__C": [0.3, 1, 3]}, "knn": {"clf__n_neighbors": [5, 15, 31]},
         "rf": {"clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", 0.5]},
         "xgb": {"clf__max_depth": [2, 4, 6], "clf__learning_rate": [0.05, 0.3]}}


def task(name, fid, outdir):
    path = Path(outdir) / f"{name}_fold{fid:02d}.json"
    if path.exists():
        return str(path)
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    s = importlib.util.spec_from_file_location("run", ROOT / "scripts" / "04_run_confirmatory.py")
    run = importlib.util.module_from_spec(s); s.loader.exec_module(run)
    if not hasattr(run.models, "_orig_make_component"):   # loky workers are reused: patch only once
        run.models._orig_make_component = run.models.make_component
    base = run.models._orig_make_component
    chosen = []

    def tuned(c, ft, cols, seed, record_fits=False):
        return GridSearchCV(base(c, ft, cols, seed), GRIDS[c], scoring="roc_auc", n_jobs=1, refit=True,
                            cv=StratifiedKFold(3, shuffle=True, random_state=seed))
    run.models.make_component = tuned
    run.check_equivalence = lambda *a, **k: {}
    orig = run.FoldModels.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        chosen.append({c: {p: (v if isinstance(v, (int, float, str)) else str(v)) for p, v in self.comp[c].best_params_.items()} for c in self.comp})
    run.FoldModels.__init__ = init
    X, y, folds, ft = run.load_processed(name)
    fold = [f for f in folds if f["fold_id"] == fid][0]
    p = run.task_fold(name, fold, False, Path(outdir))
    d = json.loads(Path(p).read_text()); d["best_params"] = chosen[-1] if chosen else None
    Path(p).write_text(json.dumps(d, default=float))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=ALL)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    od = ROOT / "results" / "study7" / "raw"; od.mkdir(parents=True, exist_ok=True)
    jobs = [(d, k) for k in range(5) for d in a.datasets]
    for p in Parallel(n_jobs=a.jobs, verbose=5)(delayed(task)(d, k, str(od)) for d, k in jobs):
        print(p, flush=True)
