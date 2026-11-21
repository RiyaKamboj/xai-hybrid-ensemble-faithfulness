"""Outer fold generation, saved once and reloaded by every later step (Stage 2.6 sec. 11)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from .data import CONFIG


def make_outer_folds(y, seed: int) -> list[dict]:
    cv = CONFIG["cv"]
    rskf = RepeatedStratifiedKFold(n_splits=cv["outer_splits"], n_repeats=cv["outer_repeats"],
                                   random_state=seed)
    folds = []
    for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        folds.append({"fold_id": i, "repeat": i // cv["outer_splits"],
                      "split": i % cv["outer_splits"], "train": tr.tolist(), "test": te.tolist()})
    return folds


def save_folds(folds, path: Path):
    Path(path).write_text(json.dumps(folds))


def load_folds(path: Path) -> list[dict]:
    folds = json.loads(Path(path).read_text())
    for f in folds:
        assert not set(f["train"]) & set(f["test"]), "train/test overlap in saved folds"
    return folds
