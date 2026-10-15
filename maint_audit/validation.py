"""Splitters, metrics and out-of-fold prediction helpers.

Two CV schemes on purpose:
  * stratified_5fold : shuffled, the usual (optimistic here: air/process temperature are random
                       walks, so neighbouring rows are near-duplicates in those columns).
  * block_5fold      : contiguous blocks of rows held out (training uses the remaining blocks,
                       including later rows). Honest name: BLOCKED CV, not forward-chaining.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold

from .config import RANDOM_STATE


def get_splitter(name: str, n_splits: int = 5):
    if name == "stratified_5fold":
        return StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    if name == "block_5fold":
        return KFold(n_splits, shuffle=False)
    raise ValueError(f"unknown scheme {name}")


def oof_scores(model, X: pd.DataFrame, y: np.ndarray, splitter) -> np.ndarray:
    """Out-of-fold P(failure) for every row."""
    out = np.zeros(len(y), dtype=float)
    for tr, te in splitter.split(X, y):
        m = clone(model).fit(X.iloc[tr], y[tr])
        out[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out


def metrics(y, score, thr: float = 0.5) -> dict:
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y)) == 2
    return {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4) if has_both else float("nan"),
        "pr_auc": round(float(average_precision_score(y, score)), 4) if has_both else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
