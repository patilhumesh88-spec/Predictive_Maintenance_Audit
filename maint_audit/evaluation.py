"""
Evaluation utilities: LOFMO stress test, blocked CV comparison, residual analysis.

Extracted from scripts/compute_evidence.py so these functions are importable,
testable, and reusable from the dashboard and API.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .config import MODE_COLS, RANDOM_STATE, REPORTS_DIR, TARGET
from .features import ENGINEERED_FEATURES, build_features
from .models import make_rf
from .rules import rule_flags


# ---------------------------------------------------------------------------
# Leave-One-Failure-Mode-Out (LOFMO)
# ---------------------------------------------------------------------------

def lofmo_recall(
    df: pd.DataFrame,
    threshold: float = 0.515,
) -> dict[str, dict[str, Any]]:
    """Run Leave-One-Failure-Mode-Out stress test.

    For each failure mode in MODE_COLS (HDF, PWF, OSF), train a Random Forest
    on development rows that exclude that mode entirely, then evaluate recall
    exclusively on the held-out mode samples.

    Parameters
    ----------
    df:
        Full labelled DataFrame containing all MODE_COLS columns.
    threshold:
        Decision threshold for RF binary classification.

    Returns
    -------
    dict mapping mode name → result dict with keys:
        - ``ml_recall``: float
        - ``rules_recall``: float
        - ``n_test``: int
        - ``ml_tp``: int
        - ``rules_tp``: int
    """
    results: dict[str, dict[str, Any]] = {}
    for mode in ["HDF", "PWF", "OSF"]:
        if mode not in df.columns:
            continue
        # Test set: rows where this mode is the failure cause
        test_mask = df[mode] == 1
        train_mask = ~test_mask
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(test_df) == 0:
            continue

        # Train RF without seeing this mode
        X_train = build_features(train_df)[ENGINEERED_FEATURES]
        y_train = train_df[TARGET].values
        model = make_rf().fit(X_train, y_train)

        # Score test set
        X_test = build_features(test_df)[ENGINEERED_FEATURES]
        ml_proba = model.predict_proba(X_test)[:, 1]
        ml_pred = (ml_proba >= threshold).astype(int)

        # Rule predictions
        rule_pred = rule_flags(test_df)["rule_any"].astype(int).values

        n_test = int(len(test_df))
        ml_tp = int(ml_pred.sum())
        rules_tp = int(rule_pred.sum())

        results[mode] = {
            "ml_recall": round(ml_tp / n_test, 4) if n_test > 0 else 0.0,
            "rules_recall": round(rules_tp / n_test, 4) if n_test > 0 else 0.0,
            "n_test": n_test,
            "ml_tp": ml_tp,
            "rules_tp": rules_tp,
        }
    return results


# ---------------------------------------------------------------------------
# Load pre-computed reports (used by dashboard for fast rendering)
# ---------------------------------------------------------------------------

def load_lofmo_report(reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    """Load pre-computed LOFMO results from reports/leave_one_mode_out.json."""
    path = reports_dir / "leave_one_mode_out.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_cost_threshold_report(reports_dir: Path = REPORTS_DIR) -> list[dict[str, Any]]:
    """Load pre-computed cost-threshold sweep from reports/cost_threshold.json."""
    path = reports_dir / "cost_threshold.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("benchmark_summary", [])


def load_metrics_report(reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    """Load pre-computed metrics from reports/metrics.json."""
    path = reports_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_lofmo_dataframe(lofmo_report: dict[str, Any]) -> pd.DataFrame:
    """Convert LOFMO JSON report into a tidy DataFrame for plotting.

    Parameters
    ----------
    lofmo_report:
        Output of ``load_lofmo_report()``

    Returns
    -------
    DataFrame with columns: mode, system, recall
    """
    rows = []
    modes_data = lofmo_report.get("tested_modes", {})
    for mode, data in modes_data.items():
        rows.append({
            "Held-out Failure Mode": mode,
            "System": "Physics Rules",
            "Recall (%)": round(data["physics_rules"]["rule_specific_recall"] * 100, 1),
        })
        rows.append({
            "Held-out Failure Mode": mode,
            "System": "Random Forest ML",
            "Recall (%)": round(data["ml_recall_at_0515"] * 100, 1),
        })
    return pd.DataFrame(rows)


def build_cost_dataframe(cost_report: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert cost-threshold JSON into a tidy DataFrame for plotting.

    Parameters
    ----------
    cost_report:
        Output of ``load_cost_threshold_report()``

    Returns
    -------
    DataFrame with columns: cost_ratio_label, threshold, recall, cost_savings_pct
    """
    rows = []
    for entry in cost_report:
        metrics = entry.get("optimal_metrics", {})
        rows.append({
            "Cost Ratio (Miss:FA)": entry["cost_ratio"],
            "Optimal Threshold": entry["optimal_threshold"],
            "Recall (%)": round(metrics.get("recall", 0) * 100, 1),
            "Cost Savings vs 0.515 (%)": entry.get("percentage_savings_vs_0515", 0.0),
        })
    return pd.DataFrame(rows)


def build_cv_comparison_dataframe(metrics_report: dict[str, Any]) -> pd.DataFrame:
    """Build blocked vs. shuffled CV comparison DataFrame for plotting.

    Parameters
    ----------
    metrics_report:
        Output of ``load_metrics_report()``

    Returns
    -------
    DataFrame with columns: feature_set, cv_type, ap
    """
    rows = []
    shuffled_ap = metrics_report.get("cv_stratified", {}).get("RandomForest", {}).get("pr_auc")
    blocked_ap = metrics_report.get("cv_block", {}).get("RandomForest", {}).get("pr_auc")
    if shuffled_ap is not None:
        rows.append({"Feature Set": "Physics-Informed Features", "CV Type": "Shuffled (Standard)", "Average Precision": shuffled_ap})
    if blocked_ap is not None:
        rows.append({"Feature Set": "Physics-Informed Features", "CV Type": "Blocked (Distribution Shift)", "Average Precision": blocked_ap})
    return pd.DataFrame(rows)
