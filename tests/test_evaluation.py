"""Tests for the evaluation utilities module (LOFMO helpers, report loaders)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from maint_audit.evaluation import (
    build_cost_dataframe,
    build_cv_comparison_dataframe,
    build_lofmo_dataframe,
    load_cost_threshold_report,
    load_lofmo_report,
    load_metrics_report,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal mock report dicts
# ---------------------------------------------------------------------------

MOCK_LOFMO = {
    "tested_modes": {
        "HDF": {
            "physics_rules": {"rule_specific_recall": 1.0, "rule_specific_tp": 115, "rule_any_recall": 1.0, "rule_any_tp": 115},
            "ml_recall_at_0515": 0.0783,
        },
        "PWF": {
            "physics_rules": {"rule_specific_recall": 1.0, "rule_specific_tp": 85, "rule_any_recall": 1.0, "rule_any_tp": 85},
            "ml_recall_at_0515": 0.1412,
        },
    }
}

MOCK_COST = [
    {"cost_ratio": "1:1", "optimal_threshold": 0.515,
     "optimal_metrics": {"recall": 0.84}, "percentage_savings_vs_0515": 0.0},
    {"cost_ratio": "10:1", "optimal_threshold": 0.30,
     "optimal_metrics": {"recall": 0.873}, "percentage_savings_vs_0515": 13.8},
]

MOCK_METRICS = {
    "cv_stratified": {"RandomForest": {"pr_auc": 0.906}},
    "cv_block": {"RandomForest": {"pr_auc": 0.847}},
}


# ---------------------------------------------------------------------------
# build_lofmo_dataframe
# ---------------------------------------------------------------------------

def test_build_lofmo_dataframe_shape():
    df = build_lofmo_dataframe(MOCK_LOFMO)
    # 2 modes x 2 systems = 4 rows
    assert len(df) == 4
    assert "Held-out Failure Mode" in df.columns
    assert "System" in df.columns
    assert "Recall (%)" in df.columns


def test_build_lofmo_dataframe_physics_recall_is_100():
    df = build_lofmo_dataframe(MOCK_LOFMO)
    physics_rows = df[df["System"] == "Physics Rules"]
    assert (physics_rows["Recall (%)"] == 100.0).all()


def test_build_lofmo_dataframe_ml_recall_low():
    df = build_lofmo_dataframe(MOCK_LOFMO)
    ml_hdf = df[(df["System"] == "Random Forest ML") & (df["Held-out Failure Mode"] == "HDF")]
    assert ml_hdf["Recall (%)"].iloc[0] == pytest.approx(7.83, abs=0.1)


def test_build_lofmo_dataframe_empty_input():
    df = build_lofmo_dataframe({})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# build_cost_dataframe
# ---------------------------------------------------------------------------

def test_build_cost_dataframe_shape():
    df = build_cost_dataframe(MOCK_COST)
    assert len(df) == 2
    assert "Optimal Threshold" in df.columns
    assert "Recall (%)" in df.columns
    assert "Cost Savings vs 0.515 (%)" in df.columns


def test_build_cost_dataframe_values():
    df = build_cost_dataframe(MOCK_COST)
    row_10_1 = df[df["Cost Ratio (Miss:FA)"] == "10:1"].iloc[0]
    assert row_10_1["Optimal Threshold"] == 0.30
    assert row_10_1["Cost Savings vs 0.515 (%)"] == pytest.approx(13.8)


# ---------------------------------------------------------------------------
# build_cv_comparison_dataframe
# ---------------------------------------------------------------------------

def test_build_cv_comparison_dataframe_shape():
    df = build_cv_comparison_dataframe(MOCK_METRICS)
    assert len(df) == 2
    assert "CV Type" in df.columns
    assert "Average Precision" in df.columns


def test_build_cv_comparison_dataframe_values():
    df = build_cv_comparison_dataframe(MOCK_METRICS)
    shuffled = df[df["CV Type"] == "Shuffled (Standard)"]["Average Precision"].iloc[0]
    blocked = df[df["CV Type"] == "Blocked (Distribution Shift)"]["Average Precision"].iloc[0]
    assert shuffled == pytest.approx(0.906)
    assert blocked == pytest.approx(0.847)
    assert shuffled > blocked  # shuffled always higher than blocked


# ---------------------------------------------------------------------------
# File loaders with real reports/ directory
# ---------------------------------------------------------------------------

def test_load_lofmo_report_from_file():
    data = load_lofmo_report()
    # If file exists, must have tested_modes key
    if data:
        assert "tested_modes" in data


def test_load_cost_threshold_report_from_file():
    data = load_cost_threshold_report()
    if data:
        assert isinstance(data, list)
        assert "cost_ratio" in data[0]


def test_load_metrics_report_from_file():
    data = load_metrics_report()
    if data:
        assert "holdout" in data


def test_load_report_missing_file_returns_empty(tmp_path):
    """Missing report file should return empty dict/list, not raise."""
    assert load_lofmo_report(tmp_path) == {}
    assert load_cost_threshold_report(tmp_path) == []
    assert load_metrics_report(tmp_path) == {}
