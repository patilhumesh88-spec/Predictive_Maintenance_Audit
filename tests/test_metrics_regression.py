import json
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"
README_PATH = ROOT / "README.md"


@pytest.fixture(scope="module")
def metrics():
    if not METRICS_PATH.exists():
        pytest.fail(f"metrics.json missing at {METRICS_PATH}. Run python scripts/generate_metrics.py first.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_dataset_cardinality(metrics):
    ds = metrics["dataset"]
    assert ds["rows"] == 10000
    assert ds["failures"] == 339
    assert abs(ds["failure_rate"] - 0.0339) < 1e-4
    assert ds["label1_without_any_mode"] == 9
    assert ds["label0_with_a_mode"] == 18


def test_rules_full_data_metrics(metrics):
    r = metrics["rules_full_data"]
    assert r["TP"] == 287
    assert r["FP"] == 0
    assert r["FN"] == 52
    assert r["precision"] == 1.0
    assert abs(r["recall"] - 0.8466) < 1e-4


def test_holdout_exact_confusion_matrix(metrics):
    h = metrics["holdout"]
    assert h["n_rows"] == 2000
    assert h["n_failures"] == 39

    # All three systems tie on binary confusion matrix
    for sys_name in ["PhysicsRules", "RandomForest", "Hybrid"]:
        res = h[sys_name]
        assert res["TP"] == 28
        assert res["FP"] == 0
        assert res["FN"] == 11
        assert res["TN"] == 1961
        assert res["precision"] == 1.0
        assert abs(res["recall"] - 0.7179) < 1e-4


def test_holdout_ranking_metrics(metrics):
    h = metrics["holdout"]
    assert abs(h["PhysicsRules"]["pr_auc"] - 0.7234) < 1e-3
    assert abs(h["RandomForest"]["pr_auc"] - 0.7851) < 1e-3
    assert abs(h["RandomForest"]["roc_auc"] - 0.9720) < 1e-3


def test_rule_unexplained_dev_split(metrics):
    ru = metrics["rule_unexplained_dev"]
    assert ru["total"] == 41
    comp = ru["composition"]
    assert comp["TWF"] == 34
    assert comp["RNF"] == 1
    assert comp["no_flag"] == 7


def test_readme_numbers_match_metrics(metrics):
    if not README_PATH.exists():
        pytest.skip("README.md not found")
    readme = README_PATH.read_text(encoding="utf-8")

    # Verify key numbers in README match metrics.json
    assert "10,000" in readme or "10000" in readme
    assert "339" in readme
    assert "287" in readme
    assert "84.66%" in readme or "84.7%" in readme
    assert "0.515" in readme
    assert "28" in readme
    assert "11" in readme
