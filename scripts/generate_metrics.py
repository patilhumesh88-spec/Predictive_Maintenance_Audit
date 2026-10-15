#!/usr/bin/env python3
"""Generate reports/metrics.json — the single source of truth for all numbers.

Usage:  python scripts/generate_metrics.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maint_audit.config import DATA_RAW, PRODUCTION_THRESHOLD, TARGET, MODE_COLS
from maint_audit.data import label_consistency_report, load_raw
from maint_audit.features import ENGINEERED_FEATURES, build_features
from maint_audit.models import make_rf
from maint_audit.rules import rule_flags
from maint_audit.validation import get_splitter, metrics, oof_scores

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def get_git_sha():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT))
        return r.stdout.strip()[:12] if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main():
    REPORTS.mkdir(exist_ok=True)
    df = load_raw()
    data_hash = hashlib.sha256(DATA_RAW.read_bytes()).hexdigest()

    # Dataset stats
    label_report = label_consistency_report(df)

    # Dev / holdout
    dev = df.iloc[:8000].copy()
    hold = df.iloc[8000:].copy()
    y_dev = dev[TARGET].values
    y_hold = hold[TARGET].values

    # Rules on full data
    rules_full = rule_flags(df)["rule_any"]
    tp_rules = int(((rules_full) & (df[TARGET] == 1)).sum())
    fp_rules = int(((rules_full) & (df[TARGET] == 0)).sum())
    fn_rules = int(((~rules_full) & (df[TARGET] == 1)).sum())

    # Dev OOF scores
    X_dev = build_features(dev)[ENGINEERED_FEATURES]
    splitter = get_splitter("stratified_5fold")
    model = make_rf()
    oof = oof_scores(model, X_dev, y_dev, splitter)
    rules_dev = rule_flags(dev)["rule_any"].astype(float).values
    hybrid_dev = np.maximum(rules_dev, oof)

    cv_rules = metrics(y_dev, rules_dev, thr=0.5)
    cv_ml = metrics(y_dev, oof, thr=PRODUCTION_THRESHOLD)
    cv_hybrid = metrics(y_dev, hybrid_dev, thr=PRODUCTION_THRESHOLD)

    # Block CV
    block_splitter = get_splitter("block_5fold")
    oof_block = oof_scores(make_rf(), X_dev, y_dev, block_splitter)
    cv_ml_block = metrics(y_dev, oof_block, thr=PRODUCTION_THRESHOLD)

    # Holdout (dev-trained model)
    model_dev = make_rf().fit(X_dev, y_dev)
    X_hold = build_features(hold)[ENGINEERED_FEATURES]
    ph = model_dev.predict_proba(X_hold)[:, 1]
    rh = rule_flags(hold)["rule_any"].astype(float).values
    holdout_rules = metrics(y_hold, rh, thr=0.5)
    holdout_ml = metrics(y_hold, ph, thr=PRODUCTION_THRESHOLD)
    holdout_hybrid = metrics(y_hold, np.maximum(rh, ph), thr=PRODUCTION_THRESHOLD)

    # Rule-unexplained
    dev_rule_flags = rule_flags(dev)["rule_any"]
    dev_fail_no_rule = dev[(dev[TARGET] == 1) & (~dev_rule_flags)]
    unexplained_composition = {}
    for m in MODE_COLS:
        c = int((dev_fail_no_rule[m] == 1).sum())
        if c > 0:
            unexplained_composition[m] = c
    unexplained_composition["no_flag"] = int((dev_fail_no_rule[MODE_COLS].sum(axis=1) == 0).sum())

    # Holdout mode breakdown
    holdout_modes = {m: int(hold[m].sum()) for m in MODE_COLS}

    import sklearn
    result = {
        "generated_by": "scripts/generate_metrics.py",
        "git_sha": get_git_sha(),
        "data_sha256": data_hash,
        "sklearn_version": sklearn.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "threshold": PRODUCTION_THRESHOLD,
        "dataset": label_report,
        "rules_full_data": {
            "TP": tp_rules, "FP": fp_rules, "FN": fn_rules,
            "precision": round(tp_rules / (tp_rules + fp_rules), 4) if (tp_rules + fp_rules) > 0 else 0,
            "recall": round(tp_rules / (tp_rules + fn_rules), 4) if (tp_rules + fn_rules) > 0 else 0,
            "coverage": f"{tp_rules}/{label_report['failures']}",
        },
        "cv_stratified": {
            "PhysicsRules": cv_rules,
            "RandomForest": cv_ml,
            "Hybrid": cv_hybrid,
        },
        "cv_block": {
            "RandomForest": cv_ml_block,
        },
        "holdout": {
            "n_rows": len(hold),
            "n_failures": int(y_hold.sum()),
            "modes": holdout_modes,
            "PhysicsRules": holdout_rules,
            "RandomForest": holdout_ml,
            "Hybrid": holdout_hybrid,
        },
        "rule_unexplained_dev": {
            "total": len(dev_fail_no_rule),
            "composition": unexplained_composition,
        },
    }

    out_path = REPORTS / "metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[OK] Written {out_path}")
    print(f"  Dataset: {label_report['rows']} rows, {label_report['failures']} failures")
    print(f"  Holdout ML AP: {holdout_ml['pr_auc']}, AUC: {holdout_ml['roc_auc']}")
    print(f"  Git SHA: {result['git_sha']}")


if __name__ == "__main__":
    main()
