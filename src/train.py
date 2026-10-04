"""Baseline ladder + leakage-aware validation + final holdout.

Run from the repo root:   python -m src.train
Outputs: reports/cv_results.csv, reports/per_mode_recall.csv, reports/holdout_metrics.json,
         models/rf_engineered.joblib
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (DATA_RAW, MODE_COLS, MODEL_PATH, MODELS_DIR,
                     PRODUCTION_THRESHOLD, RANDOM_STATE, REPORTS_DIR, TARGET)
from .data import label_consistency_report, load_raw
from .features import ENGINEERED_FEATURES, RAW_FEATURES, build_features
from .models import make_models, make_rf
from .rules import rule_flags
from .validation import get_splitter, metrics, oof_scores


def split_holdout(df, mode: str, frac: float):
    y = df[TARGET].values
    if mode == "tail":  # last rows by UDI order; harder, time-like
        n = int(len(df) * (1 - frac))
        return df.iloc[:n].copy(), df.iloc[n:].copy()
    dev, hold = train_test_split(df, test_size=frac, stratify=y, random_state=RANDOM_STATE)
    return dev.sort_index().copy(), hold.sort_index().copy()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_RAW))
    ap.add_argument("--holdout", choices=["random", "tail"], default="random")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--fit-all", action="store_true",
                    help="after reporting, refit the saved model on ALL rows (dev + holdout)")
    args = ap.parse_args(argv)

    REPORTS_DIR.mkdir(exist_ok=True); MODELS_DIR.mkdir(exist_ok=True)
    df = load_raw(args.data)
    print("Label audit:", json.dumps(label_consistency_report(df)))

    dev, hold = split_holdout(df, args.holdout, args.holdout_frac)
    y_dev = dev[TARGET].values
    feats = {"raw": RAW_FEATURES, "engineered": ENGINEERED_FEATURES}
    X_dev_all = build_features(dev)
    rules_dev = rule_flags(dev)["rule_any"].astype(float).values

    rows, oof_store = [], {}
    for scheme in ("stratified_5fold", "block_5fold"):
        splitter = get_splitter(scheme)
        # physics-rule baseline needs no fitting
        rows.append({"scheme": scheme, "features": "rules", "model": "PhysicsRules",
                     **metrics(y_dev, rules_dev)})
        for fs_name, cols in feats.items():
            X = X_dev_all[cols]
            for name, model in make_models().items():
                p = oof_scores(model, X, y_dev, splitter)
                rows.append({"scheme": scheme, "features": fs_name, "model": name, **metrics(y_dev, p)})
                if name == "RandomForest" and fs_name == "engineered":
                    oof_store[scheme] = p
        hyb = np.maximum(rules_dev, oof_store[scheme])
        rows.append({"scheme": scheme, "features": "engineered", "model": "Hybrid(rules OR RF)",
                     **metrics(y_dev, hyb)})
    cv = pd.DataFrame(rows)
    cv.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    show = ["scheme", "features", "model", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    print("\n=== CV results on DEV set (threshold 0.5; ROC/PR are threshold-free) ===")
    print(cv[show].to_string(index=False))

    # per-failure-mode recall (this exposes the unpredictable TWF/RNF ceiling)
    ml_pred = oof_store["stratified_5fold"] >= 0.5
    rule_pred = rules_dev.astype(bool)
    pm = []
    for m in MODE_COLS:
        mask = dev[m].values == 1
        if mask.sum():
            pm.append({"mode": m, "n": int(mask.sum()), "rules_recall": round(float(rule_pred[mask].mean()), 3),
                       "rf_recall": round(float(ml_pred[mask].mean()), 3),
                       "hybrid_recall": round(float((rule_pred | ml_pred)[mask].mean()), 3)})
    pm = pd.DataFrame(pm); pm.to_csv(REPORTS_DIR / "per_mode_recall.csv", index=False)
    print("\n=== Per-failure-mode recall (dev, stratified OOF) ===")
    print(pm.to_string(index=False))

    # final holdout: fitted on dev only, scored ONCE
    model = make_rf().fit(X_dev_all[ENGINEERED_FEATURES], y_dev)
    Xh = build_features(hold)[ENGINEERED_FEATURES]; yh = hold[TARGET].values
    ph = model.predict_proba(Xh)[:, 1]; rh = rule_flags(hold)["rule_any"].astype(float).values
    out = {"holdout_mode": args.holdout, "n_holdout": int(len(hold)), "n_failures": int(yh.sum()),
           "PhysicsRules": metrics(yh, rh), "RandomForest": metrics(yh, ph),
           "Hybrid": metrics(yh, np.maximum(rh, ph))}
    (REPORTS_DIR / "holdout_metrics.json").write_text(json.dumps(out, indent=2))
    print("\n=== FINAL HOLDOUT (scored once) ===")
    for k in ("PhysicsRules", "RandomForest", "Hybrid"):
        print(k, out[k])

    if args.fit_all:
        full = build_features(df)[ENGINEERED_FEATURES]
        model = make_rf().fit(full, df[TARGET].values)
    joblib.dump({"model": model, "features": ENGINEERED_FEATURES,
                 "threshold": PRODUCTION_THRESHOLD,
                 "fit_on": "all" if args.fit_all else "dev"}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
