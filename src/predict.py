"""Score ANY csv in the AI4I schema.

    python -m src.predict --input some.csv --output scored.csv

Bad rows are never silently scored: they get status='invalid: ...' and no risk value.
"""
import argparse
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import ID_COLS, MODEL_PATH, PRODUCTION_THRESHOLD
from .data import validate_inputs
from .features import build_features
from .rules import rule_flags


def predict_frame(df: pd.DataFrame, bundle: dict, threshold: Optional[float] = None) -> pd.DataFrame:
    thr = bundle.get("threshold", PRODUCTION_THRESHOLD) if threshold is None else threshold
    status = validate_inputs(df)            # raises ValueError if required columns are missing
    ok = (status == "ok").values

    out = pd.DataFrame(index=df.index)
    for c in ID_COLS:
        if c in df.columns:
            out[c] = df[c]
    flags = pd.DataFrame(False, index=df.index,
                         columns=["rule_hdf", "rule_pwf", "rule_osf", "rule_any"])
    risk = pd.Series(np.nan, index=df.index, dtype=float)
    if ok.any():
        sub = df.loc[ok]
        flags.loc[ok] = rule_flags(sub).values
        X = build_features(sub)[bundle["features"]]
        risk.loc[ok] = bundle["model"].predict_proba(X)[:, 1]
    out = out.join(flags)
    out["ml_risk"] = risk.round(4)
    flag = pd.Series(pd.NA, index=df.index, dtype="boolean")
    flag.loc[ok] = (out.loc[ok, "rule_any"] | (out.loc[ok, "ml_risk"] >= thr)).astype(bool)
    out["risk_flag"] = flag

    def explain(i):
        if not ok[i]:
            return "unscored"
        r = out.iloc[i]
        modes = [m for m, c in (("HDF", "rule_hdf"), ("PWF", "rule_pwf"), ("OSF", "rule_osf")) if r[c]]
        if modes:
            return "Rule-triggered failure risk (" + "+".join(modes) + ")"
        return "Residual statistical risk" if r["risk_flag"] else "No documented rule triggered"
    out["likely_mode"] = [explain(i) for i in range(len(out))]
    out["status"] = status.values
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="scored.csv")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args(argv)
    try:
        df = pd.read_csv(a.input)
        bundle = joblib.load(a.model)
        res = predict_frame(df, bundle, a.threshold)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    res.to_csv(a.output, index=False)
    n_bad = int((res["status"] != "ok").sum())
    print(f"Scored {len(res) - n_bad} rows, {n_bad} invalid -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
