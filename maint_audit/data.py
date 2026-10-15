"""Loading and input validation."""
from pathlib import Path

import pandas as pd

from .config import DATA_RAW, PLAUSIBLE, REQUIRED_INPUT_COLS, SENSOR_COLS, TYPE, TYPE_MAP


def load_raw(path=DATA_RAW) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put ai4i2020.csv from UCI into data/raw/."
        )
    return pd.read_csv(path)


def validate_inputs(df: pd.DataFrame) -> pd.Series:
    """Row-level status: 'ok' or a reason string. Raises ValueError if columns are missing."""
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    status = pd.Series("ok", index=df.index, dtype=object)

    def mark(mask, reason):
        hit = mask & (status == "ok")
        status[hit] = reason

    nums = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")
    mark(nums.isna().any(axis=1), "invalid: missing or non-numeric sensor value")
    mark(~df[TYPE].isin(list(TYPE_MAP)), "invalid: unknown product Type (expected L, M or H)")
    for col, (lo, hi) in PLAUSIBLE.items():
        mark((nums[col] < lo) | (nums[col] > hi), f"invalid: {col} outside plausible range")
    return status


def label_consistency_report(df: pd.DataFrame) -> dict:
    """Documented label defects in AI4I (flags vs. Machine failure)."""
    from .config import MODE_COLS, TARGET
    modes = df[MODE_COLS].sum(axis=1)
    return {
        "rows": int(len(df)),
        "failures": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
        "label1_without_any_mode": int(((df[TARGET] == 1) & (modes == 0)).sum()),
        "label0_with_a_mode": int(((df[TARGET] == 0) & (modes > 0)).sum()),
        "rows_with_multiple_modes": int((modes > 1).sum()),
    }
