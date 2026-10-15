"""Feature engineering. Uses ONLY inputs that exist before/independent of the failure label."""
import numpy as np
import pandas as pd

from .config import AIR, PROC, RPM, SENSOR_COLS, TORQUE, TYPE, WEAR

RAW_FEATURES = ["Type"] + SENSOR_COLS
ENGINEERED_FEATURES = RAW_FEATURES + ["temp_diff", "power_w", "wear_torque", "strain_ratio"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Type"] = df[TYPE]
    for c in SENSOR_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["temp_diff"] = X[PROC] - X[AIR]
    X["power_w"] = X[TORQUE] * X[RPM] * 2 * np.pi / 60.0
    X["wear_torque"] = X[WEAR] * X[TORQUE]
    X["strain_ratio"] = X["wear_torque"] / df[TYPE].map(
        {"L": 11000, "M": 12000, "H": 13000}
    ).astype(float)
    return X
