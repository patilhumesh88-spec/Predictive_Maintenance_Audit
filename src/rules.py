"""Documented physics rules. On the real AI4I file these reproduce HDF/PWF/OSF flags exactly."""
import pandas as pd

from .config import HDF_MAX_DIFF, HDF_MAX_RPM, PWF_MAX_W, PWF_MIN_W, RPM
from .features import build_features


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    out = pd.DataFrame(index=df.index)
    out["rule_hdf"] = (X["temp_diff"] < HDF_MAX_DIFF) & (X[RPM] < HDF_MAX_RPM)
    out["rule_pwf"] = (X["power_w"] < PWF_MIN_W) | (X["power_w"] > PWF_MAX_W)
    out["rule_osf"] = X["strain_ratio"] > 1.0
    out["rule_any"] = out[["rule_hdf", "rule_pwf", "rule_osf"]].any(axis=1)
    return out.fillna(False).astype(bool)
