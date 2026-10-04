import numpy as np
import pandas as pd
import pytest

from src.config import PRODUCTION_THRESHOLD
from src.features import ENGINEERED_FEATURES, build_features
from src.models import make_rf
from src.predict import predict_frame
from tests.helpers import frame, row, synthetic


def _bundle(threshold=0.5):
    df = synthetic()
    model = make_rf().fit(build_features(df)[ENGINEERED_FEATURES],
                          df["Machine failure"].values)
    return {"model": model, "features": ENGINEERED_FEATURES,
            "threshold": threshold}


def test_output_length_and_columns():
    b = _bundle()
    df = frame(row(), row(torque=5.0, rpm=1000))
    out = predict_frame(df, b)
    assert len(out) == 2
    for c in ["rule_hdf", "rule_pwf", "rule_osf", "ml_risk", "risk_flag",
              "likely_mode", "status"]:
        assert c in out.columns
    assert bool(out.loc[1, "risk_flag"]) is True and "PWF" in out.loc[1, "likely_mode"]


def test_residual_risk_wording():
    b = _bundle()
    df = frame(row(torque=5.0, rpm=1000))  # triggers PWF rule
    out = predict_frame(df, b)
    assert out.loc[0, "likely_mode"].startswith("Rule-triggered")


def test_no_unexplained_wording():
    b = _bundle()
    df = frame(row())
    out = predict_frame(df, b)
    assert not out["likely_mode"].astype(str).str.contains("unexplained").any()


def test_invalid_rows_are_unscored_not_silently_zero():
    b = _bundle()
    df = frame(row(), row(air=float("nan")), row(typ="Z"))
    out = predict_frame(df, b)
    assert out.loc[0, "status"] == "ok" and not np.isnan(out.loc[0, "ml_risk"])
    for i in (1, 2):
        assert out.loc[i, "status"].startswith("invalid")
        assert np.isnan(out.loc[i, "ml_risk"])
        assert pd.isna(out.loc[i, "risk_flag"])
        assert out.loc[i, "likely_mode"] == "unscored"


def test_empty_frame_with_columns_returns_empty():
    out = predict_frame(frame(), _bundle())
    assert len(out) == 0


def test_missing_columns_raise_value_error():
    with pytest.raises(ValueError):
        predict_frame(pd.DataFrame({"Type": ["L"]}), _bundle())


def test_extra_columns_and_ids_pass_through_safely():
    b = _bundle()
    df = frame(row()); df["UDI"] = 7; df["Product ID"] = "M1"; df["TWF"] = 1
    out = predict_frame(df, b)
    assert out.loc[0, "UDI"] == 7 and out.loc[0, "Product ID"] == "M1"
    assert "TWF" not in out.columns


def test_leakage_columns_are_never_model_features():
    from src.config import ID_COLS, MODE_COLS
    assert not set(ENGINEERED_FEATURES) & set(MODE_COLS + ID_COLS)


def test_production_threshold_matches_config():
    assert PRODUCTION_THRESHOLD == 0.515
