import pandas as pd
import pytest

from src.config import DATA_RAW, MODE_COLS, TARGET
from src.data import label_consistency_report, load_raw, validate_inputs
from src.rules import rule_flags
from tests.helpers import frame, row


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        validate_inputs(pd.DataFrame({"foo": [1]}))


def test_bad_rows_are_flagged_with_reasons():
    df = frame(row(),
               row(air=float("nan")),
               row(typ="X"),
               row(torque=-3.0),
               row(rpm=0))
    s = validate_inputs(df).tolist()
    assert s[0] == "ok"
    assert all(x.startswith("invalid") for x in s[1:])


def test_non_numeric_value_is_invalid():
    df = frame(row())
    df["Torque [Nm]"] = df["Torque [Nm]"].astype(object)
    df.loc[0, "Torque [Nm]"] = "abc"
    assert validate_inputs(df).iloc[0].startswith("invalid")


def test_load_raw_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_documented_rules_reproduce_real_flags():
    """Key finding: HDF/PWF/OSF rules match the dataset flags exactly (needs data/raw/ai4i2020.csv)."""
    if not DATA_RAW.exists():
        pytest.skip("dataset not present")
    df = load_raw()
    r = rule_flags(df)
    assert (r["rule_hdf"].astype(int) == df["HDF"]).all()
    assert (r["rule_pwf"].astype(int) == df["PWF"]).all()
    assert (r["rule_osf"].astype(int) == df["OSF"]).all()


def test_real_label_defects_are_reported():
    if not DATA_RAW.exists():
        pytest.skip("dataset not present")
    rep = label_consistency_report(load_raw())
    assert rep["rows"] == 10000 and rep["failures"] == 339
    assert rep["label1_without_any_mode"] == 9 and rep["label0_with_a_mode"] == 18
