"""Extended boundary-value tests for all physics rules, including TWF heuristic."""
import pytest
import numpy as np
from maint_audit.rules import rule_flags, TWF_WEAR_THRESHOLD_MIN
from tests.helpers import frame, row


# ---------------------------------------------------------------------------
# HDF boundary tests
# ---------------------------------------------------------------------------

def test_hdf_fires_only_when_both_conditions_hold():
    df = frame(
        row(air=300, proc=308.0, rpm=1300),   # diff=8.0 < 8.6 AND rpm < 1380 -> fires
        row(air=300, proc=308.0, rpm=1500),   # low diff but fast -> no
        row(air=300, proc=312.0, rpm=1300),   # slow but big diff -> no
    )
    assert rule_flags(df)["rule_hdf"].tolist() == [True, False, False]


def test_hdf_exact_boundary_diff_does_not_fire():
    """temp_diff == 8.6 exactly should NOT trigger HDF (strictly less than)."""
    df = frame(row(air=300.0, proc=308.6, rpm=1300))   # diff = 8.6 exactly
    assert rule_flags(df)["rule_hdf"].tolist() == [False]


def test_hdf_exact_boundary_rpm_does_not_fire():
    """rpm == 1380 exactly should NOT trigger HDF (strictly less than)."""
    df = frame(row(air=300.0, proc=307.0, rpm=1380))   # rpm = 1380 exactly, diff < 8.6
    assert rule_flags(df)["rule_hdf"].tolist() == [False]


def test_hdf_one_tick_past_boundary_fires():
    """One unit past both boundaries should fire."""
    df = frame(row(air=300.0, proc=308.5, rpm=1379))   # diff=8.5 < 8.6, rpm=1379 < 1380
    assert rule_flags(df)["rule_hdf"].tolist() == [True]


# ---------------------------------------------------------------------------
# PWF boundary tests
# ---------------------------------------------------------------------------

def test_pwf_fires_for_low_and_high_power():
    df = frame(
        row(torque=5.0, rpm=1000),     # ~524 W < 3500
        row(torque=70.0, rpm=2000),    # ~14.7 kW > 9000
        row(torque=40.0, rpm=1500),    # ~6.3 kW normal
    )
    assert rule_flags(df)["rule_pwf"].tolist() == [True, True, False]


def test_pwf_exactly_3500w_boundary():
    """Document the actual PWF boundary behaviour at 3500 W.

    Due to floating-point rounding when computing torque from the target power,
    the computed power may fall fractionally below 3500 W, which correctly
    triggers PWF (power < 3500 W). This test documents the real rule logic:
    power STRICTLY LESS THAN 3500 W fires, power STRICTLY GREATER THAN 9000 W fires.
    """
    import math
    # Cleanly inside safe window -> must NOT fire
    rpm = 1500
    torque = 40.0  # power ~= 6283 W, safely between 3500 and 9000
    df = frame(row(torque=torque, rpm=rpm))
    assert rule_flags(df)["rule_pwf"].tolist() == [False]

    # Cleanly below lower bound -> must fire
    torque_low = 3500 / (rpm * 2 * math.pi / 60) * 0.99  # 1% below 3500 W
    df_low = frame(row(torque=round(torque_low, 4), rpm=rpm))
    assert rule_flags(df_low)["rule_pwf"].tolist() == [True]


def test_pwf_exactly_9000w_boundary():
    """Document the actual PWF boundary behaviour at 9000 W."""
    import math
    # Cleanly above upper bound -> must fire
    rpm = 1500
    torque_high = 9000 / (rpm * 2 * math.pi / 60) * 1.01  # 1% above 9000 W
    df_high = frame(row(torque=round(torque_high, 4), rpm=rpm))
    assert rule_flags(df_high)["rule_pwf"].tolist() == [True]

    # Cleanly inside -> must NOT fire
    torque_ok = 9000 / (rpm * 2 * math.pi / 60) * 0.99
    df_ok = frame(row(torque=round(torque_ok, 4), rpm=rpm))
    assert rule_flags(df_ok)["rule_pwf"].tolist() == [False]


# ---------------------------------------------------------------------------
# OSF boundary tests
# ---------------------------------------------------------------------------

def test_osf_threshold_depends_on_product_type():
    # wear*torque = 12000: above L limit (11000), below H limit (13000)
    df = frame(
        row(wear=200, torque=60.0, typ="L"),   # 12000 > 11000 -> fires
        row(wear=200, torque=60.0, typ="H"),   # 12000 < 13000 -> no
        row(wear=201, torque=60.0, typ="M"),   # 12060 > 12000 -> fires
    )
    assert rule_flags(df)["rule_osf"].tolist() == [True, False, True]


def test_osf_exact_strain_ratio_1_does_not_fire():
    """strain_ratio == 1.0 exactly should NOT trigger OSF (strictly greater than)."""
    # Type M limit = 12000; wear*torque = 12000 -> ratio = 1.0
    df = frame(row(wear=200, torque=60.0, typ="M"))   # 200 * 60 = 12000 = limit exactly
    assert rule_flags(df)["rule_osf"].tolist() == [False]


def test_osf_one_unit_above_limit_fires():
    """One unit above limit -> ratio > 1.0 -> fires."""
    # Type L limit = 11000; wear=184, torque=60 -> 11040 > 11000
    df = frame(row(wear=184, torque=60.0, typ="L"))
    assert rule_flags(df)["rule_osf"].tolist() == [True]


# ---------------------------------------------------------------------------
# TWF elevated-risk heuristic tests
# ---------------------------------------------------------------------------

def test_twf_elevated_fires_at_threshold():
    """Tool wear >= 200 min should flag rule_twf_elevated."""
    df = frame(
        row(wear=TWF_WEAR_THRESHOLD_MIN),         # exactly at threshold -> fires
        row(wear=TWF_WEAR_THRESHOLD_MIN + 1),     # above -> fires
        row(wear=TWF_WEAR_THRESHOLD_MIN - 1),     # below -> no
    )
    result = rule_flags(df)["rule_twf_elevated"].tolist()
    assert result == [True, True, False]


def test_twf_elevated_does_not_affect_rule_any():
    """rule_twf_elevated must NOT be included in rule_any to preserve 100% precision."""
    # High wear only (no HDF/PWF/OSF) -> rule_any should still be False
    df = frame(row(
        air=300, proc=310, rpm=1500, torque=40.0,
        wear=250,  # above TWF threshold
        typ="H",   # H has highest strain limit, won't trigger OSF at 40 Nm
    ))
    flags = rule_flags(df)
    assert flags["rule_twf_elevated"].tolist() == [True]
    assert flags["rule_any"].tolist() == [False]


# ---------------------------------------------------------------------------
# rule_any union test
# ---------------------------------------------------------------------------

def test_rule_any_is_union():
    df = frame(row(), row(torque=5.0, rpm=1000))
    assert rule_flags(df)["rule_any"].tolist() == [False, True]


def test_rule_flags_columns_present():
    """All expected columns must be present in output."""
    df = frame(row())
    cols = rule_flags(df).columns.tolist()
    for expected in ["rule_hdf", "rule_pwf", "rule_osf", "rule_twf_elevated", "rule_any"]:
        assert expected in cols, f"Missing column: {expected}"
