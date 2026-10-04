from src.rules import rule_flags
from tests.helpers import frame, row


def test_hdf_fires_only_when_both_conditions_hold():
    df = frame(row(air=300, proc=308.0, rpm=1300),   # diff 8.0 <8.6 and rpm<1380 -> fires
               row(air=300, proc=308.0, rpm=1500),   # low diff but fast -> no
               row(air=300, proc=312.0, rpm=1300))   # slow but big diff -> no
    assert rule_flags(df)["rule_hdf"].tolist() == [True, False, False]


def test_pwf_fires_for_low_and_high_power():
    df = frame(row(torque=5.0, rpm=1000),     # ~524 W
               row(torque=70.0, rpm=2000),    # ~14.7 kW
               row(torque=40.0, rpm=1500))    # ~6.3 kW normal
    assert rule_flags(df)["rule_pwf"].tolist() == [True, True, False]


def test_osf_threshold_depends_on_product_type():
    # wear*torque = 12000: above L limit (11000), below H limit (13000), equal-not-above M limit
    df = frame(row(wear=200, torque=60.0, typ="L"),
               row(wear=200, torque=60.0, typ="H"),
               row(wear=201, torque=60.0, typ="M"))
    assert rule_flags(df)["rule_osf"].tolist() == [True, False, True]


def test_rule_any_is_union():
    df = frame(row(), row(torque=5.0, rpm=1000))
    assert rule_flags(df)["rule_any"].tolist() == [False, True]
