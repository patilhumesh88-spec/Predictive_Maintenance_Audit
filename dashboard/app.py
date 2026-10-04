import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.config import (
    MODEL_PATH, PRODUCTION_THRESHOLD, AIR, PROC, RPM, TORQUE, WEAR, TYPE,
    SENSOR_COLS, REQUIRED_INPUT_COLS
)
from src.predict import predict_frame
from src.features import build_features

# Set page config
st.set_page_config(
    page_title="ALGOTHON '26 — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

# Load production model bundle
@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        st.error(f"Production model not found at {MODEL_PATH}. Run python -m src.train --fit-all first.")
        st.stop()
    return joblib.load(MODEL_PATH)

bundle = load_bundle()

# Title & Header
st.title("⚙️ ALGOTHON '26 Predictive Maintenance Dashboard")
st.markdown("### **ALG-DATA-02: Predict What Happens Next**")
st.markdown(
    "**Physics-informed predictive maintenance with documented-rule analysis + ML risk scoring**"
)
st.write(
    "This system audits a synthetic predictive-maintenance benchmark (UCI AI4I 2020), "
    "evaluates physical failure mechanisms, adds physics-informed features, "
    "and integrates a machine learning risk layer to capture residual anomalies safely."
)

st.write("---")

# Layout: 2 Columns for inputs & live prediction
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.header("🔌 Machine Inputs")
    st.write("Adjust sensor inputs to compute real-time failure risk.")

    # Inputs with default standard operating values
    typ = st.selectbox("Machine Product Type", ["L", "M", "H"], index=1,
                       help="Product quality level (Low, Medium, High) affecting failure thresholds.")

    air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=400.0, value=300.0, step=0.1,
                               help="Ambient temperature surrounding the machine.")
    proc_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=450.0, value=310.0, step=0.1,
                                help="Internal manufacturing process temperature.")
    rot_speed = st.number_input("Rotational speed [rpm]", min_value=1.0, max_value=10000.0, value=1500.0, step=10.0,
                                help="Spindle rotational speed.")
    torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=500.0, value=40.0, step=0.5,
                             help="Spindle torque.")
    tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=1000.0, value=50.0, step=1.0,
                                help="Cumulative time the tool has been active.")

    # Construct single-row DataFrame matching schema
    input_data = pd.DataFrame([{
        AIR: air_temp,
        PROC: proc_temp,
        RPM: rot_speed,
        TORQUE: torque,
        WEAR: tool_wear,
        TYPE: typ
    }])

# Run predictive pipeline on input data
with col_results:
    st.header("⚡ Live Diagnostics")

    try:
        # Predict using official src pipeline
        res = predict_frame(input_data, bundle)

        # Display large status indicator
        is_fail = bool(res.loc[0, "risk_flag"])
        status_label = res.loc[0, "likely_mode"]

        if is_fail:
            st.error(f"🚨 **FAILURE RISK DETECTED** — Reason: {status_label}")
        else:
            st.success("✅ **SYSTEM HEALTHY** — No immediate failure risk detected.")

        # Display derived physics metrics
        derived = build_features(input_data)
        limit = {"L": 11000, "M": 12000, "H": 13000}[typ]

        st.subheader("📊 Derived Physical Variables")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Temperature Difference", f"{derived.loc[0, 'temp_diff']:.1f} K",
                     help="Process Temp - Air Temp. (HDF triggers if < 8.6 K and Speed < 1380 RPM)")
        col_p2.metric("Mechanical Power", f"{derived.loc[0, 'power_w']:.1f} W",
                     help="Torque * RPM * 2*pi/60. (PWF triggers if < 3500 W or > 9000 W)")
        col_p3.metric("Tool Strain (Wear * Torque)", f"{derived.loc[0, 'wear_torque']:.1f} min·Nm",
                     help=f"Cumulative mechanical strain. (OSF triggers if > {limit} min·Nm for Type {typ})")
        col_p4.metric("Strain Ratio", f"{derived.loc[0, 'strain_ratio']:.3f}",
                     help="Ratio of current tool strain to the type-specific physical threshold.")

        # Detailed Diagnostic Tabs
        tab_rules, tab_ml = st.tabs(["🕵️ Rule-Engine Diagnostics", "🧠 Machine Learning Risk Profile"])

        with tab_rules:
            st.write("Deterministic physics-rule evaluation based on documented failure mechanisms:")

            hdf_trig = bool(res.loc[0, "rule_hdf"])
            pwf_trig = bool(res.loc[0, "rule_pwf"])
            osf_trig = bool(res.loc[0, "rule_osf"])

            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.markdown(f"**Heat Dissipation (HDF)**")
                st.write("Triggered" if hdf_trig else "Normal")
                st.caption("Criteria: Temp Diff < 8.6 K & Speed < 1380 RPM")
                if hdf_trig:
                    st.info(f"Speed ({rot_speed:.1f} RPM) < 1380 and Temp Diff ({derived.loc[0, 'temp_diff']:.1f} K) < 8.6")

            with col_r2:
                st.markdown(f"**Power Failure (PWF)**")
                st.write("Triggered" if pwf_trig else "Normal")
                st.caption("Criteria: Power < 3500 W or > 9000 W")
                if pwf_trig:
                    st.info(f"Power ({derived.loc[0, 'power_w']:.1f} W) is outside safe operating window [3500 W, 9000 W]")

            with col_r3:
                st.markdown(f"**Overstrain Failure (OSF)**")
                st.write("Triggered" if osf_trig else "Normal")
                st.caption(f"Criteria: Strain > {limit} min·Nm for Type {typ}")
                if osf_trig:
                    st.info(f"Strain ({derived.loc[0, 'wear_torque']:.1f} min·Nm) exceeds the physical threshold ({limit} min·Nm) for Type {typ}")

        with tab_ml:
            st.write("Continuous soft failure-risk estimated by the leakage-safe Random Forest pipeline:")

            ml_risk = float(res.loc[0, "ml_risk"])
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Predicted Probability", f"{ml_risk * 100:.1f} %")
            col_m2.metric("Decision Threshold", f"{PRODUCTION_THRESHOLD * 100:.1f} %")
            col_m3.metric("ML Decision", "FAIL" if ml_risk >= PRODUCTION_THRESHOLD else "NORMAL")

            st.progress(ml_risk)
            st.caption("The ML model evaluates non-linear statistical risks that are not fully captured by deterministic physical equations.")

    except Exception as e:
        st.error(f"Error executing prediction pipeline: {e}")

st.write("---")

# Layout: Model Benchmarks & Project Findings side-by-side
col_bench, col_findings = st.columns(2)

with col_bench:
    st.header("🏆 Model Performance (Untouched Holdout)")
    st.write(
        "To guarantee generalization, all architectural decisions were frozen on development data. "
        "The final systems were scored **exactly once** on the untouched 20% holdout (last 2,000 rows in original order)."
    )

    # Verified metrics table from Notebook 08
    metrics_df = pd.DataFrame([
        {"System": "Documented Rules Only", "Precision": "1.0000", "Recall": "0.7179", "F1": "0.8358", "AP": "0.7234", "ROC-AUC": "0.8590"},
        {"System": "Leakage-safe ML (RF)", "Precision": "1.0000", "Recall": "0.7179", "F1": "0.8358", "AP": "0.7851", "ROC-AUC": "0.9720"},
        {"System": "Hybrid (Rules OR ML)", "Precision": "1.0000", "Recall": "0.7179", "F1": "0.8358", "AP": "0.7851", "ROC-AUC": "0.9720"}
    ])
    st.table(metrics_df)

    st.markdown(
        "**Untouched Holdout Confusion Matrix (Identical for all 3 systems):**\n"
        "- **True Negatives (Normal correctly flagged):** 1,961\n"
        "- **False Positives (False alarms):** 0\n"
        "- **False Negatives (Undetected failures):** 11 (all represent rule-unexplained failures)\n"
        "- **True Positives (Correctly identified failures):** 28"
    )

with col_findings:
    st.header("📝 Core Project Findings")
    st.markdown(
        "- **High Physics Coverage:** Documented physical failure mechanisms (HDF, PWF, OSF) perfectly "
        "reproduce their corresponding raw column flags and account for **84.66%** of all failures "
        "in the full dataset (287 out of 339).\n"
        "- **Physics-Informed ML is Robust:** Adding physics-informed features (such as `power_w` and `strain_ratio`) "
        "provides a **+0.139** AP boost for Random Forest (0.907 vs 0.767 raw) and stabilizes performance against block cross-validation drops.\n"
        "- **Stricter Validation Stress Tests:** Standard random-shuffled CV overestimates performance (AP ~0.907) due "
        "to temporal temperature dependency (random walks). Block CV serves as an honest robustness stress test (AP ~0.862).\n"
        "- **The Recall Performance Ceiling:** On the development set, 41 failures (13.67%) are completely unexplained "
        "by HDF/PWF/OSF rules. Of these, 34 are Tool Wear Failures (TWF). Since tool wear failures are stochastic, "
        "sensor streams contain no physical warning indicators. The ML model correctly assigns low probability to these, "
        "confirming a **stochastic recall ceiling (~86%)** that generic ML cannot bypass."
    )
