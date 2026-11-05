import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Ensure the repository root is importable when running under Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maint_audit.config import AIR, MODEL_PATH, PROC, PRODUCTION_THRESHOLD, RPM, TORQUE, TYPE, WEAR
from maint_audit.evaluation import (
    build_cost_dataframe,
    build_cv_comparison_dataframe,
    build_lofmo_dataframe,
    load_cost_threshold_report,
    load_lofmo_report,
    load_metrics_report,
)
from maint_audit.features import build_features
from maint_audit.predict import predict_frame

# Set page config
st.set_page_config(
    page_title="Predictive Maintenance Audit",
    page_icon="⚙️",
    layout="wide",
)

# Load production model bundle
@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        st.error(f"Production model not found at {MODEL_PATH}. Run python -m maint_audit.train --fit-all first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reports():
    return (
        load_lofmo_report(),
        load_cost_threshold_report(),
        load_metrics_report(),
    )


bundle = load_bundle()
lofmo_report, cost_report, metrics_report = load_reports()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚙️ Predictive Maintenance Audit")
st.markdown("**Leakage-aware failure-risk scoring on the UCI AI4I 2020 dataset**")
st.markdown(
    "> **What makes this different:** Most ML projects report metrics and stop. "
    "This project audits *whether ML is even the right tool* by benchmarking "
    "against deterministic physics — then stress-tests both under distribution shift "
    "and unseen failure modes."
)
st.write("---")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_live, tab_benchmark, tab_findings = st.tabs([
    "🔌 Live Diagnostics",
    "🏆 Model Benchmarks",
    "🔬 Research Findings",
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — Live Diagnostics
# ═══════════════════════════════════════════════════════════
with tab_live:
    col_inputs, col_results = st.columns([1, 2])

    with col_inputs:
        st.header("Machine Inputs")
        st.write("Adjust sensor inputs to compute real-time failure risk.")

        typ = st.selectbox("Machine Product Type", ["L", "M", "H"], index=1,
                           help="Product quality level (Low, Medium, High) affecting failure thresholds.")
        air_temp = st.number_input("Air temperature [K]", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
        proc_temp = st.number_input("Process temperature [K]", min_value=250.0, max_value=450.0, value=310.0, step=0.1)
        rot_speed = st.number_input("Rotational speed [rpm]", min_value=1.0, max_value=10000.0, value=1500.0, step=10.0)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=500.0, value=40.0, step=0.5)
        tool_wear = st.number_input("Tool wear [min]", min_value=0.0, max_value=1000.0, value=50.0, step=1.0)

        input_data = pd.DataFrame([{
            AIR: air_temp, PROC: proc_temp, RPM: rot_speed,
            TORQUE: torque, WEAR: tool_wear, TYPE: typ,
        }])

    with col_results:
        st.header("⚡ Live Diagnostics")
        try:
            res = predict_frame(input_data, bundle)
            is_fail = bool(res.loc[0, "risk_flag"])
            status_label = res.loc[0, "likely_mode"]
            twf_elevated = bool(res.loc[0, "rule_twf_elevated"])

            if is_fail:
                st.error(f"🚨 **FAILURE RISK DETECTED** — {status_label}")
            elif twf_elevated:
                st.warning("⚠️ **ELEVATED TWF WATCH** — Tool wear ≥ 200 min. Monitor closely; stochastic TWF risk elevated.")
            else:
                st.success("✅ **SYSTEM HEALTHY** — No immediate failure risk detected.")

            # Derived physics metrics
            derived = build_features(input_data)
            limit = {"L": 11000, "M": 12000, "H": 13000}[typ]

            st.subheader("📊 Derived Physical Variables")
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Temperature Difference", f"{derived.loc[0, 'temp_diff']:.1f} K",
                         help="Process Temp - Air Temp. (HDF triggers if < 8.6 K and Speed < 1380 RPM)")
            col_p2.metric("Mechanical Power", f"{derived.loc[0, 'power_w']:.1f} W",
                         help="Torque * RPM * 2*pi/60. (PWF triggers if < 3500 W or > 9000 W)")
            col_p3.metric("Tool Strain (Wear × Torque)", f"{derived.loc[0, 'wear_torque']:.1f} min·Nm",
                         help=f"Cumulative mechanical strain. (OSF triggers if > {limit} min·Nm for Type {typ})")
            col_p4.metric("Strain Ratio", f"{derived.loc[0, 'strain_ratio']:.3f}",
                         help="Ratio of current tool strain to the type-specific physical threshold.")

            # Diagnostic tabs
            tab_rules, tab_ml = st.tabs(["🕵️ Rule-Engine Diagnostics", "🧠 Machine Learning Risk Profile"])

            with tab_rules:
                st.write("Deterministic physics-rule evaluation based on documented failure mechanisms:")
                hdf_trig = bool(res.loc[0, "rule_hdf"])
                pwf_trig = bool(res.loc[0, "rule_pwf"])
                osf_trig = bool(res.loc[0, "rule_osf"])

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.markdown("**Heat Dissipation (HDF)**")
                    st.write("🔴 Triggered" if hdf_trig else "🟢 Normal")
                    st.caption("Criteria: Temp Diff < 8.6 K & Speed < 1380 RPM")
                    if hdf_trig:
                        st.info(f"Speed ({rot_speed:.1f} RPM) < 1380 and Temp Diff ({derived.loc[0, 'temp_diff']:.1f} K) < 8.6")

                with col_r2:
                    st.markdown("**Power Failure (PWF)**")
                    st.write("🔴 Triggered" if pwf_trig else "🟢 Normal")
                    st.caption("Criteria: Power < 3500 W or > 9000 W")
                    if pwf_trig:
                        st.info(f"Power ({derived.loc[0, 'power_w']:.1f} W) outside [3500, 9000] W")

                with col_r3:
                    st.markdown("**Overstrain Failure (OSF)**")
                    st.write("🔴 Triggered" if osf_trig else "🟢 Normal")
                    st.caption(f"Criteria: Strain > {limit} min·Nm for Type {typ}")
                    if osf_trig:
                        st.info(f"Strain ({derived.loc[0, 'wear_torque']:.1f} min·Nm) exceeds threshold ({limit} min·Nm)")

                st.divider()
                st.markdown("**Tool Wear Failure (TWF) — Elevated Risk Heuristic**")
                st.write("🟠 Elevated Watch" if twf_elevated else "🟢 Normal")
                st.caption(
                    "Heuristic: Tool wear ≥ 200 min. TWF is stochastic (200–240 min window); "
                    "no deterministic threshold exists without vibration telemetry. "
                    "Tradeoff: ~60% TWF recall at ~2.8% false-positive rate."
                )
                if twf_elevated:
                    st.warning(f"Tool wear ({tool_wear:.0f} min) ≥ 200 min threshold — schedule inspection.")

            with tab_ml:
                st.write("Continuous soft failure-risk estimated by the leakage-safe Random Forest pipeline:")
                ml_risk = float(res.loc[0, "ml_risk"])
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Predicted Probability", f"{ml_risk * 100:.1f} %")
                col_m2.metric("Decision Threshold", f"{PRODUCTION_THRESHOLD * 100:.1f} %")
                col_m3.metric("ML Decision", "FAIL" if ml_risk >= PRODUCTION_THRESHOLD else "NORMAL")
                st.progress(ml_risk)
                st.caption(
                    "The ML model provides well-calibrated continuous risk probabilities "
                    "(Brier Score = 0.0085, ECE = 1.23%). "
                    "While binary decisions tie with physics rules on the holdout, ML enables "
                    "alert tiering and cost-sensitive threshold adjustment."
                )

        except Exception as e:
            st.error(f"Error executing prediction pipeline: {e}")


# ═══════════════════════════════════════════════════════════
# TAB 2 — Model Benchmarks
# ═══════════════════════════════════════════════════════════
with tab_benchmark:
    st.header("🏆 Model Performance (Untouched Holdout)")
    st.write(
        "All architectural decisions were frozen on development data. "
        "The final systems were scored **exactly once** on the untouched 20% holdout (last 2,000 rows). "
        "**Note:** The shipped model artifact was refit on all 10,000 rows for inference; "
        "holdout metrics below come from a dev-trained model (see model card)."
    )

    if metrics_report and "holdout" in metrics_report:
        h = metrics_report["holdout"]
        metrics_df = pd.DataFrame([
            {"System": "Documented Rules", **{k: f"{v:.4f}" for k, v in h["PhysicsRules"].items() if k in ("precision", "recall", "f1", "pr_auc", "roc_auc")}},
            {"System": "Leakage-safe ML (RF)", **{k: f"{v:.4f}" for k, v in h["RandomForest"].items() if k in ("precision", "recall", "f1", "pr_auc", "roc_auc")}},
            {"System": "Hybrid (Rules OR ML)", **{k: f"{v:.4f}" for k, v in h["Hybrid"].items() if k in ("precision", "recall", "f1", "pr_auc", "roc_auc")}},
        ])
    else:
        metrics_df = pd.DataFrame([
            {"System": "Documented Rules", "precision": "1.0000", "recall": "0.7179", "f1": "0.8358", "pr_auc": "0.7234", "roc_auc": "0.8590"},
            {"System": "Leakage-safe ML (RF)", "precision": "1.0000", "recall": "0.7179", "f1": "0.8358", "pr_auc": "0.7851", "roc_auc": "0.9720"},
            {"System": "Hybrid (Rules OR ML)", "precision": "1.0000", "recall": "0.7179", "f1": "0.8358", "pr_auc": "0.7851", "roc_auc": "0.9720"},
        ])
    st.table(metrics_df)

    st.markdown(
        "**Untouched Holdout Confusion Matrix (Identical for all 3 systems):**\n"
        "- **True Negatives:** 1,961\n"
        "- **False Positives:** 0\n"
        "- **False Negatives:** 11 (rule-unexplained failures)\n"
        "- **True Positives:** 28"
    )


# ═══════════════════════════════════════════════════════════
# TAB 3 — Research Findings (NEW)
# ═══════════════════════════════════════════════════════════
with tab_findings:
    st.header("🔬 Research Findings")
    st.markdown(
        "These are the project's most important insights — visualised from the pre-computed reports. "
        "All numbers are reproduced directly from code and regression-tested."
    )

    # --- Chart 1: LOFMO ---
    st.subheader("1. Leave-One-Failure-Mode-Out: ML vs Physics Rules")
    st.markdown(
        "When a failure mode is completely withheld from training, "
        "ML recall **collapses to 7–14%** while physics rules maintain **100% recall** unconditionally — "
        "because thermodynamics is not a training distribution."
    )

    if lofmo_report:
        lofmo_df = build_lofmo_dataframe(lofmo_report)
        # Pivot for side-by-side bar chart
        pivot = lofmo_df.pivot(index="Held-out Failure Mode", columns="System", values="Recall (%)")
        # Sanitize index name and column names — Altair parse_shorthand chokes on parentheses
        pivot.index.name = "Held-out Failure Mode"
        pivot.columns.name = None
        pivot = pivot.reset_index().rename(columns={"Held-out Failure Mode": "Failure Mode"})
        pivot = pivot.set_index("Failure Mode")
        st.bar_chart(pivot, color=["#EF4444", "#22C55E"])
        st.caption(
            "🟢 Physics Rules — always 100% recall regardless of held-out mode.\n"
            "🔴 Random Forest ML — recall collapses when it has never seen the failure pattern."
        )

        # Show the raw numbers too
        with st.expander("📋 Raw Numbers"):
            modes_data = lofmo_report.get("tested_modes", {})
            for mode, data in modes_data.items():
                ml_r = data["ml_recall_at_0515"] * 100
                phy_r = data["physics_rules"]["rule_specific_recall"] * 100
                n = data["n_test_failures"]
                st.markdown(
                    f"**Held-out {mode}** ({n} events): "
                    f"Rules = **{phy_r:.1f}%** | ML = **{ml_r:.1f}%** | Gap = **{phy_r - ml_r:.1f}%**"
                )
    else:
        st.info("Run `python scripts/compute_evidence.py` to generate `reports/leave_one_mode_out.json`.")

    st.divider()

    # --- Chart 2: Cost-Sensitive Threshold ---
    st.subheader("2. Cost-Sensitive Threshold Optimisation")
    st.markdown(
        "Under asymmetric operational loss, lowering the ML decision threshold dramatically reduces "
        "total plant downtime cost. At **C_miss:C_fa = 20:1**, threshold 0.160 reduces costs by 18.2%."
    )

    if cost_report:
        cost_df = build_cost_dataframe(cost_report)
        # Show recall vs threshold
        col_cost1, col_cost2 = st.columns(2)
        with col_cost1:
            st.markdown("**Recall at Optimal Threshold per Cost Ratio**")
            chart_df = cost_df.set_index("Cost Ratio (Miss:FA)")[["Recall (%)"]].head(12)
            chart_df.index.name = "Cost Ratio"
            chart_df.columns = ["Recall %"]
            st.bar_chart(chart_df)
        with col_cost2:
            st.markdown("**Cost Savings vs Default 0.515 Threshold**")
            chart_df2 = cost_df.set_index("Cost Ratio (Miss:FA)")[["Cost Savings vs 0.515 (%)"]].head(12)
            chart_df2.index.name = "Cost Ratio"
            chart_df2.columns = ["Cost Savings vs 0.515 %"]
            st.bar_chart(chart_df2)

        with st.expander("📋 Key Operating Points"):
            key_ratios = ["1:1", "10:1", "20:1", "50:1", "100:1"]
            key_rows = cost_df[cost_df["Cost Ratio (Miss:FA)"].isin(key_ratios)]
            if not key_rows.empty:
                st.dataframe(key_rows.reset_index(drop=True), use_container_width=True)
    else:
        st.info("Run `python scripts/compute_evidence.py` to generate `reports/cost_threshold.json`.")

    st.divider()

    # --- Chart 3: Blocked vs Shuffled CV ---
    st.subheader("3. Blocked CV vs Shuffled CV — Distribution Shift Stability")
    st.markdown(
        "Standard random-shuffle CV **overestimates** performance due to temperature sensor "
        "random-walk drift between rows. Blocked CV is the honest stress test. "
        "Physics-informed features reduce the AP degradation from **−0.133** to **−0.045**."
    )

    if metrics_report:
        cv_df = build_cv_comparison_dataframe(metrics_report)
        if not cv_df.empty:
            # Sanitize CV Type values — parentheses in index values confuse Altair
            cv_chart = cv_df.copy()
            cv_chart["CV Type"] = cv_chart["CV Type"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
            st.bar_chart(
                cv_chart.set_index("CV Type")[["Average Precision"]],
            )

            shuffled = metrics_report.get("cv_stratified", {}).get("RandomForest", {}).get("pr_auc", "N/A")
            blocked = metrics_report.get("cv_block", {}).get("RandomForest", {}).get("pr_auc", "N/A")
            col_cv1, col_cv2, col_cv3 = st.columns(3)
            col_cv1.metric("Shuffled CV — AP", f"{shuffled:.4f}" if isinstance(shuffled, float) else shuffled)
            col_cv2.metric("Blocked CV — AP", f"{blocked:.4f}" if isinstance(blocked, float) else blocked)
            if isinstance(shuffled, float) and isinstance(blocked, float):
                col_cv3.metric("AP Drop", f"{shuffled - blocked:.3f}", delta=f"-{shuffled - blocked:.3f}", delta_color="inverse")
        else:
            st.info("CV comparison data not available in metrics.json.")
    else:
        st.info("Run `python scripts/generate_metrics.py` to populate metrics.json.")

    st.divider()

    # --- Core Findings Summary ---
    st.subheader("📝 Core Project Findings")
    st.markdown(
        "- **84.66% of failures** are explained by 3 deterministic physics equations with **100% precision** and zero parameters.\n"
        "- On the untouched holdout, ML and physics rules produce an **exact tie** on binary decisions (28 TP / 0 FP / 11 FN).\n"
        "- ML excels at **continuous risk ranking**: ROC-AUC 0.972 vs 0.859, enabling alert tiering and cost-sensitive thresholding.\n"
        "- Under LOFMO stress, ML recall collapses to **7–14%** on unseen failure modes; physics rules hold **100%**.\n"
        "- The 11 missed failures are **83% Tool Wear (TWF)** — stochastic events at 200–240 min wear, "
        "unpredictable without vibration telemetry. A `rule_twf_elevated` heuristic now flags wear ≥ 200 min "
        "for monitoring (~60% TWF recall, ~2.8% FP rate)."
    )
