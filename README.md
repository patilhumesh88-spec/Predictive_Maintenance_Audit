# Predictive Maintenance Audit

[![CI](https://github.com/patilhumesh88-spec/predictive-maintenance-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/patilhumesh88-spec/predictive-maintenance-audit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**A leakage-aware investigation into physical failure mechanisms vs. machine learning risk scoring on industrial milling telemetry.**

> **The 60-Second Summary:** Can machine learning improve upon documented physics equations for predicting mechanical equipment failure? By auditing the synthetic UCI AI4I 2020 benchmark (10,000 observations, 339 failure events), this project reconstructs deterministic thermodynamic and mechanical rules that explain **84.66% of all failures with 100% precision**. On an untouched holdout set, Random Forest ML ties the rules exactly on binary decisions (**28 TP / 0 FP / 11 FN**). However, ML provides well-calibrated continuous risk ranking (ROC-AUC 0.972 vs. 0.859), and physics-informed features protect the model against distribution shift under blocked validation. Stress-testing reveals that when an unseen physical failure mode is held out, ML recall collapses to **7.8%–14.1%**, whereas deterministic physics rules maintain **100% recall**.

---

## Quickstart in 3 Commands

```bash
git clone https://github.com/patilhumesh88-spec/predictive-maintenance-audit.git && cd predictive-maintenance-audit
pip install -e ".[all]"
python scripts/verify_all.py
```

To run individual entry points:
- **Run test suite (20+ tests):** `pytest tests/ -v`
- **Regenerate metrics single source of truth:** `python scripts/generate_metrics.py`
- **Launch interactive diagnostic dashboard:** `streamlit run dashboard/app.py`
- **Launch FastAPI inference service:** `uvicorn maint_audit.api:app --reload`

---

## Core Results & Evidence

All metrics are regenerated directly from code into `reports/metrics.json` and validated by regression tests.

### 1. Cross-Validation Benchmark with 95% Bootstrap Confidence Intervals
*Evaluated on the 8,000 development rows across 2,000 bootstrap resamples using Stratified 5-Fold Cross-Validation (threshold = 0.515):*

| System | Average Precision (AP) | ROC-AUC | F1 Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Physics Rules Alone** | 0.8685 `[0.828, 0.903]` | 0.9317 `[0.911, 0.950]` | 0.9267 `[0.902, 0.947]` | **1.0000** `[1.000, 1.000]` | **0.8633** `[0.821, 0.900]` |
| **Random Forest ML** | 0.9060 `[0.876, 0.932]` | 0.9785 `[0.967, 0.988]` | 0.8936 `[0.863, 0.918]` | 0.9545 `[0.929, 0.978]` | 0.8400 `[0.794, 0.879]` |
| **Hybrid (Rules $\cup$ ML)** | **0.9155** `[0.886, 0.941]` | **0.9788** `[0.967, 0.989]` | 0.9072 `[0.880, 0.931]` | 0.9557 `[0.930, 0.978]` | **0.8633** `[0.821, 0.900]` |

### 2. Final Untouched Holdout Evaluation (Scored Exactly Once)
*Evaluated on the final 2,000 rows (39 failure events) using a model trained strictly on the 8,000 development rows:*

| System | AP | ROC-AUC | F1 | Precision | Recall | True Positives | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Documented Rules** | 0.7234 | 0.8590 | 0.8358 | 1.0000 | 0.7179 | 28 | 0 | 11 |
| **Leakage-Safe ML (RF)** | 0.7851 | 0.9720 | 0.8358 | 1.0000 | 0.7179 | 28 | 0 | 11 |
| **Hybrid System** | 0.7851 | 0.9720 | 0.8358 | 1.0000 | 0.7179 | 28 | 0 | 11 |

> **Model Artifact Integrity Disclosure:** The shipped model artifact (`models/rf_engineered.joblib`) was refit on all 10,000 observations to maximize parameter coverage for live inference. The holdout performance numbers reported above were evaluated strictly with a development-trained model (8,000 rows). Detailed provenance is documented in [model_card.md](model_card.md).

---

## Architecture & Data Flow

```mermaid
flowchart TD
    A[Sensor Telemetry] --> B[Strict Schema & Plausibility Validation]
    B --> C[Physics Feature Engineering]
    C --> D[Deterministic Rule Engine\nHDF | PWF | OSF]
    C --> E[Leakage-Safe Random Forest\nContinuous Risk Ranking]
    D --> F{Decision Logic Layer}
    E --> F
    F -->|rule_triggered OR risk >= 0.515| G[Calibrated Alarm & Attribution]
    G --> H[FastAPI Endpoint / Streamlit UI]
```

### Deterministic Physics Equations
- **Heat Dissipation Failure (HDF):** `(Process temp - Air temp) < 8.6 K` AND `Rotational speed < 1380 RPM`
- **Power Failure (PWF):** Mechanical Power $P = \tau \cdot \omega \notin [3500\text{ W}, 9000\text{ W}]$
- **Overstrain Failure (OSF):** $\text{Tool Wear} \times \text{Torque} > \text{Limit}_{\text{Type}}$ (Type L: 11k, M: 12k, H: 13k $\text{min}\cdot\text{Nm}$)

Across all 10,000 rows in the raw dataset, these equations reconstruct the dataset's ground truth failure mode flags with **zero mismatches**.

---

## What I Found: Key Technical Insights

1. **Deterministic Rules Account for 84.66% of Failures:** On the full dataset, physical rules alone detect 287 out of 339 failures with zero false alarms. On holdout binary decisions, generic ML does not detect any failure that physics rules missed.
2. **ML Excels at Continuous Ranking and Alert Tiering:** While binary decisions tie, ML provides well-calibrated probabilities ($ECE = 1.23\%$, $Brier = 0.0085$), boosting ROC-AUC from 0.859 to 0.972. Under asymmetric operational loss ($C_{miss} : C_{fa} = 20:1$), lowering threshold to 0.160 reduces plant downtime costs by 18.2%.
3. **ML Fails to Generalize to Unseen Failure Modes:** In Leave-One-Failure-Mode-Out testing, Random Forest trained without HDF catches only **7.83%** of HDF events; trained without PWF, it catches only **14.12%**. In contrast, physics rules achieve **100% recall** across all held-out modes because thermodynamics and energy conservation hold independently of training data distributions.
4. **Physics Features Stabilize Validation Shift:** Because temperature measurements exhibit random-walk drift, standard random-shuffled CV overestimates performance. Under blocked CV, models trained on raw features lose 0.133 AP, whereas physics-informed features reduce this degradation to 0.045 AP.
5. **Residual Failures are Stochastic Tool Wear Events:** The 11 failures missed by both systems on the holdout are predominantly Tool Wear Failures (TWF). Mann-Whitney U testing reveals that while cumulative tool wear is significantly elevated ($p = 1.22 \times 10^{-17}$), operational sensors (RPM, torque, temps) remain within normal envelopes ($p > 0.10$). Without high-frequency vibration signals, statistical ML cannot predict the exact moment of failure without unacceptable false alarms.

---

## What I Would Do With Real-World Data

- **High-Frequency Vibration Telemetry:** Real industrial predictive maintenance relies on fast Fourier transforms (FFT), spectral kurtosis, and envelope analysis of accelerometer data to detect sub-surface bearing spalls and gear pitting.
- **Survival Analysis / Remaining Useful Life (RUL):** Rather than static binary classification, operational deployments require continuous hazard estimation (e.g., Cox Proportional Hazards or Weibull survival regression) conditioned on operating history.
- **Hierarchical Production Cascading:** Implement a multi-tier architecture where deterministic physics limits trigger immediate zero-latency safety interlocks, while statistical ML models evaluate long-term drift in the residual subspace.

---

## Project Structure

```
predictive-maintenance-audit/
├── .github/workflows/ci.yml  # GitHub Actions CI matrix (Py 3.11-3.13)
├── dashboard/app.py          # Interactive Streamlit UI
├── data/raw/ai4i2020.csv     # Verified UCI benchmark dataset
├── docs/
│   ├── CHANGELOG.md          # Comprehensive modification log
│   ├── PROJECT_DECISIONS.md  # Methodology contract & locked thresholds
│   ├── WRITEUP.md            # 1,000-word engineering narrative
│   └── architecture.md       # Detailed data flow and decision logic
├── maint_audit/              # Installable Python package
│   ├── __init__.py           # Package namespace
│   ├── api.py                # FastAPI REST endpoints (/health, /predict)
│   ├── config.py             # Physical constants and boundary thresholds
│   ├── data.py               # Data ingestion and row-level schema checks
│   ├── features.py           # Physics-informed feature engineering
│   ├── models.py             # Scikit-learn ML pipeline factories
│   ├── predict.py            # CLI and DataFrame prediction pipeline
│   ├── rules.py              # Exact physical rule reconstructions
│   ├── train.py              # Cross-validation and model training
│   └── validation.py         # Blocked CV splitters and OOF scoring
├── models/
│   └── rf_engineered.joblib  # Serialized production pipeline
├── reports/
│   ├── metrics.json          # Single source of truth for all project numbers
│   ├── bootstrap_ci.json     # 2,000-sample bootstrap confidence intervals
│   ├── calibration.json      # Brier score, ECE, reliability bins
│   ├── cost_threshold.json   # Cost-sensitive threshold optimization sweep
│   ├── leave_one_mode_out.json # Generalization across unseen failure modes
│   └── residual_analysis.json  # Statistical characterization of TWF residuals
├── scripts/
│   ├── download_data.py      # Checksum-validated data retrieval
│   ├── generate_metrics.py   # Single-source-of-truth generation pipeline
│   └── verify_all.py         # End-to-end quality and verification script
├── tests/                    # Pytest test suite (100% passing)
├── Dockerfile                # Slim, non-root container image
├── Makefile                  # Developer workflow automation
├── LICENSE                   # MIT License
├── model_card.md             # Model documentation and disclosures
├── pyproject.toml            # PEP 621 package metadata and dependencies
└── requirements.txt          # Pinned environment requirements
```

---

## Limitations

- **Synthetic Telemetry:** The AI4I 2020 dataset was synthetically generated. While based on real milling machine equations, it does not capture complex factory noise or non-stationary sensor wear.
- **Holdout Imbalance:** The 2,000-row holdout contains 0 Heat Dissipation Failures (HDF), leaving ML's holdout generalization on HDF untested.
- **Spent Holdout:** The holdout set has been examined and must be considered spent. Any future modeling exploration must use development cross-validation.

---

## Dataset Attribution & Licensing

- **Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset (DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)).
- **Dataset License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Code License:** [MIT License](LICENSE).

---

## Tool Disclosure

Built with AI coding assistance under direct human methodology oversight; every reported numerical result is regenerated and verified by code.