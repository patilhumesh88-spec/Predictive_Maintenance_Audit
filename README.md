# ALGOTHON '26 — ALG-DATA-02
## Predict What Happens Next

**Physics-informed predictive maintenance with documented-rule analysis + ML risk scoring**

> This project audits the **synthetic** UCI AI4I 2020 Predictive Maintenance benchmark. The dataset has no real timestamp; ordered/block validation is a robustness stress test only, not temporal forecasting. No real-factory deployment claim is made.

---

### Problem
Predict machine failure from sensor telemetry while investigating whether generic ML can improve on documented physical failure mechanisms — without leaking the failure labels into the model.

### Project Thesis
"A leakage-aware predictive-maintenance study that audits a synthetic benchmark, reproduces its documented failure mechanisms, adds physics-informed features, compares ML models, stress-tests validation assumptions, and investigates failures not explained by the deterministic rules."

---

## Dataset
- Source: UCI AI4I 2020 Predictive Maintenance Dataset (`data/raw/ai4i2020.csv`)
- 10,000 rows · 14 original columns
- 0 missing · 0 duplicates
- **339 failures** — overall failure rate **3.39%**
- Final untouched holdout: last 2,000 rows, **39 failures**
- Synthetic benchmark — not a real factory

## Physics Rule Audit (Documented Failure Mechanisms)
- **HDF**: `(process temp - air temp) < 8.6 K` **AND** `RPM < 1380`
- **PWF**: `mechanical power < 3500 W` OR `> 9000 W`
- **OSF**: `tool wear × torque > type-specific threshold` (L=11,000 / M=12,000 / H=13,000)
- Full-dataset reproduction: **287 / 339** failures covered → **84.66%**
- Precision **1.00** · F1 **0.9169**
- **52** failures remain rule-unexplained across the full dataset (do not call them "unpredictable")

## Rule-Unexplained Failure Analysis (Development Only)
- 8,000 development rows · 300 failures
- 259 rule-covered · **41 rule-unexplained**
- Composition: **34 TWF**, **1 RNF**, 6 anomalous (flag present but Machine failure = 0)
- OOF Random Forest detection:
  - threshold ≥ 0.30 → **3 / 41**
  - threshold ≥ 0.40 / 0.50 / 0.60 / 0.70 → **0 / 41**
- Median OOF risk: normal = 0.000 · rule-covered = 0.854 · rule-unexplained = 0.061

## Feature Engineering
Physics-informed features used by the ML pipeline (no target-derived features):
- `temp_diff` = process temperature - air temperature
- `power_w` = torque × RPM × 2π/60
- `wear_torque` = tool wear × torque
- `strain_ratio` = wear_torque / type-specific limit

## Model Comparison
Engineered Random Forest under stratified shuffled 5-fold CV:
- F1 ≈ **0.890** · AP ≈ **0.907** · ROC-AUC ≈ **0.979**

Other models benchmarked for documentation only: Logistic Regression, HistGradientBoosting, XGBoost.
Production model remains the engineered Random Forest.

## Validation Stress Test
- Stratified shuffled CV (optimistic IID baseline)
- Ordered block CV (robustness stress test — **not** temporal forecasting; dataset has no real timestamp)
- Engineered RF shuffled AP ≈ 0.907 vs ordered-block AP ≈ 0.862

## Final Holdout Evaluation (Notebook 08 — untouched holdout, scored once)
Holdout: 2,000 rows · 39 failures · threshold = **0.515**

| System | AP | ROC-AUC | TP | FP | FN |
|---|---|---|---|---|---|
| Documented Rules | 0.723 | 0.859 | 28 | 0 | 11 |
| Leakage-safe ML (RF) | 0.785 | 0.972 | 28 | 0 | 11 |
| Hybrid (Rules OR ML) | 0.785 | 0.972 | 28 | 0 | 11 |

These are the **final untouched-holdout** results. The holdout labels were never used for model or threshold selection.

## Production System
- Random Forest + OneHotEncoder(`Type`, handle_unknown="ignore") + engineered features
- Frozen threshold **0.515**
- Final decision: `rule_triggered OR ml_risk >= 0.515`
- Artifact: `models/rf_engineered.joblib`

## Dashboard
```bash
streamlit run dashboard/app.py
```
Live demo: physics rule engine + ML risk + explanations + verified holdout metrics.

---

## Project Structure
```
ALGOTHON/
├── dashboard/app.py          # Streamlit demo (inference only)
├── docs/
│   ├── PROJECT_DECISIONS.md  # Locked methodology contract
│   └── architecture.md       # Architecture + data-flow diagram
├── data/raw/ai4i2020.csv     # UCI AI4I 2020 dataset
├── models/rf_engineered.joblib  # Production model bundle
├── notebooks/                # Audited analysis notebooks 01-08
├── reports/                  # Verified audit outputs
├── src/                      # Training/inference pipeline
├── tests/                    # pytest suite
├── README.md
├── requirements.txt
└── pytest.ini
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Tests
```bash
pytest -q
```

## Run Dashboard
```bash
streamlit run dashboard/app.py
```

## Reproducibility
- Random state = 42 everywhere
- Threshold 0.515 frozen from development OOF optimization (Notebook 08)
- Final holdout scored exactly once after all decisions were frozen

## Limitations
- Synthetic dataset — not a real factory
- No real timestamp; ordered block ≠ temporal forecasting
- Holdout failure rate (1.95%) is lower than development (3.75%)
- TWF/RNF residual failures remain hard; tested ML detects very few of these residual cases
- ML offers stronger continuous risk ranking but does not improve binary recall over the documented rules on the holdout

## Dataset / Source Disclosure
UCI AI4I 2020 Predictive Maintenance Dataset — provided for benchmarking. This is a synthetic benchmark.

## AI / Tool Disclosure
Assisted by AI coding tools (Claude Code) under direct human methodology oversight. All numerical results were verified against executed notebooks before inclusion.

## Submission Links
- GitHub repo: *(to be added after deployment)*
- Demo video: *(to be added)*
- Deployed prototype: *(to be added)*

## Future Improvements
- Investigate TWF-specific sensor patterns without leaking failure-mode flags
- Add uncertainty / calibration reporting to the ML risk score
- Document the residual failure composition in more detail once additional sensor channels are available
