# Project Decisions — ALGOTHON ALG-DATA-02

## Dataset
- UCI AI4I 2020 Predictive Maintenance Dataset
- 10,000 rows, 14 original columns, no missing values, no duplicates
- Overall failure rate: 3.39% (339 failures)

## Target & Leakage Columns
- Target: `Machine failure`
- Leakage / label-derived columns (NEVER model inputs): `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
- Identifier columns (NEVER model inputs): `UDI`, `Product ID`

## Final ML Input Features
- Raw: `Type` (one-hot), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
- Engineered: `temp_diff`, `power_w`, `wear_torque`, `strain_ratio`

## Derived-Feature Formulas
- `temp_diff = Process temperature [K] - Air temperature [K]`
- `power_w = Torque * RPM * 2*pi/60`
- `wear_torque = Tool wear * Torque`
- `strain_ratio = wear_torque / strain_limit`
- `strain_limit` by Type: L=11000, M=12000, H=13000 (min*Nm, from dataset documentation)

## Documented Physics Rules (analysis layer ONLY)
- HDF: `(temp_diff < 8.6) AND (RPM < 1380)`
- PWF: `power_w < 3500 OR power_w > 9000`
- OSF: `wear_torque > strain_limit_by_Type`
- No deterministic formula for TWF or RNF; do not invent one.

## Validation Protocol
- Development split: first 80% of rows (8,000 rows, 300 failures)
- Final holdout: last 20% of rows (2,000 rows, 39 failures) — untouched until final evaluation
- Stratified 5-fold shuffled CV: development model selection / reporting
- Ordered block 5-fold CV: robustness stress test ONLY; NOT temporal forecasting

## Selected Model & Threshold
- Production model: Random Forest with OneHotEncoder(Type, handle_unknown="ignore") + engineered features
- Threshold: **0.515** — F1-optimal on development OOF predictions (Notebook 08); frozen, NOT holdout-tuned

## Final Decision Logic
- `rule_triggered = HDF OR PWF OR OSF`
- `ml_risk = RandomForest probability`
- `ml_decision = ml_risk >= 0.515`
- `final_decision = rule_triggered OR ml_decision`

## Verified Holdout Results (Notebook 08, single scoring)
- Rules: AP=0.723, ROC-AUC=0.859, Precision=1.0, Recall=0.718, F1=0.836, 28 TP / 0 FP / 11 FN
- ML: AP=0.785, ROC-AUC=0.972, same binary result as rules on this holdout
- Hybrid: identical binary result (rules already cover the 28 detected failures)

## Terminology We Do NOT Use
- "unpredictable" / "provably unpredictable" for residual failures
- "temporal forecasting" for ordered-block CV
- "ML-only (unexplained)" — use "ML risk detected", "No documented rule triggered", "Residual statistical risk"

## Limitations
- Dataset is synthetic; no real factory deployment claim
- Ordered block is harder than shuffled CV but is a stress test, not forecasting
- Final holdout failure rate (1.95%) is lower than development (3.75%) — report as distribution limitation
- TWF/RNF residual failures are not explained by documented rules
