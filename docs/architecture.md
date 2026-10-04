# Architecture — ALGOTHON ALG-DATA-02

## Data flow (production)

```
Sensor Input
    ↓
Input Validation
    ↓
Physics Feature Engineering
    ↓
 ┌───────────────┬─────────────────┐
 ↓               ↓
Rule Engine      Random Forest
HDF/PWF/OSF      ML Risk
 └───────────────┴─────────────────┘
             ↓
       Decision Layer
             ↓
      Explanation Layer
             ↓
      Streamlit Dashboard
```

## What is deterministic
- HDF rule: `(temp_diff < 8.6 K) AND (RPM < 1380)`
- PWF rule: `power_w < 3500 W OR power_w > 9000 W`
- OSF rule: `wear_torque > type-specific threshold` (L=11,000 / M=12,000 / H=13,000)
- Feature formulas: `temp_diff`, `power_w`, `wear_torque`, `strain_ratio`

## What is ML
- Random Forest trained on engineered features with `Type` one-hot encoded
- Outputs a continuous failure probability and a binary decision at the frozen threshold 0.515

## Final decision logic
```
rule_triggered = HDF OR PWF OR OSF
ml_decision    = ml_risk >= 0.515
final_decision = rule_triggered OR ml_decision
```

## Final evaluation
- Notebook 08 final untouched holdout (last 2,000 rows, 39 failures) scored once after all modeling decisions were frozen.

## Production model
- `models/rf_engineered.joblib`
- Loaded only for inference by the dashboard / `src/predict.py`
- No training happens inside the dashboard.
