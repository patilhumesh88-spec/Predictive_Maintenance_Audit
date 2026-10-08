# Submission Checklist — ALGOTHON '26 ALG-DATA-02

✅ **GitHub repository**
- Publicly accessible at: https://github.com/patilhumesh88-spec/algothon26-data-science
- Contains all source code, notebooks, data, docs, tests, models, dashboard

✅ **Live Streamlit dashboard**
- Deployed (to be completed after this checklist)
- URL: [to be filled]
- Uses `dashboard/app.py` as entry point
- Loads `models/rf_engineered.joblib` for inference only (no retraining)

✅ **Demo video**
- Script prepared in `docs/DEMO_SCRIPT.md`
- Video to be uploaded (YouTube, Streamlit app gallery, or similar)
- Length 2–4 minutes, follows the script exactly
- Does not claim: true temporal forecasting, real-factory deployment, unpredictable residual failures, hybrid superiority not demonstrated

✅ **README.md**
- Complete rewrite with all required sections
- Every numeric claim verified against executed notebooks or test suite
- Includes: project title, problem, thesis, dataset, why this approach, system architecture, data audit, physics rule audit, feature engineering, model comparison, validation stress test, rule-unexplained failure analysis, final holdout evaluation, production system, dashboard, project structure, setup, run tests, run dashboard, reproducibility, limitations, dataset/source disclosure, AI/tool disclosure, submission links
- After deployment: add GitHub repo link, live dashboard link, demo video link

✅ **Architecture**
- `docs/architecture.md` — detailed description
- `docs/architecture.mmd` — Mermaid diagram
- Shows: Sensor Input → Input Validation → Physics Feature Engineering → [Rule Engine | Random Forest] → Decision Layer → Explanation Layer → Streamlit Dashboard
- Documents what is deterministic, what is ML, what is explanatory, what is evaluated, what is not claimed

✅ **Tests passing**
- `pytest -q` passes (19/19)
- Test suite covers: missing columns, invalid/non-numeric input, empty frames, extra ID columns, rule boundaries, label defects, output schema, leakage exclusion, model loading, single-row prediction, rule explanation correctness, prediction reproducibility

✅ **Model artifact**
- `models/rf_engineered.joblib` committed and loadable
- Contains: trained preprocessing + Random Forest pipeline, threshold=0.515, feature metadata
- Verified in a fresh Python process: normal input, rule-triggering input (HDF/PWF), invalid input, unknown Type all handled correctly
- No training occurs during inference

✅ **Dataset/source disclosure**
- Clearly states: UCI AI4I 2020 Predictive Maintenance Dataset, synthetic benchmark
- Provided in `data/raw/ai4i2020.csv`
- No real timestamp; ordered block validation is a robustness stress test only

✅ **AI/tool disclosure**
- Acknowledges use of AI coding tools (Claude Code) under direct human methodology oversight
- All numerical results verified against executed notebooks before inclusion

✅ **Limitations**
- Clearly stated in README: synthetic dataset, no real timestamp, holdout failure rate lower than development, residual failures remain hard, no claim of real-factory deployment, no claim of hybrid superiority beyond holdout results

✅ **Final holdout metrics**
- From Notebook 08 (scored once on untouched holdout):
  - Holdout size: 2,000 rows
  - Holdout failures: 39
  - Threshold: 0.515 (F1-optimal on development OOF)
  - TP: 28, FP: 0, FN: 11
  - AP: 0.785, ROC-AUC: 0.972
  - These numbers appear in README, dashboard, and architecture documentation