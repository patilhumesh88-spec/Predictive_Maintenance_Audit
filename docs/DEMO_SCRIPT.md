# Demo Video Script — ALGOTHON '26 ALG-DATA-02

**Target length:** 2–4 minutes  
**Tone:** Clear, judge-friendly, focused on the verified project story.

---

[0:00–0:20] Opening — Problem & Thesis
- “Today we present ALGOTHON’26’s solution to ALG-DATA-02: Predict What Happens Next.”
- “We audit the synthetic UCI AI4I 2020 dataset to ask: Can generic ML improve on documented physical failure mechanisms?”
- “Our thesis: a leakage-aware study that reproduces the rules, adds physics-informed features, and evaluates ML without overclaiming.”

[0:20–0:45] Dataset & Benchmark Audit
- “The dataset: 10,000 rows, 339 failures (3.39%). No missing values, no duplicates.”
- “We identified label defects: 9 unexplained failures (Machine failure=1, all flags=0) and 18 RNF false positives.”
- “Crucially, we never use the failure-mode columns (TWF, HDF, PWF, OSF, RNF) or identifiers (UDI, Product ID) as model inputs.”

[0:45–1:10] Physics Rule Discovery
- “We reconstructed three deterministic rules from the documentation.”
- “HDF: (process temp – air temp) < 8.6 K AND RPM < 1380 → reproduces the HDF column perfectly.”
- “PWF: power < 3500 W OR > 9000 W → reproduces PWF perfectly.”
- “OSF: wear × torque > type‑specific threshold (L=11k, M=12k, H=13k) → reproduces OSF perfectly.”
- “Together these rules explain 287/339 failures → 84.66% coverage, precision 1.00, F1 0.9169.”
- “The remaining 52 failures are rule-unexplained — we do not call them unpredictable.”

[1:10–1:35] ML + Engineered Features
- “To help ML learn physical boundaries, we engineered four physics-informed features:”
- “temp_diff, power_w, wear_torque, strain_ratio (wear_torque divided by type‑specific limit).”
- “Using these features, an engineered Random Forest achieves:”
- “F1 ≈ 0.890, AP ≈ 0.907, ROC‑AUC ≈ 0.979 under shuffled 5‑fold CV.”
- “Other models (LogReg, HistGB, XGB) are benchmarked but not selected for production.”

[1:35–1:55] Validation Stress Test
- “Because adjacent rows share temperature drift (random walks), standard shuffled CV overestimates performance.”
- “We therefore use ordered block CV as a robustness stress test — **not** temporal forecasting.”
- “Engineered RF shuffled AP ≈ 0.907 vs ordered‑block AP ≈ 0.862 — a small gap confirming that physics‑informed features stabilize generalization.”
- “Raw features show a much larger gap (~0.13–0.18 AP), proving the value of the physics layer.”

[1:55–2:35] Live Dashboard Demo
- “Here is the live dashboard (switch to screen share or picture‑in‑picture).”
- “Inputs: Type, Air/Process temp, RPM, Torque, Tool wear.”
- “Outputs: derived physics, rule‑engine diagnostics (HDF/PWF/OSF), ML risk (probability, threshold 0.515), final decision.”
- “Demonstrate three cases:”
- “1) Normal input → ‘No documented rule triggered’.”
- “2) HDF‑triggering input (low temp diff, low RPM) → ‘Rule‑triggered failure risk (HDF)’.”
- “3) Invalid input (e.g., negative torque) → proper error message, no silent scoring.”
- “Show the Model Performance section: final untouched holdout (2,000 rows, 39 failures) yields TP=28, FP=0, FN=11, AP=0.785, ROC‑AUC=0.972 for both ML and hybrid.”
- “Emphasize: the holdout was never used for model or threshold selection.”

[2:35–3:00] Closing — Results & Limitations
- “Summary: Documented rules explain the majority of failures with perfect precision. Engineered features improve ML’s continuous risk ranking. Ordered block validation is stricter than shuffled CV. Residual failures are predominantly TWF/RNF; the tested ML model detects very few of these residual cases.”
- “Limitations: Synthetic benchmark, no real timestamp, holdout failure rate lower than development, no claim of real‑factory deployment or hybrid superiority beyond what the holdout shows.”
- “Thank you.”