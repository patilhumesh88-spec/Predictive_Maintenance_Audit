# Engineering Narrative: When Machine Learning Matches a Lookup Table

*An empirical audit of physical failure mechanisms vs. machine learning risk scoring on industrial telemetry.*

---

## 1. The Premise: Can ML Outperform Basic Physics?

Predictive maintenance is frequently framed as a standard tabular classification benchmark: ingest continuous sensor streams, engineer interaction terms, train an ensemble of gradient-boosted trees or deep nets, and optimize the F1 score.

When approaching the UCI AI4I 2020 Predictive Maintenance benchmark—a synthetic dataset of 10,000 milling machine operating cycles containing 339 failure events—the initial goal was conventional: build a production-grade failure prediction pipeline with strict data leakage discipline (excluding label-derived consequence flags like `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`).

However, an early exploratory audit of the dataset's generating documentation revealed an uncomfortable question: **If the underlying mechanical degradation modes are governed by basic physical equations, how much predictive value does a statistical machine learning model actually provide over deterministic physics rules?**

To answer this honestly, I designed a rigorous comparative study: reverse-engineer the physical rules from first principles, evaluate them as a standalone baseline, benchmark ML models with and without physics-informed feature representations, and stress-test both under severe out-of-distribution conditions.

---

## 2. Reverse-Engineering the Deterministic Physics Engine

Careful examination of the domain mechanics allowed exact mathematical reconstruction of three primary failure modes:

1. **Heat Dissipation Failure (HDF):** Occurs when the difference between process temperature and ambient temperature falls below 8.6 K while rotational spindle speed is under 1,380 RPM:
   $$\Delta T = T_{\text{process}} - T_{\text{air}} < 8.6\text{ K} \quad \wedge \quad \omega < 1380\text{ RPM}$$
2. **Power Failure (PWF):** Mechanical power drawn by the cutter spindle falls outside its safe operating window ($[3500\text{ W}, 9000\text{ W}]$):
   $$P = \tau \times \left(\omega \cdot \frac{2\pi}{60}\right) \notin [3500, 9000]\text{ W}$$
3. **Overstrain Failure (OSF):** The product of cumulative tool wear and cutting torque exceeds a structural threshold specific to the machine product variant (Type L: 11,000; Type M: 12,000; Type H: 13,000 $\text{min}\cdot\text{Nm}$):
   $$\text{Strain} = w \times \tau > \text{Limit}_{\text{Type}}$$

When evaluated against all 10,000 rows in the raw dataset, **these three rules reproduced their corresponding raw dataset flags with zero mismatches across all 10,000 rows**.

More importantly, evaluating these simple algebraic checks as a standalone binary classifier across the entire 10,000-observation dataset yielded:
- **True Positives:** 287 / 339 failures
- **False Positives:** 0
- **Precision:** 1.0000
- **Recall:** 0.8466
- **F1 Score:** 0.9169

A zero-parameter deterministic rule engine accounted for nearly 85% of all equipment failures with 100% precision.

---

## 3. The Holdout Result: An Unflattering Tie

To evaluate whether a Random Forest classifier could capture residual failures beyond the rules, all architectural decisions, feature definitions, and decision thresholds were frozen on the first 8,000 development rows. A binary threshold of $\tau = 0.515$ was locked based on out-of-fold F1 optimization.

The remaining 2,000 rows (containing 39 failures) were preserved as an untouched holdout and scored **exactly once**:

| System | Precision | Recall | F1 Score | Average Precision (AP) | ROC-AUC | TP | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physics Rules Only** | 1.0000 | 0.7179 | 0.8358 | 0.7234 | 0.8590 | 28 | 0 | 11 |
| **Random Forest ML** | 1.0000 | 0.7179 | 0.8358 | 0.7851 | 0.9720 | 28 | 0 | 11 |
| **Hybrid (Rules $\cup$ ML)** | 1.0000 | 0.7179 | 0.8358 | 0.7851 | 0.9720 | 28 | 0 | 11 |

**The result was an exact tie.**

All three systems produced identical confusion matrices: 28 True Positives, 0 False Positives, and 11 False Negatives. Every single failure identified by the Random Forest was already caught by the deterministic rules. Combining them into a hybrid system changed zero decisions.

In terms of binary decisions, machine learning added zero net recall over elementary physics equations.

---

## 4. Where Machine Learning Actually Adds Value

While ML failed to improve binary decision metrics on the holdout, deeper evaluation revealed two critical capabilities:

### A. Continuous Risk Ranking & Calibration
While physics rules output binary step functions (0 or 1), the Random Forest outputs well-calibrated continuous failure probabilities ($Brier = 0.0085$, $ECE = 1.23\%$). This continuous ranking yielded an Average Precision boost over the rules on development data ($AP = 0.906$ vs. $0.869$, paired bootstrap $\Delta = +0.038$, $p < 0.01$) and elevated ROC-AUC from $0.859$ to $0.972$.

In an operational setting, this enables alert tiering (e.g., maintenance watchlists at $P > 0.20$ vs. emergency shutdowns at $P > 0.50$). Furthermore, cost-sensitive threshold analysis proved that if a missed failure is 20x more expensive than an inspection false alarm ($C_{miss} : C_{fa} = 20:1$), lowering the operational threshold to $0.160$ reduces total plant downtime costs by 18.2% (achieving 89.3% recall).

### B. Physics-Informed Features Stabilize Validation Shift
Raw sensor values (RPM, Torque, Temps) suffered dramatic performance degradation under blocked cross-validation—a test where contiguous blocks of rows are held out to test non-stationary shifts. Random Forest on raw features plunged by 0.133 in AP. In contrast, feeding the model derived physics features ($\Delta T$, $P = \tau \cdot \omega$, and $\text{strain\_ratio}$) reduced this gap to just 0.045, demonstrating that embedding physical invariances protects models against environmental drift.

---

## 5. The Stress Test: Leave-One-Failure-Mode-Out

To test the hypothesis that statistical models are brittle when confronted with unseen failure mechanisms, I executed a **Leave-One-Failure-Mode-Out (LOFMO)** experiment: train the Random Forest on development rows with one failure mode completely removed, then evaluate recall exclusively on the held-out mode:

- **Held-out HDF (Heat Dissipation):** Rules Recall = **100.0%** (115/115) | ML Recall = **7.83%** (9/115)
- **Held-out PWF (Power Failure):** Rules Recall = **100.0%** (85/85) | ML Recall = **14.12%** (12/85)
- **Held-out OSF (Overstrain Failure):** Rules Recall = **100.0%** (77/77) | ML Recall = **10.39%** (8/77)

When ML has not observed a specific mechanical failure pattern in training, it misses 86%–92% of occurrences because statistical correlations cannot extrapolate beyond their support. In contrast, deterministic physics equations generalize unconditionally because the conservation of energy and thermodynamics hold regardless of the training sample distribution.

---

## 6. The Residual Failures: Why the 11 Holdout Failures Were Missed

Why did both systems miss the remaining 11 failures?

An audit of the residual failures (41 in development, 52 in full data) revealed that **83% are Tool Wear Failures (TWF)**. Non-parametric hypothesis testing (Mann-Whitney U) confirmed that tool wear is significantly elevated in residual failures compared to normal machines (median 211 min vs. 107 min, $U = 279,879.5$, $p = 1.22 \times 10^{-17}$).

However, operational sensors (torque, RPM, temperature difference) operate completely within normal envelopes ($p > 0.10$). In this synthetic dataset, tool wear failure is modeled as a stochastic point process between 200 and 240 minutes rather than a progressive physical degradation with continuous acoustic or vibrational warnings. Without continuous degradation telemetry, statistical models cannot predict the exact moment of failure without generating severe false alarm rates.

---

## 7. What I Would Do Differently with Real Data

1. **High-Frequency Vibration Telemetry:** Real predictive maintenance relies on fast-Fourier transform (FFT) spectral features (bearing pass frequencies, kurtosis, crest factors) from accelerometers, not single snapshot averages.
2. **Survival Analysis & Remaining Useful Life (RUL):** Rather than binary classification on arbitrary time-slices, real applications require time-to-event modeling (Cox proportional hazards or Weibull survival regression) conditioned on operating history.
3. **Hybrid Architecture:** The optimal production design is not an ensemble, but a hierarchical cascade: deterministic physical safety filters execute first with zero latency; statistical anomaly detection operates downstream only on the residual subspace.
