# PredictGuard — Probability Calibration & Trust Report

> **Phase 2: Stage 7 (Calibration Evaluation) & Stage 8 (Probability Calibration)**  
> Generated: 2026-08-08 03:14

---

## 1. What Calibration Means & Why It Matters

In predictive maintenance systems, high **ROC-AUC (0.999)** or **PR-AUC (0.973)** guarantees
good ranking order, but does **NOT** ensure trustworthy probabilities.

When PredictGuard outputs a **0.70 failure probability** for a machine over the next 24 hours,
fleet operators and maintenance dispatch teams require that **approximately 70 out of 100** machines
receiving that score actually fail. Raw probabilities from tree models trained with `scale_pos_weight`
tend to be **over-confident**, distorting financial risk calculations.

**Phase 2 Goal**: Transform raw model outputs into well-calibrated, operational risk probabilities.

---

## 2. Stage 7 — Raw Model Calibration Benchmarks

| Model Name | Brier Score (Lower Best) | ECE (Lower Best) | MCE | Log Loss | Calibration Status |
|---|---|---|---|---|---|
| **XGBoost** | 0.0021 | **0.0027** | 0.6105 | 0.0101 | Well-calibrated |
| **Random Forest** | 0.0023 | **0.0033** | 0.3004 | 0.0090 | Well-calibrated |
| **Logistic Regression** | 0.0168 | **0.0240** | 0.6281 | 0.0849 | Over-confident (predicted prob > actual frequency) |

---

## 3. Stage 8 — Post-Hoc Calibration Comparison

Calibration was fitted **ONLY on the 80 Development machines** using `CalibratedClassifierCV` with
`StratifiedGroupKFold(5)` — the 20 Final Test machines remained completely untouched.

| Model Variant | Calibration Method | Brier Score | ECE | MCE | Log Loss | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **XGBoost** | None (Raw) | **0.0021** | **0.0027** | 0.6105 | 0.0101 | 0.9737 | 0.9996 |
| **XGBoost (Isotonic)** | Isotonic | **0.0018** | **0.0008** | 0.3043 | 0.0069 | 0.9713 | 0.9995 |
| **XGBoost (Sigmoid)** | Sigmoid | **0.0018** | **0.0007** | 0.3505 | 0.0076 | 0.9718 | 0.9995 |

### Selected Calibration Method: **SIGMOID REGRESSION**

---

## 4. Probability Audit Table (Raw vs Calibrated Sample)

| Machine ID | Raw Risk Prob | Calibrated Risk Prob | Risk Delta | Actual Outcome |
|---|---|---|---|---|
| Machine 90 | `0.9982` | **`0.9434`** | `-0.0548` | ✅ NORMAL (0) |
| Machine 54 | `0.9978` | **`0.9434`** | `-0.0544` | 🚨 FAIL (1) |
| Machine 16 | `0.9969` | **`0.9425`** | `-0.0544` | 🚨 FAIL (1) |
| Machine 54 | `0.9961` | **`0.9418`** | `-0.0543` | 🚨 FAIL (1) |
| Machine 85 | `0.9430` | **`0.8471`** | `-0.0959` | 🚨 FAIL (1) |
| Machine 85 | `0.6648` | **`0.3474`** | `-0.3174` | ✅ NORMAL (0) |
| Machine 95 | `0.6353` | **`0.5360`** | `-0.0993` | ✅ NORMAL (0) |
| Machine 85 | `0.4416` | **`0.1365`** | `-0.3051` | ✅ NORMAL (0) |
| Machine 16 | `0.3640` | **`0.0209`** | `-0.3431` | ✅ NORMAL (0) |
| Machine 62 | `0.3005` | **`0.0659`** | `-0.2346` | ✅ NORMAL (0) |
| Machine 22 | `0.0009` | **`0.0003`** | `-0.0006` | ✅ NORMAL (0) |
| Machine 39 | `0.0008` | **`0.0003`** | `-0.0005` | ✅ NORMAL (0) |
| Machine 54 | `0.0004` | **`0.0003`** | `-0.0001` | ✅ NORMAL (0) |
| Machine 90 | `0.0002` | **`0.0003`** | `+0.0001` | ✅ NORMAL (0) |
| Machine 53 | `0.0001` | **`0.0003`** | `+0.0002` | ✅ NORMAL (0) |

---

## 5. Important Engineering Clarification: SHAP & Raw Models

> ⚠️ **Why SHAP Explainability (Phase 3) Must Use the RAW Model:**  
> Post-hoc calibration wrappers (Isotonic/Sigmoid) apply a monotonic 1D transformation `g(f(x))`
> to the final probability output. Feature importance and TreeSHAP interaction values must be computed
> directly on the underlying **RAW tree structure** (`raw_model.pkl`) to preserve exact game-theoretic mathematical axioms.

---

## 6. Output Artifacts Saved

| Artifact | Path |
|---|---|
| Raw XGBoost Model | `models/raw_model.pkl` |
| Calibrated XGBoost Model | `models/calibrated_model.pkl` |
| Calibration Metadata Card | `models/calibration_metadata.json` |
| Raw Calibration Metrics | `reports/calibration_metrics.csv` |
| Before/After Metrics | `reports/calibration_before_after.csv` |
| Unified Calibration Dashboard | `reports/figures/calibration_dashboard.png` |
| Reliability Diagrams | `reports/figures/reliability_diagrams.png` |
| Probability Histograms | `reports/figures/probability_histograms.png` |
