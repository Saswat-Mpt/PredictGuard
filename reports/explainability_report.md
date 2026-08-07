# PredictGuard — Explainability Report

> Generated: 2026-08-07 22:11 UTC

---

## 1. Overview

This report documents **why** the PredictGuard XGBoost model predicts failure for any given machine-hour, using SHAP (SHapley Additive exPlanations). It covers:

- **Global explanations**: which features matter most across all predictions
- **Local explanations**: why a specific machine was flagged
- **Native vs SHAP importance**: a critical comparison

---

## 2. Why SHAP Instead of Built-in Feature Importance?

| Property | Native Gain Importance | SHAP |
|---|---|---|
| Handles feature interactions | ❌ No | ✅ Yes |
| Consistent across models | ❌ No | ✅ Yes (Shapley axioms) |
| Per-prediction explanations | ❌ No | ✅ Yes |
| Direction of effect (push up/down) | ❌ No | ✅ Yes |
| Accounts for correlated features | ❌ Partially | ✅ Better |

**Conclusion**: Native gain importance can over-count high-cardinality features and ignores feature interactions. SHAP provides a theoretically grounded, consistent measure guaranteed by the Shapley value axioms (Efficiency, Symmetry, Dummy, Additivity).

---

## 3. Important Note on Raw vs Calibrated Model

> ⚠️ **SHAP values are computed exclusively on the RAW (un-calibrated) XGBoost model.**
>
> Reason: `CalibratedClassifierCV` (Platt Scaling) wraps the tree with an isotonic/sigmoid > layer that maps raw scores → calibrated probabilities. The SHAP `TreeExplainer` explains > the **tree structure** — not the post-hoc sigmoid layer. Using it on the calibrated wrapper > would produce incorrect or misleading SHAP values.

| Quantity | Source |
|---|---|
| Raw Score (log-odds) | Raw XGBoost model |
| Calibrated Probability | Sigmoid-calibrated wrapper |
| SHAP Values | Raw XGBoost model ✅ |

---

## 4. Global Feature Importance (SHAP)

> Computed on a stratified random sample of **5,000 rows** from the Development set.

### Top 10 Features by Mean |SHAP|

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | `errors_last_24h` | 2.840110 |
| 2 | `rolling_error_rate_24h` | 0.869348 |
| 3 | `volt_rolling_mean_24h` | 0.506991 |
| 4 | `pressure_rolling_min_24h` | 0.360081 |
| 5 | `maintenance_count_last_year` | 0.340815 |
| 6 | `vibration_rolling_mean_24h` | 0.319382 |
| 7 | `error1_count_24h` | 0.288985 |
| 8 | `error2_count_24h` | 0.272600 |
| 9 | `volt_rolling_median_24h` | 0.270684 |
| 10 | `error5_count_24h` | 0.270081 |

---

## 5. SHAP vs Native Importance Comparison

### Top 10 Features — Rank Comparison

| Feature | SHAP Rank | Native Rank | |Rank Delta| |
|---|---|---|---|
| `errors_last_24h` | 1 | 2 | 1 ✅ |
| `rolling_error_rate_24h` | 2 | 1 | 1 ✅ |
| `volt_rolling_mean_24h` | 3 | 9 | 6 ⚠️ |
| `pressure_rolling_min_24h` | 4 | 6 | 2 ✅ |
| `maintenance_count_last_year` | 5 | 27 | 22 ⚠️ |
| `vibration_rolling_mean_24h` | 6 | 8 | 2 ✅ |
| `error1_count_24h` | 7 | 14 | 7 ⚠️ |
| `error2_count_24h` | 8 | 12 | 4 ✅ |
| `volt_rolling_median_24h` | 9 | 17 | 8 ⚠️ |
| `error5_count_24h` | 10 | 11 | 1 ✅ |

**Discussion:**
Features with large rank deltas (marked ⚠️) indicate cases where native gain importance is misleading relative to SHAP. This commonly occurs for features that appear frequently in splits but have low marginal impact (high-frequency, low-gain leaves).

---

## 6. Example Local Explanations

### Machine 16 — 2015-07-01 01:00:00

| Property | Value |
|---|---|
| Raw Score | `0.9998` |
| Calibrated Probability | `94.4%` |
| Risk Level | **CRITICAL** |

**Plain English:**

```
Machine 16 — Failure Risk 94.4% (CRITICAL)

Risk increased because:
  • errors last 24h = 3.000 (SHAP +4.2529)
  • rolling error rate 24h = 0.125 (SHAP +1.3682)
  • rotate rolling mean 24h = 372.548 (SHAP +0.8887)
  • pressure rolling min 24h = 113.285 (SHAP +0.7351)
  • rotate rolling median 24h = 377.556 (SHAP +0.5673)

Risk decreased because:
  • error2 count 24h = 1.000 (SHAP -0.4136)
  • error3 count 24h = 1.000 (SHAP -0.2398)
  • age = 3.000 (SHAP -0.1939)
  • error5 count 24h = 0.000 (SHAP -0.1858)
  • volt rolling mean 24h = 171.061 (SHAP -0.1599)
```

**Top Positive Drivers (increase failure risk):**

- `errors_last_24h` = 3.0000 — SHAP: **+4.2529**
- `rolling_error_rate_24h` = 0.1250 — SHAP: **+1.3682**
- `rotate_rolling_mean_24h` = 372.5477 — SHAP: **+0.8887**
- `pressure_rolling_min_24h` = 113.2851 — SHAP: **+0.7351**
- `rotate_rolling_median_24h` = 377.5563 — SHAP: **+0.5673**

**Top Negative Drivers (decrease failure risk):**

- `error2_count_24h` = 1.0000 — SHAP: **-0.4136**
- `error3_count_24h` = 1.0000 — SHAP: **-0.2398**
- `age` = 3.0000 — SHAP: **-0.1939**
- `error5_count_24h` = 0.0000 — SHAP: **-0.1858**
- `volt_rolling_mean_24h` = 171.0611 — SHAP: **-0.1599**

---

### Machine 16 — 2015-06-30 20:00:00

| Property | Value |
|---|---|
| Raw Score | `0.9998` |
| Calibrated Probability | `94.4%` |
| Risk Level | **CRITICAL** |

**Plain English:**

```
Machine 16 — Failure Risk 94.4% (CRITICAL)

Risk increased because:
  • errors last 24h = 3.000 (SHAP +4.2749)
  • rolling error rate 24h = 0.125 (SHAP +1.3710)
  • rotate rolling mean 24h = 373.730 (SHAP +0.8913)
  • pressure rolling min 24h = 113.285 (SHAP +0.7296)
  • rotate rolling median 24h = 379.222 (SHAP +0.5656)

Risk decreased because:
  • error2 count 24h = 1.000 (SHAP -0.4128)
  • error3 count 24h = 1.000 (SHAP -0.2374)
  • age = 3.000 (SHAP -0.2011)
  • error5 count 24h = 0.000 (SHAP -0.1871)
  • volt rolling min 24h = 136.884 (SHAP -0.1628)
```

**Top Positive Drivers (increase failure risk):**

- `errors_last_24h` = 3.0000 — SHAP: **+4.2749**
- `rolling_error_rate_24h` = 0.1250 — SHAP: **+1.3710**
- `rotate_rolling_mean_24h` = 373.7300 — SHAP: **+0.8913**
- `pressure_rolling_min_24h` = 113.2851 — SHAP: **+0.7296**
- `rotate_rolling_median_24h` = 379.2221 — SHAP: **+0.5656**

**Top Negative Drivers (decrease failure risk):**

- `error2_count_24h` = 1.0000 — SHAP: **-0.4128**
- `error3_count_24h` = 1.0000 — SHAP: **-0.2374**
- `age` = 3.0000 — SHAP: **-0.2011**
- `error5_count_24h` = 0.0000 — SHAP: **-0.1871**
- `volt_rolling_min_24h` = 136.8838 — SHAP: **-0.1628**

---

### Machine 16 — 2015-06-30 23:00:00

| Property | Value |
|---|---|
| Raw Score | `0.9998` |
| Calibrated Probability | `94.4%` |
| Risk Level | **CRITICAL** |

**Plain English:**

```
Machine 16 — Failure Risk 94.4% (CRITICAL)

Risk increased because:
  • errors last 24h = 3.000 (SHAP +4.2624)
  • rolling error rate 24h = 0.125 (SHAP +1.3714)
  • rotate rolling mean 24h = 374.865 (SHAP +0.8972)
  • pressure rolling min 24h = 113.285 (SHAP +0.7355)
  • rotate rolling median 24h = 379.222 (SHAP +0.5697)

Risk decreased because:
  • error2 count 24h = 1.000 (SHAP -0.4131)
  • error3 count 24h = 1.000 (SHAP -0.2529)
  • age = 3.000 (SHAP -0.1939)
  • error5 count 24h = 0.000 (SHAP -0.1875)
  • volt rolling mean 24h = 171.180 (SHAP -0.1619)
```

**Top Positive Drivers (increase failure risk):**

- `errors_last_24h` = 3.0000 — SHAP: **+4.2624**
- `rolling_error_rate_24h` = 0.1250 — SHAP: **+1.3714**
- `rotate_rolling_mean_24h` = 374.8649 — SHAP: **+0.8972**
- `pressure_rolling_min_24h` = 113.2851 — SHAP: **+0.7355**
- `rotate_rolling_median_24h` = 379.2221 — SHAP: **+0.5697**

**Top Negative Drivers (decrease failure risk):**

- `error2_count_24h` = 1.0000 — SHAP: **-0.4131**
- `error3_count_24h` = 1.0000 — SHAP: **-0.2529**
- `age` = 3.0000 — SHAP: **-0.1939**
- `error5_count_24h` = 0.0000 — SHAP: **-0.1875**
- `volt_rolling_mean_24h` = 171.1801 — SHAP: **-0.1619**

---

### Machine 62 — 2015-10-25 06:00:00

| Property | Value |
|---|---|
| Raw Score | `0.0000` |
| Calibrated Probability | `0.0%` |
| Risk Level | **LOW** |

**Plain English:**

```
Machine 62 — Failure Risk 0.0% (LOW)

Risk increased because:
  • rotation div voltage = 2.027 (SHAP +0.0200)
  • age = 20.000 (SHAP +0.0167)
  • model model3 = 0.000 (SHAP +0.0089)
  • pressure div rotation = 0.290 (SHAP +0.0083)
  • pressure x vibration = 4869.991 (SHAP +0.0066)

Risk decreased because:
  • maintenance count last 30 days = 5.000 (SHAP -3.2161)
  • errors last 24h = 0.000 (SHAP -2.2111)
  • maintenance count last year = 30.000 (SHAP -0.9561)
  • rolling error rate 24h = 0.000 (SHAP -0.6487)
  • volt rolling mean 24h = 169.784 (SHAP -0.3959)
```

**Top Positive Drivers (increase failure risk):**

- `rotation_div_voltage` = 2.0266 — SHAP: **+0.0200**
- `age` = 20.0000 — SHAP: **+0.0167**
- `model_model3` = 0.0000 — SHAP: **+0.0089**
- `pressure_div_rotation` = 0.2897 — SHAP: **+0.0083**
- `pressure_x_vibration` = 4869.9906 — SHAP: **+0.0066**

**Top Negative Drivers (decrease failure risk):**

- `maintenance_count_last_30_days` = 5.0000 — SHAP: **-3.2161**
- `errors_last_24h` = 0.0000 — SHAP: **-2.2111**
- `maintenance_count_last_year` = 30.0000 — SHAP: **-0.9561**
- `rolling_error_rate_24h` = 0.0000 — SHAP: **-0.6487**
- `volt_rolling_mean_24h` = 169.7845 — SHAP: **-0.3959**

---

### Machine 95 — 2015-11-01 06:00:00

| Property | Value |
|---|---|
| Raw Score | `0.0000` |
| Calibrated Probability | `0.0%` |
| Risk Level | **LOW** |

**Plain English:**

```
Machine 95 — Failure Risk 0.0% (LOW)

Risk increased because:
  • model model4 = 0.000 (SHAP +0.0644)
  • model model2 = 1.000 (SHAP +0.0169)
  • model model3 = 0.000 (SHAP +0.0106)
  • rotate rolling std 24h = 48.695 (SHAP +0.0072)
  • age = 18.000 (SHAP +0.0051)

Risk decreased because:
  • maintenance count last 30 days = 6.000 (SHAP -3.3400)
  • errors last 24h = 0.000 (SHAP -2.2265)
  • maintenance count last year = 28.000 (SHAP -0.9790)
  • rolling error rate 24h = 0.000 (SHAP -0.6584)
  • volt rolling mean 24h = 165.341 (SHAP -0.3539)
```

**Top Positive Drivers (increase failure risk):**

- `model_model4` = 0.0000 — SHAP: **+0.0644**
- `model_model2` = 1.0000 — SHAP: **+0.0169**
- `model_model3` = 0.0000 — SHAP: **+0.0106**
- `rotate_rolling_std_24h` = 48.6951 — SHAP: **+0.0072**
- `age` = 18.0000 — SHAP: **+0.0051**

**Top Negative Drivers (decrease failure risk):**

- `maintenance_count_last_30_days` = 6.0000 — SHAP: **-3.3400**
- `errors_last_24h` = 0.0000 — SHAP: **-2.2265**
- `maintenance_count_last_year` = 28.0000 — SHAP: **-0.9790**
- `rolling_error_rate_24h` = 0.0000 — SHAP: **-0.6584**
- `volt_rolling_mean_24h` = 165.3413 — SHAP: **-0.3539**

---

## 7. Figures Generated

| Figure | Description |
|---|---|
| `shap_summary.png` | Global SHAP dot summary — distribution of feature impacts |
| `shap_beeswarm.png` | Beeswarm — feature value vs SHAP value colour-coded |
| `shap_bar.png` | Mean |SHAP| bar chart — top features by global importance |
| `shap_waterfall.png` | Waterfall for highest-risk machine-hour |
| `shap_force.png` | Force plot for highest-risk machine-hour |
| `native_importance.png` | XGBoost native gain-based importance |
| `importance_comparison.png` | SHAP vs native side-by-side comparison |

---

## 8. Limitations

- SHAP values explain **the model's decision**, not the ground truth.
- TreeExplainer with `interventional` perturbation assumes feature independence; correlated features (e.g., rolling windows at different horizons) may share SHAP mass.
- Local explanations are computed on individual rows; they do not represent the machine's overall maintenance risk over time.
