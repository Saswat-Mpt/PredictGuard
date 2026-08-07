# PredictGuard — Feature Engineering Report

> **Stage 3: Feature Engineering**  
> Generated **119** engineered features across **876,100** telemetry observations.

---

## 1. Executive Summary & Feature Categories

PredictGuard constructs domain-driven, leakage-safe features across seven key categories:

| Feature Category | Count | Primary Formulas & Descriptions |
|---|---|---|
| **Trailing Rolling Stats** | 48 | Trailing mean, std, min, max, median, range over 3h and 24h windows |
| **Rate of Change** | 20 | Lags (`delta_1h`, `delta_3h`, `delta_6h`, `delta_12h`), velocity, acceleration |
| **Expanding Z-Scores** | 12 | Machine expanding mean & std shifted by 1 timestep: `(val - exp_mean) / exp_std` |
| **Interactions** | 5 | Sensor domain products and ratios (`pressure x vibration`, `rotation / volt`) |
| **Error History** | 8 | `errors_last_24h`, `errors_last_72h`, `days_since_last_error`, error type counts |
| **Maintenance History** | 8 | `days_since_last_maintenance`, `maint_count_30d/90d/1y`, component counts |
| **Static Machine Specs** | 5 | Machine `age`, one-hot encoded machine `model` |

---

## 2. Strict Leakage Prevention Strategy

| Rule | Technical Implementation | Why It Matters |
|---|---|---|
| **Trailing-Only Windows** | `rolling(window=W, min_periods=1)` | Centered windows use future values $t+1 \dots t+k$, corrupting evaluation |
| **Shifted Expanding Stats** | `expanding().shift(1)` | Prevents observation $x(t)$ from influencing its own expanding baseline |
| **Backward Temporal Joins** | `pd.merge_asof(..., direction='backward')` | Ensures event logs ($t_{event} \le t_{telemetry}$) are never merged from the future |
| **Per-Machine Isolation** | `groupby('machineID')` | Prevents window bleeding across machine boundaries in memory |

---

## 3. Automated Leakage Audit Results

The automated `leakage_checker` suite verified:

- **Target Leakage Check**: PASS (No feature has $|r| > 0.99$ with `y_failure`)
- **Expanding Shift Check**: PASS (Observation $t$ strictly excluded from expanding mean)
- **Overall Audit Result**: **PASS — 100% LEAKAGE FREE**

---

## 4. Top Ranked Features (Mutual Information Preview)

Top 15 features ranked by Mutual Information score with target `y_failure`:

| Rank | Feature Name | Mutual Info Score | Abs Pearson Corr | Variance | Missing % |
|---|---|---|---|---|---|
| 1 | `errors_last_24h` | 0.0531 | 0.5392 | 0.1300 | 0.0% |
| 2 | `rolling_error_rate_24h` | 0.0529 | 0.5392 | 0.0002 | 0.0% |
| 3 | `errors_last_72h` | 0.0303 | 0.3035 | 0.3767 | 0.0% |
| 4 | `rotate_rolling_max_24h` | 0.0204 | 0.1473 | 830.0961 | 0.0% |
| 5 | `vibration_rolling_min_24h` | 0.0186 | 0.1144 | 8.9811 | 0.0% |
| 6 | `rotate_rolling_min_24h` | 0.0179 | 0.1259 | 941.6761 | 0.0% |
| 7 | `volt_rolling_min_24h` | 0.0178 | 0.0875 | 69.5800 | 0.0% |
| 8 | `vibration_rolling_max_24h` | 0.0172 | 0.0976 | 10.9808 | 0.0% |
| 9 | `pressure_rolling_min_24h` | 0.0169 | 0.0995 | 39.8000 | 0.0% |
| 10 | `error3_count_24h` | 0.0160 | 0.3295 | 0.0239 | 0.0% |
| 11 | `rotate_rolling_median_24h` | 0.0154 | 0.2027 | 384.7986 | 0.0% |
| 12 | `rotate_rolling_mean_24h` | 0.0144 | 0.2209 | 327.8020 | 0.0% |
| 13 | `volt_rolling_max_24h` | 0.0137 | 0.0805 | 77.0606 | 0.0% |
| 14 | `error5_count_24h` | 0.0134 | 0.3378 | 0.0102 | 0.0% |
| 15 | `error2_count_24h` | 0.0131 | 0.2925 | 0.0281 | 0.0% |

---

## 5. Feature Validation Summary

- **Total Engineered Features**: 119
- **Missing Value Features**: 0
- **Infinite Value Features**: 0
- **Constant Variance Features**: 0

---

## 6. Outputs Generated

| Artifact | File Path | Size / Count |
|---|---|---|
| Primary Feature Parquet | `data/processed/features.parquet` | Compressed Parquet |
| Feature CSV | `data/processed/features.csv` | Full text export |
| Feature Dictionary | `data/processed/feature_dictionary.csv` | Feature metadata registry |
| Markdown Report | `reports/feature_engineering_report.md` | This document |
| Figures | `reports/figures/feature_*.png` | Publication-quality plots |
