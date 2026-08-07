# PredictGuard — Data Splitting Strategy Report

> **Stage 4: Leakage-Safe Data Splitting**  
> Evaluates machine-level generalization across **100 machines**.

---

## 1. Why Machine-Level Splitting is Required

Standard random row-level splitting (`train_test_split(X, y)`) is a major flaw in predictive maintenance projects.
Because sensor readings are recorded hourly for the same physical equipment, consecutive rows are highly correlated.
Randomly placing adjacent timestamps from the same machine into both train and test sets leads to **severe temporal data leakage**,
giving artificially high test accuracy while failing when deployed to unseen machines.

**PredictGuard enforces strict Machine-Level Splitting:**
- 80% of Machine IDs $\rightarrow$ **Development Set** (for training & 5-fold CV)
- 20% of Machine IDs $\rightarrow$ **Final Test Set** (held out, completely untouched)


---

## 2. Dataset Split Statistics

| Metric | Development Set | Final Test Set | Total / Combined |
|---|---|---|---|
| Unique Machines | **80** (80%) | **20** (20%) | 100 |
| Total Telemetry Rows | 700,880 | 175,220 | 876,100 |
| Positive Samples (y=1) | 13,377 | 3,807 | 17,184 |
| Positive Class % | **1.91%** | **2.17%** | Overall ~1.96% |
| Avg Rows per Machine | 8761.0 | 8761.0 | 8,761 |

---

## 3. Grouped Cross-Validation Summary (5 Folds)

Cross-validation is performed using `StratifiedGroupKFold` grouped by `machineID` on the Development set:

| Fold | Train Machines | Val Machines | Train Rows | Val Rows | Train Pos % | Val Pos % |
|---|---|---|---|---|---|---|
| Fold 1 | 64 | 16 | 560,704 | 140,176 | 1.94% | 1.77% |
| Fold 2 | 64 | 16 | 560,704 | 140,176 | 1.93% | 1.83% |
| Fold 3 | 64 | 16 | 560,704 | 140,176 | 1.90% | 1.93% |
| Fold 4 | 64 | 16 | 560,704 | 140,176 | 1.90% | 1.96% |
| Fold 5 | 64 | 16 | 560,704 | 140,176 | 1.87% | 2.05% |

---

## 4. Automated Leakage Audit Summary

- **Machine Overlap Check**: PASS (0 machine IDs shared between Dev & Test)
- **Row Overlap Check**: PASS (0 exact timestamp-machine pairs shared)
- **CV Fold Isolation**: PASS (0 machines shared between train and validation within any fold)
- **Overall Status**: **100% LEAKAGE-FREE**

---

## 5. Generated Artifacts

| File | Description |
|---|---|
| `data/processed/development.parquet` | Development dataset (80 machines) |
| `data/processed/test.parquet` | Final test dataset (20 machines) |
| `data/processed/cv_folds.json` | Cross-validation fold assignments |
| `data/processed/split_manifest.json` | Split configuration & machine lists |
| `reports/split_strategy_report.md` | This report |
| `reports/figures/split_summary_report.png` | Unified Senior Split Report figure |
