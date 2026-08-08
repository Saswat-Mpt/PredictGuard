# PredictGuard — Cross-Validation & Hyperparameter Tuning Report

> **Stage 6: Grouped Cross-Validation & Light Hyperparameter Tuning**  
> Generated: 2026-08-08 03:02

---

## 1. Methodology

### Why Grouped Cross-Validation is Required

Standard k-fold CV splits data rows randomly. In predictive maintenance, each machine
produces thousands of sequential hourly observations. A random row split allows the same
machine's data to appear in both train and validation, causing **machine leakage** —
inflated CV scores that collapse in production when the model encounters new machines.

**Solution**: `StratifiedGroupKFold(groups=machineID)` guarantees that every row for
a machine appears in exactly ONE fold — either train or validation, never both.

### Why PR-AUC is the Primary Metric

With ~1.96% positive failure rate, **Accuracy is deceptive**: a dummy model predicting
all-zero achieves 98.1% accuracy with 0% recall. PR-AUC (Average Precision) specifically
measures performance on the rare failure class across all decision thresholds.

---

## 2. Fold Statistics (Development Set — 80 Machines)

| Fold | Train Machines | Val Machines | Train Rows | Val Rows | Val Positives | Val Negatives | Val Failure Rate | Leakage |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 16 | 560,704 | 140,176 | 2,484 | 137,692 | 1.772% | ✅ None |
| 2 | 64 | 16 | 560,704 | 140,176 | 2,562 | 137,614 | 1.828% | ✅ None |
| 3 | 64 | 16 | 560,704 | 140,176 | 2,706 | 137,470 | 1.930% | ✅ None |
| 4 | 64 | 16 | 560,704 | 140,176 | 2,751 | 137,425 | 1.963% | ✅ None |
| 5 | 64 | 16 | 560,704 | 140,176 | 2,874 | 137,302 | 2.050% | ✅ None |

---

## 3. Cross-Validation Results (Mean ± Std [95% CI])

| Model | PR-AUC | ROC-AUC | F1-SCORE | RECALL | PRECISION |
|---|---|---|---|---|---|
| XGBoost | 0.9717 ± 0.0031 (±0.0038) | 0.9991 ± 0.0003 (±0.0004) | 0.9453 ± 0.0085 (±0.0105) | 0.9590 ± 0.0142 (±0.0176) | 0.9324 ± 0.0180 (±0.0224) |
| Random Forest | 0.9707 ± 0.0032 (±0.0040) | 0.9990 ± 0.0013 (±0.0016) | 0.9476 ± 0.0033 (±0.0041) | 0.9490 ± 0.0076 (±0.0094) | 0.9463 ± 0.0125 (±0.0155) |
| Logistic Regression | 0.7776 ± 0.0430 (±0.0534) | 0.9938 ± 0.0032 (±0.0040) | 0.6813 ± 0.0221 (±0.0275) | 0.9885 ± 0.0059 (±0.0073) | 0.5202 ± 0.0266 (±0.0331) |

---

## 4. Hyperparameter Search

### Search Configuration

| Parameter | Description |
|---|---|
| Model | XGBoost (best Stage 5 PR-AUC: 0.9702) |
| Strategy | `RandomizedSearchCV` |
| Inner CV | `StratifiedGroupKFold(n_splits=5)` grouped by `machineID` |
| Scoring | `average_precision` (PR-AUC) |
| Iterations | 5 |
| Random Seed | 42 |

### Search Space

| Parameter | Values |
|---|---|
| `max_depth` | [3, 4, 5, 6, 7, 8] |
| `learning_rate` | [0.01, 0.05, 0.08, 0.1, 0.15, 0.2] |
| `n_estimators` | [100, 150, 200, 250, 300] |
| `min_child_weight` | [1, 3, 5, 7] |
| `subsample` | [0.6, 0.7, 0.8, 0.9, 1.0] |
| `colsample_bytree` | [0.6, 0.7, 0.8, 0.9, 1.0] |

### Best CV Score (PR-AUC): **0.9600**

### Best Parameters Found

```json
{
  "subsample": 0.6,
  "n_estimators": 200,
  "min_child_weight": 5,
  "max_depth": 5,
  "learning_rate": 0.05,
  "colsample_bytree": 0.6
}
```

---

## 5. Before vs After Tuning (Final Test Set — 20 Unseen Machines)

| Metric | Default XGBoost | Tuned XGBoost | Delta |
|---|---|---|---|
| PR-AUC | 0.9702 | 0.9737 | +0.0035 |
| ROC-AUC | 0.9992 | 0.9996 | +0.0004 |
| F1-SCORE | 0.9463 | 0.9474 | +0.0011 |
| RECALL | 0.9648 | 0.9869 | +0.0221 |
| PRECISION | 0.9285 | 0.9110 | -0.0175 |

---

## 6. Limitations

1. **No calibration applied**: Raw probabilities may be miscalibrated. Calibration begins in Phase 2.
2. **Light search only**: 5 RandomizedSearchCV iterations explore a focused region of the search space.
3. **XGBoost only tuned**: LR and RF were not tuned (acceptable for Stage 6 baselines).
4. **Static `scale_pos_weight`**: Fixed globally; per-fold recomputation is used in CV only.

---

## 7. Artifacts Saved

| Artifact | Path |
|---|---|
| Best Tuned Model | `models/best_model.pkl` |
| Best Model Metadata | `models/best_model_metadata.json` |
| CV Results | `reports/cv_results.csv` |
| RandomSearch Results | `reports/random_search_results.csv` |
| Best Parameters | `reports/best_parameters.json` |
| CV Summary | `reports/cv_summary.json` |
| CV Metric Distribution | `reports/figures/cv_metric_distribution.png` |
| Fold-wise PR-AUC | `reports/figures/cv_foldwise_prauc.png` |
| Fold-wise ROC-AUC | `reports/figures/cv_foldwise_rocauc.png` |
| Before vs After Tuning | `reports/figures/cv_before_vs_after_tuning.png` |
| Hyperparameter Importance | `reports/figures/cv_hyperparam_importance.png` |
| Training Time Comparison | `reports/figures/cv_training_time_comparison.png` |
