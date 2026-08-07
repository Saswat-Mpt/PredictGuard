# PredictGuard — MLflow Experiment Summary

> Generated: 2026-08-07 22:36 UTC

---

## 1. Experiment Configuration

| Property | Value |
|---|---|
| Experiment Name | `PredictGuard` |
| Run Name | `PredictGuard_Production` |
| Run ID | `0787f014c9234bf88774b48fbdd2c74a` |
| Tracking URI | `sqlite:///C:/Users/hp/Downloads/ML project/PredictGuard/mlruns/mlflow.db` |

---

## 2. Logged Parameters

| Parameter | Value |
|---|---|
| `calibration_method` | `sigmoid` |
| `cost_C_FN` | `10000.0` |
| `cost_C_FP` | `500.0` |
| `cv_folds` | `5` |
| `feature_count` | `119` |
| `model` | `XGBoostClassifier` |
| `optimal_threshold` | `0.68` |
| `prediction_horizon_hours` | `24` |
| `primary_metric` | `pr_auc` |
| `random_seed` | `42` |
| `split_strategy` | `machine_level` |

---

## 3. Logged Metrics

### Training & CV (Phase 1)

| Metric | Value |
|---|---|
| `train_pr_auc` | `0.970200` |
| `train_precision` | `0.928500` |
| `train_recall` | `0.964800` |
| `train_roc_auc` | `0.999200` |

### Calibration (Phase 2)

| Metric | Value |
|---|---|
| `cal_brier_after` | `0.001800` |
| `cal_brier_before` | `0.002100` |
| `cal_ece_after` | `0.000700` |
| `cal_ece_before` | `0.002700` |
| `cal_pr_auc_after` | `0.971800` |
| `cal_pr_auc_before` | `0.973700` |

### Component Classifier (Stage 10)

| Metric | Value |
|---|---|
| `comp_comp1_f1` | `0.988300` |
| `comp_comp1_precision` | `1.000000` |
| `comp_comp1_recall` | `0.976800` |
| `comp_comp2_f1` | `0.994300` |
| `comp_comp2_precision` | `0.997400` |
| `comp_comp2_recall` | `0.991300` |
| `comp_comp3_f1` | `0.983700` |
| `comp_comp3_precision` | `0.970600` |
| `comp_comp3_recall` | `0.997200` |
| `comp_comp4_f1` | `0.999300` |
| `comp_comp4_precision` | `1.000000` |
| `comp_comp4_recall` | `0.998600` |

### Fleet & Decision (Stages 11–12)

| Metric | Value |
|---|---|
| `decision_dispatch_rate` | `0.020000` |
| `decision_expected_cost` | `337500.000000` |
| `decision_f1` | `0.977400` |
| `decision_precision` | `0.956000` |
| `decision_recall` | `0.999800` |
| `fleet_avg_failure_rate` | `0.021727` |
| `fleet_avg_risk_score` | `2.151107` |
| `fleet_max_risk_score` | `94.000000` |
| `fleet_n_machines` | `20.000000` |

---

## 4. Logged Artifacts

| Category | Count |
|---|---|
| Figures (PNG) | 60 |
| Reports & Data (MD/CSV/JSON) | 25 |
| Model Files (PKL/JSON) | 13 |

---

## 5. How to View

```bash
# From project root:
mlflow ui
# Then open: http://localhost:5000
```

Navigate to the **PredictGuard** experiment and click on the run to see:
- All logged parameters
- Interactive metric charts
- Artifact browser (figures, reports, models)
