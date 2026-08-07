# PredictGuard — Component Classifier Report

> Generated: 2026-08-07 22:11 UTC

---

## 1. Objective

Stage 10 predicts **which component will fail** given that the binary model
has already predicted a failure. This converts a binary alert into an
actionable diagnosis: *'Component 3 is predicted to fail with 71% confidence.'*

The component classifier is **only invoked on failure-predicted rows**,
never on normal operation rows.

---

## 2. Dataset Construction

| Property | Value |
|---|---|
| Source | `data/processed/development.parquet` |
| Filter | `y_failure == 1` AND `failure_component` not null |
| Training samples | 13,377 |
| Feature count | 119 |

### Class Distribution (Training Set)

| Component | Count | % |
|---|---|---|
| `comp2` | 4,842 | 36.2% |
| `comp1` | 3,675 | 27.5% |
| `comp4` | 3,018 | 22.6% |
| `comp3` | 1,842 | 13.8% |

> ℹ️ Class weights (inverse frequency) applied during training to handle imbalance.

---

## 3. Model Architecture

| Property | Value |
|---|---|
| Algorithm | XGBoost multiclass (`multi:softprob`) |
| Objective | Predict probability of each component class |
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.05 |
| Class balancing | `sample_weight` (inverse frequency) |
| tree_method | `hist` (memory-efficient) |

---

## 4. Evaluation Results

> ⚠️ **Primary metric: Per-class Recall** — missing a real failure component is more costly than a false alarm.

### Aggregate Metrics

| Metric | Value |
|---|---|
| Accuracy | 0.9908 |
| Macro F1 | 0.9914 |
| Weighted F1 | 0.9908 |

### Per-Class Metrics

| Component | Samples | Precision | Recall | F1 |
|---|---|---|---|---|
| `comp1` | 906 | 1.0000 | **0.9768** | 0.9883 |
| `comp2` | 1,149 | 0.9974 | **0.9913** | 0.9943 |
| `comp3` | 1,059 | 0.9706 | **0.9972** | 0.9837 |
| `comp4` | 693 | 1.0000 | **0.9986** | 0.9993 |

---

## 5. Example Diagnosis Cards

### Machine 16 — 2015-07-01 01:00:00

```
Failure Probability : 94.4%  [CRITICAL]
Predicted Component : comp2
Component Confidence: 99.9%

Component Probabilities:
  comp1:                                0.0%
  comp2: █████████████████████████████  99.9%
  comp3:                                0.1%
  comp4:                                0.0%

Recommended: Inspect rotary bearing assembly and lubrication system.
```

### Machine 16 — 2015-06-30 20:00:00

```
Failure Probability : 94.4%  [CRITICAL]
Predicted Component : comp2
Component Confidence: 99.9%

Component Probabilities:
  comp1:                                0.0%
  comp2: █████████████████████████████  99.9%
  comp3:                                0.1%
  comp4:                                0.0%

Recommended: Inspect rotary bearing assembly and lubrication system.
```

### Machine 16 — 2015-06-30 23:00:00

```
Failure Probability : 94.4%  [CRITICAL]
Predicted Component : comp2
Component Confidence: 99.9%

Component Probabilities:
  comp1:                                0.0%
  comp2: █████████████████████████████  99.9%
  comp3:                                0.1%
  comp4:                                0.0%

Recommended: Inspect rotary bearing assembly and lubrication system.
```

### Machine 16 — 2015-07-01 02:00:00

```
Failure Probability : 94.4%  [CRITICAL]
Predicted Component : comp2
Component Confidence: 99.9%

Component Probabilities:
  comp1:                                0.0%
  comp2: █████████████████████████████  99.9%
  comp3:                                0.1%
  comp4:                                0.0%

Recommended: Inspect rotary bearing assembly and lubrication system.
```

---

## 6. Failure Modes & Limitations

- The component classifier relies entirely on the same sensor features as the binary model. It **cannot** distinguish components whose failure signatures overlap significantly.
- Very rare components (< 10 samples) are excluded from training.
- When confidence is low (< 40%), treat the output as a preliminary screening signal, not a definitive diagnosis.
- This classifier is trained on **Development machines only** — performance may vary on machines with unusual operating profiles.

---

## 7. Figures

| Figure | Description |
|---|---|
| `component_confusion_matrix.png` | Confusion matrix across all 4 component classes |
| `component_per_class_recall.png` | Per-class recall bar chart |
| `component_distribution.png` | Class distribution (bar + pie) |
| `component_prediction_distribution.png` | Predicted probability histograms per class |
| `component_roc_ovr.png` | One-vs-Rest ROC curves (AUC per component) |
| `component_feature_importance.png` | Top features for component discrimination |
