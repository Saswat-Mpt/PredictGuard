# PredictGuard — Baseline Model Training & Evaluation Report

> **Stage 5: Baseline and Stronger Models**  
> Models evaluated on held-out **Final Test Set (20 machines, 175,220 rows)**.

---

## 1. Why PR-AUC is the Primary Metric (Not Accuracy)

In predictive maintenance datasets with extreme class imbalance (~1.9% positive failure rate),
**Accuracy is a misleading metric**.

A naive dummy model predicting `y=0` for every single timestamp achieves **98.1% Accuracy**,
yet catches **0% of machine failures** (0% Recall, 0% F1), causing catastrophic unscheduled downtime.

**PR-AUC (Precision-Recall AUC / Average Precision)** measures the area under the Precision-Recall curve
specifically for the rare positive failure class. It directly reflects how effectively the system detects
real failure risks while controlling false alarms.


---

## 2. Model Performance Comparison Table

| Model Name | PR-AUC (Primary) | ROC-AUC | F1 Score | Recall | Precision | Bal Acc | MCC | Log Loss | Fit Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.7417** | 0.9953 | 0.6821 | 0.9974 | 0.5183 | 0.9884 | 0.7115 | 0.0850 | 18.17s |
| **Random Forest** | **0.9644** | 0.9995 | 0.9457 | 0.9464 | 0.9449 | 0.9726 | 0.9445 | 0.0090 | 96.67s |
| **XGBoost** | **0.9702** | 0.9992 | 0.9463 | 0.9648 | 0.9285 | 0.9816 | 0.9452 | 0.0106 | 15.32s |

---

## 3. Confusion Matrix Breakdown

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|---|---|---|---|---|
| **Logistic Regression** | 167,884 | 3,529 | 10 | **3,797** |
| **Random Forest** | 171,203 | 210 | 204 | **3,603** |
| **XGBoost** | 171,130 | 283 | 134 | **3,673** |

---

## 4. Key Engineering Takeaways

1. **Tree-based models (XGBoost & Random Forest)** significantly outperform linear Logistic Regression because failure risk is governed by non-linear interactions between sensor degradation, error history, and maintenance elapsed time.
2. **Class imbalance weighting** (`scale_pos_weight` in XGBoost and `class_weight='balanced'` in Random Forest) is essential for enabling models to output meaningful probability distributions for rare failures.
3. **Zero Machine Leakage**: Because evaluation was conducted on 20 completely unseen machines, these metrics reflect true operational generalization capability.

---

## 5. Artifacts & Outputs Saved

| Artifact | Location |
|---|---|
| Logistic Regression Model | `models/logistic_regression.pkl` |
| Random Forest Model | `models/random_forest.pkl` |
| XGBoost Model | `models/xgboost.pkl` |
| Training Metadata Registry | `models/training_metadata.json` |
| Unified Evaluation Dashboard | `reports/figures/model_evaluation_dashboard.png` |
| Markdown Report | `reports/model_training_report.md` |
