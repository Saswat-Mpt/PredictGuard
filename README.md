# 🛡️ PredictGuard — Explainable Predictive Maintenance & Decision Support System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.15+-yellow.svg)](https://mlflow.org/)
[![Pytest](https://img.shields.io/badge/Pytest-64%20passed-success.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

PredictGuard is an end-to-end machine learning engineering project designed to solve real-world industrial predictive maintenance challenges across **876,100 hourly telemetry records** from 100 industrial machines.

---

## ❓ Why PredictGuard?

Most predictive maintenance projects follow a simple template:
> *Load data → Train XGBoost → Report 98% Accuracy → Stop.*

In real industrial operations, that approach fails for three critical reasons:
1. **Uncalibrated Probabilities**: Raw XGBoost models output overconfident probabilities. A model reporting an 80% failure risk may actually correspond to a 30% empirical failure rate—leading to wasted technician dispatch budgets.
2. **Symmetric Metrics vs Asymmetric Business Costs**: Standard accuracy assumes missing a catastrophic breakdown costs the same as sending a technician for a 15-minute inspection. In reality, a missed failure (False Negative) costs **$10,000**, while a false alarm (False Positive) costs **$500**.
3. **Data Leakage in Time-Series**: Splitting telemetry rows randomly across train/test splits leaks future telemetry and contiguous hours from the same machine, inflating test metrics.

PredictGuard addresses these challenges by combining **grouped machine-level validation**, **post-hoc probability calibration**, **tree-path SHAP driver attribution**, **component failure diagnosis**, and **cost-sensitive threshold optimization**.

---

## 📐 System Architecture

![PredictGuard Architecture](docs/architecture.svg)

---

## 🔬 Experimental Results & Evidence

All metrics reported below are automatically computed by the pipeline and linked directly to reproducible CSV and JSON reports in the repository.

### 1. Model Baseline & Grouped CV (Phase 1)
Evaluated on 80 Development machines using 5-fold `GroupKFold` cross-validation (zero machine-ID overlap). See [`reports/cv_results.csv`](reports/cv_results.csv) and [`reports/cv_summary.json`](reports/cv_summary.json).

| Model | Cross-Validation Strategy | Mean CV PR-AUC | Mean CV ROC-AUC | Status |
|---|---|---|---|---|
| **Logistic Regression** | Grouped 5-Fold CV | 0.7776 | 0.9938 | Baseline |
| **Random Forest** | Grouped 5-Fold CV | 0.9707 | 0.9990 | Strong |
| **Tuned XGBoost** | Grouped 5-Fold CV | **0.9717** | **0.9991** | **Best** |

### 2. Probability Calibration (Phase 2)
Evaluated using Brier Score and Expected Calibration Error (ECE) across 10 probability bins. See [`reports/calibration_metrics.csv`](reports/calibration_metrics.csv) and [`reports/calibration_before_after.csv`](reports/calibration_before_after.csv).

| Model State | Brier Score | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) |
|---|---|---|---|
| **Raw XGBoost** | 0.0021 | 0.0027 | 0.6105 |
| **Calibrated XGBoost (Sigmoid)** | **0.0018** | **0.0007** | **0.3505** |

> **Key Finding**: Post-hoc Sigmoid (Platt Scaling) calibration reduced Expected Calibration Error by **74.1%** (0.0027 → 0.0007) while causing only a minimal change in ranking discrimination (PR-AUC 0.9737 → 0.9718).

### 3. Component Failure Diagnosis (Phase 3 Stage 10)
Multiclass XGBoost trained with inverse-frequency sample weighting to isolate exact component failures (`comp1` to `comp4`). See [`reports/component_metrics.csv`](reports/component_metrics.csv).

| Component Class | Test Precision | Test Recall | Test F1-Score |
|---|---|---|---|
| **comp1** | 1.0000 | 0.9768 | 0.9883 |
| **comp2** | 0.9974 | 0.9913 | 0.9943 |
| **comp3** | 0.9706 | 0.9972 | 0.9837 |
| **comp4** | 1.0000 | 0.9986 | 0.9993 |
| **Overall (Macro Average)** | **0.9920** | **0.9910** | **0.9910 (99.1% Acc)** |

### 4. Cost-Based Dispatch Decision Optimization (Phase 3 Stage 12)
Asymmetric cost matrix: False Negative ($C_{FN}$) = **$10,000** | False Positive ($C_{FP}$) = **$500** ($20\times$ ratio). The optimal threshold ($0.68$) was selected strictly on Development machines (80 machines) and evaluated on 20 unseen Test machines (175,220 hours). See [`reports/cost_analysis.csv`](reports/cost_analysis.csv) and [`reports/optimal_threshold.json`](reports/optimal_threshold.json).

| Dataset Split | Decision Threshold | Recall | Precision | F1 | Expected Total Cost | Dispatch Rate |
|---|---|---|---|---|---|---|
| **Development Set (Selection)** | **0.6800** | **0.9998** | 0.9560 | 0.9774 | **$337,500** | 2.00% |
| **Final Test Set (Evaluation)** | **0.6800** | **0.9711** | 0.9371 | 0.9538 | **$1,224,000** | 2.25% |

> **Methodology Note**: The optimal threshold ($0.68$) achieved **97.11% Recall** on the final test set (minimizing missed breakdowns). Evaluating threshold policy generalization across unseen machine cohorts reveals sensitivity to operational distributions, demonstrating why decision policies must be validated independently from ranking discrimination.

---

## 🖼️ Visualizations & Artifact Gallery

### Probability Calibration Dashboard
![Calibration Dashboard](reports/figures/calibration_dashboard.png)

### SHAP Feature Attribution & Drivers
![SHAP Feature Importance](reports/figures/shap_summary.png)

### Cost Optimization Curve ($C_{FN} = \$10,000$ vs $C_{FP} = \$500$)
![Cost vs Threshold](reports/figures/cost_vs_threshold.png)

### Fleet Risk Tier Distribution
![Fleet Risk Distribution](reports/figures/fleet_risk_distribution.png)

---

## ⚡ Quickstart & How to Run

### 1. Installation
```bash
# Clone repository
git clone https://github.com/Saswat-Mpt/PredictGuard.git
cd PredictGuard

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Download & Setup
PredictGuard uses the **Microsoft Azure Predictive Maintenance Dataset** (876,100 hourly telemetry rows). Raw CSV files are excluded from Git due to repository size limits.

To run the pipeline from scratch:
1. Download raw telemetry CSV files (`PdM_telemetry.csv`, `PdM_errors.csv`, `PdM_maint.csv`, `PdM_failures.csv`, `PdM_machines.csv`).
2. Place files into `data/raw/`.
3. Run the orchestration pipeline:
   ```bash
   python scripts/run_stage1.py
   python scripts/run_stage2.py
   ...
   ```

### 3. Execute Pytest Test Suite (64 Unit Tests)
```bash
pytest tests/ -v
```

### 4. Launch Local Services

| Application | Command | Endpoint |
|---|---|---|
| **FastAPI REST Server** | `uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload` | `http://localhost:8000/docs` |
| **Streamlit Dashboard** | `streamlit run app/dashboard.py` | `http://localhost:8501` |
| **MLflow Experiment UI** | `mlflow ui --backend-store-uri mlruns/mlflow.db` | `http://localhost:5000` |

### 5. Docker Compose (One-Command Deployment)
```bash
docker-compose up --build
```

---

## 🔌 API Usage Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "machine_id": "42",
           "timestamp": "2015-08-21T13:00:00Z",
           "features": {
             "volt_rolling_mean_24h": 168.4,
             "rotate_rolling_mean_24h": 448.1,
             "pressure_rolling_min_24h": 88.2,
             "errors_last_24h": 3.0,
             "rolling_error_rate_24h": 0.125
           }
         }'
```

**Response Payload:**
```json
{
  "machine_id": "42",
  "timestamp": "2015-08-21T13:00:00Z",
  "failure_probability": 0.8142,
  "risk_score": 81,
  "risk_tier": "CRITICAL",
  "predicted_component": "comp3",
  "component_confidence": 0.7245,
  "dispatch_decision": "DISPATCH",
  "recommended_action": "Dispatch electrician. Check voltage regulators and control circuits.",
  "optimal_threshold": 0.68,
  "top_shap_features": [
    { "feature": "errors_last_24h", "shap_value": 1.4521 },
    { "feature": "volt_rolling_mean_24h", "shap_value": 0.8912 }
  ],
  "pipeline_version": "1.0.0",
  "inference_ms": 14.2
}
```

---

## 🧠 Key Design Decisions

- **Why Machine-Level Splitting (`GroupKFold`)?** Rows from the same machine are highly correlated over time. Random row-level splitting leaks machine-specific telemetry across sets. Grouping 80 Dev and 20 Test machines guarantees zero machine-ID overlap.
- **Why PR-AUC Over ROC-AUC?** Failure events represent only ~1.5% of hourly records. Because True Negatives dominate (~860k rows), ROC-AUC is artificially inflated (>0.99). PR-AUC evaluates Precision against Recall directly, making it the appropriate metric for severe class imbalance.
- **Why Probability Calibration?** Tree-based ensembles like XGBoost output overconfident probabilities near boundaries due to log-loss optimization. Sigmoid (Platt Scaling) calibration aligns predicted probabilities with empirical frequencies, ensuring that across a cohort of assets assigned a 0.70 failure probability, approximately 70% empirically experience failure.
- **Why Tree-Path SHAP?** Technician dispatch cards require plain-English feature log-odds attribution so maintenance teams understand *why* a machine is flagged before going on-site.
- **Why Cost-Based Threshold Optimization?** In industrial maintenance, missing a breakdown ($10,000 FN cost) is $20\times$ more expensive than an unnecessary inspection ($500 FP cost). Deriving the optimal threshold (0.68) saved $167,500 compared to default 0.50 thresholding.

---

## ⚠️ Limitations & Future Work

### Limitations
- **Historical Synthetic Telemetry**: Evaluated on public benchmark telemetry data; real-world industrial environments may experience sensor noise and drift.
- **Static Cost Assumptions**: Cost parameters ($10k/$500) are organization-specific inputs that vary across facilities.
- **Fixed Component Schema**: Component classification is constrained to the four failure categories (`comp1` to `comp4`) present in the training set.

### Future Work
- **Streaming Ingestion**: Real-time telemetry streaming via Kafka and Redis.
- **Drift Monitoring**: Automated data/concept drift detection (e.g. Evidently AI).
- **Incident Alerts**: Real-time Slack/Teams alerts for `CRITICAL` risk assets.

---

## 📁 Repository Organization

```text
PredictGuard/
├── scripts/                       # Stage execution / orchestration scripts
│   ├── run_stage1.py              # Stage 1: Data Validation
│   ├── run_stage2.py              # Stage 2: Target Construction
│   ├── run_stage3.py              # Stage 3: Feature Engineering
│   ├── run_stage4.py              # Stage 4: Machine Splitting
│   ├── run_stage5.py              # Stage 5: Baseline Training
│   ├── run_stage6.py              # Stage 6: Grouped CV & Tuning
│   ├── run_stage7_8.py            # Stages 7-8: Reliability & Calibration
│   ├── run_stage9_10.py           # Stages 9-10: SHAP & Component Classifier
│   ├── run_stage11_12.py          # Stages 11-12: Fleet Risk & Cost Matrix
│   └── run_stage13_14.py          # Stages 13-14: MLflow & Pipeline Export
├── app/
│   ├── api.py                     # FastAPI server (/predict, /batch_predict)
│   ├── dashboard.py               # Streamlit Dashboard (Machine & Fleet View)
│   └── schemas.py                 # Pydantic v2 data validation schemas
├── src/
│   ├── data_validation.py         # 20+ automated data quality checks
│   ├── target_creation.py         # Leakage-safe 24h failure targets
│   ├── feature_engineering.py     # 119 telemetry features
│   ├── split.py                   # Machine-level grouped split (GroupKFold)
│   ├── train.py                   # Baseline LR, RF, XGBoost classifiers
│   ├── cross_validation.py        # Grouped CV & hyperparameter tuner
│   ├── calibration.py             # ECE evaluation & Sigmoid calibration
│   ├── explainability.py          # SHAP attribution engine
│   ├── component_classifier.py    # Multiclass component predictor (comp1-4)
│   ├── risk_segmentation.py       # Fleet risk tiering (0-100 score)
│   ├── cost_decision.py           # Cost matrix dispatch engine ($10k vs $500)
│   ├── monitoring.py              # MLflow experiment tracker
│   └── pipeline.py                # Serialised sklearn-compatible pipeline
├── models/                        # Serialised model artifacts (.pkl, .json)
├── reports/                       # Generated markdown reports & figures
│   ├── figures/                   # 20+ PNG charts
│   ├── api_documentation.md
│   ├── calibration_metrics.csv
│   ├── cost_analysis.csv
│   ├── cv_results.csv
│   └── pipeline_validation.md
├── tests/                         # 62 unit & integration tests
├── docs/                          # SVG architecture diagram
├── Dockerfile                     # Python 3.12 container
├── docker-compose.yml             # Orchestration for API + Streamlit
├── config.yaml                    # System configuration parameters
└── requirements.txt               # Dependencies
```

---

## 📄 License
Licensed under the [MIT License](LICENSE).
