# 🛡️ PredictGuard — Explainable Predictive Maintenance System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.15+-yellow.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

PredictGuard is an end-to-end, production-grade **Predictive Maintenance System** engineered to transition predictive maintenance from generic accuracy metrics to trustworthy decision-support intelligence.

Unlike traditional ML projects that stop at binary classification, PredictGuard provides:
1. **Calibrated Probabilities** — Post-hoc Sigmoid probability calibration reducing Expected Calibration Error (ECE) by **74.1%**.
2. **Explainable AI (XAI)** — Tree-path SHAP driver attribution for every prediction.
3. **Component-Level Failure Diagnosis** — Multiclass XGBoost isolating exact component failure (`comp1`–`comp4`) with **99.1% accuracy**.
4. **Fleet Risk Tiering** — Real-time 4-tier risk score (0–100) segmentation.
5. **Cost-Optimal Dispatch Decisions** — Asymmetric FP/FN cost matrix optimization achieving **97.1% Recall** and saving **15.8%** over baseline thresholds.

---

## 📐 System Architecture

```text
               Telemetry Data (Telemetry, Maintenance, Errors, Failures)
                                         │
                                         ▼
                               Stage 1: Data Validation (20+ Quality Checks)
                                         │
                                         ▼
                         Stage 2–3: Leakage-Safe Feature Engineering (119 Features)
                                         │
                                         ▼
                         Stage 4–6: Grouped CV & Tuned XGBoost Binary Model
                                         │
                                         ▼
                         Stage 7–8: Post-Hoc Sigmoid Calibration (ECE = 0.0007)
                                         │
                                         ▼
               ┌─────────────────────────┴─────────────────────────┐
               │                                                   │
               ▼                                                   ▼
Stage 9: Tree-Path SHAP Attribution              Stage 10: Multiclass Component Classifier
 (Global Summary & Local Force)                   (comp1–comp4 Diagnosis, 99.1% Acc)
               │                                                   │
               └─────────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                       Stage 11: Fleet Risk Score Tiering (0–100)
                                         │
                                         ▼
                     Stage 12: Cost-Sensitive Dispatch Decision Engine
                                         │
                                         ▼
                   Stage 13–14: MLflow Tracking & Serialised Pipeline
                                         │
                                         ▼
               ┌─────────────────────────┴─────────────────────────┐
               │                                                   │
               ▼                                                   ▼
     Stage 15: FastAPI REST API                       Stage 15: Streamlit Dashboard
    (POST /predict, POST /batch)                   (Machine View & Fleet View)
               │                                                   │
               └─────────────────────────┬─────────────────────────┘
                                         │
                                         ▼
                    Stage 16: Docker Compose & GitHub Actions CI/CD
```

---

## 📊 Performance & Calibration Benchmark

| Evaluation Stage | Primary Metric | Baseline | PredictGuard Model | Gain / Status |
|---|---|---|---|---|
| **Binary Model CV** | PR-AUC | 0.8210 (LR) | **0.9737 (XGBoost)** | +18.6% PR-AUC |
| **Probability Calibration** | ECE | 0.0027 (Raw) | **0.0007 (Calibrated)** | **74.1% Error Reduction** |
| **Component Diagnosis** | Macro-F1 | 0.7500 | **0.9910** | **99.1% Test Accuracy** |
| **Cost Optimization** | Expected Cost | $1,391,500 (t=0.5) | **$1,224,000 (t=0.68)** | **15.8% Cost Savings** |
| **Dispatch Reliability** | Test Recall | 0.8500 | **0.9711** | **3,697 True Positives** |

---

## ⚡ Quickstart & Local Setup

### 1. Prerequisites
- Python 3.12+
- Git

### 2. Installation
```bash
# Clone repository
git clone https://github.com/your-username/PredictGuard.git
cd PredictGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Pipeline Validation Test Suite
```bash
pytest tests/ -v
```

### 4. Start the Application

#### Launch FastAPI Server
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
# Interactive API Documentation: http://localhost:8000/docs
```

#### Launch Streamlit Dashboard
```bash
streamlit run app/dashboard.py
# Streamlit Dashboard UI: http://localhost:8501
```

#### Launch MLflow Tracking UI
```bash
mlflow ui --backend-store-uri mlruns/mlflow.db
# MLflow Experiment Dashboard: http://localhost:5000
```

---

## 🐳 Docker Deployment

Run the full stack (FastAPI + Streamlit Dashboard) with Docker Compose:

```bash
docker-compose up --build
```
- **REST API:** `http://localhost:8000`
- **Dashboard:** `http://localhost:8501`

---

## 🔌 REST API Endpoints

### Single Machine Prediction (`POST /predict`)

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

## 📁 Repository Structure

```text
PredictGuard/
├── app/
│   ├── api.py                     # FastAPI REST server
│   ├── dashboard.py               # Streamlit interactive dashboard
│   └── schemas.py                 # Pydantic v2 data validation schemas
├── src/
│   ├── data_validation.py         # Automated data quality pipeline
│   ├── features.py                # Leakage-safe feature engineering
│   ├── model.py                   # XGBoost binary predictor
│   ├── calibration.py             # Sigmoid probability calibration
│   ├── explainability.py          # SHAP attribution engine
│   ├── component_classifier.py    # Multiclass component predictor
│   ├── risk_segmentation.py       # Fleet risk tiering
│   ├── cost_decision.py           # Cost matrix dispatch engine
│   ├── monitoring.py              # MLflow experiment tracking
│   └── pipeline.py                # Serialised sklearn-compatible pipeline
├── models/                        # Serialised model artifacts (.pkl, .json)
├── reports/                       # Generated markdown reports & figures
│   ├── figures/                   # 20+ publication-quality plots
│   ├── api_documentation.md
│   ├── mlflow_summary.md
│   ├── pipeline_validation.md
│   └── reproducibility_report.md
├── tests/                         # Pytest unit & integration test suite
├── Dockerfile                     # Container definition
├── docker-compose.yml             # Service orchestration
├── config.yaml                    # System configuration
└── requirements.txt               # Dependencies
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
