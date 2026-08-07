# PredictGuard — Pipeline Architecture Report

> Generated: 2026-08-07 22:36 UTC

---

## 1. Overview

The `PredictGuardPipeline` is a serialised, end-to-end inference system that
takes pre-engineered machine telemetry features and produces structured,
actionable maintenance dispatch decisions.

---

## 2. Pipeline Version

| Component | Version |
|---|---|
| `pipeline_version` | `1.0.0` |
| `feature_version` | `3.1` |
| `model_version` | `2.0` |
| `calibration_version` | `1.0` |
| `feature_count` | `119` |
| `optimal_threshold` | `0.68` |
| `created_at` | `2026-08-07T22:36:14.132625+00:00` |
| `python_version` | `3.12.6` |
| `sklearn_version` | `1.9.0` |
| `xgboost_version` | `3.4.0` |

---

## 3. Data Flow

```
┌─────────────────────────────────────────────────┐
│           Input: Feature-Engineered Data         │
│  (119 cols: rolling stats, error rates, etc.)   │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│              FeatureAligner                      │
│  Aligns columns to training order               │
│  Fills missing features with 0.0                │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│        CalibratedClassifierCV (XGBoost)          │
│  Output: P(failure | features)   ∈ [0, 1]       │
│  Calibrated with Sigmoid (Phase 2 Stage 8)       │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│          RiskScoreTransformer                    │
│  risk_score = round(100 × P)                    │
│  risk_tier  ∈ {LOW, MEDIUM, HIGH, CRITICAL}     │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│          ComponentPredictor (XGBoost)            │
│  Output: predicted_component ∈ {comp1..comp4}   │
│  + per-class probability scores                  │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│        DispatchDecisionTransformer               │
│  P ≥ 0.68 → DISPATCH + action narrative         │
│  P <  0.68 → MONITOR                            │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────┐
│           predict_report() Output                │
│  machine_id, timestamp, failure_probability,     │
│  risk_score, risk_tier, predicted_component,     │
│  dispatch_decision, top_shap_features           │
└─────────────────────────────────────────────────┘
```

---

## 4. Custom Transformers

| Transformer | Input | Output | Purpose |
|---|---|---|---|
| `FeatureAligner` | Raw DataFrame | Aligned float matrix | Column alignment, fill-missing |
| `RiskScoreTransformer` | `calibrated_prob` | + `risk_score`, `risk_tier` | Tier assignment |
| `DispatchDecisionTransformer` | `calibrated_prob`, `predicted_component` | + `dispatch_decision`, `recommended_action` | Business decision |

---

## 5. Prediction Interface

| Method | Returns | Use Case |
|---|---|---|
| `predict_proba(X)` | `np.ndarray` — P(failure) | Ranking / scoring |
| `predict(X)` | `np.ndarray` — 0/1 at optimal threshold | Binary classification |
| `predict_risk(X)` | DataFrame — score + tier | Fleet dashboard |
| `predict_component(X)` | DataFrame — component + confidence | Maintenance routing |
| `predict_report(X, ...)` | `List[Dict]` — full structured report | API / integration |

---

## 6. Persistence

```python
# Save
pipeline.save('models/')
# → models/predictguard_pipeline.pkl       (joblib, compress=3)
# → models/predictguard_pipeline_metadata.json

# Load
from src.pipeline import PredictGuardPipeline
pipeline = PredictGuardPipeline.load('models/')
```

---

## 7. Limitations

- **Feature engineering** (rolling windows, lag features) is NOT included in the
  pipeline. This step requires ordered time-series streaming per machine and
  is implemented in `src/features.py`. The pipeline takes pre-engineered features
  as input — exactly like the production inference pattern.
- For Stage 15 (FastAPI), the API will wrap feature engineering + pipeline together.
