# PredictGuard REST API Documentation

The PredictGuard REST API exposes the serialised end-to-end Machine Learning pipeline (`predictguard_pipeline.pkl`) for real-time asset telemetry scoring, risk tiering, component failure diagnosis, SHAP driver attribution, and cost-optimal maintenance dispatch decisions.

---

## Base URL

```text
http://localhost:8000
```

---

## Endpoints Summary

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| `GET` | `/` | Health check & system metadata | None |
| `POST` | `/predict` | Single machine risk prediction | None |
| `POST` | `/batch_predict` | Batch fleet prediction (up to 500 assets) | None |

---

## 1. Health Check

### Request
`GET /`

### Response `200 OK`
```json
{
  "status": "running",
  "project": "PredictGuard",
  "pipeline_version": "1.0.0",
  "feature_count": 119,
  "optimal_threshold": 0.68
}
```

---

## 2. Single Machine Prediction

### Request
`POST /predict`

**Headers:** `Content-Type: application/json`

```json
{
  "machine_id": "42",
  "timestamp": "2015-08-21T13:00:00Z",
  "features": {
    "volt_rolling_mean_24h": 168.4,
    "rotate_rolling_mean_24h": 448.1,
    "pressure_rolling_min_24h": 88.2,
    "errors_last_24h": 3.0,
    "rolling_error_rate_24h": 0.125,
    "maintenance_count_last_year": 2.0
  }
}
```

> **Note:** Any missing features are automatically aligned and filled with `0.0` by the `FeatureAligner`.

### Response `200 OK`
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
    {
      "feature": "errors_last_24h",
      "shap_value": 1.4521
    },
    {
      "feature": "volt_rolling_mean_24h",
      "shap_value": 0.8912
    }
  ],
  "pipeline_version": "1.0.0",
  "inference_ms": 14.2
}
```

---

## 3. Batch Prediction

### Request
`POST /batch_predict`

```json
{
  "machines": [
    {
      "machine_id": "1",
      "timestamp": "2015-08-21T13:00:00Z",
      "features": { "volt_rolling_mean_24h": 170.1 }
    },
    {
      "machine_id": "2",
      "timestamp": "2015-08-21T13:00:00Z",
      "features": { "errors_last_24h": 5.0 }
    }
  ]
}
```

### Response `200 OK`
```json
{
  "n_machines": 2,
  "n_dispatch": 1,
  "n_monitor": 1,
  "predictions": [ ... ],
  "total_inference_ms": 28.5
}
```

---

## Error Codes

| Code | Description | Example |
|---|---|---|
| `422 Unprocessable Entity` | Validation error (e.g. missing `machine_id` or batch > 500) | `{ "detail": "machines list must not be empty." }` |
| `500 Internal Server Error` | Pipeline prediction exception | `{ "detail": "Prediction error: ..." }` |
