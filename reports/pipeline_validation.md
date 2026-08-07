# PredictGuard — Pipeline Validation Report

> Generated: 2026-08-07 22:36 UTC

---

## Status: ✅ ALL CHECKS PASSED

---

## Validation Checks

| Check | Result |
|---|---|
| Calibrated probability diff < tolerance | ✅ PASS |
| Binary prediction match rate ≥ 99.9% | ✅ PASS |
| Component prediction match rate ≥ 99% | ✅ PASS |
| Risk tier match rate ≥ 99.9% | ✅ PASS |

---

## Numerical Reproducibility

| Metric | Value |
|---|---|
| Max probability difference | `0.00e+00` |
| Tolerance | `1.00e-05` |
| Binary prediction match | `100.0000%` |
| Component match | `100.0000%` |
| Risk tier match | `100.0000%` |
| Validation sample size | `5000` |

---

## Pipeline Architecture

```
Input Features (pre-engineered, 119 columns)
       │
       ▼
FeatureAligner (column alignment + zero-fill missing)
       │
       ▼
CalibratedClassifierCV (XGBoost + Sigmoid)
       │  predict_proba()[:, 1]
       ▼
RiskScoreTransformer (risk_score = round(100 × prob))
       │  risk_tier ∈ {LOW, MEDIUM, HIGH, CRITICAL}
       ▼
ComponentPredictor (XGBoost multiclass: comp1–comp4)
       │
       ▼
DispatchDecisionTransformer (prob ≥ optimal_threshold → DISPATCH)
       │
       ▼
predict_report() → JSON
```

---

## Edge Case Handling

| Scenario | Behaviour |
|---|---|
| Missing feature columns | `FeatureAligner` fills with 0.0 and warns |
| Unknown machine ID | Treated as new machine; no history assumed |
| All rows same probability | Risk tiers still correctly assigned |
| Very high threshold | All predictions → MONITOR |
| Very low threshold | All predictions → DISPATCH |
