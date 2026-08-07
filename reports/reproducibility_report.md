# PredictGuard — Reproducibility Report

> Generated: 2026-08-07 22:36 UTC

## Result

**Pipeline reproducibility: CONFIRMED**

The serialised `predictguard_pipeline.pkl` produces predictions numerically identical to the original Stages 1–12 models (max probability error = `0.00e+00`, well within tolerance `1.00e-05`).

## How to Use the Pipeline

```python
from src.pipeline import PredictGuardPipeline
import pandas as pd

# Load saved pipeline
pipeline = PredictGuardPipeline.load('models/')

# Run inference on new feature-engineered data
df = pd.read_parquet('data/processed/new_data.parquet')
reports = pipeline.predict_report(df,
                                  machine_ids=df['machineID'],
                                  timestamps=df['datetime'])

# Each report is a structured dict:
print(reports[0])
# {
#   'machine_id': 47,
#   'failure_probability': 0.8142,
#   'risk_score': 81,
#   'risk_tier': 'CRITICAL',
#   'predicted_component': 'comp3',
#   'dispatch_decision': 'DISPATCH',
#   'recommended_action': 'Dispatch electrician ...',
#   'top_shap_features': [...]
# }
```
