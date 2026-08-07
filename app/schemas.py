"""
schemas.py
==========
PredictGuard — FastAPI request / response schemas using Pydantic v2.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TelemetryInput(BaseModel):
    """
    Flat key-value map of feature-engineered columns for a single machine-hour.
    Unknown columns are ignored; missing columns are filled with 0.0.
    """
    machine_id: str = Field(..., description="Machine identifier")
    timestamp: Optional[str] = Field(None, description="ISO-8601 timestamp of observation")
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Feature-engineered telemetry values keyed by column name. "
            "Must contain at least the core sensor rolling-window features. "
            "Missing keys are filled with 0.0 by the FeatureAligner."
        ),
    )

    model_config = {"json_schema_extra": {
        "example": {
            "machine_id": "42",
            "timestamp": "2015-08-21T13:00:00Z",
            "features": {
                "volt_rolling_mean_24h": 168.4,
                "rotate_rolling_mean_24h": 448.1,
                "pressure_rolling_min_24h": 88.2,
                "errors_last_24h": 3.0,
                "rolling_error_rate_24h": 0.125,
                "maintenance_count_last_year": 2.0,
            }
        }
    }}


class BatchTelemetryInput(BaseModel):
    """A list of TelemetryInput for batch prediction."""
    machines: List[TelemetryInput] = Field(
        ..., description="List of machine observations (max 500 per request)", max_length=500
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class SHAPFeature(BaseModel):
    feature: str
    shap_value: float


class PredictionOutput(BaseModel):
    machine_id: str
    timestamp: Optional[str]
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    risk_score: int = Field(..., ge=0, le=100)
    risk_tier: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    predicted_component: str
    component_confidence: float = Field(..., ge=0.0, le=1.0)
    dispatch_decision: str = Field(..., pattern="^(DISPATCH|MONITOR)$")
    recommended_action: str
    optimal_threshold: float
    top_shap_features: List[SHAPFeature] = []
    pipeline_version: str
    inference_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchPredictionOutput(BaseModel):
    n_machines: int
    n_dispatch: int
    n_monitor: int
    predictions: List[PredictionOutput]
    total_inference_ms: float


class HealthResponse(BaseModel):
    status: str
    project: str
    pipeline_version: str
    feature_count: int
    optimal_threshold: float
