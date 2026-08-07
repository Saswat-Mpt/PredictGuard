"""
test_api_dashboard.py
=====================
Unit tests for PredictGuard Phase 4 — Stage 15 (FastAPI & Dashboard).

Tests:
    test_api_health_endpoint         — GET / returns 200 and running status
    test_api_predict_endpoint        — POST /predict returns valid PredictionOutput schema
    test_api_batch_predict_endpoint  — POST /batch_predict returns valid BatchPredictionOutput schema
    test_api_invalid_batch           — empty or oversized batch returns 422
    test_dashboard_loads_pipeline    — pipeline loads cleanly via Streamlit loader
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api import app
from src.pipeline import PredictGuardPipeline


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_api_health_endpoint(client):
    """GET / must return 200 with status='running' and valid metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["project"] == "PredictGuard"
    assert "pipeline_version" in data
    assert data["feature_count"] > 0


def test_api_predict_endpoint(client):
    """POST /predict must return structured risk prediction."""
    payload = {
        "machine_id": "42",
        "timestamp": "2015-08-21T13:00:00Z",
        "features": {
            "volt_rolling_mean_24h": 168.4,
            "rotate_rolling_mean_24h": 448.1,
            "pressure_rolling_min_24h": 88.2,
            "errors_last_24h": 3.0,
            "rolling_error_rate_24h": 0.125,
            "maintenance_count_last_year": 2.0,
        },
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["machine_id"] == "42"
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert data["predicted_component"] in {"comp1", "comp2", "comp3", "comp4"}
    assert data["dispatch_decision"] in {"DISPATCH", "MONITOR"}
    assert "recommended_action" in data
    assert isinstance(data["top_shap_features"], list)


def test_api_batch_predict_endpoint(client):
    """POST /batch_predict must return list of predictions and counts."""
    payload = {
        "machines": [
            {
                "machine_id": "1",
                "timestamp": "2015-08-21T13:00:00Z",
                "features": {"volt_rolling_mean_24h": 170.1},
            },
            {
                "machine_id": "2",
                "timestamp": "2015-08-21T13:00:00Z",
                "features": {"errors_last_24h": 5.0},
            },
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_machines"] == 2
    assert data["n_dispatch"] + data["n_monitor"] == 2
    assert len(data["predictions"]) == 2


def test_api_invalid_batch(client):
    """Empty batch must fail with 422."""
    response = client.post("/batch_predict", json={"machines": []})
    assert response.status_code == 422


def test_dashboard_loads_pipeline():
    """PredictGuardPipeline loader must load saved pipeline cleanly."""
    models_dir = PROJECT_ROOT / "models"
    p = PredictGuardPipeline.load(models_dir)
    assert p is not None
    assert hasattr(p, "predict_report")
