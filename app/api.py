"""
api.py
======
PredictGuard — FastAPI REST API.

Endpoints:
    GET  /                  Health check
    POST /predict           Single machine prediction
    POST /batch_predict     Batch prediction (up to 500 machines)

Usage:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import site

# Bootstrap user site-packages (handles --user pip installs)
for _sp in ([site.getusersitepackages()]
            if isinstance(site.getusersitepackages(), str)
            else site.getusersitepackages()):
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PredictGuardPipeline
from app.schemas import (
    BatchPredictionOutput,
    BatchTelemetryInput,
    HealthResponse,
    PredictionOutput,
    SHAPFeature,
    TelemetryInput,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predictguard.api")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {}
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Lifespan — load pipeline once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    import pathlib
    try:
        pathlib.WindowsPath()
    except NotImplementedError:
        pathlib.WindowsPath = pathlib.PosixPath

    logger.info("Loading PredictGuard pipeline from %s ...", MODELS_DIR)
    t0 = time.time()
    _state["pipeline"] = PredictGuardPipeline.load(MODELS_DIR)
    logger.info(
        "Pipeline loaded in %.2fs: v%s, %d features, threshold=%.2f",
        time.time() - t0,
        _state["pipeline"].metadata.get("pipeline_version", "?"),
        len(_state["pipeline"].feature_cols),
        _state["pipeline"].optimal_threshold,
    )
    yield
    _state.clear()
    logger.info("PredictGuard API shut down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PredictGuard API",
    description=(
        "Explainable Predictive Maintenance System — "
        "provides calibrated failure probabilities, risk tiers, "
        "component diagnosis, and cost-optimal dispatch decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_feature_df(features: Dict[str, float], pipeline: PredictGuardPipeline) -> pd.DataFrame:
    """Convert features dict → aligned DataFrame ready for the pipeline."""
    df = pd.DataFrame([features])
    # FeatureAligner handles missing cols (fills with 0.0)
    X_aligned = pipeline._feature_aligner.transform(df)
    return X_aligned


def _make_prediction(item: TelemetryInput, pipeline: PredictGuardPipeline) -> Dict[str, Any]:
    """Run full prediction for a single TelemetryInput. Returns raw dict."""
    t0 = time.perf_counter()
    X = _build_feature_df(item.features, pipeline)
    reports = pipeline.predict_report(
        X,
        machine_ids=[item.machine_id],
        timestamps=[item.timestamp],
        include_shap_top_n=5,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    rep = reports[0]
    rep["inference_ms"] = round(elapsed_ms, 2)
    return rep


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check — returns API status and pipeline metadata."""
    pipeline: PredictGuardPipeline = _state["pipeline"]
    return HealthResponse(
        status="running",
        project="PredictGuard",
        pipeline_version=pipeline.metadata.get("pipeline_version", "1.0.0"),
        feature_count=len(pipeline.feature_cols),
        optimal_threshold=pipeline.optimal_threshold,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(item: TelemetryInput, request: Request):
    """
    Predict failure risk for a single machine observation.

    Returns calibrated failure probability, risk tier, predicted failing
    component, SHAP top-5 feature drivers, and dispatch recommendation.
    """
    pipeline: PredictGuardPipeline = _state["pipeline"]
    logger.info("POST /predict | machine=%s", item.machine_id)
    try:
        rep = _make_prediction(item, pipeline)
    except Exception as exc:
        logger.exception("Prediction failed for machine %s", item.machine_id)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    shap_feats = [
        SHAPFeature(feature=f["feature"], shap_value=f["shap_value"])
        for f in rep.get("top_shap_features", [])
    ]
    logger.info(
        "  machine=%s | prob=%.3f | tier=%s | comp=%s | decision=%s | %.1fms",
        rep["machine_id"], rep["failure_probability"], rep["risk_tier"],
        rep["predicted_component"], rep["dispatch_decision"], rep["inference_ms"],
    )
    return PredictionOutput(
        machine_id=str(rep["machine_id"]),
        timestamp=rep.get("timestamp"),
        failure_probability=rep["failure_probability"],
        risk_score=rep["risk_score"],
        risk_tier=rep["risk_tier"],
        predicted_component=rep["predicted_component"],
        component_confidence=rep.get("component_confidence", 0.0),
        dispatch_decision=rep["dispatch_decision"],
        recommended_action=rep["recommended_action"],
        optimal_threshold=rep["optimal_threshold"],
        top_shap_features=shap_feats,
        pipeline_version=rep["pipeline_version"],
        inference_ms=rep["inference_ms"],
    )


@app.post("/batch_predict", response_model=BatchPredictionOutput, tags=["Prediction"])
async def batch_predict(batch: BatchTelemetryInput, request: Request):
    """
    Batch prediction for up to 500 machines.

    Accepts a list of machine observations and returns structured predictions
    for each, including fleet-level dispatch summary counts.
    """
    pipeline: PredictGuardPipeline = _state["pipeline"]
    n = len(batch.machines)
    logger.info("POST /batch_predict | n=%d machines", n)
    if n == 0:
        raise HTTPException(status_code=422, detail="machines list must not be empty.")
    if n > 500:
        raise HTTPException(status_code=422, detail="Maximum 500 machines per batch request.")

    t0 = time.perf_counter()
    predictions = []
    for item in batch.machines:
        try:
            rep = _make_prediction(item, pipeline)
            shap_feats = [
                SHAPFeature(feature=f["feature"], shap_value=f["shap_value"])
                for f in rep.get("top_shap_features", [])
            ]
            predictions.append(PredictionOutput(
                machine_id=str(rep["machine_id"]),
                timestamp=rep.get("timestamp"),
                failure_probability=rep["failure_probability"],
                risk_score=rep["risk_score"],
                risk_tier=rep["risk_tier"],
                predicted_component=rep["predicted_component"],
                component_confidence=rep.get("component_confidence", 0.0),
                dispatch_decision=rep["dispatch_decision"],
                recommended_action=rep["recommended_action"],
                optimal_threshold=rep["optimal_threshold"],
                top_shap_features=shap_feats,
                pipeline_version=rep["pipeline_version"],
                inference_ms=rep["inference_ms"],
            ))
        except Exception as exc:
            logger.warning("Skipping machine %s — prediction error: %s", item.machine_id, exc)

    total_ms = (time.perf_counter() - t0) * 1000
    n_dispatch = sum(1 for p in predictions if p.dispatch_decision == "DISPATCH")
    logger.info(
        "Batch done: %d/%d predicted | DISPATCH=%d MONITOR=%d | %.1fms total",
        len(predictions), n, n_dispatch, len(predictions) - n_dispatch, total_ms
    )
    return BatchPredictionOutput(
        n_machines=len(predictions),
        n_dispatch=n_dispatch,
        n_monitor=len(predictions) - n_dispatch,
        predictions=predictions,
        total_inference_ms=round(total_ms, 2),
    )
