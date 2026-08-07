"""
test_pipeline_monitoring.py
===========================
Unit tests for PredictGuard Phase 4 — Stage 13 & 14.

Tests:
    test_feature_aligner_fills_missing      — missing cols filled with 0
    test_feature_aligner_drops_extra        — extra cols removed
    test_feature_aligner_reorders           — output order matches expected
    test_risk_score_transformer_bounds      — risk_score in [0, 100]
    test_risk_score_tier_assignment         — tiers correctly assigned
    test_dispatch_above_threshold           — high prob → DISPATCH
    test_dispatch_below_threshold           — low prob → MONITOR
    test_pipeline_predict_proba_shape       — output shape correct
    test_pipeline_predict_binary            — binary output is 0/1
    test_pipeline_predict_report_fields     — report contains required keys
    test_pipeline_save_load_roundtrip       — load produces same predictions
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    FeatureAligner,
    RiskScoreTransformer,
    DispatchDecisionTransformer,
    PredictGuardPipeline,
    PIPELINE_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [f"feat_{i}" for i in range(20)]


def _make_feature_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        rng.randn(n, len(FEATURE_COLS)),
        columns=FEATURE_COLS,
    )


def _make_calibrated_model(proba: np.ndarray) -> MagicMock:
    model = MagicMock()
    model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    return model


def _make_component_predictor(n: int) -> MagicMock:
    predictor = MagicMock()
    predictor.feature_cols = FEATURE_COLS
    comps = ["comp1", "comp2", "comp3", "comp4"]
    rng = np.random.RandomState(99)
    comp_series = pd.Series(rng.choice(comps, size=n))
    conf_series = pd.Series(rng.uniform(0.5, 1.0, size=n))
    predictor.predict_proba_df.return_value = pd.DataFrame({
        "predicted_component": comp_series,
        "component_confidence": conf_series,
    })
    return predictor


def _make_pipeline(n: int = 50, threshold: float = 0.5) -> PredictGuardPipeline:
    rng = np.random.RandomState(7)
    proba = rng.rand(n)
    cal_model = _make_calibrated_model(proba)
    comp_pred = _make_component_predictor(n)
    raw_model = MagicMock()
    return PredictGuardPipeline(
        calibrated_model=cal_model,
        raw_model=raw_model,
        component_predictor=comp_pred,
        feature_cols=FEATURE_COLS,
        optimal_threshold=threshold,
    )


# ===========================================================================
# FeatureAligner Tests
# ===========================================================================

def test_feature_aligner_fills_missing():
    """Missing columns must be filled with fill_value."""
    aligner = FeatureAligner(expected_cols=["a", "b", "c"], fill_value=0.0)
    df = pd.DataFrame({"a": [1.0, 2.0]})   # b and c are missing
    out = aligner.transform(df)
    assert list(out.columns) == ["a", "b", "c"]
    assert (out["b"] == 0.0).all()
    assert (out["c"] == 0.0).all()


def test_feature_aligner_drops_extra():
    """Extra columns not in expected_cols must be dropped."""
    aligner = FeatureAligner(expected_cols=["x", "y"])
    df = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [99.0]})
    out = aligner.transform(df)
    assert "z" not in out.columns
    assert list(out.columns) == ["x", "y"]


def test_feature_aligner_reorders():
    """Output columns must be in the same order as expected_cols."""
    aligner = FeatureAligner(expected_cols=["b", "a", "c"])
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    out = aligner.transform(df)
    assert list(out.columns) == ["b", "a", "c"]


# ===========================================================================
# RiskScoreTransformer Tests
# ===========================================================================

def test_risk_score_transformer_bounds():
    """risk_score must always be in [0, 100]."""
    transformer = RiskScoreTransformer()
    df = pd.DataFrame({"calibrated_prob": [0.0, 0.5, 1.0, 0.999]})
    out = transformer.transform(df)
    assert out["risk_score"].between(0, 100).all()


def test_risk_score_tier_assignment():
    """Tiers must match the configured thresholds."""
    transformer = RiskScoreTransformer()
    df = pd.DataFrame({"calibrated_prob": [0.05, 0.35, 0.60, 0.85]})
    out = transformer.transform(df)
    assert out.iloc[0]["risk_tier"] == "LOW"
    assert out.iloc[1]["risk_tier"] == "MEDIUM"
    assert out.iloc[2]["risk_tier"] == "HIGH"
    assert out.iloc[3]["risk_tier"] == "CRITICAL"


# ===========================================================================
# DispatchDecisionTransformer Tests
# ===========================================================================

def test_dispatch_above_threshold():
    """Prob >= threshold → DISPATCH."""
    t = DispatchDecisionTransformer(optimal_threshold=0.4)
    df = pd.DataFrame({
        "calibrated_prob": [0.9, 0.4],
        "predicted_component": ["comp1", "comp2"],
    })
    out = t.transform(df)
    assert out.iloc[0]["dispatch_decision"] == "DISPATCH"
    assert out.iloc[1]["dispatch_decision"] == "DISPATCH"   # exactly at threshold


def test_dispatch_below_threshold():
    """Prob < threshold → MONITOR."""
    t = DispatchDecisionTransformer(optimal_threshold=0.4)
    df = pd.DataFrame({
        "calibrated_prob": [0.1, 0.39],
        "predicted_component": ["comp3", "comp4"],
    })
    out = t.transform(df)
    assert (out["dispatch_decision"] == "MONITOR").all()


# ===========================================================================
# PredictGuardPipeline Tests
# ===========================================================================

def test_pipeline_predict_proba_shape():
    """predict_proba output shape must match input rows."""
    n = 30
    pipeline = _make_pipeline(n=n)
    X = _make_feature_df(n=n)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (n,)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_pipeline_predict_binary():
    """predict() must return only 0 or 1."""
    pipeline = _make_pipeline(n=40, threshold=0.5)
    X = _make_feature_df(n=40)
    y = pipeline.predict(X)
    assert set(y).issubset({0, 1})


def test_pipeline_predict_report_fields():
    """predict_report() must contain all required fields."""
    n = 5
    pipeline = _make_pipeline(n=n)
    X = _make_feature_df(n=n)
    reports = pipeline.predict_report(
        X,
        machine_ids=[f"M{i}" for i in range(n)],
        timestamps=[f"2015-01-{i+1:02d}" for i in range(n)],
        include_shap_top_n=0,   # skip SHAP to avoid model mock issues
    )
    required_keys = {
        "machine_id", "timestamp", "failure_probability",
        "risk_score", "risk_tier", "predicted_component",
        "dispatch_decision", "recommended_action",
        "pipeline_version",
    }
    assert len(reports) == n
    for rep in reports:
        assert required_keys.issubset(rep.keys()), (
            f"Missing keys: {required_keys - rep.keys()}"
        )
        assert rep["pipeline_version"] == PIPELINE_VERSION
        assert rep["risk_score"] in range(0, 101)
        assert rep["risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert rep["dispatch_decision"] in {"DISPATCH", "MONITOR"}


# ---------------------------------------------------------------------------
# Picklable stubs for round-trip test
# ---------------------------------------------------------------------------
class _StubCalModel:
    """Minimal picklable calibrated-model stub."""
    def predict_proba(self, X):
        n = len(X) if hasattr(X, '__len__') else 1
        rng = np.random.RandomState(42)
        p = rng.rand(n)
        return np.column_stack([1 - p, p])


class _StubCompPredictor:
    """Minimal picklable component-predictor stub."""
    feature_cols = FEATURE_COLS
    classes_ = ["comp1", "comp2", "comp3", "comp4"]

    def predict_proba_df(self, X):
        n = len(X)
        rng = np.random.RandomState(7)
        return pd.DataFrame({
            "predicted_component": rng.choice(self.classes_, size=n),
            "component_confidence": rng.uniform(0.5, 1.0, size=n),
        })


def test_pipeline_save_load_roundtrip(tmp_path):
    """Saved and re-loaded pipeline must produce identical predict_proba output."""
    n = 20
    pipeline = PredictGuardPipeline(
        calibrated_model=_StubCalModel(),
        raw_model=_StubCalModel(),
        component_predictor=_StubCompPredictor(),
        feature_cols=FEATURE_COLS,
        optimal_threshold=0.5,
    )
    X = _make_feature_df(n=n)
    proba_before = pipeline.predict_proba(X)

    pipeline.save(tmp_path)
    loaded = PredictGuardPipeline.load(tmp_path)
    proba_after = loaded.predict_proba(X)

    np.testing.assert_array_almost_equal(proba_before, proba_after, decimal=6)
    assert loaded.metadata["pipeline_version"] == PIPELINE_VERSION
