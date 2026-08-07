"""
test_explainability_component.py
=================================
Unit tests for PredictGuard Phase 3 — Stage 9 & Stage 10.

Tests:
    test_shap_explainer_local        — local SHAP values shape & types
    test_feature_importance_analyzer — comparison table has required columns
    test_prediction_interpreter_card — card structure and risk levels
    test_component_predictor_dataset — dataset builder filters correctly
    test_component_predictor_fit     — classifier trains and predicts
    test_component_evaluator_metrics — per-class metrics computation
    test_diagnosis_card_builder      — diagnosis card structure

All tests use synthetic data (~1s each, no disk I/O for models).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.utils import estimator_checks

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explainability import (
    FeatureImportanceAnalyzer,
    PredictionInterpreter,
)
from src.component_classifier import (
    ComponentPredictor,
    ComponentEvaluator,
    ComponentDiagnosisReport,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_FEATURES = 20
FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]
COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]


def _make_synthetic_df(n_rows: int = 500, failure_frac: float = 0.3) -> pd.DataFrame:
    """Create a synthetic DataFrame that matches PredictGuard's expected schema."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_rows, N_FEATURES)
    y_failure = (rng.rand(n_rows) < failure_frac).astype(int)
    comp_labels = np.where(
        y_failure == 1,
        rng.choice(COMPONENTS, size=n_rows),
        None
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["datetime"] = pd.date_range("2015-01-01", periods=n_rows, freq="h")
    df["machineID"] = rng.randint(1, 20, size=n_rows)
    df["y_failure"] = y_failure
    df["failure_component"] = pd.array(comp_labels, dtype=object)
    df["time_to_failure_hours"] = np.where(y_failure == 1, 24.0, np.nan)
    return df


def _train_tiny_xgb(X: np.ndarray, y: np.ndarray):
    """Train a tiny XGBoost binary classifier for testing."""
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=10, max_depth=3, random_state=42,
        verbosity=0, eval_metric="logloss"
    )
    clf.fit(X, y)
    return clf


# ===========================================================================
# Test 1 — FeatureImportanceAnalyzer
# ===========================================================================

def test_feature_importance_analyzer_comparison_table(tmp_path):
    """Comparison table must have required columns and correct row count."""
    df = _make_synthetic_df(300, failure_frac=0.2)
    X = df[FEATURE_NAMES].values
    y = df["y_failure"].values
    model = _train_tiny_xgb(X, y)

    # Mock SHAP importance (normally from SHAPExplainer)
    rng = np.random.RandomState(0)
    shap_imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_abs_shap": np.abs(rng.randn(N_FEATURES)),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    analyzer = FeatureImportanceAnalyzer(
        raw_model=model,
        feature_names=FEATURE_NAMES,
        figures_dir=tmp_path,
    )
    comp_df = analyzer.build_comparison_table(shap_imp)

    required_cols = {"feature", "mean_abs_shap", "shap_rank", "native_rank", "rank_delta"}
    assert required_cols.issubset(comp_df.columns), (
        f"Missing columns: {required_cols - set(comp_df.columns)}"
    )
    assert len(comp_df) == N_FEATURES
    assert comp_df["rank_delta"].ge(0).all(), "rank_delta must be non-negative"


# ===========================================================================
# Test 2 — PredictionInterpreter card structure
# ===========================================================================

def test_prediction_interpreter_card_structure(tmp_path):
    """Card must contain all required keys and valid risk level."""
    interpreter = PredictionInterpreter(
        feature_names=FEATURE_NAMES,
        reports_dir=tmp_path,
    )
    rng = np.random.RandomState(42)
    X_row = pd.DataFrame([rng.randn(N_FEATURES)], columns=FEATURE_NAMES)
    shap_values = rng.randn(N_FEATURES)

    card = interpreter.build_explanation_card(
        machine_id=42,
        timestamp="2015-06-15 12:00",
        X_row=X_row,
        shap_values=shap_values,
        raw_prob=0.75,
        calibrated_prob=0.68,
    )

    required = {
        "machine_id", "timestamp", "raw_score", "calibrated_probability",
        "risk_level", "top_positive_features", "top_negative_features",
        "plain_english_explanation", "generated_at",
    }
    assert required.issubset(card.keys()), f"Missing keys: {required - card.keys()}"
    assert card["machine_id"] == 42
    assert card["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0.0 <= card["calibrated_probability"] <= 1.0
    assert isinstance(card["plain_english_explanation"], str)


def test_prediction_interpreter_risk_levels():
    """Risk level thresholds are correctly applied."""
    interpreter = PredictionInterpreter(feature_names=[], reports_dir=Path("."))
    assert interpreter._risk_level(0.90) == "CRITICAL"
    assert interpreter._risk_level(0.65) == "HIGH"
    assert interpreter._risk_level(0.35) == "MEDIUM"
    assert interpreter._risk_level(0.10) == "LOW"


# ===========================================================================
# Test 3 — ComponentPredictor dataset builder
# ===========================================================================

def test_component_predictor_build_dataset(tmp_path):
    """Dataset builder must return only failure rows with valid labels."""
    df = _make_synthetic_df(500, failure_frac=0.4)
    predictor = ComponentPredictor(models_dir=tmp_path)
    fail_df = predictor.build_dataset(df, min_class_samples=1)

    assert (fail_df["y_failure"] == 1).all(), "All rows must be failures"
    assert fail_df["failure_component"].notna().all(), "All rows must have component labels"
    assert set(fail_df["failure_component"].unique()).issubset(set(COMPONENTS))
    assert len(fail_df) > 0


# ===========================================================================
# Test 4 — ComponentPredictor fit and predict
# ===========================================================================

def test_component_predictor_fit_predict(tmp_path):
    """Predictor should train without error and produce valid predictions."""
    df = _make_synthetic_df(600, failure_frac=0.5)
    predictor = ComponentPredictor(models_dir=tmp_path, random_state=42)
    fail_df = predictor.build_dataset(df, min_class_samples=1)

    # Need at least 2 classes to train
    if fail_df["failure_component"].nunique() < 2:
        pytest.skip("Not enough component classes in synthetic data")

    predictor.fit(fail_df, n_estimators=10)

    y_pred = predictor.predict(fail_df)
    assert len(y_pred) == len(fail_df)
    assert set(y_pred).issubset(set(predictor.classes_))

    proba_df = predictor.predict_proba_df(fail_df)
    assert "predicted_component" in proba_df.columns
    assert "component_confidence" in proba_df.columns
    for cls in predictor.classes_:
        assert cls in proba_df.columns
    # Probabilities must sum to 1
    prob_sums = proba_df[predictor.classes_].sum(axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)


# ===========================================================================
# Test 5 — ComponentEvaluator metrics
# ===========================================================================

def test_component_evaluator_metrics(tmp_path):
    """Per-class metrics should be computed correctly."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.choice(COMPONENTS[:3], size=n)
    y_pred = rng.choice(COMPONENTS[:3], size=n)
    y_proba = rng.dirichlet([1, 1, 1], size=n)

    evaluator = ComponentEvaluator(figures_dir=tmp_path, reports_dir=tmp_path)
    metrics_df, aggregate = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=COMPONENTS[:3],
    )

    required_cols = {"component", "n_samples", "precision", "recall", "f1_score"}
    assert required_cols.issubset(metrics_df.columns)
    assert "accuracy" in aggregate
    assert "macro_f1" in aggregate
    assert "weighted_f1" in aggregate
    assert 0.0 <= aggregate["accuracy"] <= 1.0
    assert (metrics_df["recall"].between(0.0, 1.0)).all()


# ===========================================================================
# Test 6 — ComponentDiagnosisReport card builder
# ===========================================================================

def test_diagnosis_card_builder():
    """Diagnosis card should have correct structure and inspection advice."""
    writer = ComponentDiagnosisReport(reports_dir=Path("."))
    card = writer.build_diagnosis_card(
        machine_id=88,
        timestamp="2015-09-20 08:00",
        calibrated_prob=0.85,
        component_proba={"comp1": 0.10, "comp2": 0.15, "comp3": 0.65, "comp4": 0.10},
        predicted_component="comp3",
        top_positive_features=["pressure_zscore", "vibration_rolling_mean_24h"],
    )

    assert card["machine_id"] == 88
    assert card["failure_probability"] == pytest.approx(0.85, abs=1e-4)
    assert card["predicted_component"] == "comp3"
    assert card["risk_level"] == "CRITICAL"
    assert card["component_confidence"] == pytest.approx(0.65, abs=1e-4)
    assert "inspection_advice" in card
    assert isinstance(card["inspection_advice"], str)
    assert len(card["inspection_advice"]) > 0
