"""
test_calibration.py
===================
Unit tests for PredictGuard Phase 2 — src/calibration.py.

Fast (<10s) synthetic tests for calibration metrics, ECE/MCE,
Brier Score bounds, Isotonic/Sigmoid post-hoc calibration, and probability audit.

Author: PredictGuard Contributors
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (
    CalibrationEvaluator,
    ProbabilityCalibrator,
    ReliabilityAnalyzer,
)


@pytest.fixture()
def synthetic_calibration_data() -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic ground truth y_true and predicted probabilities y_prob."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.choice([0, 1], size=n, p=[0.90, 0.10])
    # Uncalibrated probabilities (pushed towards 0/1)
    y_prob = np.where(y_true == 1, rng.uniform(0.6, 0.99, size=n), rng.uniform(0.01, 0.40, size=n))
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Test 1: CalibrationEvaluator — ECE & MCE computation
# ---------------------------------------------------------------------------

def test_ece_mce_perfect_calibration() -> None:
    """ECE and MCE should be ~0.0 for perfectly calibrated predictions."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    evaluator = CalibrationEvaluator(n_bins=5)
    ece, mce, bin_df = evaluator.compute_ece_mce(y_true, y_prob, n_bins=5)

    assert ece == pytest.approx(0.0, abs=1e-3)
    assert mce == pytest.approx(0.0, abs=1e-3)
    assert not bin_df.empty


# ---------------------------------------------------------------------------
# Test 2: Brier Score Loss bounds
# ---------------------------------------------------------------------------

def test_brier_score_bounds(synthetic_calibration_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Brier score must be between 0.0 and 1.0."""
    y_true, y_prob = synthetic_calibration_data
    brier = CalibrationEvaluator.compute_brier_score(y_true, y_prob)

    assert 0.0 <= brier <= 1.0


# ---------------------------------------------------------------------------
# Test 3: CalibrationEvaluator — full evaluation dict
# ---------------------------------------------------------------------------

def test_calibration_evaluator_dict(synthetic_calibration_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Evaluate returns required metrics and status strings."""
    y_true, y_prob = synthetic_calibration_data
    evaluator = CalibrationEvaluator(n_bins=10)
    res = evaluator.evaluate("Test Model", y_true, y_prob)

    required_keys = {"model_name", "brier_score", "ece", "mce", "log_loss", "status", "bin_table"}
    assert required_keys.issubset(set(res.keys()))

    assert 0.0 <= res["ece"] <= 1.0
    assert 0.0 <= res["mce"] <= 1.0


# ---------------------------------------------------------------------------
# Test 4: ProbabilityCalibrator — Isotonic calibration fit
# ---------------------------------------------------------------------------

def test_probability_calibrator_isotonic() -> None:
    """CalibratedClassifierCV fits Isotonic calibrator with StratifiedGroupKFold."""
    from sklearn.dummy import DummyClassifier

    rng = np.random.default_rng(42)
    n = 200
    X_dev = pd.DataFrame({"feat1": rng.normal(size=n), "feat2": rng.normal(size=n)})
    y_dev = pd.Series(rng.choice([0, 1], size=n, p=[0.85, 0.15]))
    groups_dev = pd.Series(rng.choice(range(1, 11), size=n))

    base_model = DummyClassifier(strategy="prior")
    calibrator = ProbabilityCalibrator(n_folds=3, random_seed=42)

    cal_model = calibrator.fit_calibrated_model(base_model, X_dev, y_dev, groups_dev, method="isotonic")

    probs = cal_model.predict_proba(X_dev.values)[:, 1]
    assert len(probs) == n
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ---------------------------------------------------------------------------
# Test 5: ReliabilityAnalyzer — probability audit table
# ---------------------------------------------------------------------------

def test_reliability_analyzer_audit() -> None:
    """ReliabilityAnalyzer outputs audit table with raw, calibrated, and delta."""
    y_true = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    raw = np.array([0.90, 0.40, 0.85, 0.20, 0.95, 0.10, 0.80, 0.30, 0.75, 0.15])
    cal = np.array([0.75, 0.25, 0.70, 0.12, 0.80, 0.05, 0.65, 0.18, 0.60, 0.08])
    m_ids = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    ts = pd.Series([pd.Timestamp("2015-01-01")] * 10)

    audit_df = ReliabilityAnalyzer.audit_probabilities(
        y_true, raw, cal, machine_ids=m_ids, timestamps=ts, n_samples=6, random_seed=42
    )

    assert not audit_df.empty
    assert "raw_prob" in audit_df.columns
    assert "calibrated_prob" in audit_df.columns
    assert "prob_delta" in audit_df.columns
    assert "actual_failure" in audit_df.columns
