"""
test_risk_decision.py
=====================
Unit tests for PredictGuard Phase 3 — Stage 11 & 12.

Tests:
    test_risk_segmenter_score_range       — risk scores are in [0, 100]
    test_risk_segmenter_tier_assignment   — tier thresholds are respected
    test_fleet_analyzer_tier_summary      — tier summary has required columns
    test_fleet_analyzer_machine_summary   — machine summary deduplication
    test_threshold_optimizer_sweep        — sweep covers correct range
    test_cost_evaluator_zero_fn           — perfect recall → FN cost = 0
    test_cost_evaluator_zero_fp           — predict all negative → no FP cost
    test_decision_engine_dispatch         — high-risk → DISPATCH
    test_decision_engine_monitor          — low-risk → MONITOR
    test_cost_savings_direction           — optimal saves cost vs baseline
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_segmentation import RiskSegmenter, FleetAnalyzer, DEFAULT_TIER_THRESHOLDS
from src.cost_decision import (
    CostEvaluator,
    ThresholdOptimizer,
    DecisionEngine,
    DEFAULT_COST_MATRIX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_COLS = [f"feat_{i}" for i in range(10)]


def _make_scored_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic scored DataFrame (already has calibrated_prob etc.)."""
    rng = np.random.RandomState(seed)
    probs = rng.beta(1, 5, size=n)          # skewed toward low risk
    scores = (probs * 100).round().astype(int).clip(0, 100)
    df = pd.DataFrame({
        "machineID": rng.randint(1, 21, size=n),
        "datetime": pd.date_range("2015-01-01", periods=n, freq="h"),
        "y_failure": (rng.rand(n) < 0.05).astype(int),
        "calibrated_prob": probs,
        "risk_score": scores,
        "risk_tier": ["LOW"] * n,           # will be overwritten
        "predicted_component": rng.choice(["comp1", "comp2", "comp3", "comp4"], size=n),
        "component_confidence": rng.uniform(0.5, 1.0, size=n),
    })
    segmenter = RiskSegmenter()
    df["risk_tier"] = df["risk_score"].apply(segmenter._assign_tier)
    return df


class _FakeCalModel:
    """Stub calibrated model returning fixed proba column."""
    def __init__(self, proba: np.ndarray):
        self._proba = proba

    def predict_proba(self, X):
        n = len(X)
        p = self._proba[:n] if len(self._proba) >= n else np.full(n, 0.5)
        return np.column_stack([1 - p, p])


# ===========================================================================
# Tests — Stage 11 RiskSegmenter
# ===========================================================================

def test_risk_segmenter_score_range():
    """Risk scores must always be in [0, 100]."""
    n = 100
    rng = np.random.RandomState(1)
    probs = rng.rand(n)
    df = pd.DataFrame(
        {**{c: rng.randn(n) for c in FEATURE_COLS},
         "machineID": 1, "datetime": pd.date_range("2015-01-01", periods=n, freq="h"),
         "y_failure": 0}
    )
    fake_model = _FakeCalModel(probs)
    segmenter = RiskSegmenter()
    scored = segmenter.assign_risk_scores(df, fake_model, FEATURE_COLS)
    assert scored["risk_score"].between(0, 100).all()
    assert scored["calibrated_prob"].between(0.0, 1.0).all()


def test_risk_segmenter_tier_assignment():
    """Tier boundaries must be respected: LOW < 20, MEDIUM 20-49, HIGH 50-74, CRITICAL >=75."""
    segmenter = RiskSegmenter()
    assert segmenter._assign_tier(10) == "LOW"
    assert segmenter._assign_tier(19) == "LOW"
    assert segmenter._assign_tier(20) == "MEDIUM"
    assert segmenter._assign_tier(49) == "MEDIUM"
    assert segmenter._assign_tier(50) == "HIGH"
    assert segmenter._assign_tier(74) == "HIGH"
    assert segmenter._assign_tier(75) == "CRITICAL"
    assert segmenter._assign_tier(100) == "CRITICAL"


def test_fleet_analyzer_tier_summary(tmp_path):
    """Tier summary must have all 4 tiers and required columns."""
    df = _make_scored_df(400)
    # Force some into each tier
    df.loc[:50, "risk_score"] = 10
    df.loc[:50, "risk_tier"] = "LOW"
    df.loc[51:100, "risk_score"] = 30
    df.loc[51:100, "risk_tier"] = "MEDIUM"
    df.loc[101:150, "risk_score"] = 60
    df.loc[101:150, "risk_tier"] = "HIGH"
    df.loc[151:200, "risk_score"] = 85
    df.loc[151:200, "risk_tier"] = "CRITICAL"

    analyzer = FleetAnalyzer(figures_dir=tmp_path, reports_dir=tmp_path)
    tier_df = analyzer.build_tier_summary(df)

    required_cols = {"tier", "n_machine_hours", "n_unique_machines",
                     "avg_risk_score", "failure_rate"}
    assert required_cols.issubset(tier_df.columns)
    assert set(tier_df["tier"]) == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_fleet_analyzer_machine_summary(tmp_path):
    """Machine summary should have exactly one row per unique machine."""
    df = _make_scored_df(300)
    analyzer = FleetAnalyzer(figures_dir=tmp_path, reports_dir=tmp_path)
    summary = analyzer.build_machine_summary(df)
    assert len(summary) == df["machineID"].nunique()
    assert "latest_risk_score" in summary.columns
    assert "risk_tier" in summary.columns


# ===========================================================================
# Tests — Stage 12 CostEvaluator / ThresholdOptimizer
# ===========================================================================

def test_threshold_optimizer_sweep():
    """Sweep must cover the correct range and have expected columns."""
    rng = np.random.RandomState(42)
    y = rng.randint(0, 2, size=500)
    p = rng.rand(500)
    optimizer = ThresholdOptimizer(threshold_min=0.1, threshold_max=0.5, threshold_step=0.1)
    sweep_df = optimizer.sweep(y, p)

    assert len(sweep_df) == pytest.approx(5, abs=1)   # 0.1, 0.2, 0.3, 0.4, 0.5
    required = {"threshold", "TP", "FP", "TN", "FN", "expected_cost", "recall", "precision"}
    assert required.issubset(sweep_df.columns)
    assert (sweep_df["threshold"] >= 0.1).all()
    assert (sweep_df["threshold"] <= 0.51).all()


def test_cost_evaluator_zero_fn():
    """If threshold = 0.0 (predict all positive), FN = 0 → FN cost = 0."""
    evaluator = CostEvaluator({"C_FN": 10_000, "C_FP": 500})
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.9])  # all predicted positive
    result = evaluator.evaluate(y, p, threshold=0.0)
    assert result["FN"] == 0
    assert result["recall"] == pytest.approx(1.0)
    # Expected cost = C_FP × FP only
    assert result["expected_cost"] == pytest.approx(500 * result["FP"])


def test_cost_evaluator_zero_fp():
    """If threshold = 1.0 (predict all negative), FP = 0 → FP cost = 0."""
    evaluator = CostEvaluator({"C_FN": 10_000, "C_FP": 500})
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    result = evaluator.evaluate(y, p, threshold=1.01)   # nothing predicted positive
    assert result["FP"] == 0
    assert result["expected_cost"] == pytest.approx(10_000 * result["FN"])


def test_decision_engine_dispatch():
    """Machine with prob above threshold → DISPATCH."""
    engine = DecisionEngine(optimal_threshold=0.4, cost_matrix=DEFAULT_COST_MATRIX)
    machine_summary = pd.DataFrame([{
        "machineID": 42,
        "latest_risk_score": 85,
        "max_calibrated_prob": 0.85,
        "latest_calibrated_prob": 0.85,
        "risk_tier": "CRITICAL",
        "predicted_component": "comp3",
        "component_confidence": 0.72,
        "avg_risk_score": 80.0,
        "failure_rate": 0.15,
    }])
    decisions = engine.generate_decisions(machine_summary, df_scored=machine_summary)
    assert decisions.iloc[0]["decision"] == "DISPATCH"


def test_decision_engine_monitor():
    """Machine with prob below threshold → MONITOR."""
    engine = DecisionEngine(optimal_threshold=0.4, cost_matrix=DEFAULT_COST_MATRIX)
    machine_summary = pd.DataFrame([{
        "machineID": 18,
        "latest_risk_score": 12,
        "max_calibrated_prob": 0.12,
        "latest_calibrated_prob": 0.12,
        "risk_tier": "LOW",
        "predicted_component": "comp1",
        "component_confidence": 0.50,
        "avg_risk_score": 10.0,
        "failure_rate": 0.01,
    }])
    decisions = engine.generate_decisions(machine_summary, df_scored=machine_summary)
    assert decisions.iloc[0]["decision"] == "MONITOR"


def test_cost_savings_direction():
    """Optimal threshold should save cost compared to naive 0.5 baseline."""
    rng = np.random.RandomState(99)
    y = rng.randint(0, 2, size=1000)
    p = np.clip(y * 0.6 + rng.randn(1000) * 0.15, 0, 1)
    optimizer = ThresholdOptimizer(cost_matrix={"C_FN": 10_000, "C_FP": 500})
    optimizer.sweep(y, p)
    opt_t, opt_m = optimizer.find_optimal()

    engine = DecisionEngine(opt_t, {"C_FN": 10_000, "C_FP": 500})
    savings = engine.compute_cost_savings(y, p, baseline_threshold=0.5)

    # Optimal should not increase cost over baseline
    assert savings["cost_savings"] >= 0, (
        f"Optimal threshold increased cost by ${-savings['cost_savings']:.0f}"
    )
