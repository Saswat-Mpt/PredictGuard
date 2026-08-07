"""
test_cross_validation.py
========================
Unit tests for PredictGuard Stage 6 — src/cross_validation.py.

Tests are fast (<30s) and work with tiny synthetic DataFrames —
they do NOT require any processed parquet files.

Author: PredictGuard Contributors
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable from the project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_validation import (
    CrossValidator,
    FoldAnalyzer,
    HyperparameterTuner,
    MetricAggregator,
    _compute_fold_metrics,
    _prepare_xy,
)


# ---------------------------------------------------------------------------
# Fixture: tiny synthetic development DataFrame (10 machines x 50 rows each)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_dev_df() -> pd.DataFrame:
    """
    Synthetic DataFrame that mimics the development.parquet structure.
    10 machines x 50 rows = 500 rows total.
    ~10% positive failure rate (artificially high to ensure each fold has positives).
    """
    rng = np.random.default_rng(42)
    n_machines = 10
    rows_per_machine = 50
    records = []

    for machine_id in range(1, n_machines + 1):
        n_rows = rows_per_machine
        y = rng.choice([0, 1], size=n_rows, p=[0.90, 0.10])
        for i in range(n_rows):
            records.append({
                "machineID": machine_id,
                "datetime": pd.Timestamp("2015-01-01") + pd.Timedelta(hours=i),
                "y_failure": int(y[i]),
                "failure_component": "comp1" if y[i] else None,
                "time_to_failure_hours": float(rng.uniform(1, 24)) if y[i] else np.nan,
                # A handful of numeric features
                "volt_mean_3h": rng.normal(170, 10),
                "rotate_mean_3h": rng.normal(400, 20),
                "pressure_mean_3h": rng.normal(100, 5),
                "vibration_mean_3h": rng.normal(40, 2),
                "volt_std_3h": rng.uniform(0, 5),
                "error1_count_24h": rng.integers(0, 3),
                "maint_comp1_hours_since": rng.uniform(0, 500),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Test 1: _prepare_xy strips non-feature columns correctly
# ---------------------------------------------------------------------------

def test_prepare_xy_strips_metadata(synthetic_dev_df: pd.DataFrame) -> None:
    """Feature matrix must NOT contain machineID, datetime, y_failure, etc."""
    X, y, groups, feature_names = _prepare_xy(synthetic_dev_df)

    excluded = {"machineID", "datetime", "y_failure", "failure_component", "time_to_failure_hours"}
    for col in excluded:
        assert col not in X.columns, f"Metadata column '{col}' leaked into feature matrix!"

    assert len(X) == len(synthetic_dev_df)
    assert set(groups.unique()) == set(synthetic_dev_df["machineID"].unique())
    assert X.isna().sum().sum() == 0   # NaNs should be filled
    assert len(feature_names) == X.shape[1]


# ---------------------------------------------------------------------------
# Test 2: FoldAnalyzer — zero leakage guarantee
# ---------------------------------------------------------------------------

def test_fold_analyzer_no_leakage(synthetic_dev_df: pd.DataFrame) -> None:
    """No machine ID must appear in both train and validation of any fold."""
    analyzer = FoldAnalyzer(n_folds=5, random_seed=42)
    stats_df, raw_folds = analyzer.analyze(synthetic_dev_df)

    assert not stats_df.empty, "Fold stats DataFrame must not be empty."
    assert "fold" in stats_df.columns

    for fold_info in raw_folds:
        train_machines = set(fold_info["train_machines"])
        val_machines   = set(fold_info["val_machines"])
        overlap = train_machines & val_machines
        assert len(overlap) == 0, (
            f"Fold {fold_info['fold']}: machine leakage detected! "
            f"Shared machine IDs: {overlap}"
        )

    # Leakage column should all be False
    assert stats_df["leakage"].sum() == 0, "Leakage detected in fold stats!"


# ---------------------------------------------------------------------------
# Test 3: FoldAnalyzer — failure rate sanity
# ---------------------------------------------------------------------------

def test_fold_analyzer_failure_rate(synthetic_dev_df: pd.DataFrame) -> None:
    """Validation failure rate must be between 0% and 100% for every fold."""
    analyzer = FoldAnalyzer(n_folds=5, random_seed=42)
    stats_df, _ = analyzer.analyze(synthetic_dev_df)

    assert (stats_df["val_failure_rate_pct"] >= 0.0).all()
    assert (stats_df["val_failure_rate_pct"] <= 100.0).all()


# ---------------------------------------------------------------------------
# Test 4: CrossValidator — returns valid results DataFrame
# ---------------------------------------------------------------------------

def test_cross_validator_returns_dataframe(synthetic_dev_df: pd.DataFrame) -> None:
    """CrossValidator must return a non-empty DataFrame with required columns."""
    # Use only 2 folds and LR only to keep test fast
    cv = CrossValidator(n_folds=2, random_seed=42, models_to_eval=["Logistic Regression"])
    results_df = cv.run(synthetic_dev_df)

    assert isinstance(results_df, pd.DataFrame), "CV must return a DataFrame."
    assert not results_df.empty, "CV results must not be empty."

    required_cols = {"model", "fold", "pr_auc", "roc_auc", "f1_score", "recall", "precision"}
    assert required_cols.issubset(set(results_df.columns)), (
        f"Missing columns: {required_cols - set(results_df.columns)}"
    )

    # PR-AUC must be between 0 and 1
    assert (results_df["pr_auc"] >= 0.0).all()
    assert (results_df["pr_auc"] <= 1.0).all()

    # Should have exactly n_folds rows for 1 model
    assert len(results_df) == 2


# ---------------------------------------------------------------------------
# Test 5: MetricAggregator — correct shape and CI bounds
# ---------------------------------------------------------------------------

def test_metric_aggregator_shape_and_ci(synthetic_dev_df: pd.DataFrame) -> None:
    """Aggregated summary must have one row per model and valid CI values."""
    cv = CrossValidator(n_folds=2, random_seed=42, models_to_eval=["Logistic Regression", "XGBoost"])
    results_df = cv.run(synthetic_dev_df)

    agg = MetricAggregator()
    summary = agg.aggregate(results_df)

    n_models = results_df["model"].nunique()
    assert len(summary) == n_models, "One summary row expected per model."

    # CI must be non-negative
    ci_cols = [c for c in summary.columns if c.endswith("_ci95")]
    for col in ci_cols:
        assert (summary[col] >= 0.0).all(), f"Negative CI in column {col}!"

    # Formatted table must have correct shape
    formatted = agg.format_table(summary)
    assert len(formatted) == n_models
    assert "Model" in formatted.columns
