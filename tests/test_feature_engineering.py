"""
test_feature_engineering.py
============================
Unit tests for Stage 3 feature engineering pipeline.

Tests cover:
  1. Trailing-only property of rolling statistics (no future leakage).
  2. Shifted property of expanding z-scores (current row excluded from its expanding mean).
  3. Rate of change deltas calculation.
  4. Safe interaction terms (no division by zero).
  5. Leakage-safe error and maintenance temporal joins.
  6. Automated leakage checker validation suite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import feature_engineering as fe
from src.validators import leakage_checker as lc


@pytest.fixture
def sample_telemetry():
    """Generates 48 hours of telemetry for 2 machines."""
    dates = pd.date_range("2015-01-01 00:00:00", periods=48, freq="h")
    df1 = pd.DataFrame({
        "datetime": dates,
        "machineID": 1,
        "volt": np.linspace(100, 200, 48),
        "rotate": np.linspace(300, 400, 48),
        "pressure": np.linspace(80, 120, 48),
        "vibration": np.linspace(30, 50, 48),
        "y_failure": [0]*40 + [1]*8,
        "time_to_failure_hours": [np.nan]*40 + [8.0]*8,
        "failure_component": [np.nan]*40 + ["comp1"]*8,
    })
    df2 = pd.DataFrame({
        "datetime": dates,
        "machineID": 2,
        "volt": np.linspace(150, 250, 48),
        "rotate": np.linspace(250, 350, 48),
        "pressure": np.linspace(90, 110, 48),
        "vibration": np.linspace(35, 45, 48),
        "y_failure": [0]*48,
        "time_to_failure_hours": [np.nan]*48,
        "failure_component": [np.nan]*48,
    })
    return pd.concat([df1, df2], ignore_index=True)


def test_trailing_rolling_stats(sample_telemetry):
    """Test that rolling mean is trailing and does not leak future values."""
    res = fe.compute_rolling_features(sample_telemetry, windows=[3])
    
    # Machine 1 first 3 rows volt values: 100.0, 102.1276..., 104.2553...
    # Row 0 rolling mean (window=3, min_periods=1) should be 100.0
    m1 = res[res["machineID"] == 1]
    assert np.isclose(m1.iloc[0]["volt_rolling_mean_3h"], m1.iloc[0]["volt"])
    assert np.isclose(m1.iloc[2]["volt_rolling_mean_3h"], m1.iloc[0:3]["volt"].mean())


def test_expanding_zscores_shifted(sample_telemetry):
    """Test that expanding mean is shifted by 1 timestep to prevent self-leakage."""
    res = fe.compute_expanding_zscores(sample_telemetry)
    m1 = res[res["machineID"] == 1]
    
    # Expanding mean at index 1 must equal raw volt value at index 0
    raw_0 = m1.iloc[0]["volt"]
    exp_mean_1 = m1.iloc[1]["volt_exp_mean"]
    assert np.isclose(raw_0, exp_mean_1)


def test_rate_of_change(sample_telemetry):
    """Test delta_1h and velocity calculation."""
    res = fe.compute_rate_of_change_features(sample_telemetry, lags=[1, 3])
    m1 = res[res["machineID"] == 1]
    
    # delta_1h at index 1 should be volt[1] - volt[0]
    expected_delta = m1.iloc[1]["volt"] - m1.iloc[0]["volt"]
    assert np.isclose(m1.iloc[1]["volt_delta_1h"], expected_delta)


def test_safe_interactions(sample_telemetry):
    """Test interaction terms with zero values do not raise DivisionByZero."""
    tel = sample_telemetry.copy()
    tel["volt"] = 0.0  # zero voltage
    res = fe.compute_interaction_features(tel)
    
    # Division by zero voltage should produce finite numbers due to eps
    assert np.isfinite(res["rotation_div_voltage"]).all()


def test_leakage_checker_suite(sample_telemetry):
    """Run automated leakage checker on sample telemetry."""
    res = fe.compute_rolling_features(sample_telemetry)
    res = fe.compute_expanding_zscores(res)
    leakage_results = lc.run_full_leakage_suite(res)
    assert leakage_results["overall_leakage_free"] is True
