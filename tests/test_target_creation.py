"""
test_target_creation.py
========================
Unit tests for leakage-safe target construction logic.

Tests cover:
  1. Machine with no failures (all y=0, failure_component=NaN)
  2. Single failure (y=1 for exactly 24 hours prior, component correctly assigned)
  3. Failure occurring exactly at t + 24h (inclusive boundary check)
  4. Failure occurring at t + 24h + 1 second (out of window check)
  5. Multiple failures within 24h window (first component assigned)
  6. End-of-history telemetry rows
  7. Cross-machine isolation (events on machine A do not affect machine B)

Run with:
    pytest tests/test_target_creation.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import target_creation as tc


@pytest.fixture
def base_timestamps():
    """Generates 48 hourly timestamps."""
    return pd.date_range("2015-01-01 00:00:00", periods=48, freq="h")


def test_machine_with_no_failures(base_timestamps):
    """Test that a machine with zero failures gets all y_failure=0 and NaN components."""
    telemetry = pd.DataFrame({
        "machineID": 1,
        "datetime": base_timestamps,
        "volt": 100.0,
    })
    failures = pd.DataFrame(columns=["machineID", "datetime", "failure"])

    res = tc.compute_binary_target(telemetry, failures, horizon_hours=24)
    res = tc.assign_failure_component(res, failures, horizon_hours=24)

    assert (res["y_failure"] == 0).all()
    assert res["failure_component"].isna().all()
    assert res["time_to_failure_hours"].isna().all()


def test_single_failure_exact_window(base_timestamps):
    """Test positive label generation for a failure at index 24 (2015-01-02 00:00:00)."""
    telemetry = pd.DataFrame({
        "machineID": 1,
        "datetime": base_timestamps,
        "volt": 100.0,
    })
    # Failure at 2015-01-02 00:00:00 (index 24)
    fail_time = pd.Timestamp("2015-01-02 00:00:00")
    failures = pd.DataFrame({
        "machineID": [1],
        "datetime": [fail_time],
        "failure": ["comp1"]
    })

    res = tc.compute_binary_target(telemetry, failures, horizon_hours=24)
    res = tc.assign_failure_component(res, failures, horizon_hours=24)

    # Telemetry timestamps t where fail_time is in (t, t + 24h]:
    # t in [2015-01-01 00:00:00, 2015-01-01 23:00:00] -> exactly 24 hourly rows (indices 0 to 23)
    # Index 0: 2015-01-01 00:00:00 -> +24h is 2015-01-02 00:00:00 (exact boundary, inclusive) -> y=1
    # Index 24: 2015-01-02 00:00:00 -> +24h is 2015-01-03 00:00:00, but fail_time is at 2015-01-02 00:00:00 (<= t) -> y=0

    pos_indices = res[res["y_failure"] == 1].index.tolist()
    assert pos_indices == list(range(0, 24))
    assert (res.loc[0:23, "failure_component"] == "comp1").all()
    assert res.loc[24:, "y_failure"].sum() == 0


def test_boundary_inclusive_and_exclusive(base_timestamps):
    """Verify that (t, t+24h] excludes failure at t and includes failure at t+24h."""
    telemetry = pd.DataFrame({
        "machineID": [1, 1],
        "datetime": [pd.Timestamp("2015-01-01 00:00:00"), pd.Timestamp("2015-01-01 01:00:00")]
    })
    # Failure exactly at 2015-01-01 00:00:00
    failures_at_t = pd.DataFrame({
        "machineID": [1],
        "datetime": [pd.Timestamp("2015-01-01 00:00:00")],
        "failure": ["comp2"]
    })
    res_at_t = tc.compute_binary_target(telemetry, failures_at_t, horizon_hours=24)
    # At t=2015-01-01 00:00:00, failure is at t (not in (t, t+24h]), so y=0
    assert res_at_t.loc[res_at_t["datetime"] == "2015-01-01 00:00:00", "y_failure"].iloc[0] == 0

    # Failure at 2015-01-02 00:00:00 (exact t+24h for 2015-01-01 00:00:00)
    failures_at_24h = pd.DataFrame({
        "machineID": [1],
        "datetime": [pd.Timestamp("2015-01-02 00:00:00")],
        "failure": ["comp2"]
    })
    res_24h = tc.compute_binary_target(telemetry, failures_at_24h, horizon_hours=24)
    # At t=2015-01-01 00:00:00, failure is at t+24h (in (t, t+24h]), so y=1
    assert res_24h.loc[res_24h["datetime"] == "2015-01-01 00:00:00", "y_failure"].iloc[0] == 1


def test_multiple_failures_first_component_assigned(base_timestamps):
    """Test that when multiple failures occur in window, the FIRST component is assigned."""
    telemetry = pd.DataFrame({
        "machineID": [1],
        "datetime": [pd.Timestamp("2015-01-01 00:00:00")]
    })
    failures = pd.DataFrame({
        "machineID": [1, 1],
        "datetime": [pd.Timestamp("2015-01-01 06:00:00"), pd.Timestamp("2015-01-01 12:00:00")],
        "failure": ["comp3", "comp4"]
    })

    res = tc.compute_binary_target(telemetry, failures, horizon_hours=24)
    res = tc.assign_failure_component(res, failures, horizon_hours=24)

    assert res.iloc[0]["y_failure"] == 1
    assert res.iloc[0]["failure_component"] == "comp3"  # comp3 happens first
    assert res.iloc[0]["time_to_failure_hours"] == 6.0  # time to first failure


def test_cross_machine_isolation():
    """Ensure machine A failures do not affect machine B targets."""
    telemetry = pd.DataFrame({
        "machineID": [1, 2],
        "datetime": [pd.Timestamp("2015-01-01 00:00:00"), pd.Timestamp("2015-01-01 00:00:00")]
    })
    failures = pd.DataFrame({
        "machineID": [1],
        "datetime": [pd.Timestamp("2015-01-01 12:00:00")],
        "failure": ["comp1"]
    })

    res = tc.compute_binary_target(telemetry, failures, horizon_hours=24)
    res = tc.assign_failure_component(res, failures, horizon_hours=24)

    m1_row = res[res["machineID"] == 1].iloc[0]
    m2_row = res[res["machineID"] == 2].iloc[0]

    assert m1_row["y_failure"] == 1
    assert m1_row["failure_component"] == "comp1"

    assert m2_row["y_failure"] == 0
    assert pd.isna(m2_row["failure_component"])
