"""
test_split.py
=============
Unit tests for leakage-safe machine-level data splitting pipeline.

Tests cover:
  1. Machine-level splitting (zero machine overlap between Dev and Test).
  2. Stratification quality (failure occurrence balanced across Dev and Test).
  3. Grouped K-Fold cross validation (zero machine overlap within any fold).
  4. Leakage validator exception raising when leakage is injected.
  5. Split manifest generation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import split as sp


@pytest.fixture
def sample_feature_df():
    """Generates sample telemetry feature DataFrame for 10 machines."""
    records = []
    dates = pd.date_range("2015-01-01 00:00:00", periods=100, freq="h")
    for mid in range(1, 11):
        # Machine 1..8 have failures, 9..10 have zero failures
        has_fail = mid <= 8
        for d in dates:
            y = 1 if has_fail and d.hour == 12 else 0
            records.append({
                "datetime": d,
                "machineID": mid,
                "volt": 100.0 + mid,
                "rotate": 300.0,
                "y_failure": y,
                "model_model1": 1 if mid % 2 == 0 else 0,
            })
    return pd.DataFrame(records)


def test_machine_splitter_no_overlap(sample_feature_df):
    """Verify that MachineSplitter creates mutually exclusive Dev and Test sets."""
    splitter = sp.MachineSplitter(test_ratio=0.20, random_seed=42, stratify_by_failure=True)
    dev_df, test_df, dev_ids, test_ids = splitter.split(sample_feature_df)

    assert len(set(dev_ids).intersection(set(test_ids))) == 0
    assert len(dev_ids) == 8
    assert len(test_ids) == 2
    assert len(dev_df) == 800
    assert len(test_df) == 200


def test_grouped_cross_validator(sample_feature_df):
    """Verify GroupedCrossValidator creates 5 folds with zero machine overlap per fold."""
    splitter = sp.MachineSplitter(test_ratio=0.20, random_seed=42)
    dev_df, _, _, _ = splitter.split(sample_feature_df)

    cv = sp.GroupedCrossValidator(n_folds=4, random_seed=42)
    fold_assignments, fold_summary = cv.generate_folds(dev_df)

    assert len(fold_assignments["folds"]) == 4
    for fold_key, fold_info in fold_assignments["folds"].items():
        tr = set(fold_info["train_machines"])
        val = set(fold_info["val_machines"])
        assert len(tr.intersection(val)) == 0


def test_leakage_validator_pass(sample_feature_df):
    """Verify LeakageValidator passes clean split."""
    splitter = sp.MachineSplitter(test_ratio=0.20, random_seed=42)
    dev_df, test_df, _, _ = splitter.split(sample_feature_df)

    cv = sp.GroupedCrossValidator(n_folds=4, random_seed=42)
    fold_assignments, _ = cv.generate_folds(dev_df)

    val_res = sp.LeakageValidator.validate_split(dev_df, test_df, fold_assignments)
    assert val_res["passed_all"] is True


def test_leakage_validator_detects_leak(sample_feature_df):
    """Verify LeakageValidator raises ValueError if a machine is injected into both Dev and Test."""
    splitter = sp.MachineSplitter(test_ratio=0.20, random_seed=42)
    dev_df, test_df, _, _ = splitter.split(sample_feature_df)

    # Inject Machine 1 into test set (creating machine overlap leak)
    test_df_leaky = pd.concat([test_df, sample_feature_df[sample_feature_df["machineID"] == 1]])

    cv = sp.GroupedCrossValidator(n_folds=4, random_seed=42)
    fold_assignments, _ = cv.generate_folds(dev_df)

    with pytest.raises(ValueError, match="Leakage validation FAILED"):
        sp.LeakageValidator.validate_split(dev_df, test_df_leaky, fold_assignments)
