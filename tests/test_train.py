"""
test_train.py
=============
Unit tests for Stage 5 baseline model training, evaluation, and saving pipeline.

Tests cover:
  1. Data preparation & non-feature column filtering.
  2. Logistic Regression Pipeline construction & fitting.
  3. Random Forest Classifier construction & fitting.
  4. XGBoost Classifier scale_pos_weight computation & fitting.
  5. ModelEvaluator metric computation (PR-AUC, ROC-AUC, F1, Confusion Matrix).
  6. ModelSaver pickle & JSON metadata registry export.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import train as tr


@pytest.fixture
def sample_train_test_data():
    """Generates synthetic feature matrices for training and testing."""
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 50

    X_tr = pd.DataFrame({
        "volt": rng.normal(100, 10, n_train),
        "rotate": rng.normal(300, 50, n_train),
        "errors_last_24h": rng.choice([0, 1, 2], size=n_train, p=[0.8, 0.15, 0.05]),
        "y_failure": rng.choice([0, 1], size=n_train, p=[0.9, 0.1]),
        "machineID": rng.choice([1, 2, 3], size=n_train),
        "datetime": pd.date_range("2015-01-01", periods=n_train, freq="h"),
    })

    X_te = pd.DataFrame({
        "volt": rng.normal(100, 10, n_test),
        "rotate": rng.normal(300, 50, n_test),
        "errors_last_24h": rng.choice([0, 1, 2], size=n_test, p=[0.8, 0.15, 0.05]),
        "y_failure": rng.choice([0, 1], size=n_test, p=[0.9, 0.1]),
        "machineID": [4]*n_test,
        "datetime": pd.date_range("2015-01-01", periods=n_test, freq="h"),
    })

    return X_tr, X_te


def test_prepare_data(sample_train_test_data):
    """Test non-feature column removal in prepare_data."""
    X_tr, _ = sample_train_test_data
    X, y, feature_names = tr.ModelTrainer.prepare_data(X_tr)

    assert "datetime" not in X.columns
    assert "machineID" not in X.columns
    assert "y_failure" not in X.columns
    assert len(feature_names) == 3
    assert len(y) == len(X_tr)


def test_logistic_regression_pipeline(sample_train_test_data):
    """Test fitting Logistic Regression Pipeline."""
    X_tr, X_te = sample_train_test_data
    X_train, y_train, _ = tr.ModelTrainer.prepare_data(X_tr)
    X_test, y_test, _ = tr.ModelTrainer.prepare_data(X_te)

    trainer = tr.ModelTrainer(random_seed=42)
    model = trainer.build_logistic_regression()
    model.fit(X_train, y_train)

    eval_res = tr.ModelEvaluator.evaluate("Logistic Regression", model, X_test, y_test, fit_time_sec=0.5)

    assert "pr_auc" in eval_res
    assert "roc_auc" in eval_res
    assert eval_res["pr_auc"] >= 0.0
    assert eval_res["roc_auc"] >= 0.0


def test_random_forest_classifier(sample_train_test_data):
    """Test fitting Random Forest Classifier."""
    X_tr, X_te = sample_train_test_data
    X_train, y_train, _ = tr.ModelTrainer.prepare_data(X_tr)
    X_test, y_test, _ = tr.ModelTrainer.prepare_data(X_te)

    trainer = tr.ModelTrainer(random_seed=42)
    model = trainer.build_random_forest(n_estimators=10)
    model.fit(X_train, y_train)

    eval_res = tr.ModelEvaluator.evaluate("Random Forest", model, X_test, y_test, fit_time_sec=0.8)

    assert eval_res["f1_score"] >= 0.0
    assert eval_res["confusion_matrix"]["TP"] >= 0


def test_xgboost_classifier(sample_train_test_data):
    """Test fitting XGBoost Classifier."""
    X_tr, X_te = sample_train_test_data
    X_train, y_train, _ = tr.ModelTrainer.prepare_data(X_tr)
    X_test, y_test, _ = tr.ModelTrainer.prepare_data(X_te)

    trainer = tr.ModelTrainer(random_seed=42)
    model = trainer.build_xgboost(y_train, n_estimators=10)
    model.fit(X_train, y_train)

    eval_res = tr.ModelEvaluator.evaluate("XGBoost", model, X_test, y_test, fit_time_sec=1.2)

    assert eval_res["pr_auc"] >= 0.0
    assert eval_res["log_loss"] >= 0.0


def test_model_saver(sample_train_test_data, tmp_path):
    """Test saving model pickle and JSON metadata registry."""
    X_tr, X_te = sample_train_test_data
    X_train, y_train, _ = tr.ModelTrainer.prepare_data(X_tr)
    X_test, y_test, _ = tr.ModelTrainer.prepare_data(X_te)

    trainer = tr.ModelTrainer(random_seed=42)
    model = trainer.build_random_forest(n_estimators=5)
    model.fit(X_train, y_train)

    eval_res = tr.ModelEvaluator.evaluate("Test Model", model, X_test, y_test, fit_time_sec=0.1)

    pkl_path = tr.ModelSaver.save(model, "Test Model", eval_res, X_train, y_train, tmp_path)

    assert pkl_path.exists()
    assert (tmp_path / "test_model_metadata.json").exists()
