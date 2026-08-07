"""
run_stage7_8.py
===============
Headless pipeline runner for PredictGuard — Phase 2:
Stage 7 (Calibration Evaluation) & Stage 8 (Probability Calibration).

Usage:
    python run_stage7_8.py

Execution order:
  1. Load development.parquet and test.parquet.
  2. Load trained models (Logistic Regression, Random Forest, XGBoost / best_model).
  3. Evaluate raw probability calibration (Stage 7) -> Brier, ECE, MCE, Log Loss.
  4. Fit Isotonic & Sigmoid post-hoc calibration on Development set using StratifiedGroupKFold.
  5. Evaluate calibrated models on Final Test set (Stage 8).
  6. Select best calibrated model -> save models/calibrated_model.pkl + metadata.
  7. Save raw_model.pkl + calibration reports + probability audit table.
  8. Export publication-quality figures (calibration_dashboard.png, etc.).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (
    CalibrationEvaluator,
    ProbabilityCalibrator,
    ReliabilityAnalyzer,
    generate_calibration_report,
    plot_calibration_dashboard,
    plot_probability_histograms,
    plot_reliability_diagrams,
)

NON_FEATURE_COLS = {
    "datetime",
    "machineID",
    "y_failure",
    "failure_component",
    "time_to_failure_hours",
}


def _prepare_data(df: pd.DataFrame) -> tuple:
    """Extract feature matrix X, target y, and groups machineID."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df["y_failure"].astype(int)
    groups = df["machineID"]

    if X.isna().sum().sum() > 0:
        X = X.fillna(0.0)
    if np.isinf(X.values).any():
        X = X.replace([np.inf, -np.inf], 0.0)

    return X, y, groups


def main() -> None:
    """Execute Phase 2 Calibration pipeline."""
    config_path = PROJECT_ROOT / "config.yaml"
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    log_cfg       = config.get("logging", {})
    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_dir", "data/processed")
    reports_dir   = PROJECT_ROOT / config.get("reports", {}).get("output_dir", "reports")
    figures_dir   = PROJECT_ROOT / config.get("reports", {}).get("figures_dir", "reports/figures")
    models_dir    = PROJECT_ROOT / "models"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        datefmt=log_cfg.get("datefmt", "%H:%M:%S"),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(reports_dir / "phase2_run.log", mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_stage7_8")

    logger.info("=" * 65)
    logger.info(" PredictGuard -- Phase 2: Probability Calibration & Trust")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load Data & Models
    # ------------------------------------------------------------------
    dev_path  = processed_dir / "development.parquet"
    test_path = processed_dir / "test.parquet"

    if not dev_path.exists() or not test_path.exists():
        logger.error("Data parquets not found. Run earlier stages first.")
        sys.exit(1)

    dev_df  = pd.read_parquet(dev_path)
    test_df = pd.read_parquet(test_path)

    X_dev, y_dev, groups_dev = _prepare_data(dev_df)
    X_test, y_test, groups_test = _prepare_data(test_df)

    logger.info("Loaded Development (%d rows) and Test (%d rows).", len(dev_df), len(test_df))

    # Load baseline/best models
    models: Dict[str, Any] = {}
    
    # Best model from Stage 6 or Stage 5 XGBoost
    best_model_path = models_dir / "best_model.pkl"
    if not best_model_path.exists():
        best_model_path = models_dir / "xgboost.pkl"
    if not best_model_path.exists():
        logger.error("No trained XGBoost / best model found in %s", models_dir)
        sys.exit(1)

    xgb_raw = joblib.load(best_model_path)
    models["XGBoost"] = xgb_raw

    for m_name, fname in [("Random Forest", "random_forest.pkl"), ("Logistic Regression", "logistic_regression.pkl")]:
        fpath = models_dir / fname
        if fpath.exists():
            models[m_name] = joblib.load(fpath)

    logger.info("Loaded %d models for calibration evaluation.", len(models))

    # ------------------------------------------------------------------
    # 2. Stage 7: Evaluate Raw Model Calibration
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("STAGE 7: Calibration Evaluation of Raw Probabilities")
    logger.info("-" * 50)

    evaluator = CalibrationEvaluator(n_bins=10)
    raw_results: Dict[str, Dict[str, Any]] = {}
    raw_records = []

    for m_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        res = evaluator.evaluate(m_name, y_test.values, y_prob)
        res["y_prob"] = y_prob
        raw_results[m_name] = res

        raw_records.append({
            "model_name": m_name,
            "brier_score": res["brier_score"],
            "ece": res["ece"],
            "mce": res["mce"],
            "log_loss": res["log_loss"],
            "slope": res["slope"],
            "intercept": res["intercept"],
            "pr_auc": res["pr_auc"],
            "roc_auc": res["roc_auc"],
            "status": res["status"],
        })

    raw_metrics_df = pd.DataFrame(raw_records).sort_values("ece").reset_index(drop=True)
    logger.info("\nRaw Model Calibration Metrics:\n%s\n", raw_metrics_df.to_string(index=False))

    # Save Stage 7 CSV
    raw_metrics_df.to_csv(reports_dir / "calibration_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Stage 8: Post-Hoc Calibration (Isotonic vs Sigmoid)
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("STAGE 8: Post-Hoc Probability Calibration (XGBoost)")
    logger.info("-" * 50)

    calibrator = ProbabilityCalibrator(n_folds=5, random_seed=42)

    # A. Isotonic Calibration
    logger.info("Fitting Isotonic Calibration on Development set (80 machines) ...")
    t0 = time.time()
    iso_model = calibrator.fit_calibrated_model(xgb_raw, X_dev, y_dev, groups_dev, method="isotonic")
    t_iso = time.time() - t0
    iso_prob = iso_model.predict_proba(X_test.values)[:, 1]
    iso_eval = evaluator.evaluate("XGBoost (Isotonic)", y_test.values, iso_prob, fit_time_sec=t_iso)
    iso_eval["y_prob"] = iso_prob
    iso_eval["method"] = "Isotonic"

    # B. Sigmoid / Platt Scaling
    logger.info("Fitting Sigmoid (Platt Scaling) Calibration on Development set (80 machines) ...")
    t0 = time.time()
    sig_model = calibrator.fit_calibrated_model(xgb_raw, X_dev, y_dev, groups_dev, method="sigmoid")
    t_sig = time.time() - t0
    sig_prob = sig_model.predict_proba(X_test.values)[:, 1]
    sig_eval = evaluator.evaluate("XGBoost (Sigmoid)", y_test.values, sig_prob, fit_time_sec=t_sig)
    sig_eval["y_prob"] = sig_prob
    sig_eval["method"] = "Sigmoid"

    # Raw XGBoost baseline for comparison
    raw_xgb_eval = raw_results["XGBoost"]
    raw_xgb_eval["method"] = "None (Raw)"

    # Before vs After Comparison DataFrame
    before_after_df = pd.DataFrame([
        raw_xgb_eval,
        iso_eval,
        sig_eval,
    ])[["model_name", "method", "brier_score", "ece", "mce", "log_loss", "pr_auc", "roc_auc", "recall", "precision", "status"]]

    logger.info("\nBefore vs After Calibration Comparison:\n%s\n", before_after_df.to_string(index=False))
    before_after_df.to_csv(reports_dir / "calibration_before_after.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Select Best Calibrated Model & Save Artifacts
    # ------------------------------------------------------------------
    best_cal_model = iso_model if iso_eval["ece"] <= sig_eval["ece"] else sig_model
    best_cal_eval  = iso_eval if iso_eval["ece"] <= sig_eval["ece"] else sig_eval
    best_method    = best_cal_eval["method"]

    logger.info("Best calibration method selected: %s (ECE=%.4f, Brier=%.4f)",
                best_method.upper(), best_cal_eval["ece"], best_cal_eval["brier_score"])

    # Save raw model backup
    joblib.dump(xgb_raw, models_dir / "raw_model.pkl")
    logger.info("Saved raw model backup: models/raw_model.pkl")

    # Save calibrated model
    joblib.dump(best_cal_model, models_dir / "calibrated_model.pkl")
    logger.info("Saved calibrated model: models/calibrated_model.pkl")

    # Calibration Metadata JSON
    cal_metadata: Dict[str, Any] = {
        "base_model": "XGBoost",
        "calibration_method": best_method,
        "fitting_cv": "StratifiedGroupKFold(n_splits=5)",
        "fit_machines": dev_df["machineID"].nunique(),
        "fit_rows": len(dev_df),
        "test_machines": test_df["machineID"].nunique(),
        "test_rows": len(test_df),
        "brier_before": raw_xgb_eval["brier_score"],
        "brier_after": best_cal_eval["brier_score"],
        "ece_before": raw_xgb_eval["ece"],
        "ece_after": best_cal_eval["ece"],
        "mce_before": raw_xgb_eval["mce"],
        "mce_after": best_cal_eval["mce"],
        "pr_auc_before": raw_xgb_eval["pr_auc"],
        "pr_auc_after": best_cal_eval["pr_auc"],
        "pr_auc_change": round(best_cal_eval["pr_auc"] - raw_xgb_eval["pr_auc"], 4),
        "calibration_timestamp": datetime.now().isoformat(),
    }

    cal_meta_path = models_dir / "calibration_metadata.json"
    with open(cal_meta_path, "w", encoding="utf-8") as f:
        json.dump(cal_metadata, f, indent=2)
    logger.info("Saved calibration metadata card: %s", cal_meta_path)

    # ------------------------------------------------------------------
    # 5. Probability Audit Table
    # ------------------------------------------------------------------
    audit_df = ReliabilityAnalyzer.audit_probabilities(
        y_true=y_test,
        raw_probs=raw_xgb_eval["y_prob"],
        calibrated_probs=best_cal_eval["y_prob"],
        machine_ids=test_df["machineID"],
        timestamps=test_df["datetime"],
        n_samples=15,
        random_seed=42,
    )
    logger.info("\nProbability Audit Sample (Raw vs Calibrated):\n%s\n", audit_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Figures & Reports
    # ------------------------------------------------------------------
    logger.info("Generating publication-quality figures ...")
    plot_calibration_dashboard(
        eval_raw_models=raw_results,
        raw_eval=raw_xgb_eval,
        iso_eval=iso_eval,
        sig_eval=sig_eval,
        y_test=y_test.values,
        raw_probs=raw_xgb_eval["y_prob"],
        calibrated_probs=best_cal_eval["y_prob"],
        figures_dir=figures_dir,
    )
    plot_reliability_diagrams(
        eval_results={"Raw XGBoost": raw_xgb_eval, "Isotonic XGBoost": iso_eval, "Sigmoid XGBoost": sig_eval},
        figures_dir=figures_dir,
    )
    plot_probability_histograms(
        raw_probs=raw_xgb_eval["y_prob"],
        calibrated_probs=best_cal_eval["y_prob"],
        figures_dir=figures_dir,
    )

    generate_calibration_report(
        raw_eval_df=raw_metrics_df,
        before_after_df=before_after_df,
        audit_sample_df=audit_df,
        best_method=best_method,
        output_path=reports_dir / "calibration_report.md",
    )

    logger.info("=" * 65)
    logger.info(" Phase 2 (Stage 7 & Stage 8) COMPLETE")
    logger.info(" Calibration Improvement:")
    logger.info("   ECE         : %.4f -> %.4f (%.1f%% reduction)",
                raw_xgb_eval["ece"], best_cal_eval["ece"],
                100.0 * (raw_xgb_eval["ece"] - best_cal_eval["ece"]) / raw_xgb_eval["ece"])
    logger.info("   Brier Score : %.4f -> %.4f", raw_xgb_eval["brier_score"], best_cal_eval["brier_score"])
    logger.info("   PR-AUC      : %.4f -> %.4f", raw_xgb_eval["pr_auc"], best_cal_eval["pr_auc"])
    logger.info(" Saved models to %s", models_dir)
    logger.info(" Saved reports to %s", reports_dir)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
