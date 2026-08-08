"""
run_stage13_14.py
=================
PredictGuard — Phase 4 Orchestrator: Stage 13 (MLflow Monitoring) + Stage 14 (Pipeline)

Usage:
    python run_stage13_14.py

Stage 13 Outputs:
    mlruns/                         ← local MLflow tracking store
    reports/mlflow_summary.md       ← human-readable run summary

Stage 14 Outputs:
    models/predictguard_pipeline.pkl
    models/predictguard_pipeline_metadata.json
    reports/pipeline_validation.md
    reports/reproducibility_report.md
    reports/pipeline_report.md

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_stage13_14")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEV_PARQUET  = DATA_DIR / "development.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"

NON_FEATURE_COLS = {
    "datetime", "machineID", "y_failure",
    "failure_component", "time_to_failure_hours",
    "calibrated_prob", "risk_score", "risk_tier",
    "predicted_component", "component_confidence",
}


# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
from src.monitoring import (
    ExperimentTracker,
    ArtifactLogger,
    MetricLoader,
    MLflowSummaryWriter,
)
from src.pipeline import (
    PredictGuardPipeline,
    PipelineValidator,
    PipelineReportWriter,
)
from src.component_classifier import ComponentPredictor


# ===========================================================================
# Helpers
# ===========================================================================

def _banner(msg: str) -> None:
    logger.info("=" * 66)
    logger.info(" %s", msg)
    logger.info("=" * 66)


def _section(msg: str) -> None:
    logger.info("-" * 50)
    logger.info("%s", msg)
    logger.info("-" * 50)


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _load_models() -> Dict[str, Any]:
    logger.info("Loading trained models ...")
    models = {
        "raw_model":       joblib.load(MODELS_DIR / "raw_model.pkl"),
        "calibrated_model": joblib.load(MODELS_DIR / "calibrated_model.pkl"),
        "component_predictor": ComponentPredictor.load(MODELS_DIR),
    }
    logger.info(
        "Loaded: raw=%s  calibrated=%s  component=%d classes",
        type(models["raw_model"]).__name__,
        type(models["calibrated_model"]).__name__,
        len(models["component_predictor"].classes_),
    )
    return models


def _load_optimal_threshold() -> float:
    ot_path = REPORTS_DIR / "optimal_threshold.json"
    if ot_path.exists():
        with open(ot_path) as f:
            return float(json.load(f)["optimal_threshold"])
    logger.warning("optimal_threshold.json not found — using default 0.68")
    return 0.68


# ===========================================================================
# Stage 13 — MLflow Experiment Tracking
# ===========================================================================

def run_stage13(models: Dict[str, Any], feature_cols: List[str]) -> str:
    """Log all Phase 1–12 metrics, params, and artifacts to MLflow."""
    _banner("STAGE 13: MLflow Experiment Monitoring")
    t_stage = time.time()

    loader  = MetricLoader(reports_dir=REPORTS_DIR, models_dir=MODELS_DIR)
    tracker = ExperimentTracker(project_root=PROJECT_ROOT)
    artifact_logger = ArtifactLogger(
        figures_dir=FIGURES_DIR, reports_dir=REPORTS_DIR, models_dir=MODELS_DIR
    )
    summary_writer = MLflowSummaryWriter(reports_dir=REPORTS_DIR)

    # -----------------------------------------------------------------------
    # PART 1-3 — Start run, log params
    # -----------------------------------------------------------------------
    _section("PART 1-3: Start MLflow Run & Log Parameters")
    run_id = tracker.start_run("PredictGuard_Production")

    all_params = loader.load_all_params()
    all_params["feature_count"] = len(feature_cols)
    cost_data = loader.load_cost_metrics()
    all_params.update(cost_data["params"])
    tracker.log_params(all_params)

    # -----------------------------------------------------------------------
    # PART 2 — Log all metrics
    # -----------------------------------------------------------------------
    _section("PART 2: Log Metrics from Stages 1-12")

    all_metrics: Dict[str, float] = {}

    training_metrics = loader.load_training_metrics()
    all_metrics.update(training_metrics)
    logger.info("Training metrics: %d entries", len(training_metrics))

    cal_metrics = loader.load_calibration_metrics()
    all_metrics.update(cal_metrics)
    logger.info("Calibration metrics: %d entries", len(cal_metrics))

    comp_metrics = loader.load_component_metrics()
    all_metrics.update(comp_metrics)
    logger.info("Component metrics: %d entries", len(comp_metrics))

    fleet_metrics = loader.load_fleet_metrics()
    all_metrics.update(fleet_metrics)
    logger.info("Fleet metrics: %d entries", len(fleet_metrics))

    cost_metrics = cost_data["metrics"]
    all_metrics.update(cost_metrics)
    logger.info("Decision metrics: %d entries", len(cost_metrics))

    tracker.log_metrics(all_metrics)

    # Log cost-vs-threshold sweep as metric series
    cost_csv = REPORTS_DIR / "cost_analysis.csv"
    if cost_csv.exists():
        sweep_df = pd.read_csv(cost_csv)
        tracker.log_metric_series(
            "threshold_expected_cost",
            (sweep_df["expected_cost"] / 1000).tolist()
        )
        logger.info("Logged cost sweep series (%d points).", len(sweep_df))

    # -----------------------------------------------------------------------
    # PART 4 — Log artifacts
    # -----------------------------------------------------------------------
    _section("PART 4: Log Artifacts (Figures, Reports, Models)")
    n_figures = artifact_logger.log_figures()
    n_reports = artifact_logger.log_reports()
    n_models  = artifact_logger.log_models()

    artifact_counts = {
        "Figures (PNG)": n_figures,
        "Reports & Data (MD/CSV/JSON)": n_reports,
        "Model Files (PKL/JSON)": n_models,
    }

    # -----------------------------------------------------------------------
    # PART 5 — End run & summary
    # -----------------------------------------------------------------------
    tracker.end_run()

    tracking_uri = tracker.tracking_uri
    summary_writer.write(
        run_id=run_id,
        run_name="PredictGuard_Production",
        all_metrics=all_metrics,
        all_params=all_params,
        artifact_counts=artifact_counts,
        tracking_uri=tracking_uri,
    )

    elapsed = time.time() - t_stage
    logger.info("Stage 13 complete in %.1fs.", elapsed)
    logger.info(
        "MLflow summary: %d metrics, %d params, %d+%d+%d artifacts logged.",
        len(all_metrics), len(all_params), n_figures, n_reports, n_models
    )
    logger.info("To view: run  mlflow ui  from project root → http://127.0.0.1:5000")
    return run_id


# ===========================================================================
# Stage 14 — Pipeline & Model Persistence
# ===========================================================================

def run_stage14(
    models: Dict[str, Any],
    test: pd.DataFrame,
    dev: pd.DataFrame,
    feature_cols: List[str],
    optimal_threshold: float,
) -> PredictGuardPipeline:
    """Build, validate, and persist the end-to-end inference pipeline."""
    _banner("STAGE 14: Reliable Pipeline & Model Persistence")
    t_stage = time.time()

    report_writer = PipelineReportWriter(reports_dir=REPORTS_DIR)

    # -----------------------------------------------------------------------
    # PART 7 — Build Pipeline
    # -----------------------------------------------------------------------
    _section("PART 7: Building PredictGuardPipeline")
    pipeline = PredictGuardPipeline(
        calibrated_model=models["calibrated_model"],
        raw_model=models["raw_model"],
        component_predictor=models["component_predictor"],
        feature_cols=feature_cols,
        optimal_threshold=optimal_threshold,
        cost_matrix={"C_FN": 10_000.0, "C_FP": 500.0},
    )
    logger.info("Pipeline built: %s", repr(pipeline))

    # -----------------------------------------------------------------------
    # PART 10 — Smoke-test predict_report()
    # -----------------------------------------------------------------------
    _section("PART 10: Prediction Interface Smoke Test")
    sample = test.sample(5, random_state=42)
    reports = pipeline.predict_report(
        sample,
        machine_ids=sample["machineID"].tolist(),
        timestamps=sample["datetime"].tolist(),
        include_shap_top_n=5,
    )

    logger.info("Sample prediction reports:")
    for rep in reports:
        logger.info(
            "  Machine %-4s | Risk: %3d (%s) | Comp: %-6s | Prob: %.3f | Decision: %s",
            rep["machine_id"],
            rep["risk_score"],
            rep["risk_tier"],
            rep["predicted_component"],
            rep["failure_probability"],
            rep["dispatch_decision"],
        )
        if rep.get("top_shap_features"):
            top3 = [f["feature"] for f in rep["top_shap_features"][:3]]
            logger.info("    Top SHAP: %s", ", ".join(top3))

    # Save sample report JSON
    sample_json_path = REPORTS_DIR / "prediction_cards" / "pipeline_sample_reports.json"
    sample_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=str)
    logger.info("Saved sample pipeline reports: %s", sample_json_path)

    # -----------------------------------------------------------------------
    # PART 9 — Save Pipeline
    # -----------------------------------------------------------------------
    _section("PART 9: Saving Pipeline to Disk")
    pipeline_path, metadata_path = pipeline.save(MODELS_DIR)
    logger.info("Pipeline saved: %s", pipeline_path)
    logger.info("Metadata saved: %s", metadata_path)

    # -----------------------------------------------------------------------
    # PART 11 — Validate Pipeline
    # -----------------------------------------------------------------------
    _section("PART 11: Pipeline Reproducibility Validation")

    # Load from disk to verify serialisation round-trip
    loaded_pipeline = PredictGuardPipeline.load(MODELS_DIR)
    logger.info("Pipeline round-trip load successful: %s", repr(loaded_pipeline))

    validator = PipelineValidator(
        pipeline=loaded_pipeline,
        calibrated_model=models["calibrated_model"],
        component_predictor=models["component_predictor"],
        tolerance=1e-5,
    )
    val_results = validator.validate(test, feature_cols=feature_cols, n_rows=5000)
    validator.write_report(val_results, reports_dir=REPORTS_DIR)

    if not val_results["passed"]:
        logger.error("Pipeline validation FAILED -- check reports/pipeline_validation.md")
    else:
        logger.info(
            "Pipeline validation PASSED [OK] -- max_prob_diff=%.2e, "
            "binary=%.4f, comp=%.4f, tier=%.4f",
            val_results["max_prob_diff"],
            val_results["match_rates"]["binary_predictions"],
            val_results["match_rates"]["component_predictions"],
            val_results["match_rates"]["risk_tiers"],
        )

    # -----------------------------------------------------------------------
    # PART 13 — Pipeline Report
    # -----------------------------------------------------------------------
    _section("PART 13: Writing Pipeline Documentation")
    report_writer.write(pipeline)

    elapsed = time.time() - t_stage
    logger.info("Stage 14 complete in %.1fs.", elapsed)
    return loaded_pipeline


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    pipeline_start = time.time()
    _banner("PredictGuard — Phase 4: Stage 13 (MLflow) + Stage 14 (Pipeline)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data ...")
    dev  = pd.read_parquet(DEV_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    logger.info("dev=%s  test=%s", dev.shape, test.shape)

    feature_cols = _get_feature_cols(test)
    logger.info("Feature columns: %d", len(feature_cols))

    # Load models
    models = _load_models()
    optimal_threshold = _load_optimal_threshold()
    logger.info("Optimal threshold: %.2f", optimal_threshold)

    # Stage 13
    run_id = run_stage13(models, feature_cols)

    # Stage 14
    pipeline = run_stage14(
        models=models,
        test=test,
        dev=dev,
        feature_cols=feature_cols,
        optimal_threshold=optimal_threshold,
    )

    total = time.time() - pipeline_start
    _banner("Phase 4 (Stage 13 & 14) COMPLETE")
    logger.info("  Total runtime               : %.1fs (%.1f min)", total, total / 60)
    logger.info("  MLflow run ID               : %s", run_id[:12])
    logger.info("  Pipeline                    : models/predictguard_pipeline.pkl")
    logger.info("  Pipeline metadata           : models/predictguard_pipeline_metadata.json")
    logger.info("  MLflow summary              : reports/mlflow_summary.md")
    logger.info("  Pipeline validation         : reports/pipeline_validation.md")
    logger.info("  Reproducibility report      : reports/reproducibility_report.md")
    logger.info("  Pipeline documentation      : reports/pipeline_report.md")
    logger.info("  Sample report cards         : reports/prediction_cards/pipeline_sample_reports.json")
    logger.info("  View MLflow UI              : mlflow ui --backend-store-uri mlruns/mlflow.db  -> http://127.0.0.1:5000")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
