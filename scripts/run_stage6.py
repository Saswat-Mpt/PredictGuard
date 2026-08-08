"""
run_stage6.py
=============
Headless pipeline runner for PredictGuard — Stage 6:
Grouped Cross-Validation & Light Hyperparameter Tuning.

Usage:
    python run_stage6.py

Execution order:
  1.  Load development.parquet and test.parquet.
  2.  Run FoldAnalyzer  → fold statistics (leakage audit).
  3.  Run CrossValidator on all 3 baseline models → per-fold metrics.
  4.  Run MetricAggregator → mean ± std ± 95% CI summary.
  5.  Run HyperparameterTuner on XGBoost (15 iterations).
  6.  Evaluate Default XGBoost vs Tuned XGBoost on Final Test set.
  7.  Save models/best_model.pkl + metadata.
  8.  Save all CSV / JSON reports.
  9.  Generate 6 publication-quality figures.
  10. Generate reports/cross_validation_report.md.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_validation import (
    CrossValidator,
    FoldAnalyzer,
    HyperparameterTuner,
    MetricAggregator,
    _build_xgb,
    _prepare_xy,
    generate_cv_report,
    plot_before_vs_after_tuning,
    plot_cv_metric_distribution,
    plot_foldwise_metrics,
    plot_hyperparam_importance,
    plot_training_time_comparison,
)
from src.train import ModelEvaluator

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
MetricsDict = Dict[str, Any]


def main() -> None:
    """Execute Stage 6 Grouped Cross-Validation & Hyperparameter Tuning."""

    # ------------------------------------------------------------------
    # 0. Configuration
    # ------------------------------------------------------------------
    config_path = PROJECT_ROOT / "config.yaml"
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    cv_cfg  = config.get("cross_validation", {})
    log_cfg = config.get("logging", {})

    N_FOLDS     = int(cv_cfg.get("n_folds", 5))
    N_ITER      = int(cv_cfg.get("n_iter_search", 15))
    RANDOM_SEED = int(cv_cfg.get("random_seed", 42))
    SCORING     = str(cv_cfg.get("scoring", "average_precision"))

    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_dir", "data/processed")
    reports_dir   = PROJECT_ROOT / config.get("reports", {}).get("output_dir", "reports")
    figures_dir   = PROJECT_ROOT / config.get("reports", {}).get("figures_dir", "reports/figures")
    models_dir    = PROJECT_ROOT / "models"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        datefmt=log_cfg.get("datefmt", "%H:%M:%S"),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(reports_dir / "stage6_run.log", mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_stage6")

    logger.info("=" * 65)
    logger.info(" PredictGuard -- Stage 6: CV & Hyperparameter Tuning")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    dev_path  = processed_dir / "development.parquet"
    test_path = processed_dir / "test.parquet"

    if not dev_path.exists() or not test_path.exists():
        logger.error("development.parquet or test.parquet not found. Run Stage 4 first.")
        sys.exit(1)

    logger.info("Loading development and test datasets ...")
    dev_df  = pd.read_parquet(dev_path)
    test_df = pd.read_parquet(test_path)

    logger.info("Development: %d machines, %d rows", dev_df["machineID"].nunique(), len(dev_df))
    logger.info("Final Test : %d machines, %d rows", test_df["machineID"].nunique(), len(test_df))

    # Prepare feature matrices
    X_dev, y_dev, groups_dev, feature_names = _prepare_xy(dev_df)
    X_test, y_test, _, _                    = _prepare_xy(test_df)

    logger.info("Feature matrix: %d features", len(feature_names))

    # ------------------------------------------------------------------
    # 2. Fold Analysis — Leakage Audit
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 1: Fold Analysis & Leakage Audit")
    logger.info("-" * 50)

    analyzer = FoldAnalyzer(n_folds=N_FOLDS, random_seed=RANDOM_SEED)
    fold_stats_df, raw_folds = analyzer.analyze(dev_df)

    logger.info("\n%s", fold_stats_df.to_string(index=False))

    if fold_stats_df["leakage"].any():
        logger.error("CRITICAL: Machine leakage detected! Aborting pipeline.")
        sys.exit(2)
    logger.info("Leakage audit PASSED. Zero machine overlap across all %d folds.", N_FOLDS)

    # ------------------------------------------------------------------
    # 3. Cross-Validation on All 3 Baseline Models
    #    Checkpoint: if cv_results.csv already exists from a prior run,
    #    load it directly and skip the ~20-minute CV refit.
    # ------------------------------------------------------------------
    cv_checkpoint = reports_dir / "cv_results.csv"

    if cv_checkpoint.exists():
        logger.info("CV checkpoint found at %s — loading cached results.", cv_checkpoint)
        cv_results_df = pd.read_csv(cv_checkpoint)
        logger.info("Loaded %d cached fold-model records.", len(cv_results_df))
    else:
        logger.info("-" * 50)
        logger.info("PART 2 & 3: 5-Fold Grouped Cross-Validation (3 Models)")
        logger.info("-" * 50)

        cv = CrossValidator(n_folds=N_FOLDS, random_seed=RANDOM_SEED)
        cv_results_df = cv.run(dev_df)

        # Save checkpoint immediately so a crash in later steps doesn't lose CV work
        cv_results_df.to_csv(cv_checkpoint, index=False)
        logger.info("CV checkpoint saved to %s", cv_checkpoint)

    # ------------------------------------------------------------------
    # 4. Aggregate Metrics
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 3: Metric Aggregation (Mean +/- Std +/- 95 CI)")
    logger.info("-" * 50)

    agg          = MetricAggregator()
    summary_df   = agg.aggregate(cv_results_df)
    formatted_df = agg.format_table(summary_df)

    logger.info("\n%s\n", formatted_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Hyperparameter Tuning — XGBoost
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 4 & 5: RandomizedSearchCV on XGBoost (%d iterations)", N_ITER)
    logger.info("-" * 50)

    tuner = HyperparameterTuner(
        n_folds=N_FOLDS,
        n_iter=N_ITER,
        scoring=SCORING,
        random_seed=RANDOM_SEED,
    )

    t_tune_start = time.time()
    best_estimator, best_params, search_results_df = tuner.tune(X_dev, y_dev, groups_dev)
    tune_elapsed = time.time() - t_tune_start

    best_cv_score = search_results_df["mean_test_score"].iloc[0]
    logger.info("Tuning complete in %.1fs | Best CV PR-AUC: %.4f", tune_elapsed, best_cv_score)

    # ------------------------------------------------------------------
    # 6. Evaluate Default vs Tuned XGBoost on Final Test Set
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 5: Default vs Tuned XGBoost — Final Test Set Evaluation")
    logger.info("-" * 50)

    evaluator = ModelEvaluator()

    # Default XGBoost
    logger.info("Training DEFAULT XGBoost on full Development set ...")
    default_xgb = _build_xgb(y_dev.values, seed=RANDOM_SEED)
    t0 = time.time()
    default_xgb.fit(X_dev, y_dev)
    default_fit_time = time.time() - t0
    default_eval = evaluator.evaluate("XGBoost (Default)", default_xgb, X_test, y_test,
                                      fit_time_sec=default_fit_time)

    # Tuned XGBoost (already fitted by RandomizedSearchCV with refit=True)
    logger.info("Evaluating TUNED XGBoost on Final Test set ...")
    t0 = time.time()
    tuned_xgb = best_estimator
    tuned_fit_time = tune_elapsed   # total search time (approx)
    tuned_eval = evaluator.evaluate("XGBoost (Tuned)", tuned_xgb, X_test, y_test,
                                    fit_time_sec=tuned_fit_time)

    logger.info("")
    logger.info("%-30s  PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  Recall=%.4f",
                "Default XGBoost",
                default_eval["pr_auc"], default_eval["roc_auc"],
                default_eval["f1_score"], default_eval["recall"])
    logger.info("%-30s  PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  Recall=%.4f",
                "Tuned XGBoost",
                tuned_eval["pr_auc"], tuned_eval["roc_auc"],
                tuned_eval["f1_score"], tuned_eval["recall"])

    pr_delta = tuned_eval["pr_auc"] - default_eval["pr_auc"]
    logger.info("PR-AUC delta (Tuned - Default): %+.4f", pr_delta)

    # ------------------------------------------------------------------
    # 7. Save Best Model
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 6: Saving Best Model Artifacts")
    logger.info("-" * 50)

    best_pkl = models_dir / "best_model.pkl"
    joblib.dump(tuned_xgb, best_pkl)
    logger.info("Saved best_model.pkl (%.2f MB)", best_pkl.stat().st_size / 1_048_576)

    best_meta: Dict[str, Any] = {
        "model_name": "XGBoost (Tuned)",
        "stage": "Phase 1 Stage 6",
        "version": "2.0.0",
        "training_date": datetime.now().isoformat(),
        "feature_count": len(feature_names),
        "training_rows": len(X_dev),
        "positive_class_pct": round(100.0 * float(y_dev.mean()), 3),
        "cv_pr_auc_mean": round(
            float(summary_df.loc[summary_df["model"] == "XGBoost", "pr_auc_mean"].iloc[0])
            if "XGBoost" in summary_df["model"].values else 0.0, 4
        ),
        "test_pr_auc": tuned_eval["pr_auc"],
        "test_roc_auc": tuned_eval["roc_auc"],
        "test_f1_score": tuned_eval["f1_score"],
        "test_recall": tuned_eval["recall"],
        "test_precision": tuned_eval["precision"],
        "best_params": best_params,
        "best_cv_score": round(float(best_cv_score), 4),
        "n_search_iterations": N_ITER,
    }

    meta_path = models_dir / "best_model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2)
    logger.info("Saved best_model_metadata.json")

    # ------------------------------------------------------------------
    # 8. Save Reports (CSV / JSON)
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 7: Saving Report Files")
    logger.info("-" * 50)

    # cv_results.csv
    cv_path = reports_dir / "cv_results.csv"
    cv_results_df.to_csv(cv_path, index=False)
    logger.info("Saved cv_results.csv (%d rows)", len(cv_results_df))

    # random_search_results.csv
    rs_path = reports_dir / "random_search_results.csv"
    search_results_df.to_csv(rs_path, index=False)
    logger.info("Saved random_search_results.csv (%d rows)", len(search_results_df))

    # best_parameters.json
    bp_path = reports_dir / "best_parameters.json"
    with open(bp_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Saved best_parameters.json")

    # cv_summary.json
    cv_xgb_row = summary_df[summary_df["model"] == "XGBoost"]
    cv_summary_json: Dict[str, Any] = {
        "cv_strategy": "StratifiedGroupKFold",
        "folds": N_FOLDS,
        "primary_metric": "PR-AUC",
        "best_model": "XGBoost",
        "best_parameters": best_params,
        "mean_cv_pr_auc": round(float(cv_xgb_row["pr_auc_mean"].iloc[0]), 4) if len(cv_xgb_row) else 0.0,
        "std_cv_pr_auc": round(float(cv_xgb_row["pr_auc_std"].iloc[0]), 4) if len(cv_xgb_row) else 0.0,
        "best_cv_score_randomsearch": round(float(best_cv_score), 4),
        "final_test_pr_auc_default": default_eval["pr_auc"],
        "final_test_pr_auc_tuned": tuned_eval["pr_auc"],
        "generated_at": datetime.now().isoformat(),
    }

    cs_path = reports_dir / "cv_summary.json"
    with open(cs_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary_json, f, indent=2)
    logger.info("Saved cv_summary.json")

    # ------------------------------------------------------------------
    # 9. Visualizations
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 6: Generating Publication-Quality Figures")
    logger.info("-" * 50)

    plot_cv_metric_distribution(cv_results_df, figures_dir)
    plot_foldwise_metrics(cv_results_df, figures_dir)
    plot_before_vs_after_tuning(
        default_metrics={
            "pr_auc":    default_eval["pr_auc"],
            "roc_auc":   default_eval["roc_auc"],
            "f1_score":  default_eval["f1_score"],
            "recall":    default_eval["recall"],
            "precision": default_eval["precision"],
        },
        tuned_metrics={
            "pr_auc":    tuned_eval["pr_auc"],
            "roc_auc":   tuned_eval["roc_auc"],
            "f1_score":  tuned_eval["f1_score"],
            "recall":    tuned_eval["recall"],
            "precision": tuned_eval["precision"],
        },
        figures_dir=figures_dir,
    )
    plot_hyperparam_importance(search_results_df, best_params, figures_dir)
    plot_training_time_comparison(cv_results_df, figures_dir)

    # ------------------------------------------------------------------
    # 10. Markdown Report
    # ------------------------------------------------------------------
    logger.info("-" * 50)
    logger.info("PART 8: Generating cross_validation_report.md")
    logger.info("-" * 50)

    generate_cv_report(
        fold_stats_df=fold_stats_df,
        cv_summary_df=summary_df,
        formatted_table=formatted_df,
        best_params=best_params,
        best_cv_score=float(best_cv_score),
        default_test_metrics={
            "pr_auc":    default_eval["pr_auc"],
            "roc_auc":   default_eval["roc_auc"],
            "f1_score":  default_eval["f1_score"],
            "recall":    default_eval["recall"],
            "precision": default_eval["precision"],
        },
        tuned_test_metrics={
            "pr_auc":    tuned_eval["pr_auc"],
            "roc_auc":   tuned_eval["roc_auc"],
            "f1_score":  tuned_eval["f1_score"],
            "recall":    tuned_eval["recall"],
            "precision": tuned_eval["precision"],
        },
        output_path=reports_dir / "cross_validation_report.md",
    )

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info(" Stage 6 COMPLETE")
    logger.info("=" * 65)
    logger.info(" Best Model (Tuned XGBoost)")
    logger.info("   PR-AUC   : %.4f  (default: %.4f, delta: %+.4f)",
                tuned_eval["pr_auc"], default_eval["pr_auc"], pr_delta)
    logger.info("   ROC-AUC  : %.4f", tuned_eval["roc_auc"])
    logger.info("   F1 Score : %.4f", tuned_eval["f1_score"])
    logger.info("   Recall   : %.4f", tuned_eval["recall"])
    logger.info("   Precision: %.4f", tuned_eval["precision"])
    logger.info("-" * 65)
    logger.info(" Artifacts saved to:")
    logger.info("   models/best_model.pkl")
    logger.info("   models/best_model_metadata.json")
    logger.info("   reports/cv_results.csv")
    logger.info("   reports/random_search_results.csv")
    logger.info("   reports/best_parameters.json")
    logger.info("   reports/cv_summary.json")
    logger.info("   reports/cross_validation_report.md")
    logger.info("   reports/figures/  (6 new figures)")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
