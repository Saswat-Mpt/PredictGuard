"""
run_stage5.py
=============
Headless pipeline runner for PredictGuard — Stage 5: Baseline & Stronger Models.

Usage:
    python run_stage5.py

Trains Logistic Regression, Random Forest, and XGBoost baseline models on the Development
dataset (80 machines), evaluates them on the held-out Final Test dataset (20 machines),
saves trained model pickles, writes the Model Registry metadata, and exports figures.
"""

import json
import logging
import sys
import time
import yaml
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import train as tr


def main() -> None:
    """Execute Stage 5 Baseline Training & Evaluation pipeline."""
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    log_level = config.get("logging", {}).get("level", "INFO")
    log_format = config.get("logging", {}).get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log_datefmt = config.get("logging", {}).get("datefmt", "%H:%M:%S")

    reports_dir = PROJECT_ROOT / config.get("reports", {}).get("output_dir", "reports")
    figures_dir = PROJECT_ROOT / config.get("reports", {}).get("figures_dir", "reports/figures")
    processed_dir = PROJECT_ROOT / config.get("data", {}).get("processed_dir", "data/processed")
    models_dir = PROJECT_ROOT / "models"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt=log_datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(reports_dir / "stage5_run.log", mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_stage5")

    logger.info("=" * 60)
    logger.info(" PredictGuard -- Stage 5: Baseline and Stronger Models")
    logger.info("=" * 60)

    # 1. Load Development and Test Datasets
    dev_parquet = processed_dir / "development.parquet"
    test_parquet = processed_dir / "test.parquet"

    if not dev_parquet.exists() or not test_parquet.exists():
        logger.error("Splitted datasets not found at %s. Run Stage 4 first!", processed_dir)
        sys.exit(1)

    dev_df = pd.read_parquet(dev_parquet)
    test_df = pd.read_parquet(test_parquet)

    logger.info(
        "Loaded Development set (%d machines, %d rows) and Final Test set (%d machines, %d rows).",
        dev_df["machineID"].nunique(),
        len(dev_df),
        test_df["machineID"].nunique(),
        len(test_df),
    )

    # 2. Extract Feature Matrices and Target Vectors
    trainer = tr.ModelTrainer(random_seed=config.get("validation", {}).get("random_seed", 42))
    X_train, y_train, feature_names = trainer.prepare_data(dev_df)
    X_test, y_test, _ = trainer.prepare_data(test_df)

    logger.info("Training feature matrix: %d rows x %d features", X_train.shape[0], X_train.shape[1])
    logger.info("Test feature matrix    : %d rows x %d features", X_test.shape[0], X_test.shape[1])

    evaluator = tr.ModelEvaluator()
    saver = tr.ModelSaver()
    eval_results: Dict[str, Dict[str, Any]] = {}
    combined_registry: Dict[str, Any] = {}

    # 3. Model 1: Logistic Regression Baseline
    logger.info("--- Model 1: Logistic Regression Baseline ---")
    lr_model = trainer.build_logistic_regression()
    start_fit = time.time()
    lr_model.fit(X_train, y_train)
    fit_time_lr = time.time() - start_fit

    lr_eval = evaluator.evaluate("Logistic Regression", lr_model, X_test, y_test, fit_time_sec=fit_time_lr)
    eval_results["Logistic Regression"] = lr_eval
    saver.save(lr_model, "Logistic Regression", lr_eval, X_train, y_train, models_dir)

    # 4. Model 2: Random Forest Classifier
    logger.info("--- Model 2: Random Forest Classifier ---")
    rf_model = trainer.build_random_forest(n_estimators=100)
    start_fit = time.time()
    rf_model.fit(X_train, y_train)
    fit_time_rf = time.time() - start_fit

    rf_eval = evaluator.evaluate("Random Forest", rf_model, X_test, y_test, fit_time_sec=fit_time_rf)
    eval_results["Random Forest"] = rf_eval
    saver.save(rf_model, "Random Forest", rf_eval, X_train, y_train, models_dir)

    # 5. Model 3: XGBoost Classifier
    logger.info("--- Model 3: XGBoost Classifier ---")
    xgb_model = trainer.build_xgboost(y_train, n_estimators=100)
    start_fit = time.time()
    xgb_model.fit(X_train, y_train)
    fit_time_xgb = time.time() - start_fit

    xgb_eval = evaluator.evaluate("XGBoost", xgb_model, X_test, y_test, fit_time_sec=fit_time_xgb)
    eval_results["XGBoost"] = xgb_eval
    saver.save(xgb_model, "XGBoost", xgb_eval, X_train, y_train, models_dir)

    # 6. Combined Model Registry JSON
    for m_name, res in eval_results.items():
        combined_registry[m_name] = {
            "pr_auc": res["pr_auc"],
            "roc_auc": res["roc_auc"],
            "f1_score": res["f1_score"],
            "precision": res["precision"],
            "recall": res["recall"],
            "balanced_accuracy": res["balanced_accuracy"],
            "mcc": res["mcc"],
            "log_loss": res["log_loss"],
            "confusion_matrix": res["confusion_matrix"],
            "fit_time_sec": res["fit_time_sec"],
        }

    registry_path = models_dir / "training_metadata.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(combined_registry, f, indent=2)
    logger.info("Saved combined Model Registry metadata to %s", registry_path)

    # 7. Generate Markdown Report
    stats = {
        "test_machines": test_df["machineID"].nunique(),
        "test_rows": len(test_df),
    }
    report_path = reports_dir / "model_training_report.md"
    tr.generate_training_report(eval_results, stats, report_path)

    # 8. Visualisations — Unified Evaluation Dashboard
    logger.info("Generating publication-quality evaluation dashboard ...")
    tr.plot_model_evaluation_dashboard(eval_results, y_test, figures_dir)

    logger.info("=" * 60)
    logger.info(" Stage 5 COMPLETE")
    logger.info(" Models Saved       -> %s", models_dir)
    logger.info(" Registry Metadata  -> %s", registry_path)
    logger.info(" Report             -> %s", report_path)
    logger.info(" Figures            -> %s", figures_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
