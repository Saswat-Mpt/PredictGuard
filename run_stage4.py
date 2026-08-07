"""
run_stage4.py
=============
Headless pipeline runner for PredictGuard — Stage 4: Leakage-Safe Data Splitting.

Usage:
    python run_stage4.py

Splits telemetry features into Development (80 machines) and Final Test (20 machines)
datasets at the machine-level, generates 5-fold Grouped Cross Validation splits,
runs exhaustive leakage checks, exports Parquet datasets, and produces reports.
"""

import json
import logging
import sys
import yaml
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import split as sp


def main() -> None:
    """Execute Stage 4 Data Splitting pipeline."""
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

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        datefmt=log_datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(reports_dir / "stage4_run.log", mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_stage4")

    logger.info("=" * 60)
    logger.info(" PredictGuard -- Stage 4: Leakage-Safe Data Splitting")
    logger.info("=" * 60)

    # Load feature dataset
    features_parquet = processed_dir / "features.parquet"
    if not features_parquet.exists():
        logger.error("Feature dataset not found at %s. Run Stage 3 first!", features_parquet)
        sys.exit(1)

    df_features = pd.read_parquet(features_parquet)
    logger.info("Loaded features dataset with %d rows and %d columns.", len(df_features), df_features.shape[1])

    # Parameters
    split_config = config.get("split", {})
    test_ratio = split_config.get("test_ratio", 0.20)
    n_folds = split_config.get("n_folds", 5)
    seed = split_config.get("random_seed", 42)
    stratify = split_config.get("stratify_by_failure", True)

    # 1. Machine-Level Development / Test Split
    splitter = sp.MachineSplitter(test_ratio=test_ratio, random_seed=seed, stratify_by_failure=stratify)
    dev_df, test_df, dev_ids, test_ids = splitter.split(df_features)

    # 2. Save Machine ID CSV lists
    pd.DataFrame({"machineID": dev_ids}).to_csv(processed_dir / "development_machine_ids.csv", index=False)
    pd.DataFrame({"machineID": test_ids}).to_csv(processed_dir / "test_machine_ids.csv", index=False)
    logger.info("Saved development_machine_ids.csv (%d IDs) and test_machine_ids.csv (%d IDs)", len(dev_ids), len(test_ids))

    # 3. Grouped Cross-Validation (5 Folds on Development)
    cv = sp.GroupedCrossValidator(n_folds=n_folds, random_seed=seed)
    fold_assignments, fold_summary = cv.generate_folds(dev_df)

    # Save CV folds JSON
    cv_folds_path = processed_dir / "cv_folds.json"
    with open(cv_folds_path, "w", encoding="utf-8") as f:
        json.dump(fold_assignments, f, indent=2)
    logger.info("Saved Cross-Validation Folds to %s", cv_folds_path)

    # 4. Leakage Validation Audit
    sp.LeakageValidator.validate_split(dev_df, test_df, fold_assignments)

    # 5. Compute Statistics & Save Manifest
    stats = sp.compute_split_statistics(dev_df, test_df)
    manifest_path = processed_dir / "split_manifest.json"
    sp.save_split_manifest(split_config, dev_ids, test_ids, manifest_path)

    # 6. Persist Parquet Datasets
    dev_parquet = processed_dir / "development.parquet"
    test_parquet = processed_dir / "test.parquet"
    
    dev_df.to_parquet(dev_parquet, index=False, engine="pyarrow")
    logger.info("Saved Development Parquet: %s (%.1f MB)", dev_parquet, dev_parquet.stat().st_size / 1_048_576)

    test_df.to_parquet(test_parquet, index=False, engine="pyarrow")
    logger.info("Saved Final Test Parquet  : %s (%.1f MB)", test_parquet, test_parquet.stat().st_size / 1_048_576)

    # 7. Generate Markdown Report
    report_path = reports_dir / "split_strategy_report.md"
    sp.generate_split_report(stats, fold_summary, report_path)

    # 8. Visualisations
    logger.info("Generating publication-quality split visualisations ...")
    sp.plot_split_summary_report(dev_df, test_df, fold_assignments, figures_dir)

    logger.info("=" * 60)
    logger.info(" Stage 4 COMPLETE")
    logger.info(" Development Parquet -> %s", dev_parquet)
    logger.info(" Test Parquet        -> %s", test_parquet)
    logger.info(" CV Folds JSON       -> %s", cv_folds_path)
    logger.info(" Split Manifest      -> %s", manifest_path)
    logger.info(" Report              -> %s", report_path)
    logger.info(" Figures             -> %s", figures_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
