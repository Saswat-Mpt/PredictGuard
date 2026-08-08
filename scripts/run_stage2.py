"""
run_stage2.py
=============
Headless pipeline runner for PredictGuard — Stage 2: Target Construction.

Usage:
    python run_stage2.py

This script runs the complete Stage 2 Target Construction pipeline without requiring Jupyter.
Useful for CI/CD, automated testing, and reproducible execution.
"""

import logging
import sys
import yaml
from pathlib import Path

# Make src/ importable from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import target_creation as tc


def main() -> None:
    """Execute Stage 2 Target Construction pipeline."""
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
    raw_dir = PROJECT_ROOT / config.get("data", {}).get("raw_dir", "data/raw")
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
            logging.FileHandler(reports_dir / "stage2_run.log", mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("run_stage2")

    logger.info("=" * 60)
    logger.info(" PredictGuard -- Stage 2: Target Construction")
    logger.info("=" * 60)

    horizon_hours = config.get("target", {}).get("prediction_horizon_hours", 24)

    # 1. Load data
    datasets = tc.load_stage1_data(raw_dir)
    telemetry = datasets["telemetry"]
    failures = datasets["failures"]

    # 2. Validate inputs
    issues = tc.validate_inputs(telemetry, failures)
    if issues:
        logger.error("Input validation failed with %d issues.", len(issues))
        sys.exit(1)

    # 3. Binary target
    df = tc.compute_binary_target(telemetry, failures, horizon_hours=horizon_hours)

    # 4. Component target
    df = tc.assign_failure_component(df, failures, horizon_hours=horizon_hours)

    # 5. Validation
    spot_check_n = config.get("validation", {}).get("spot_check_n_machines", 5)
    seed = config.get("validation", {}).get("random_seed", 42)
    val_results = tc.validate_targets(
        df, failures, horizon_hours=horizon_hours, spot_check_n=spot_check_n, random_seed=seed
    )

    if not all([
        val_results["positive_check_pass"],
        val_results["negative_check_pass"],
        val_results["component_check_pass"],
        val_results["nan_consistency_pass"],
    ]):
        logger.error("Target validation failed! Check reports for details.")
        sys.exit(1)

    # 6. Statistics
    stats = tc.compute_target_statistics(df, failures)

    # 7. Persistence
    tc.save_targets(df, processed_dir)

    # 8. Report
    report_path = reports_dir / "target_construction_report.md"
    tc.generate_target_report(
        df=df,
        stats=stats,
        validation=val_results,
        output_path=report_path,
        horizon_hours=horizon_hours,
    )

    # 9. Visualisations
    logger.info("Generating publication-quality visualisations ...")
    tc.plot_class_distribution(df, figures_dir)
    tc.plot_failure_timeline(failures, figures_dir)
    tc.plot_component_frequency(df, figures_dir)
    tc.plot_positives_per_machine(df, figures_dir)
    tc.plot_time_to_failure_histogram(df, figures_dir, horizon_hours=horizon_hours)
    tc.plot_positive_windows_sample(df, failures, figures_dir, horizon_hours=horizon_hours)
    tc.plot_failure_heatmap(failures, df, figures_dir)

    logger.info("=" * 60)
    logger.info(" Stage 2 COMPLETE")
    logger.info(" Target Parquet -> %s", processed_dir / "telemetry_with_targets.parquet")
    logger.info(" Target CSV     -> %s", processed_dir / "telemetry_with_targets.csv")
    logger.info(" Report         -> %s", report_path)
    logger.info(" Figures        -> %s", figures_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
