"""
run_stage1.py
=============
Headless pipeline runner for PredictGuard — Stage 1.

Usage:
    python run_stage1.py

This script runs the full Stage 1 Data Understanding & Validation pipeline
without requiring Jupyter. Useful for CI/CD and automated reproduction.
"""

import logging
import sys
from pathlib import Path

# Make src/ importable from the project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import data_validation as dv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "reports" / "stage1_run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_stage1")


def main() -> None:
    """Execute the full Stage 1 validation pipeline."""
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(" PredictGuard — Stage 1: Data Understanding & Validation")
    logger.info("=" * 60)

    # 1. Load
    datasets = dv.load_all_datasets(DATA_DIR)

    # 2. Schema
    dv.validate_schemas(datasets)

    # 3. Missing & duplicates
    missing_df = dv.check_missing_values(datasets)
    duplicates_df = dv.check_duplicate_rows(datasets)

    # 4. Consistency
    consistency = dv.verify_machine_consistency(datasets)

    # 5. Summary
    telemetry_summary = dv.compute_telemetry_summary(
        datasets["telemetry"], datasets["machines"]
    )

    # 6. Gaps
    gaps_df = dv.detect_telemetry_gaps(datasets["telemetry"], gap_threshold_hours=3.0)

    # 7. Frozen sensors
    frozen_df = dv.detect_frozen_sensors(datasets["telemetry"], window=dv.DEFAULT_FROZEN_WINDOW)

    # 8. Range validation
    telemetry_flagged = dv.validate_sensor_ranges(datasets["telemetry"], dv.SENSOR_BOUNDS)

    # 9. Cross-table integrity
    integrity = dv.verify_cross_table_integrity(datasets)

    # 10. Report
    report_df = dv.build_quality_report(
        datasets=datasets,
        missing_df=missing_df,
        duplicates_df=duplicates_df,
        consistency=consistency,
        gaps_df=gaps_df,
        frozen_df=frozen_df,
        telemetry_flagged=telemetry_flagged,
        integrity=integrity,
        telemetry_summary=telemetry_summary,
    )
    dv.save_report_csv(report_df, REPORTS_DIR / "data_quality_report.csv")
    dv.save_report_md(
        report_df=report_df,
        datasets=datasets,
        missing_df=missing_df,
        duplicates_df=duplicates_df,
        gaps_df=gaps_df,
        frozen_df=frozen_df,
        telemetry_flagged=telemetry_flagged,
        integrity=integrity,
        telemetry_summary=telemetry_summary,
        consistency=consistency,
        output_path=REPORTS_DIR / "data_quality_report.md",
    )

    # 11. Visualisations
    logger.info("Generating visualisations …")
    dv.plot_missing_heatmap(datasets, FIGURES_DIR)
    dv.plot_sensor_histograms(datasets["telemetry"], FIGURES_DIR)
    dv.plot_boxplots(datasets["telemetry"], FIGURES_DIR)
    dv.plot_machine_timeline(datasets["telemetry"], datasets["failures"], FIGURES_DIR)
    dv.plot_records_per_machine(datasets, FIGURES_DIR)
    dv.plot_gap_distribution(gaps_df, FIGURES_DIR)
    dv.plot_sensor_correlation(datasets["telemetry"], FIGURES_DIR)
    dv.plot_failure_counts(datasets["failures"], FIGURES_DIR)
    dv.plot_maintenance_frequency(datasets["maint"], FIGURES_DIR)
    dv.plot_error_frequency(datasets["errors"], FIGURES_DIR)

    logger.info("=" * 60)
    logger.info(" Stage 1 COMPLETE")
    logger.info(" Reports  -> %s", REPORTS_DIR)
    logger.info(" Figures  -> %s", FIGURES_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
