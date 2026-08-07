"""
run_stage11_12.py
=================
PredictGuard — Phase 3 Orchestrator: Stage 11 (Fleet Risk Segmentation) + Stage 12 (Cost Decision)

Usage:
    python run_stage11_12.py

Outputs:
    reports/fleet_risk_summary.csv
    reports/cost_analysis.csv
    reports/optimal_threshold.json
    reports/dispatch_decisions.csv
    reports/fleet_segmentation_report.md
    reports/cost_decision_report.md
    reports/figures/fleet_*.png
    reports/figures/cost_*.png   / threshold_*.png / dispatch_*.png / decision_*.png
    reports/prediction_cards/fleet_status.json

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
logger = logging.getLogger("run_stage11_12")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_segmentation import (
    RiskSegmenter,
    FleetAnalyzer,
    FleetReportWriter,
    DEFAULT_TIER_THRESHOLDS,
    TIER_ORDER,
)
from src.cost_decision import (
    CostEvaluator,
    ThresholdOptimizer,
    DecisionEngine,
    CostVisualizer,
    CostReportWriter,
    DEFAULT_COST_MATRIX,
)
from src.component_classifier import ComponentPredictor

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DATA_DIR    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CARDS_DIR   = REPORTS_DIR / "prediction_cards"

DEV_PARQUET  = DATA_DIR / "development.parquet"
TEST_PARQUET = DATA_DIR / "test.parquet"
CAL_MODEL    = MODELS_DIR / "calibrated_model.pkl"
COMP_MODEL   = MODELS_DIR / "component_classifier.pkl"

NON_FEATURE_COLS = {
    "datetime", "machineID", "y_failure",
    "failure_component", "time_to_failure_hours",
    "calibrated_prob", "risk_score", "risk_tier",
    "predicted_component", "component_confidence",
}

# ---------------------------------------------------------------------------
# Cost config — change here or load from config.yaml
# ---------------------------------------------------------------------------
COST_MATRIX: Dict[str, float] = {
    "C_FN": 10_000.0,
    "C_FP":    500.0,
}


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


def _load_artifacts() -> Tuple[Any, ComponentPredictor, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading calibrated model ...")
    cal_model = joblib.load(CAL_MODEL)
    logger.info("Loading component classifier ...")
    comp_predictor = ComponentPredictor.load(MODELS_DIR)
    logger.info("Loading development parquet ...")
    dev = pd.read_parquet(DEV_PARQUET)
    logger.info("Loading test parquet ...")
    test = pd.read_parquet(TEST_PARQUET)
    logger.info(
        "Loaded: cal_model=%s  comp=%d classes  dev=%s  test=%s",
        type(cal_model).__name__, len(comp_predictor.classes_),
        dev.shape, test.shape,
    )
    return cal_model, comp_predictor, dev, test


# ===========================================================================
# Stage 11 — Fleet Risk Segmentation
# ===========================================================================

def run_stage11(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    cal_model: Any,
    comp_predictor: ComponentPredictor,
) -> Dict[str, Any]:

    _banner("STAGE 11: Fleet Risk Segmentation")
    t_stage = time.time()

    feature_cols = _get_feature_cols(test)
    logger.info("Feature columns: %d", len(feature_cols))

    segmenter = RiskSegmenter(tier_thresholds=DEFAULT_TIER_THRESHOLDS)
    analyzer = FleetAnalyzer(figures_dir=FIGURES_DIR, reports_dir=REPORTS_DIR)
    report_writer = FleetReportWriter(reports_dir=REPORTS_DIR)

    # -----------------------------------------------------------------------
    # PART 1 & 2 — Assign risk scores & tiers to TEST set
    # -----------------------------------------------------------------------
    _section("PART 1 & 2: Risk Scores & Tier Assignment (Test Set)")
    test_scored = segmenter.assign_risk_scores(test, cal_model, feature_cols)

    # -----------------------------------------------------------------------
    # Also score DEV set (needed for Stage 12 threshold optimisation)
    # -----------------------------------------------------------------------
    dev_feature_cols = _get_feature_cols(dev)
    dev_scored = segmenter.assign_risk_scores(dev, cal_model, dev_feature_cols)

    # -----------------------------------------------------------------------
    # Predict components for scored data
    # -----------------------------------------------------------------------
    _section("Component Prediction on Scored Data")
    test_scored = segmenter.assign_component_predictions(test_scored, comp_predictor)
    dev_scored = segmenter.assign_component_predictions(dev_scored, comp_predictor)

    # -----------------------------------------------------------------------
    # PART 3 — Fleet Aggregation
    # -----------------------------------------------------------------------
    _section("PART 3: Fleet Tier Summary")
    tier_df = analyzer.build_tier_summary(test_scored)

    # -----------------------------------------------------------------------
    # PART 4 — Machine Ranking
    # -----------------------------------------------------------------------
    _section("PART 4: Machine Ranking")
    machine_summary = analyzer.build_machine_summary(test_scored)
    top_machines = analyzer.build_top_machines(machine_summary, top_n=20)

    logger.info("Top 10 highest-risk machines:\n%s",
                top_machines.head(10)[["machineID", "latest_risk_score", "risk_tier",
                                       "predicted_component"]].to_string())

    # Save CSVs
    fleet_csv = REPORTS_DIR / "fleet_risk_summary.csv"
    machine_summary.to_csv(fleet_csv, index=False)
    logger.info("Saved fleet risk summary: %s", fleet_csv)

    # -----------------------------------------------------------------------
    # PART 5 — Figures
    # -----------------------------------------------------------------------
    _section("PART 5: Fleet Visualisations")
    analyzer.plot_fleet_risk_distribution(test_scored)
    analyzer.plot_tier_pie(tier_df)
    analyzer.plot_tier_bar(tier_df)
    analyzer.plot_machine_ranking(top_machines)
    analyzer.plot_failure_rate_by_tier(tier_df)
    analyzer.plot_component_by_tier(test_scored)
    analyzer.plot_risk_histogram(machine_summary)

    # -----------------------------------------------------------------------
    # Fleet Report
    # -----------------------------------------------------------------------
    _section("Writing Fleet Segmentation Report")
    report_writer.write(
        tier_df=tier_df,
        top_machines=top_machines,
        machine_summary=machine_summary,
        tier_thresholds=DEFAULT_TIER_THRESHOLDS,
    )

    elapsed = time.time() - t_stage
    logger.info("Stage 11 complete in %.1fs.", elapsed)

    return {
        "test_scored": test_scored,
        "dev_scored": dev_scored,
        "machine_summary": machine_summary,
        "top_machines": top_machines,
        "tier_df": tier_df,
        "feature_cols": feature_cols,
    }


# ===========================================================================
# Stage 12 — Cost-Based Dispatch Decision
# ===========================================================================

def run_stage12(
    dev_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    machine_summary: pd.DataFrame,
    feature_cols: List[str],
) -> None:

    _banner("STAGE 12: Cost-Based Dispatch Decision")
    t_stage = time.time()

    visualizer = CostVisualizer(figures_dir=FIGURES_DIR)
    report_writer = CostReportWriter(reports_dir=REPORTS_DIR)

    # -----------------------------------------------------------------------
    # PART 6 — Cost Matrix
    # -----------------------------------------------------------------------
    _section("PART 6: Cost Matrix")
    logger.info(
        "Cost Matrix: C_FN=$%s  C_FP=$%s  Ratio=%dx",
        f"{COST_MATRIX['C_FN']:,.0f}",
        f"{COST_MATRIX['C_FP']:,.0f}",
        int(COST_MATRIX["C_FN"] / COST_MATRIX["C_FP"]),
    )

    # -----------------------------------------------------------------------
    # PART 7 & 8 — Threshold Sweep & Optimal (DEV SET ONLY)
    # -----------------------------------------------------------------------
    _section("PART 7 & 8: Threshold Sweep on Development Set")

    y_dev = dev_scored["y_failure"].values
    proba_dev = dev_scored["calibrated_prob"].values

    optimizer = ThresholdOptimizer(cost_matrix=COST_MATRIX)
    sweep_df = optimizer.sweep(y_dev, proba_dev)
    optimal_threshold, optimal_metrics = optimizer.find_optimal()

    # Save sweep + optimal
    cost_csv = REPORTS_DIR / "cost_analysis.csv"
    sweep_df.to_csv(cost_csv, index=False)
    logger.info("Saved cost analysis CSV: %s", cost_csv)
    optimizer.save_optimal(REPORTS_DIR)

    logger.info(
        "\nThreshold sweep summary (first 5 rows around optimal):\n%s",
        sweep_df[
            sweep_df["threshold"].between(
                max(0.05, optimal_threshold - 0.02),
                min(0.95, optimal_threshold + 0.02)
            )
        ].to_string(index=False),
    )

    # -----------------------------------------------------------------------
    # PART 9 — Final Evaluation on Test Set (ONCE)
    # -----------------------------------------------------------------------
    _section("PART 9: Final Evaluation on Test Set (Single Use)")

    y_test = test_scored["y_failure"].values
    proba_test = test_scored["calibrated_prob"].values
    evaluator = CostEvaluator(cost_matrix=COST_MATRIX)
    test_metrics = evaluator.evaluate(y_test, proba_test, optimal_threshold)

    logger.info(
        "\nFinal Test Evaluation (threshold=%.2f):\n"
        "  Precision=%.4f  Recall=%.4f  F1=%.4f\n"
        "  TP=%d  FP=%d  TN=%d  FN=%d\n"
        "  Expected Cost=$%s  Dispatch Rate=%.1f%%",
        optimal_threshold,
        test_metrics["precision"], test_metrics["recall"], test_metrics["f1"],
        test_metrics["TP"], test_metrics["FP"],
        test_metrics["TN"], test_metrics["FN"],
        f"{test_metrics['expected_cost']:,.0f}",
        test_metrics["dispatch_rate"] * 100,
    )

    cost_savings = DecisionEngine(optimal_threshold, COST_MATRIX).compute_cost_savings(
        y_test, proba_test, baseline_threshold=0.5
    )
    logger.info(
        "Cost savings vs naive 0.5 threshold: $%s (%.1f%%)",
        f"{cost_savings['cost_savings']:,.0f}",
        cost_savings["pct_saving"],
    )

    # -----------------------------------------------------------------------
    # PART 10 — Decision Examples + PART 11 Dispatch Decisions
    # -----------------------------------------------------------------------
    _section("PART 10 & 11: Dispatch Decisions")

    engine = DecisionEngine(optimal_threshold, COST_MATRIX)
    decisions_df = engine.generate_decisions(machine_summary, test_scored)
    example_decisions = engine.build_example_reports(decisions_df, n_dispatch=3, n_monitor=2)

    # Save dispatch decisions CSV
    dispatch_csv = REPORTS_DIR / "dispatch_decisions.csv"
    decisions_df.to_csv(dispatch_csv, index=False)
    logger.info("Saved dispatch decisions: %s", dispatch_csv)

    logger.info("\nSample dispatch decisions:")
    logger.info("%-10s %-8s %-10s %-12s %-10s %-10s",
                "Machine", "Risk", "Tier", "Component", "Prob %", "Decision")
    for ex in example_decisions:
        logger.info("%-10s %-8s %-10s %-12s %-10.1f %-10s",
                    ex["machine_id"], ex["risk_score"], ex["risk_tier"],
                    ex["predicted_component"], ex["calibrated_prob"] * 100,
                    ex["decision"])

    # -----------------------------------------------------------------------
    # Fleet Status JSON (bonus: backend payload for Phase 4 dashboard)
    # -----------------------------------------------------------------------
    n_dispatch = (decisions_df["decision"] == "DISPATCH").sum()
    fleet_status = {
        "fleet_summary": {t: int((decisions_df["risk_tier"] == t).sum()) for t in TIER_ORDER},
        "highest_risk_machine": int(machine_summary.iloc[0]["machineID"]),
        "highest_risk_score": int(machine_summary.iloc[0]["latest_risk_score"]),
        "recommended_dispatches": int(n_dispatch),
        "optimal_threshold": optimal_threshold,
        "expected_cost_test": test_metrics["expected_cost"],
        "cost_savings_vs_baseline": cost_savings["cost_savings"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    fleet_json = CARDS_DIR / "fleet_status.json"
    with open(fleet_json, "w", encoding="utf-8") as f:
        json.dump(fleet_status, f, indent=2)
    logger.info("Saved fleet status JSON: %s", fleet_json)

    # -----------------------------------------------------------------------
    # PART 11 — Visualisations
    # -----------------------------------------------------------------------
    _section("PART 11: Cost & Decision Visualisations")
    visualizer.plot_cost_vs_threshold(sweep_df, optimal_threshold)
    visualizer.plot_threshold_comparison(sweep_df, optimal_threshold)
    visualizer.plot_cost_breakdown(sweep_df, optimal_threshold, COST_MATRIX)
    # Dispatch decisions figure (pass optimal threshold via simple attr injection)
    visualizer.optimal_threshold = optimal_threshold
    visualizer.plot_dispatch_decisions(decisions_df)
    y_pred_test = (proba_test >= optimal_threshold).astype(int)
    visualizer.plot_confusion_matrix_final(y_test, y_pred_test)

    # -----------------------------------------------------------------------
    # PART 13 — Cost Decision Report
    # -----------------------------------------------------------------------
    _section("PART 13: Writing Cost Decision Report")
    report_writer.write(
        optimal_threshold=optimal_threshold,
        cost_matrix=COST_MATRIX,
        optimal_metrics=optimal_metrics,
        test_metrics=test_metrics,
        cost_savings=cost_savings,
        example_decisions=example_decisions,
    )

    elapsed = time.time() - t_stage
    logger.info("Stage 12 complete in %.1fs.", elapsed)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    pipeline_start = time.time()
    _banner("PredictGuard — Phase 3 (Final): Fleet Segmentation & Cost Decision")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    cal_model, comp_predictor, dev, test = _load_artifacts()

    stage11_results = run_stage11(dev, test, cal_model, comp_predictor)

    run_stage12(
        dev_scored=stage11_results["dev_scored"],
        test_scored=stage11_results["test_scored"],
        machine_summary=stage11_results["machine_summary"],
        feature_cols=stage11_results["feature_cols"],
    )

    total = time.time() - pipeline_start
    _banner("Phase 3 (Stage 11 & 12) COMPLETE")
    logger.info("  Total runtime              : %.1fs (%.1f min)", total, total / 60)
    logger.info("  Fleet risk summary CSV     : reports/fleet_risk_summary.csv")
    logger.info("  Cost analysis CSV          : reports/cost_analysis.csv")
    logger.info("  Optimal threshold JSON     : reports/optimal_threshold.json")
    logger.info("  Dispatch decisions CSV     : reports/dispatch_decisions.csv")
    logger.info("  Fleet segmentation report  : reports/fleet_segmentation_report.md")
    logger.info("  Cost decision report       : reports/cost_decision_report.md")
    logger.info("  Fleet status JSON          : reports/prediction_cards/fleet_status.json")
    logger.info("  All figures                : reports/figures/")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
