"""
run_stage9_10.py
================
PredictGuard — Phase 3 Orchestrator: Stage 9 (SHAP Explainability) + Stage 10 (Component Classifier)

Usage:
    python run_stage9_10.py

Outputs:
    models/component_classifier.pkl
    reports/shap_summary.csv
    reports/component_metrics.csv
    reports/feature_importance_comparison.csv
    reports/explainability_report.md
    reports/component_classifier_report.md
    reports/figures/shap_*.png
    reports/figures/component_*.png
    reports/figures/importance_comparison.png
    reports/figures/native_importance.png
    reports/prediction_cards/prediction_card_*.json

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup — must happen before importing project modules
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_stage9_10")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import (
    SHAPExplainer,
    FeatureImportanceAnalyzer,
    PredictionInterpreter,
    ExplainabilityReportWriter,
)
from src.component_classifier import (
    ComponentPredictor,
    ComponentEvaluator,
    ComponentDiagnosisReport,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
REPORTS_DIR   = PROJECT_ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
CARDS_DIR     = REPORTS_DIR / "prediction_cards"

DEV_PARQUET   = DATA_DIR / "development.parquet"
TEST_PARQUET  = DATA_DIR / "test.parquet"
RAW_MODEL     = MODELS_DIR / "raw_model.pkl"
CAL_MODEL     = MODELS_DIR / "calibrated_model.pkl"

NON_FEATURE_COLS = {
    "datetime", "machineID", "y_failure",
    "failure_component", "time_to_failure_hours",
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


def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading development set from %s ...", DEV_PARQUET)
    dev = pd.read_parquet(DEV_PARQUET)
    logger.info("Loading test set from %s ...", TEST_PARQUET)
    test = pd.read_parquet(TEST_PARQUET)
    logger.info(
        "Loaded dev=%s  test=%s",
        dev.shape, test.shape
    )
    return dev, test


def _load_models() -> Tuple[Any, Any]:
    raw_model = joblib.load(RAW_MODEL)
    cal_model = joblib.load(CAL_MODEL)
    logger.info("Loaded raw model:        %s", type(raw_model).__name__)
    logger.info("Loaded calibrated model: %s", type(cal_model).__name__)
    return raw_model, cal_model


# ===========================================================================
# Stage 9 — SHAP Explainability
# ===========================================================================

def run_stage9(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    raw_model: Any,
    cal_model: Any,
) -> Dict[str, Any]:

    _banner("STAGE 9: SHAP Explainability & Feature Importance")
    t_stage = time.time()

    feature_cols = _get_feature_cols(dev)
    logger.info("Feature columns: %d", len(feature_cols))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    X_dev = dev[feature_cols].select_dtypes(include=[np.number])

    # -----------------------------------------------------------------------
    # PART 1 — SHAPExplainer (global)
    # -----------------------------------------------------------------------
    _section("PART 1 & 2: Global SHAP Values")

    explainer = SHAPExplainer(
        raw_model=raw_model,
        feature_names=list(X_dev.columns),
        figures_dir=FIGURES_DIR,
    )
    shap_values = explainer.compute_global(X_dev)
    n_sampled = min(5000, len(X_dev))

    logger.info("Global SHAP shape: %s", shap_values.shape)

    # -----------------------------------------------------------------------
    # PART 2 — Global plots
    # -----------------------------------------------------------------------
    _section("PART 2: SHAP Global Figures")
    explainer.plot_summary(X_dev)
    explainer.plot_beeswarm(X_dev)
    explainer.plot_bar(X_dev)

    # -----------------------------------------------------------------------
    # PART 3 — Global importance DataFrame
    # -----------------------------------------------------------------------
    shap_imp_df = explainer.get_global_importance_df()
    logger.info("Top 5 global SHAP features:\n%s", shap_imp_df.head(5).to_string(index=False))

    # Save SHAP summary CSV
    shap_csv = REPORTS_DIR / "shap_summary.csv"
    shap_imp_df.to_csv(shap_csv, index=False)
    logger.info("Saved SHAP summary CSV: %s", shap_csv)

    # -----------------------------------------------------------------------
    # PART 3 — Native Feature Importance
    # -----------------------------------------------------------------------
    _section("PART 3: Native vs SHAP Feature Importance")

    analyzer = FeatureImportanceAnalyzer(
        raw_model=raw_model,
        feature_names=list(X_dev.columns),
        figures_dir=FIGURES_DIR,
    )
    comparison_df = analyzer.build_comparison_table(shap_imp_df)
    analyzer.plot_native_importance()
    analyzer.plot_comparison(comparison_df)

    comparison_csv = REPORTS_DIR / "feature_importance_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info("Saved feature importance comparison CSV: %s", comparison_csv)

    # -----------------------------------------------------------------------
    # PART 4 — Local SHAP explanations (waterfall + force on highest-risk row)
    # -----------------------------------------------------------------------
    _section("PART 4 & 5: Local SHAP — Waterfall & Force")

    # Highest-risk test row
    X_test = test[feature_cols].select_dtypes(include=[np.number])
    cal_probs_test = cal_model.predict_proba(X_test)[:, 1]
    top_idx = int(np.argmax(cal_probs_test))
    X_top = X_test.iloc[[top_idx]]
    machine_id_top = test.iloc[top_idx].get("machineID", "N/A")
    ts_top = test.iloc[top_idx].get("datetime", "N/A")
    raw_probs_test = raw_model.predict_proba(X_test)[:, 1]

    logger.info(
        "Highest-risk row: Machine %s @ %s — CalibProb=%.4f",
        machine_id_top, ts_top, cal_probs_test[top_idx]
    )
    explainer.plot_waterfall(X_top, row_label=f"Machine {machine_id_top} @ {ts_top}")
    explainer.plot_force(X_top, row_label=f"Machine {machine_id_top} @ {ts_top}")

    # -----------------------------------------------------------------------
    # PART 5 — Prediction explanation cards
    # -----------------------------------------------------------------------
    _section("PART 5: Prediction Explanation Cards")

    interpreter = PredictionInterpreter(
        feature_names=list(X_test.columns),
        reports_dir=CARDS_DIR,
    )
    example_cards = interpreter.generate_example_cards(
        df=test,
        feature_cols=feature_cols,
        explainer=explainer,
        raw_model=raw_model,
        calibrated_model=cal_model,
        n_high=3,
        n_low=2,
    )
    # Save each card individually
    for i, card in enumerate(example_cards):
        interpreter.save_card(
            card,
            filename=f"prediction_card_{i+1}_machine{card['machine_id']}.json"
        )

    # -----------------------------------------------------------------------
    # PART 6 — Explainability Report
    # -----------------------------------------------------------------------
    _section("PART 6: Writing Explainability Report")
    report_writer = ExplainabilityReportWriter(reports_dir=REPORTS_DIR)
    report_writer.write(
        shap_importance_df=shap_imp_df,
        comparison_df=comparison_df,
        example_cards=example_cards,
        global_shap_computed_on=n_sampled,
    )

    elapsed = time.time() - t_stage
    logger.info("Stage 9 complete in %.1fs.", elapsed)

    return {
        "shap_imp_df": shap_imp_df,
        "comparison_df": comparison_df,
        "example_cards": example_cards,
        "explainer": explainer,
        "feature_cols": feature_cols,
    }


# ===========================================================================
# Stage 10 — Component Classifier
# ===========================================================================

def run_stage10(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    cal_model: Any,
    stage9_results: Dict[str, Any],
) -> None:

    _banner("STAGE 10: Failure Component Classification")
    t_stage = time.time()

    feature_cols = stage9_results["feature_cols"]
    explainer = stage9_results["explainer"]

    # -----------------------------------------------------------------------
    # PART 6 — Dataset construction
    # -----------------------------------------------------------------------
    _section("PART 6: Build Component Dataset")

    predictor = ComponentPredictor(models_dir=MODELS_DIR)
    train_fail_df = predictor.build_dataset(dev)
    logger.info("Component training set: %d rows, %d classes",
                len(train_fail_df), train_fail_df["failure_component"].nunique())

    # -----------------------------------------------------------------------
    # PART 7 — Train
    # -----------------------------------------------------------------------
    _section("PART 7: Train Component Classifier")
    predictor.fit(train_fail_df)
    predictor.save()

    # -----------------------------------------------------------------------
    # PART 8 — Evaluate on test set failure rows
    # -----------------------------------------------------------------------
    _section("PART 8: Evaluate on Test Set Failure Rows")

    evaluator = ComponentEvaluator(figures_dir=FIGURES_DIR, reports_dir=REPORTS_DIR)

    test_fail_df = predictor.build_dataset(test, min_class_samples=1)
    if len(test_fail_df) == 0:
        logger.warning("No labelled failure rows in test set — skipping evaluation.")
        return

    logger.info("Test failure rows for evaluation: %d", len(test_fail_df))

    y_true = test_fail_df["failure_component"].values
    y_pred = predictor.predict(test_fail_df)
    proba_df = predictor.predict_proba_df(test_fail_df)
    y_proba = proba_df[predictor.classes_].values

    metrics_df, aggregate_metrics = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=predictor.classes_,
    )

    # Save metrics CSV
    metrics_csv = REPORTS_DIR / "component_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info("Saved component metrics CSV: %s", metrics_csv)

    # Log summary
    logger.info(
        "\nComponent Evaluation:\n%s\nAggregate: %s",
        metrics_df.to_string(index=False),
        aggregate_metrics,
    )

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    _section("PART 10: Visualisations")

    evaluator.plot_confusion_matrix(y_true, y_pred, predictor.classes_)
    evaluator.plot_per_class_recall(metrics_df)
    evaluator.plot_component_distribution(train_fail_df["failure_component"].values)
    evaluator.plot_prediction_distribution(proba_df, predictor.classes_)
    evaluator.plot_roc_ovr(y_true, y_proba, predictor.classes_)
    evaluator.plot_component_feature_importance(predictor)

    # -----------------------------------------------------------------------
    # PART 9 — Probability output (combined binary + component)
    # -----------------------------------------------------------------------
    _section("PART 9: Probability Diagnosis Output")

    # Build combined diagnosis cards for top failure predictions from calibrated model
    X_test_all = test[feature_cols].select_dtypes(include=[np.number])
    cal_probs_all = cal_model.predict_proba(X_test_all)[:, 1]
    top_failure_idx = np.argsort(cal_probs_all)[-5:][::-1]

    diagnosis_writer = ComponentDiagnosisReport(reports_dir=CARDS_DIR)
    diagnosis_cards = []

    logger.info("\nTop 5 Failure Predictions with Component Diagnosis:")
    logger.info("%-10s %-20s %-12s %-10s %-12s %-10s",
                "Machine", "Timestamp", "Fail Prob", "Comp", "Conf", "Risk")

    for idx in top_failure_idx:
        row = test.iloc[idx]
        machine_id = row.get("machineID", -1)
        timestamp = row.get("datetime", "unknown")
        cal_prob = float(cal_probs_all[idx])

        # Component prediction for this row
        X_one = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
        X_one = X_one.select_dtypes(include=[np.number])

        # Align with predictor features
        missing = [c for c in predictor.feature_cols if c not in X_one.columns]
        for mc in missing:
            X_one[mc] = 0.0
        X_one = X_one[predictor.feature_cols]

        comp_proba_row = predictor.predict_proba_df(X_one).iloc[0]
        predicted_comp = comp_proba_row["predicted_component"]
        comp_confidence = float(comp_proba_row["component_confidence"])
        comp_proba_dict = {
            c: float(comp_proba_row[c]) for c in predictor.classes_
        }

        # SHAP for this row (top positive features)
        sv, _ = explainer.compute_local(X_one[explainer.feature_names] if all(
            f in X_one.columns for f in explainer.feature_names
        ) else X_test_all.iloc[[idx]])
        sorted_feat_idx = np.argsort(sv.flatten())[::-1]
        top_pos_feats = [
            explainer.feature_names[i]
            for i in sorted_feat_idx if sv.flatten()[i] > 0
        ][:5]

        card = diagnosis_writer.build_diagnosis_card(
            machine_id=machine_id,
            timestamp=timestamp,
            calibrated_prob=cal_prob,
            component_proba=comp_proba_dict,
            predicted_component=predicted_comp,
            top_positive_features=top_pos_feats,
        )
        diagnosis_cards.append(card)

        logger.info("%-10s %-20s %-12.1f %-10s %-12.1f %-10s",
                    machine_id, str(timestamp)[:19],
                    cal_prob * 100, predicted_comp,
                    comp_confidence * 100,
                    card["risk_level"])

    # Save diagnosis cards
    diag_out = CARDS_DIR / "diagnosis_cards.json"
    with open(diag_out, "w", encoding="utf-8") as f:
        json.dump(diagnosis_cards, f, indent=2, default=str)
    logger.info("Saved %d diagnosis cards: %s", len(diagnosis_cards), diag_out)

    # -----------------------------------------------------------------------
    # PART 12 — Component Classifier Report
    # -----------------------------------------------------------------------
    _section("PART 12: Writing Component Classifier Report")

    diagnosis_writer_reports = ComponentDiagnosisReport(reports_dir=REPORTS_DIR)
    diagnosis_writer_reports.write_report(
        predictor=predictor,
        train_df=train_fail_df,
        test_df=test_fail_df,
        metrics_df=metrics_df,
        aggregate_metrics=aggregate_metrics,
        example_cards=diagnosis_cards,
    )

    elapsed = time.time() - t_stage
    logger.info("Stage 10 complete in %.1fs.", elapsed)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    pipeline_start = time.time()
    _banner("PredictGuard — Phase 3: Explainability & Component Classification")

    # Setup output directories
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    # Load artifacts
    dev, test = _load_data()
    raw_model, cal_model = _load_models()

    # Stage 9
    stage9_results = run_stage9(dev, test, raw_model, cal_model)

    # Stage 10
    run_stage10(dev, test, cal_model, stage9_results)

    # Final summary
    total = time.time() - pipeline_start
    _banner("Phase 3 (Stage 9 & 10) COMPLETE")
    logger.info("  Total runtime         : %.1f seconds (%.1f min)", total, total / 60)
    logger.info("  SHAP report           : reports/explainability_report.md")
    logger.info("  Component report      : reports/component_classifier_report.md")
    logger.info("  Component model       : models/component_classifier.pkl")
    logger.info("  SHAP summary CSV      : reports/shap_summary.csv")
    logger.info("  Component metrics CSV : reports/component_metrics.csv")
    logger.info("  Feature comparison CSV: reports/feature_importance_comparison.csv")
    logger.info("  Prediction cards      : reports/prediction_cards/")
    logger.info("  All figures           : reports/figures/")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
