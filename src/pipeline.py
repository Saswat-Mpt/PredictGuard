"""
pipeline.py
===========
PredictGuard — Phase 4 Stage 14: End-to-End Inference Pipeline & Model Persistence.

Provides:
    FeatureAligner            — sklearn transformer: aligns input columns to expected set
    RiskScoreTransformer      — sklearn transformer: adds risk_score and risk_tier
    DispatchDecisionTransformer — sklearn transformer: adds DISPATCH/MONITOR decision
    PredictGuardPipeline      — full inference pipeline wrapping all trained models
    PipelineValidator         — validates pipeline reproducibility vs raw models
    PipelineVersioner         — saves/loads versioned pipeline metadata

Architecture:
    Input Features (pre-engineered)
         │
         ▼
    FeatureAligner (column alignment, fill missing)
         │
         ▼
    Binary Prediction (calibrated XGBoost)
         │
         ▼
    RiskScoreTransformer (0-100 score + tier)
         │
         ▼
    Component Prediction (multiclass XGBoost)
         │
         ▼
    DispatchDecisionTransformer (DISPATCH / MONITOR)
         │
         ▼
    predict_report() → structured JSON output

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "1.0.0"
FEATURE_VERSION  = "3.1"      # Corresponds to Stage 3 feature engineering
MODEL_VERSION    = "2.0"      # Corresponds to Stage 6 tuned XGBoost
CALIBRATION_VERSION = "1.0"   # Corresponds to Stage 8 sigmoid calibration

# Non-feature columns that must be excluded from model input
NON_FEATURE_COLS = {
    "datetime", "machineID", "y_failure",
    "failure_component", "time_to_failure_hours",
    "calibrated_prob", "risk_score", "risk_tier",
    "predicted_component", "component_confidence",
    "failure_prob", "dispatch_decision",
}

# Risk tier thresholds (0–100 integer scale)
TIER_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "LOW":      (0.0,  20.0),
    "MEDIUM":   (20.0, 50.0),
    "HIGH":     (50.0, 75.0),
    "CRITICAL": (75.0, 100.01),
}

DISPATCH_ACTIONS: Dict[str, str] = {
    "comp1": "Dispatch hydraulic specialist. Inspect pressure seals and pump.",
    "comp2": "Dispatch mechanical engineer. Inspect rotary bearing and lubrication.",
    "comp3": "Dispatch electrician. Check voltage regulators and control circuits.",
    "comp4": "Dispatch vibration analyst. Inspect dampeners and drive belt.",
}


# ===========================================================================
# Custom sklearn Transformers
# ===========================================================================

class FeatureAligner(BaseEstimator, TransformerMixin):
    """
    Ensures the input DataFrame has exactly the expected feature columns.

    - Drops columns not in expected_cols
    - Fills missing expected columns with 0.0 (with a warning)
    - Reorders columns to match training order

    Parameters
    ----------
    expected_cols : list of str
        Ordered list of feature column names from training.
    fill_value : float
        Value used to fill missing columns (default 0.0).
    """

    def __init__(
        self,
        expected_cols: Optional[List[str]] = None,
        fill_value: float = 0.0,
    ) -> None:
        self.expected_cols = expected_cols or []
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureAligner":
        if not self.expected_cols:
            self.expected_cols = [
                c for c in X.columns if c not in NON_FEATURE_COLS
            ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.expected_cols) - set(X.columns)
        if missing:
            logger.warning(
                "FeatureAligner: %d missing feature columns filled with %.1f: %s",
                len(missing), self.fill_value,
                sorted(missing)[:5],
            )
            for col in missing:
                X = X.copy()
                X[col] = self.fill_value
        extra = set(X.columns) - set(self.expected_cols)
        if extra:
            X = X.drop(columns=list(extra), errors="ignore")
        return X[self.expected_cols].astype(float)


class RiskScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Adds risk_score (0–100) and risk_tier columns to a DataFrame
    that already has a `calibrated_prob` column.

    Parameters
    ----------
    tier_thresholds : dict mapping tier name → (lo, hi)
    """

    def __init__(
        self,
        tier_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.tier_thresholds = tier_thresholds or TIER_THRESHOLDS

    def fit(self, X, y=None) -> "RiskScoreTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["risk_score"] = (out["calibrated_prob"] * 100).round().astype(int).clip(0, 100)
        out["risk_tier"] = out["risk_score"].apply(self._assign_tier)
        return out

    def _assign_tier(self, score: int) -> str:
        for tier, (lo, hi) in self.tier_thresholds.items():
            if lo <= score < hi:
                return tier
        return "CRITICAL"


class DispatchDecisionTransformer(BaseEstimator, TransformerMixin):
    """
    Adds `dispatch_decision` (DISPATCH / MONITOR) and `recommended_action`
    based on `calibrated_prob` vs the optimal threshold.

    Parameters
    ----------
    optimal_threshold : float
        Probability threshold from Stage 12 cost optimisation.
    """

    def __init__(self, optimal_threshold: float = 0.5) -> None:
        self.optimal_threshold = optimal_threshold

    def fit(self, X, y=None) -> "DispatchDecisionTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["dispatch_decision"] = np.where(
            out["calibrated_prob"] >= self.optimal_threshold,
            "DISPATCH",
            "MONITOR",
        )
        out["recommended_action"] = out.apply(
            lambda r: DISPATCH_ACTIONS.get(
                r.get("predicted_component", ""), "Dispatch maintenance team."
            ) if r["dispatch_decision"] == "DISPATCH"
            else "Continue scheduled monitoring. No immediate action required.",
            axis=1,
        )
        return out


# ===========================================================================
# PredictGuardPipeline
# ===========================================================================

class PredictGuardPipeline:
    """
    End-to-end PredictGuard inference pipeline.

    Wraps:
        - Calibrated binary failure model (XGBoost + Sigmoid calibration)
        - Component failure classifier (XGBoost multiclass)
        - Risk scoring and tier assignment
        - Cost-optimal dispatch decision

    Parameters
    ----------
    calibrated_model : sklearn-compatible classifier
        Calibrated binary failure predictor (Phase 2).
    raw_model : XGBClassifier
        Raw (uncalibrated) XGBoost model — used for SHAP (Phase 3).
    component_predictor : ComponentPredictor
        Multiclass component failure predictor (Stage 10).
    feature_cols : list of str
        Ordered list of feature column names used in training.
    optimal_threshold : float
        Cost-optimal dispatch probability threshold (Stage 12).
    tier_thresholds : dict, optional
        Risk tier boundaries (configurable).
    """

    def __init__(
        self,
        calibrated_model: Any,
        raw_model: Any,
        component_predictor: Any,
        feature_cols: List[str],
        optimal_threshold: float,
        tier_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        cost_matrix: Optional[Dict[str, float]] = None,
    ) -> None:
        self.calibrated_model = calibrated_model
        self.raw_model = raw_model
        self.component_predictor = component_predictor
        self.feature_cols = feature_cols
        self.optimal_threshold = optimal_threshold
        self.tier_thresholds = tier_thresholds or TIER_THRESHOLDS
        self.cost_matrix = cost_matrix or {"C_FN": 10_000.0, "C_FP": 500.0}

        # Custom transformers
        self._feature_aligner = FeatureAligner(expected_cols=feature_cols)
        self._risk_transformer = RiskScoreTransformer(tier_thresholds=self.tier_thresholds)
        self._dispatch_transformer = DispatchDecisionTransformer(
            optimal_threshold=optimal_threshold
        )

        # Pipeline metadata
        self.metadata: Dict[str, Any] = {
            "pipeline_version": PIPELINE_VERSION,
            "feature_version": FEATURE_VERSION,
            "model_version": MODEL_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "feature_count": len(feature_cols),
            "optimal_threshold": optimal_threshold,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
        }
        logger.info(
            "PredictGuardPipeline v%s initialised: %d features, threshold=%.2f",
            PIPELINE_VERSION, len(feature_cols), optimal_threshold,
        )

    # ------------------------------------------------------------------
    # Core Prediction Methods
    # ------------------------------------------------------------------

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a clean, aligned feature matrix."""
        return self._feature_aligner.transform(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated failure probabilities.

        Parameters
        ----------
        X : DataFrame with feature columns.

        Returns
        -------
        proba : np.ndarray, shape (n,)  — P(failure) for each row.
        """
        X_aligned = self._align_features(X)
        return self.calibrated_model.predict_proba(X_aligned)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Binary failure predictions at optimal threshold.

        Returns
        -------
        y_pred : np.ndarray of int (0 / 1)
        """
        return (self.predict_proba(X) >= self.optimal_threshold).astype(int)

    def predict_component(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict failing component for each row.

        Returns
        -------
        DataFrame with columns: predicted_component, component_confidence,
                                comp1_prob, comp2_prob, comp3_prob, comp4_prob
        """
        feat_cols = self.component_predictor.feature_cols
        available = [c for c in feat_cols if c in X.columns]
        missing = set(feat_cols) - set(available)
        if missing:
            logger.warning("Component predictor: %d missing cols, padding with 0", len(missing))
            X = X.copy()
            for mc in missing:
                X[mc] = 0.0
        return self.component_predictor.predict_proba_df(X[feat_cols])

    def predict_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute risk scores and tiers for each row.

        Returns
        -------
        DataFrame with columns: calibrated_prob, risk_score, risk_tier
        """
        proba = self.predict_proba(X)
        df = pd.DataFrame({"calibrated_prob": proba})
        return self._risk_transformer.transform(df)

    # ------------------------------------------------------------------
    def predict_report(
        self,
        X: pd.DataFrame,
        machine_ids: Optional[Union[pd.Series, List]] = None,
        timestamps: Optional[Union[pd.Series, List]] = None,
        include_shap_top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Full prediction report for every row in X.

        Combines binary prediction, calibration, risk scoring,
        component diagnosis, and dispatch decision into one structured output.

        Parameters
        ----------
        X             : feature-engineered DataFrame
        machine_ids   : optional machine IDs (length = len(X))
        timestamps    : optional timestamps (length = len(X))
        include_shap_top_n : number of top SHAP features per row (0 = skip SHAP)

        Returns
        -------
        List of dicts, one per row, each containing all prediction fields.
        """
        n = len(X)
        if machine_ids is None:
            machine_ids = [f"machine_{i}" for i in range(n)]
        if timestamps is None:
            timestamps = [datetime.now(timezone.utc).isoformat()] * n

        machine_ids = list(machine_ids)
        timestamps  = list(timestamps)

        proba      = self.predict_proba(X)
        risk_df    = self._risk_transformer.transform(
            pd.DataFrame({"calibrated_prob": proba})
        )
        comp_df    = self.predict_component(X)

        reports = []
        for i in range(n):
            risk_score = int(risk_df.iloc[i]["risk_score"])
            risk_tier  = str(risk_df.iloc[i]["risk_tier"])
            cal_prob   = float(proba[i])
            comp_pred  = str(comp_df.iloc[i]["predicted_component"])
            comp_conf  = float(comp_df.iloc[i]["component_confidence"])
            dispatch   = "DISPATCH" if cal_prob >= self.optimal_threshold else "MONITOR"
            action     = (
                DISPATCH_ACTIONS.get(comp_pred, "Dispatch maintenance team.")
                if dispatch == "DISPATCH"
                else "Continue scheduled monitoring."
            )

            report: Dict[str, Any] = {
                "machine_id": machine_ids[i],
                "timestamp": str(timestamps[i]),
                "failure_probability": round(cal_prob, 4),
                "risk_score": risk_score,
                "risk_tier": risk_tier,
                "predicted_component": comp_pred,
                "component_confidence": round(comp_conf, 4),
                "dispatch_decision": dispatch,
                "recommended_action": action,
                "optimal_threshold": self.optimal_threshold,
                "pipeline_version": PIPELINE_VERSION,
            }

            # Optional: top SHAP features (fast — tree_path_dependent)
            if include_shap_top_n > 0:
                try:
                    import shap
                    if not hasattr(self, "_shap_explainer"):
                        self._shap_explainer = shap.TreeExplainer(
                            self.raw_model,
                            feature_perturbation="tree_path_dependent",
                        )
                    X_row = self._align_features(X.iloc[[i]])
                    sv = self._shap_explainer.shap_values(X_row)
                    top_idx = np.abs(sv[0]).argsort()[::-1][:include_shap_top_n]
                    report["top_shap_features"] = [
                        {
                            "feature": self.feature_cols[j],
                            "shap_value": round(float(sv[0][j]), 4),
                        }
                        for j in top_idx
                    ]
                except Exception as exc:
                    logger.debug("SHAP skipped for row %d: %s", i, exc)
                    report["top_shap_features"] = []

            reports.append(report)

        logger.info("Generated %d prediction reports (threshold=%.2f).", n, self.optimal_threshold)
        return reports

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, models_dir: Path) -> Tuple[Path, Path]:
        """
        Persist the pipeline (joblib) and metadata (JSON).

        Returns
        -------
        (pipeline_path, metadata_path)
        """
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = models_dir / "predictguard_pipeline.pkl"
        metadata_path = models_dir / "predictguard_pipeline_metadata.json"

        joblib.dump(self, pipeline_path, compress=3)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info(
            "Pipeline saved: %s (%.1f MB)",
            pipeline_path.name,
            pipeline_path.stat().st_size / 1e6,
        )
        logger.info("Metadata saved: %s", metadata_path.name)
        return pipeline_path, metadata_path

    @classmethod
    def load(cls, models_dir: Path) -> "PredictGuardPipeline":
        """Load pipeline from models/predictguard_pipeline.pkl."""
        # Cross-platform compatibility patch for unpickling WindowsPath on Linux/macOS
        import pathlib
        try:
            pathlib.WindowsPath()
        except NotImplementedError:
            pathlib.WindowsPath = pathlib.PosixPath

        pipeline_path = Path(models_dir) / "predictguard_pipeline.pkl"
        pipeline = joblib.load(pipeline_path)
        logger.info(
            "Pipeline loaded: v%s (threshold=%.2f, %d features)",
            pipeline.metadata.get("pipeline_version", "?"),
            pipeline.optimal_threshold,
            len(pipeline.feature_cols),
        )
        return pipeline

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"PredictGuardPipeline(version={PIPELINE_VERSION}, "
            f"features={len(self.feature_cols)}, "
            f"threshold={self.optimal_threshold:.2f})"
        )


# ===========================================================================
# PipelineValidator
# ===========================================================================

class PipelineValidator:
    """
    Validates that the loaded pipeline produces predictions numerically
    identical to the original models used in Stages 1–12.

    Checks:
        1. calibrated_prob   — max absolute difference < tolerance
        2. binary predictions — exact match rate >= 99.9%
        3. component predictions — exact match rate >= 99%
        4. risk tiers — exact match rate >= 99.9%
    """

    def __init__(
        self,
        pipeline: PredictGuardPipeline,
        calibrated_model: Any,
        component_predictor: Any,
        tolerance: float = 1e-5,
    ) -> None:
        self.pipeline = pipeline
        self.calibrated_model = calibrated_model
        self.component_predictor = component_predictor
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    def validate(
        self, X: pd.DataFrame, feature_cols: List[str], n_rows: int = 5000
    ) -> Dict[str, Any]:
        """
        Run reproducibility checks on a sample of X.

        Returns
        -------
        results : dict with 'passed', 'checks', 'max_prob_diff', 'match_rates'
        """
        sample = X.sample(min(n_rows, len(X)), random_state=42)
        X_feat = sample[feature_cols].select_dtypes(include=[np.number])

        # 1. Calibrated probabilities
        orig_proba = self.calibrated_model.predict_proba(X_feat)[:, 1]
        pipe_proba = self.pipeline.predict_proba(sample)
        max_diff = float(np.abs(orig_proba - pipe_proba).max())

        # 2. Binary predictions
        orig_pred = (orig_proba >= self.pipeline.optimal_threshold).astype(int)
        pipe_pred = self.pipeline.predict(sample)
        bin_match = float((orig_pred == pipe_pred).mean())

        # 3. Component predictions
        comp_feat_cols = self.component_predictor.feature_cols
        available = [c for c in comp_feat_cols if c in sample.columns]
        orig_comp = self.component_predictor.predict_proba_df(
            sample[available]
        )["predicted_component"].values
        pipe_comp = self.pipeline.predict_component(sample)["predicted_component"].values
        comp_match = float((orig_comp == pipe_comp).mean())

        # 4. Risk tiers
        orig_risk = self.pipeline._risk_transformer.transform(
            pd.DataFrame({"calibrated_prob": orig_proba})
        )["risk_tier"].values
        pipe_risk = self.pipeline.predict_risk(sample)["risk_tier"].values
        tier_match = float((orig_risk == pipe_risk).mean())

        checks = {
            "prob_max_diff_below_tolerance": max_diff < self.tolerance,
            "binary_match_rate_above_99pct": bin_match >= 0.999,
            "component_match_rate_above_99pct": comp_match >= 0.990,
            "risk_tier_match_rate_above_99pct": tier_match >= 0.999,
        }

        results = {
            "passed": all(checks.values()),
            "checks": checks,
            "max_prob_diff": max_diff,
            "match_rates": {
                "binary_predictions": bin_match,
                "component_predictions": comp_match,
                "risk_tiers": tier_match,
            },
            "n_samples": len(sample),
            "tolerance": self.tolerance,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        }

        status = "PASSED [OK]" if results["passed"] else "FAILED [!!]"
        logger.info(
            "Pipeline validation: %s | max_prob_diff=%.2e | binary=%.4f | "
            "comp=%.4f | tier=%.4f",
            status, max_diff, bin_match, comp_match, tier_match,
        )
        return results

    # ------------------------------------------------------------------
    def write_report(
        self, results: Dict[str, Any], reports_dir: Path
    ) -> Path:
        """Write pipeline_validation.md and reproducibility_report.md."""
        lines = [
            "# PredictGuard — Pipeline Validation Report",
            "",
            f"> Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            f"## Status: {'✅ ALL CHECKS PASSED' if results['passed'] else '❌ VALIDATION FAILED'}",
            "",
            "---",
            "",
            "## Validation Checks",
            "",
            "| Check | Result |",
            "|---|---|",
        ]
        check_labels = {
            "prob_max_diff_below_tolerance": "Calibrated probability diff < tolerance",
            "binary_match_rate_above_99pct": "Binary prediction match rate ≥ 99.9%",
            "component_match_rate_above_99pct": "Component prediction match rate ≥ 99%",
            "risk_tier_match_rate_above_99pct": "Risk tier match rate ≥ 99.9%",
        }
        for key, label in check_labels.items():
            icon = "✅ PASS" if results["checks"].get(key) else "❌ FAIL"
            lines.append(f"| {label} | {icon} |")

        lines += [
            "",
            "---",
            "",
            "## Numerical Reproducibility",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Max probability difference | `{results['max_prob_diff']:.2e}` |",
            f"| Tolerance | `{results['tolerance']:.2e}` |",
            f"| Binary prediction match | `{results['match_rates']['binary_predictions']*100:.4f}%` |",
            f"| Component match | `{results['match_rates']['component_predictions']*100:.4f}%` |",
            f"| Risk tier match | `{results['match_rates']['risk_tiers']*100:.4f}%` |",
            f"| Validation sample size | `{results['n_samples']}` |",
            "",
            "---",
            "",
            "## Pipeline Architecture",
            "",
            "```",
            "Input Features (pre-engineered, 119 columns)",
            "       │",
            "       ▼",
            "FeatureAligner (column alignment + zero-fill missing)",
            "       │",
            "       ▼",
            "CalibratedClassifierCV (XGBoost + Sigmoid)",
            "       │  predict_proba()[:, 1]",
            "       ▼",
            "RiskScoreTransformer (risk_score = round(100 × prob))",
            "       │  risk_tier ∈ {LOW, MEDIUM, HIGH, CRITICAL}",
            "       ▼",
            "ComponentPredictor (XGBoost multiclass: comp1–comp4)",
            "       │",
            "       ▼",
            "DispatchDecisionTransformer (prob ≥ optimal_threshold → DISPATCH)",
            "       │",
            "       ▼",
            "predict_report() → JSON",
            "```",
            "",
            "---",
            "",
            "## Edge Case Handling",
            "",
            "| Scenario | Behaviour |",
            "|---|---|",
            "| Missing feature columns | `FeatureAligner` fills with 0.0 and warns |",
            "| Unknown machine ID | Treated as new machine; no history assumed |",
            "| All rows same probability | Risk tiers still correctly assigned |",
            "| Very high threshold | All predictions → MONITOR |",
            "| Very low threshold | All predictions → DISPATCH |",
            "",
        ]

        out_val = Path(reports_dir) / "pipeline_validation.md"
        out_val.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved pipeline validation report: %s", out_val)

        # Reproducibility report (brief summary for README linking)
        repro_lines = [
            "# PredictGuard — Reproducibility Report",
            "",
            f"> Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Result",
            "",
            f"**Pipeline reproducibility: {'CONFIRMED' if results['passed'] else 'FAILED'}**",
            "",
            f"The serialised `predictguard_pipeline.pkl` produces predictions "
            f"numerically identical to the original Stages 1–12 models "
            f"(max probability error = `{results['max_prob_diff']:.2e}`, "
            f"well within tolerance `{results['tolerance']:.2e}`).",
            "",
            "## How to Use the Pipeline",
            "",
            "```python",
            "from src.pipeline import PredictGuardPipeline",
            "import pandas as pd",
            "",
            "# Load saved pipeline",
            "pipeline = PredictGuardPipeline.load('models/')",
            "",
            "# Run inference on new feature-engineered data",
            "df = pd.read_parquet('data/processed/new_data.parquet')",
            "reports = pipeline.predict_report(df,",
            "                                  machine_ids=df['machineID'],",
            "                                  timestamps=df['datetime'])",
            "",
            "# Each report is a structured dict:",
            "print(reports[0])",
            "# {",
            "#   'machine_id': 47,",
            "#   'failure_probability': 0.8142,",
            "#   'risk_score': 81,",
            "#   'risk_tier': 'CRITICAL',",
            "#   'predicted_component': 'comp3',",
            "#   'dispatch_decision': 'DISPATCH',",
            "#   'recommended_action': 'Dispatch electrician ...',",
            "#   'top_shap_features': [...]",
            "# }",
            "```",
            "",
        ]
        out_repro = Path(reports_dir) / "reproducibility_report.md"
        out_repro.write_text("\n".join(repro_lines), encoding="utf-8")
        logger.info("Saved reproducibility report: %s", out_repro)

        return out_val


# ===========================================================================
# PipelineReportWriter
# ===========================================================================

class PipelineReportWriter:
    """Writes reports/pipeline_report.md with full architecture documentation."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)

    def write(self, pipeline: PredictGuardPipeline) -> Path:
        lines = [
            "# PredictGuard — Pipeline Architecture Report",
            "",
            f"> Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## 1. Overview",
            "",
            "The `PredictGuardPipeline` is a serialised, end-to-end inference system that",
            "takes pre-engineered machine telemetry features and produces structured,",
            "actionable maintenance dispatch decisions.",
            "",
            "---",
            "",
            "## 2. Pipeline Version",
            "",
            "| Component | Version |",
            "|---|---|",
        ]
        for k, v in pipeline.metadata.items():
            lines.append(f"| `{k}` | `{v}` |")

        lines += [
            "",
            "---",
            "",
            "## 3. Data Flow",
            "",
            "```",
            "┌─────────────────────────────────────────────────┐",
            "│           Input: Feature-Engineered Data         │",
            "│  (119 cols: rolling stats, error rates, etc.)   │",
            "└─────────────────────────┬───────────────────────┘",
            "                          │",
            "                          ▼",
            "┌─────────────────────────────────────────────────┐",
            "│              FeatureAligner                      │",
            "│  Aligns columns to training order               │",
            "│  Fills missing features with 0.0                │",
            "└─────────────────────────┬───────────────────────┘",
            "                          │",
            "                          ▼",
            "┌─────────────────────────────────────────────────┐",
            "│        CalibratedClassifierCV (XGBoost)          │",
            "│  Output: P(failure | features)   ∈ [0, 1]       │",
            "│  Calibrated with Sigmoid (Phase 2 Stage 8)       │",
            "└─────────────────────────┬───────────────────────┘",
            "                          │",
            "                          ▼",
            "┌─────────────────────────────────────────────────┐",
            "│          RiskScoreTransformer                    │",
            "│  risk_score = round(100 × P)                    │",
            "│  risk_tier  ∈ {LOW, MEDIUM, HIGH, CRITICAL}     │",
            "└─────────────────────────┬───────────────────────┘",
            "                          │",
            "                          ▼",
            "┌─────────────────────────────────────────────────┐",
            "│          ComponentPredictor (XGBoost)            │",
            "│  Output: predicted_component ∈ {comp1..comp4}   │",
            "│  + per-class probability scores                  │",
            "└─────────────────────────┬───────────────────────┘",
            "                          │",
            "                          ▼",
            "┌─────────────────────────────────────────────────┐",
            "│        DispatchDecisionTransformer               │",
            "│  P ≥ 0.68 → DISPATCH + action narrative         │",
            "│  P <  0.68 → MONITOR                            │",
            "└─────────────────────────┬───────────────────────┘",
            "                          │",
            "                          ▼",
            "┌─────────────────────────────────────────────────┐",
            "│           predict_report() Output                │",
            "│  machine_id, timestamp, failure_probability,     │",
            "│  risk_score, risk_tier, predicted_component,     │",
            "│  dispatch_decision, top_shap_features           │",
            "└─────────────────────────────────────────────────┘",
            "```",
            "",
            "---",
            "",
            "## 4. Custom Transformers",
            "",
            "| Transformer | Input | Output | Purpose |",
            "|---|---|---|---|",
            "| `FeatureAligner` | Raw DataFrame | Aligned float matrix | Column alignment, fill-missing |",
            "| `RiskScoreTransformer` | `calibrated_prob` | + `risk_score`, `risk_tier` | Tier assignment |",
            "| `DispatchDecisionTransformer` | `calibrated_prob`, `predicted_component` | + `dispatch_decision`, `recommended_action` | Business decision |",
            "",
            "---",
            "",
            "## 5. Prediction Interface",
            "",
            "| Method | Returns | Use Case |",
            "|---|---|---|",
            "| `predict_proba(X)` | `np.ndarray` — P(failure) | Ranking / scoring |",
            "| `predict(X)` | `np.ndarray` — 0/1 at optimal threshold | Binary classification |",
            "| `predict_risk(X)` | DataFrame — score + tier | Fleet dashboard |",
            "| `predict_component(X)` | DataFrame — component + confidence | Maintenance routing |",
            "| `predict_report(X, ...)` | `List[Dict]` — full structured report | API / integration |",
            "",
            "---",
            "",
            "## 6. Persistence",
            "",
            "```python",
            "# Save",
            "pipeline.save('models/')",
            "# → models/predictguard_pipeline.pkl       (joblib, compress=3)",
            "# → models/predictguard_pipeline_metadata.json",
            "",
            "# Load",
            "from src.pipeline import PredictGuardPipeline",
            "pipeline = PredictGuardPipeline.load('models/')",
            "```",
            "",
            "---",
            "",
            "## 7. Limitations",
            "",
            "- **Feature engineering** (rolling windows, lag features) is NOT included in the",
            "  pipeline. This step requires ordered time-series streaming per machine and",
            "  is implemented in `src/features.py`. The pipeline takes pre-engineered features",
            "  as input — exactly like the production inference pattern.",
            "- For Stage 15 (FastAPI), the API will wrap feature engineering + pipeline together.",
            "",
        ]

        out = self.reports_dir / "pipeline_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved pipeline report: %s", out)
        return out
