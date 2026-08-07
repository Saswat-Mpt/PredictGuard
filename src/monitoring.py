"""
monitoring.py
=============
PredictGuard — Phase 4 Stage 13: ML Experiment Monitoring with MLflow.

Provides:
    ExperimentTracker   — creates MLflow runs, logs metrics, params, artifacts
    ArtifactLogger      — logs figures, reports, and model files to MLflow
    MLflowSummaryWriter — generates reports/mlflow_summary.md

Design principle:
    Rather than re-running training, this module reads from the already-computed
    CSVs and JSON artifacts produced by Stages 1–12 and logs them into MLflow.
    This is the correct production pattern: MLflow captures every completed run.

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import platform
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bootstrap: ensure user site-packages is on sys.path (handles pip --user installs)
import site
for _sp in site.getusersitepackages() if isinstance(site.getusersitepackages(), list) else [site.getusersitepackages()]:
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "PredictGuard"
TRACKING_URI_TEMPLATE = "file:///{}/mlruns"


# ===========================================================================
# 1. ExperimentTracker
# ===========================================================================

class ExperimentTracker:
    """
    Manages MLflow experiment runs for PredictGuard.

    Creates a local file-based MLflow tracking server under mlruns/.
    Each call to start_run() creates a new versioned run with timestamp + commit hash.

    Usage:
        tracker = ExperimentTracker(project_root=Path("."))
        tracker.start_run("Phase1_Training")
        tracker.log_params({...})
        tracker.log_metrics({...})
        tracker.end_run()
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        # MLflow 3.x dropped file-store; use SQLite backend
        db_path = self.project_root / "mlruns" / "mlflow.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{db_path.as_posix()}"
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self._active_run: Optional[mlflow.ActiveRun] = None
        logger.info("MLflow tracking URI: %s", tracking_uri)
        logger.info("MLflow experiment: %s", EXPERIMENT_NAME)

    # ------------------------------------------------------------------
    def start_run(self, run_name: str) -> str:
        """Start a new MLflow run. Returns run_id."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        full_name = f"{run_name}_{ts}"
        commit = self._get_git_commit()
        tags = {
            "project": "PredictGuard",
            "run_name": run_name,
            "timestamp": ts,
            "git_commit": commit,
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "hostname": socket.gethostname(),
        }
        self._active_run = mlflow.start_run(run_name=full_name, tags=tags)
        run_id = self._active_run.info.run_id
        logger.info("MLflow run started: %s (id=%s)", full_name, run_id[:8])
        return run_id

    # ------------------------------------------------------------------
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dict of parameters. Values are converted to strings."""
        safe = {k: str(v) for k, v in params.items()}
        mlflow.log_params(safe)
        logger.info("Logged %d params to MLflow.", len(safe))

    # ------------------------------------------------------------------
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dict of numeric metrics."""
        numeric = {k: float(v) for k, v in metrics.items() if v is not None}
        mlflow.log_metrics(numeric, step=step)
        logger.info("Logged %d metrics to MLflow.", len(numeric))

    # ------------------------------------------------------------------
    def log_metric_series(
        self, metric_name: str, values: List[float], start_step: int = 0
    ) -> None:
        """Log a time-series metric (e.g., cost vs threshold sweep)."""
        for i, val in enumerate(values, start=start_step):
            mlflow.log_metric(metric_name, float(val), step=i)

    # ------------------------------------------------------------------
    def end_run(self) -> None:
        if self._active_run:
            mlflow.end_run()
            logger.info("MLflow run ended.")
            self._active_run = None

    # ------------------------------------------------------------------
    @staticmethod
    def _get_git_commit() -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_run()


# ===========================================================================
# 2. ArtifactLogger
# ===========================================================================

class ArtifactLogger:
    """
    Logs files and directories to the active MLflow run.

    Supports:
        - Individual files (figures, reports, model PKLs)
        - Entire directories (reports/figures/)
        - Auto-discovery of all project artifacts
    """

    def __init__(self, figures_dir: Path, reports_dir: Path, models_dir: Path) -> None:
        self.figures_dir = Path(figures_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir = Path(models_dir)

    # ------------------------------------------------------------------
    def log_figures(self, subfolder: str = "figures") -> int:
        """Log all PNG figures from reports/figures/."""
        logged = 0
        for png in sorted(self.figures_dir.glob("*.png")):
            mlflow.log_artifact(str(png), artifact_path=subfolder)
            logged += 1
        logger.info("Logged %d figures to MLflow.", logged)
        return logged

    # ------------------------------------------------------------------
    def log_reports(self, subfolder: str = "reports") -> int:
        """Log all markdown and CSV reports."""
        logged = 0
        for ext in ["*.md", "*.csv", "*.json"]:
            for f in sorted(self.reports_dir.glob(ext)):
                mlflow.log_artifact(str(f), artifact_path=subfolder)
                logged += 1
        logger.info("Logged %d report files to MLflow.", logged)
        return logged

    # ------------------------------------------------------------------
    def log_models(self, subfolder: str = "models") -> int:
        """Log model PKL files to MLflow artifacts."""
        logged = 0
        for pkl in sorted(self.models_dir.glob("*.pkl")):
            mlflow.log_artifact(str(pkl), artifact_path=subfolder)
            logged += 1
        for meta in sorted(self.models_dir.glob("*.json")):
            mlflow.log_artifact(str(meta), artifact_path=subfolder)
            logged += 1
        logger.info("Logged %d model files to MLflow.", logged)
        return logged

    # ------------------------------------------------------------------
    def log_file(self, path: Path, subfolder: str = "") -> None:
        """Log a single file."""
        mlflow.log_artifact(str(path), artifact_path=subfolder or None)
        logger.info("Logged artifact: %s", path.name)


# ===========================================================================
# 3. MetricLoader
# ===========================================================================

class MetricLoader:
    """
    Reads metrics from existing Stage 1–12 output CSVs and JSONs.
    Produces clean dicts ready for mlflow.log_metrics().
    """

    def __init__(self, reports_dir: Path, models_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.models_dir = Path(models_dir)

    # ------------------------------------------------------------------
    def load_training_metrics(self) -> Dict[str, float]:
        """Load CV and training metrics from Stage 5–6 outputs."""
        metrics: Dict[str, float] = {}
        cv_csv = self.reports_dir / "cv_results.csv"
        if cv_csv.exists():
            df = pd.read_csv(cv_csv)
            xgb_row = df[df.get("model_name", df.iloc[:, 0]).astype(str).str.contains("XGBoost")]
            if not xgb_row.empty:
                row = xgb_row.iloc[0]
                for col in ["mean_pr_auc", "std_pr_auc", "mean_roc_auc", "mean_f1", "mean_recall"]:
                    if col in row.index:
                        metrics[f"cv_{col}"] = float(row[col])

        # Best model metadata
        meta_path = self.models_dir / "xgboost_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            for k in ["pr_auc", "roc_auc", "f1", "recall", "precision",
                      "brier_score", "log_loss"]:
                if k in meta:
                    metrics[f"train_{k}"] = float(meta[k])
        return metrics

    # ------------------------------------------------------------------
    def load_calibration_metrics(self) -> Dict[str, float]:
        """Load Phase 2 calibration metrics."""
        metrics: Dict[str, float] = {}
        cal_meta = self.models_dir / "calibration_metadata.json"
        if cal_meta.exists():
            with open(cal_meta) as f:
                meta = json.load(f)
            metrics.update({
                "cal_brier_before": float(meta.get("brier_before", 0)),
                "cal_brier_after": float(meta.get("brier_after", 0)),
                "cal_ece_before": float(meta.get("ece_before", 0)),
                "cal_ece_after": float(meta.get("ece_after", 0)),
                "cal_pr_auc_before": float(meta.get("pr_auc_before", 0)),
                "cal_pr_auc_after": float(meta.get("pr_auc_after", 0)),
            })
        return metrics

    # ------------------------------------------------------------------
    def load_component_metrics(self) -> Dict[str, float]:
        """Load Stage 10 component classifier metrics."""
        metrics: Dict[str, float] = {}
        comp_csv = self.reports_dir / "component_metrics.csv"
        if comp_csv.exists():
            df = pd.read_csv(comp_csv)
            for _, row in df.iterrows():
                comp = row["component"]
                metrics[f"comp_{comp}_precision"] = float(row["precision"])
                metrics[f"comp_{comp}_recall"] = float(row["recall"])
                metrics[f"comp_{comp}_f1"] = float(row["f1_score"])
        return metrics

    # ------------------------------------------------------------------
    def load_cost_metrics(self) -> Dict[str, Any]:
        """Load Stage 12 cost and dispatch metrics."""
        metrics: Dict[str, float] = {}
        params: Dict[str, Any] = {}
        ot_path = self.reports_dir / "optimal_threshold.json"
        if ot_path.exists():
            with open(ot_path) as f:
                ot = json.load(f)
            params["optimal_threshold"] = ot.get("optimal_threshold", 0)
            params["cost_C_FN"] = ot["cost_matrix"]["C_FN"]
            params["cost_C_FP"] = ot["cost_matrix"]["C_FP"]
            m = ot.get("metrics_at_optimal", {})
            metrics.update({
                "decision_recall": float(m.get("recall", 0)),
                "decision_precision": float(m.get("precision", 0)),
                "decision_f1": float(m.get("f1", 0)),
                "decision_expected_cost": float(m.get("expected_cost", 0)),
                "decision_dispatch_rate": float(m.get("dispatch_rate", 0)),
            })
        return {"metrics": metrics, "params": params}

    # ------------------------------------------------------------------
    def load_fleet_metrics(self) -> Dict[str, float]:
        """Load Stage 11 fleet summary metrics."""
        metrics: Dict[str, float] = {}
        fleet_csv = self.reports_dir / "fleet_risk_summary.csv"
        if fleet_csv.exists():
            df = pd.read_csv(fleet_csv)
            metrics["fleet_n_machines"] = float(len(df))
            metrics["fleet_avg_risk_score"] = float(df["avg_risk_score"].mean())
            metrics["fleet_max_risk_score"] = float(df["max_risk_score"].max())
            metrics["fleet_avg_failure_rate"] = float(df["failure_rate"].mean())
        return metrics

    # ------------------------------------------------------------------
    def load_all_params(self) -> Dict[str, Any]:
        """Load training parameters for MLflow logging."""
        params: Dict[str, Any] = {
            "model": "XGBoostClassifier",
            "calibration_method": "sigmoid",
            "cv_folds": 5,
            "feature_count": 119,
            "split_strategy": "machine_level",
            "primary_metric": "pr_auc",
            "random_seed": 42,
            "prediction_horizon_hours": 24,
        }
        # Pull XGBoost hyperparams from metadata
        xgb_meta = self.models_dir / "xgboost_metadata.json"
        if xgb_meta.exists():
            with open(xgb_meta) as f:
                meta = json.load(f)
            for k in ["n_estimators", "max_depth", "learning_rate",
                      "scale_pos_weight", "positive_rate"]:
                if k in meta:
                    params[f"xgb_{k}"] = meta[k]
        return params


# ===========================================================================
# 4. MLflowSummaryWriter
# ===========================================================================

class MLflowSummaryWriter:
    """Writes a human-readable MLflow experiment summary to reports/mlflow_summary.md."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)

    def write(
        self,
        run_id: str,
        run_name: str,
        all_metrics: Dict[str, float],
        all_params: Dict[str, Any],
        artifact_counts: Dict[str, int],
        tracking_uri: str,
    ) -> Path:
        lines = [
            "# PredictGuard — MLflow Experiment Summary",
            "",
            f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## 1. Experiment Configuration",
            "",
            "| Property | Value |",
            "|---|---|",
            f"| Experiment Name | `{EXPERIMENT_NAME}` |",
            f"| Run Name | `{run_name}` |",
            f"| Run ID | `{run_id}` |",
            f"| Tracking URI | `{tracking_uri}` |",
            "",
            "---",
            "",
            "## 2. Logged Parameters",
            "",
            "| Parameter | Value |",
            "|---|---|",
        ]
        for k, v in sorted(all_params.items()):
            lines.append(f"| `{k}` | `{v}` |")

        lines += [
            "",
            "---",
            "",
            "## 3. Logged Metrics",
            "",
            "### Training & CV (Phase 1)",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        train_keys = [k for k in all_metrics if k.startswith("train_") or k.startswith("cv_")]
        for k in sorted(train_keys):
            lines.append(f"| `{k}` | `{all_metrics[k]:.6f}` |")

        lines += ["", "### Calibration (Phase 2)", "", "| Metric | Value |", "|---|---|"]
        cal_keys = [k for k in all_metrics if k.startswith("cal_")]
        for k in sorted(cal_keys):
            lines.append(f"| `{k}` | `{all_metrics[k]:.6f}` |")

        lines += ["", "### Component Classifier (Stage 10)", "", "| Metric | Value |", "|---|---|"]
        comp_keys = [k for k in all_metrics if k.startswith("comp_")]
        for k in sorted(comp_keys):
            lines.append(f"| `{k}` | `{all_metrics[k]:.6f}` |")

        lines += ["", "### Fleet & Decision (Stages 11–12)", "", "| Metric | Value |", "|---|---|"]
        fleet_keys = [k for k in all_metrics if k.startswith("fleet_") or k.startswith("decision_")]
        for k in sorted(fleet_keys):
            lines.append(f"| `{k}` | `{all_metrics[k]:.6f}` |")

        lines += [
            "",
            "---",
            "",
            "## 4. Logged Artifacts",
            "",
            "| Category | Count |",
            "|---|---|",
        ]
        for cat, cnt in artifact_counts.items():
            lines.append(f"| {cat} | {cnt} |")

        lines += [
            "",
            "---",
            "",
            "## 5. How to View",
            "",
            "```bash",
            "# From project root:",
            "mlflow ui",
            "# Then open: http://localhost:5000",
            "```",
            "",
            "Navigate to the **PredictGuard** experiment and click on the run to see:",
            "- All logged parameters",
            "- Interactive metric charts",
            "- Artifact browser (figures, reports, models)",
            "",
        ]

        out = self.reports_dir / "mlflow_summary.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved MLflow summary: %s", out)
        return out
