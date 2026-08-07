"""
calibration.py
==============
Production-grade probability calibration module for PredictGuard — Phase 2.
Covers Stage 7 (Calibration Evaluation) & Stage 8 (Probability Calibration).

Why Calibration Matters in Predictive Maintenance
-------------------------------------------------
High ROC-AUC (0.999) or PR-AUC (0.973) does NOT mean predicted probabilities
are trustworthy. Models trained on class-imbalanced datasets with class-weighting
(e.g., scale_pos_weight=51.4) often produce uncalibrated probabilities (pushed
toward extreme 0s and 1s).

If PredictGuard predicts a 70% failure probability for a wind turbine gearbox over
the next 24 hours, maintenance dispatch teams require that approximately 70% of
turbines with that risk score actually experience a failure.

Classes exported:
  1. CalibrationEvaluator    — Computes ECE, MCE, Brier Score, Log Loss, Slope, Intercept.
  2. ProbabilityCalibrator   — Fits Isotonic & Sigmoid calibration with StratifiedGroupKFold.
  3. ReliabilityAnalyzer     — Generates probability audit tables and reliability metrics.

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Non-feature metadata columns to exclude during model predictions
NON_FEATURE_COLS = {
    "datetime",
    "machineID",
    "y_failure",
    "failure_component",
    "time_to_failure_hours",
}

sns.set_theme(
    style="darkgrid",
    palette="muted",
    rc={"figure.dpi": 150, "axes.titlesize": 14, "axes.labelsize": 12},
)


# ===========================================================================
# 1. CalibrationEvaluator
# ===========================================================================

class CalibrationEvaluator:
    """
    Computes probability calibration metrics for binary classification models.

    Metrics:
      - Brier Score Loss: Mean squared error of predicted probabilities.
      - Log Loss: Cross-entropy loss.
      - Expected Calibration Error (ECE): Weighted mean absolute difference
        between predicted confidence and true empirical accuracy across bins.
      - Maximum Calibration Error (MCE): Maximum bin calibration error.
      - Calibration Slope & Intercept: Logistic regression of logit(y_prob) vs y_true.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    @staticmethod
    def compute_ece_mce(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground truth labels (0 or 1).
        y_prob : np.ndarray
            Predicted probabilities for class 1.
        n_bins : int
            Number of equal-width probability bins.

        Returns
        -------
        ece : float
        mce : float
        bin_df : pd.DataFrame
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0
        n_samples = len(y_true)
        bin_records = []

        for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Include right edge for final bin
            if i == n_bins - 1:
                in_bin = (y_prob >= lower) & (y_prob <= upper)
            else:
                in_bin = (y_prob >= lower) & (y_prob < upper)

            bin_size = int(np.sum(in_bin))

            if bin_size > 0:
                avg_confidence = float(np.mean(y_prob[in_bin]))
                avg_accuracy   = float(np.mean(y_true[in_bin]))
                abs_error      = abs(avg_confidence - avg_accuracy)

                ece += (bin_size / n_samples) * abs_error
                mce  = max(mce, abs_error)
            else:
                avg_confidence = (lower + upper) / 2.0
                avg_accuracy   = 0.0
                abs_error      = 0.0

            bin_records.append({
                "bin": i + 1,
                "bin_lower": round(float(lower), 3),
                "bin_upper": round(float(upper), 3),
                "count": bin_size,
                "confidence": round(avg_confidence, 4),
                "accuracy": round(avg_accuracy, 4),
                "abs_error": round(abs_error, 4),
            })

        bin_df = pd.DataFrame(bin_records)
        return float(ece), float(mce), bin_df

    @staticmethod
    def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute Brier Score Loss (lower is better, 0.0 = perfect)."""
        return float(brier_score_loss(y_true, y_prob))

    @staticmethod
    def compute_calibration_slope_intercept(
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute calibration slope and intercept via logistic regression.
        logit(p) = log(p / (1 - p)).
        Perfect calibration: slope = 1.0, intercept = 0.0.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
        logit_p = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)

        try:
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(logit_p, y_true)
            slope = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])
        except Exception as e:
            logger.warning("Calibration slope/intercept fit failed: %s", e)
            slope, intercept = 1.0, 0.0

        return slope, intercept

    def evaluate(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fit_time_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate full calibration metrics for a single model prediction set.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
        y_pred = (y_prob >= 0.5).astype(int)

        brier = self.compute_brier_score(y_true, y_prob)
        loss = float(log_loss(y_true, y_prob))
        ece, mce, bin_df = self.compute_ece_mce(y_true, y_prob, n_bins=self.n_bins)
        slope, intercept = self.compute_calibration_slope_intercept(y_true, y_prob)

        pr_auc  = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        prec    = float(precision_score(y_true, y_pred, zero_division=0))
        rec     = float(recall_score(y_true, y_pred, zero_division=0))

        # Under/Overconfidence classification
        mean_prob = float(np.mean(y_prob))
        observed_freq = float(np.mean(y_true))
        bias_diff = mean_prob - observed_freq
        if bias_diff > 0.02:
            calibration_status = "Over-confident (predicted prob > actual frequency)"
        elif bias_diff < -0.02:
            calibration_status = "Under-confident (predicted prob < actual frequency)"
        else:
            calibration_status = "Well-calibrated"

        return {
            "model_name": model_name,
            "brier_score": round(brier, 4),
            "log_loss": round(loss, 4),
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "mean_prob": round(mean_prob, 4),
            "observed_freq": round(observed_freq, 4),
            "status": calibration_status,
            "bin_table": bin_df,
            "fit_time_sec": round(fit_time_sec, 2),
        }


# ===========================================================================
# 2. ProbabilityCalibrator
# ===========================================================================

class ProbabilityCalibrator:
    """
    Fits post-hoc probability calibration models (Isotonic or Sigmoid/Platt)
    using StratifiedGroupKFold on the Development set to guarantee ZERO data leakage.

    Key principles:
      - Calibration MUST be fitted ONLY on development machines.
      - Uses ``CalibratedClassifierCV`` with ``cv=StratifiedGroupKFold(5)``.
      - Compares Isotonic Regression vs Sigmoid (Platt Scaling).
    """

    def __init__(self, n_folds: int = 5, random_seed: int = 42):
        self.n_folds = n_folds
        self.random_seed = random_seed

    def fit_calibrated_model(
        self,
        base_model: Any,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        groups_dev: pd.Series,
        method: str = "isotonic",
    ) -> CalibratedClassifierCV:
        """
        Fit post-hoc calibrator using StratifiedGroupKFold on Development set.

        Parameters
        ----------
        base_model : sklearn estimator / pipeline / XGBClassifier
            Pre-configured base model architecture.
        X_dev : pd.DataFrame
            Development feature matrix.
        y_dev : pd.Series
            Development target.
        groups_dev : pd.Series
            machineID for grouped split.
        method : str
            'isotonic' or 'sigmoid' (Platt scaling).

        Returns
        -------
        calibrated_model : CalibratedClassifierCV
        """
        logger.info("Fitting '%s' post-hoc calibration via CalibratedClassifierCV ...", method)

        try:
            cv_splitter = StratifiedGroupKFold(n_splits=self.n_folds)
            splits = list(cv_splitter.split(X_dev, y_dev, groups=groups_dev))
        except Exception:
            logger.warning("StratifiedGroupKFold failed — falling back to GroupKFold.")
            cv_splitter = GroupKFold(n_splits=self.n_folds)
            splits = list(cv_splitter.split(X_dev, y_dev, groups=groups_dev))

        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=splits,
            n_jobs=-1,
        )

        t0 = time.time()
        X_np = X_dev.values if hasattr(X_dev, "values") else np.array(X_dev)
        y_np = y_dev.values if hasattr(y_dev, "values") else np.array(y_dev)

        calibrated_model.fit(X_np, y_np)
        elapsed = time.time() - t0

        logger.info("'%s' calibration fit complete in %.2f seconds.", method, elapsed)
        return calibrated_model


# ===========================================================================
# 3. ReliabilityAnalyzer
# ===========================================================================

class ReliabilityAnalyzer:
    """
    Generates probability audit tables and empirical reliability analytics.
    """

    @staticmethod
    def audit_probabilities(
        y_true: pd.Series,
        raw_probs: np.ndarray,
        calibrated_probs: np.ndarray,
        machine_ids: Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
        n_samples: int = 15,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a probability audit sample table showing Raw vs Calibrated probabilities
        alongside actual ground truth outcomes.
        """
        df_audit = pd.DataFrame({
            "machineID": machine_ids.values if machine_ids is not None else np.arange(len(y_true)),
            "datetime": timestamps.values if timestamps is not None else np.zeros(len(y_true)),
            "raw_prob": np.round(raw_probs, 4),
            "calibrated_prob": np.round(calibrated_probs, 4),
            "prob_delta": np.round(calibrated_probs - raw_probs, 4),
            "actual_failure": y_true.values if hasattr(y_true, "values") else np.array(y_true),
        })

        # Stratified sample: include high risk, medium risk, and low risk examples
        rng = np.random.default_rng(random_seed)
        high_risk = df_audit[df_audit["raw_prob"] >= 0.70]
        med_risk  = df_audit[(df_audit["raw_prob"] >= 0.30) & (df_audit["raw_prob"] < 0.70)]
        low_risk  = df_audit[df_audit["raw_prob"] < 0.30]

        samples = []
        n_each = max(1, n_samples // 3)

        if not high_risk.empty:
            idx = rng.choice(high_risk.index, size=min(n_each, len(high_risk)), replace=False)
            samples.append(high_risk.loc[idx])
        if not med_risk.empty:
            idx = rng.choice(med_risk.index, size=min(n_each, len(med_risk)), replace=False)
            samples.append(med_risk.loc[idx])
        if not low_risk.empty:
            idx = rng.choice(low_risk.index, size=min(n_each, len(low_risk)), replace=False)
            samples.append(low_risk.loc[idx])

        audit_sample = pd.concat(samples).sort_values("raw_prob", ascending=False).reset_index(drop=True)
        return audit_sample


# ===========================================================================
# 4. Visualizations — Publication-Quality Figures
# ===========================================================================

def _save_fig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_calibration_dashboard(
    eval_raw_models: Dict[str, Dict[str, Any]],
    raw_eval: Dict[str, Any],
    iso_eval: Dict[str, Any],
    sig_eval: Dict[str, Any],
    y_test: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    Generate unified 4-panel Calibration Dashboard graphic for README / reports.

    Panels:
      1. Top-Left: Raw Calibration Curves (LR vs RF vs XGBoost).
      2. Top-Right: Before vs After Calibration Curves (Raw XGB vs Isotonic vs Sigmoid).
      3. Bottom-Left: Reliability Diagram & Confidence Histogram (Calibrated XGBoost).
      4. Bottom-Right: Calibration Error Metrics Bar Chart (Brier, ECE, MCE Comparison).
    """
    figs_dir = Path(figures_dir)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("PredictGuard — Phase 2 Calibration & Probability Trust Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_ba  = fig.add_subplot(gs[0, 1])
    ax_rel = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1])

    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#ff7f0e", "XGBoost": "#2ca02c"}

    # Panel 1: Raw Calibration Curves
    ax_raw.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)
    for m_name, res in eval_raw_models.items():
        prob_true, prob_pred = calibration_curve(y_test, res.get("y_prob", raw_probs), n_bins=10)
        ax_raw.plot(prob_pred, prob_true, "s-", label=f"{m_name} (ECE={res['ece']:.4f})",
                    color=colors.get(m_name, "purple"), linewidth=2, markersize=6)
    ax_raw.set_xlabel("Mean Predicted Probability")
    ax_raw.set_ylabel("Fraction of Positives (Observed)")
    ax_raw.set_title("Raw Model Calibration Curves (Stage 7)", fontsize=12, fontweight="bold")
    ax_raw.legend(loc="upper left", fontsize=9)

    # Panel 2: Before vs After Calibration Curves
    ax_ba.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)
    # Raw XGB
    pt, pp = calibration_curve(y_test, raw_probs, n_bins=10)
    ax_ba.plot(pp, pt, "o--", label=f"Raw XGBoost (ECE={raw_eval['ece']:.4f})", color="#2ca02c", alpha=0.7)
    # Isotonic
    pt_i, pp_i = calibration_curve(y_test, iso_eval["y_prob"], n_bins=10)
    ax_ba.plot(pp_i, pt_i, "s-", label=f"Isotonic XGBoost (ECE={iso_eval['ece']:.4f})", color="#d62728", linewidth=2)
    # Sigmoid
    pt_s, pp_s = calibration_curve(y_test, sig_eval["y_prob"], n_bins=10)
    ax_ba.plot(pp_s, pt_s, "^-.", label=f"Sigmoid XGBoost (ECE={sig_eval['ece']:.4f})", color="#9467bd", linewidth=2)

    ax_ba.set_xlabel("Mean Predicted Probability")
    ax_ba.set_ylabel("Fraction of Positives (Observed)")
    ax_ba.set_title("Post-Hoc Calibration Curves (Stage 8)", fontsize=12, fontweight="bold")
    ax_ba.legend(loc="upper left", fontsize=9)

    # Panel 3: Reliability Diagram & Confidence Histogram for Calibrated Model
    ax_rel_twin = ax_rel.twinx()
    bin_table = iso_eval["bin_table"]
    bin_centers = (bin_table["bin_lower"] + bin_table["bin_upper"]) / 2.0
    width = 0.08

    # Accuracy vs Confidence bars
    ax_rel.bar(bin_centers, bin_table["accuracy"], width=width, alpha=0.6, color="#1f77b4", label="Observed Accuracy")
    ax_rel.plot(bin_centers, bin_table["confidence"], "ro-", label="Predicted Confidence", linewidth=2)
    ax_rel.plot([0, 1], [0, 1], "k--", label="Ideal Diagonal")

    # Probability density histogram on right axis
    ax_rel_twin.hist(calibrated_probs, bins=20, color="gray", alpha=0.25, density=True)
    ax_rel_twin.set_ylabel("Sample Density (Gray)", color="gray")
    ax_rel_twin.grid(False)

    ax_rel.set_xlabel("Predicted Probability Bin")
    ax_rel.set_ylabel("Empirical Accuracy / Confidence")
    ax_rel.set_title("Reliability Diagram & Confidence Histogram (Isotonic XGBoost)", fontsize=12, fontweight="bold")
    ax_rel.legend(loc="upper left", fontsize=9)

    # Panel 4: Calibration Error Bar Chart (Brier, ECE, MCE)
    metric_names = ["Brier Score", "ECE", "MCE", "Log Loss"]
    raw_vals = [raw_eval["brier_score"], raw_eval["ece"], raw_eval["mce"], raw_eval["log_loss"]]
    iso_vals = [iso_eval["brier_score"], iso_eval["ece"], iso_eval["mce"], iso_eval["log_loss"]]
    sig_vals = [sig_eval["brier_score"], sig_eval["ece"], sig_eval["mce"], sig_eval["log_loss"]]

    x = np.arange(len(metric_names))
    w = 0.25

    ax_err.bar(x - w, raw_vals, w, label="Raw XGBoost", color="#2ca02c", alpha=0.7)
    ax_err.bar(x,     iso_vals, w, label="Isotonic",    color="#d62728", alpha=0.9)
    ax_err.bar(x + w, sig_vals, w, label="Sigmoid",     color="#9467bd", alpha=0.9)

    for i in range(len(metric_names)):
        ax_err.text(x[i] - w, raw_vals[i] + 0.002, f"{raw_vals[i]:.3f}", ha="center", fontsize=7)
        ax_err.text(x[i],     iso_vals[i] + 0.002, f"{iso_vals[i]:.3f}", ha="center", fontsize=7)
        ax_err.text(x[i] + w, sig_vals[i] + 0.002, f"{sig_vals[i]:.3f}", ha="center", fontsize=7)

    ax_err.set_xticks(x)
    ax_err.set_xticklabels(metric_names)
    ax_err.set_ylabel("Error Metric Value (Lower is Better)")
    ax_err.set_title("Calibration Error Comparison Across Models", fontsize=12, fontweight="bold")
    ax_err.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, figs_dir / "calibration_dashboard.png")


def plot_reliability_diagrams(
    eval_results: Dict[str, Dict[str, Any]],
    figures_dir: Path,
) -> None:
    """
    Generates standalone reliability diagrams for all evaluated models.
    """
    figs_dir = Path(figures_dir)
    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("PredictGuard — Stage 7 & 8 Reliability Diagrams", fontsize=14, fontweight="bold")

    for ax, (m_name, res) in zip(axes, eval_results.items()):
        bin_df = res["bin_table"]
        centers = (bin_df["bin_lower"] + bin_df["bin_upper"]) / 2.0
        ax.plot([0, 1], [0, 1], "k--", label="Ideal")
        ax.bar(centers, bin_df["accuracy"], width=0.08, alpha=0.5, color="#1f77b4", label="Accuracy")
        ax.plot(centers, bin_df["confidence"], "ro-", label="Confidence")

        ax.set_title(f"{m_name}\nECE={res['ece']:.4f} | Brier={res['brier_score']:.4f}", fontsize=10)
        ax.set_xlabel("Predicted Bin")
        ax.set_ylabel("Observed Accuracy")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, figs_dir / "reliability_diagrams.png")


def plot_probability_histograms(
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    figures_dir: Path,
) -> None:
    """
    Probability distribution comparison histogram (Raw vs Calibrated).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("PredictGuard — Probability Distribution: Raw vs Calibrated", fontsize=13, fontweight="bold")

    ax.hist(raw_probs, bins=40, alpha=0.5, color="#2ca02c", label="Raw XGBoost Probabilities", density=True)
    ax.hist(calibrated_probs, bins=40, alpha=0.5, color="#d62728", label="Calibrated Isotonic Probabilities", density=True)

    ax.set_xlabel("Predicted Probability of Failure")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")
    ax.set_title("Shift in Predicted Risk Distribution Post-Calibration", fontsize=11)

    fig.tight_layout()
    _save_fig(fig, Path(figures_dir) / "probability_histograms.png")


# ===========================================================================
# 5. Markdown Calibration Report Generator
# ===========================================================================

def generate_calibration_report(
    raw_eval_df: pd.DataFrame,
    before_after_df: pd.DataFrame,
    audit_sample_df: pd.DataFrame,
    best_method: str,
    output_path: Path,
) -> None:
    """Generate reports/calibration_report.md markdown file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# PredictGuard — Probability Calibration & Trust Report",
        "",
        f"> **Phase 2: Stage 7 (Calibration Evaluation) & Stage 8 (Probability Calibration)**  ",
        f"> Generated: {now}",
        "",
        "---",
        "",
        "## 1. What Calibration Means & Why It Matters",
        "",
        "In predictive maintenance systems, high **ROC-AUC (0.999)** or **PR-AUC (0.973)** guarantees",
        "good ranking order, but does **NOT** ensure trustworthy probabilities.",
        "",
        "When PredictGuard outputs a **0.70 failure probability** for a machine over the next 24 hours,",
        "fleet operators and maintenance dispatch teams require that **approximately 70 out of 100** machines",
        "receiving that score actually fail. Raw probabilities from tree models trained with `scale_pos_weight`",
        "tend to be **over-confident**, distorting financial risk calculations.",
        "",
        "**Phase 2 Goal**: Transform raw model outputs into well-calibrated, operational risk probabilities.",
        "",
        "---",
        "",
        "## 2. Stage 7 — Raw Model Calibration Benchmarks",
        "",
        "| Model Name | Brier Score (Lower Best) | ECE (Lower Best) | MCE | Log Loss | Calibration Status |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in raw_eval_df.iterrows():
        lines.append(
            f"| **{row['model_name']}** | {row['brier_score']:.4f} | **{row['ece']:.4f}** | "
            f"{row['mce']:.4f} | {row['log_loss']:.4f} | {row['status']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Stage 8 — Post-Hoc Calibration Comparison",
        "",
        "Calibration was fitted **ONLY on the 80 Development machines** using `CalibratedClassifierCV` with",
        "`StratifiedGroupKFold(5)` — the 20 Final Test machines remained completely untouched.",
        "",
        "| Model Variant | Calibration Method | Brier Score | ECE | MCE | Log Loss | PR-AUC | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, row in before_after_df.iterrows():
        lines.append(
            f"| **{row['model_name']}** | {row.get('method', 'None')} | **{row['brier_score']:.4f}** | "
            f"**{row['ece']:.4f}** | {row['mce']:.4f} | {row['log_loss']:.4f} | "
            f"{row['pr_auc']:.4f} | {row['roc_auc']:.4f} |"
        )

    lines += [
        "",
        f"### Selected Calibration Method: **{best_method.upper()} REGRESSION**",
        "",
        "---",
        "",
        "## 4. Probability Audit Table (Raw vs Calibrated Sample)",
        "",
        "| Machine ID | Raw Risk Prob | Calibrated Risk Prob | Risk Delta | Actual Outcome |",
        "|---|---|---|---|---|",
    ]

    for _, row in audit_sample_df.iterrows():
        outcome_str = "🚨 FAIL (1)" if row["actual_failure"] == 1 else "✅ NORMAL (0)"
        lines.append(
            f"| Machine {int(row['machineID'])} | `{row['raw_prob']:.4f}` | "
            f"**`{row['calibrated_prob']:.4f}`** | `{row['prob_delta']:+.4f}` | {outcome_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Important Engineering Clarification: SHAP & Raw Models",
        "",
        "> ⚠️ **Why SHAP Explainability (Phase 3) Must Use the RAW Model:**  ",
        "> Post-hoc calibration wrappers (Isotonic/Sigmoid) apply a monotonic 1D transformation `g(f(x))`",
        "> to the final probability output. Feature importance and TreeSHAP interaction values must be computed",
        "> directly on the underlying **RAW tree structure** (`raw_model.pkl`) to preserve exact game-theoretic mathematical axioms.",
        "",
        "---",
        "",
        "## 6. Output Artifacts Saved",
        "",
        "| Artifact | Path |",
        "|---|---|",
        "| Raw XGBoost Model | `models/raw_model.pkl` |",
        "| Calibrated XGBoost Model | `models/calibrated_model.pkl` |",
        "| Calibration Metadata Card | `models/calibration_metadata.json` |",
        "| Raw Calibration Metrics | `reports/calibration_metrics.csv` |",
        "| Before/After Metrics | `reports/calibration_before_after.csv` |",
        "| Unified Calibration Dashboard | `reports/figures/calibration_dashboard.png` |",
        "| Reliability Diagrams | `reports/figures/reliability_diagrams.png` |",
        "| Probability Histograms | `reports/figures/probability_histograms.png` |",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved calibration report to %s", output_path)
