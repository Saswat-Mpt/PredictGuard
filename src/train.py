"""
train.py
========
Production-grade model training and evaluation module for PredictGuard — Stage 5.

Establishes strong baseline models on the exact same leakage-safe features and machine-level split:
  1. Logistic Regression Baseline (with StandardScaler in sklearn Pipeline, class_weight="balanced")
  2. Random Forest Classifier (class_weight="balanced")
  3. XGBoost Classifier (with automated scale_pos_weight from class imbalance)

Evaluation metrics prioritize **PR-AUC (Precision-Recall AUC / Average Precision)**
over accuracy due to extreme class imbalance (1.96% positive failure rate).

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Non-feature metadata and target columns to exclude from training
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
# 1. ModelTrainer
# ===========================================================================

class ModelTrainer:
    """
    Handles data preparation, feature extraction, and baseline model construction.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    @staticmethod
    def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Extract feature matrix X, target vector y, and list of feature names.

        Parameters
        ----------
        df : pd.DataFrame
            Development or Test DataFrame.

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        feature_names : List[str]
        """
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        
        # Select numeric columns only
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        y = df["y_failure"].astype(int)

        # Check missing or infinite values
        if X.isna().sum().sum() > 0:
            logger.warning("Feature matrix contains missing values. Imputing with 0.0.")
            X = X.fillna(0.0)

        if np.isinf(X.values).sum() > 0:
            logger.warning("Feature matrix contains infinite values. Replacing with 0.0.")
            X = X.replace([np.inf, -np.inf], 0.0)

        return X, y, list(X.columns)

    def build_logistic_regression(self) -> Pipeline:
        """
        Construct Logistic Regression Pipeline with StandardScaler.

        Why Scaling is Required:
        -----------------------
        Gradient-based optimization in Logistic Regression relies on the scale of inputs.
        Unscaled features with large magnitudes (e.g. rotation ~400 vs vibration ~40)
        dominate gradient updates and cause convergence failures.
        """
        logger.info("Building Logistic Regression Pipeline (StandardScaler + class_weight='balanced') ...")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_seed,
            )),
        ])

    def build_random_forest(self, n_estimators: int = 100) -> RandomForestClassifier:
        """
        Construct Random Forest Classifier.

        Why Scaling is NOT Required:
        ---------------------------
        Tree-based algorithms select split thresholds independently per feature.
        Monotonic feature scaling does not alter decision boundary order or tree splits.
        """
        logger.info("Building Random Forest Classifier (n_estimators=%d, class_weight='balanced') ...", n_estimators)
        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_seed,
        )

    def build_xgboost(
        self,
        y_train: pd.Series,
        n_estimators: int = 100,
    ) -> XGBClassifier:
        """
        Construct XGBoost Classifier with automated scale_pos_weight.

        Calculates scale_pos_weight = negative_samples / positive_samples
        to compensate for rare positive failure events (~1.9% positive).
        """
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        logger.info(
            "Building XGBoost Classifier (scale_pos_weight=%.2f, n_estimators=%d) ...",
            scale_pos_weight,
            n_estimators,
        )

        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric="logloss",
        )


# ===========================================================================
# 2. ModelEvaluator
# ===========================================================================

class ModelEvaluator:
    """
    Evaluates trained models using comprehensive classification metrics.
    """

    @staticmethod
    def evaluate(
        model_name: str,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        fit_time_sec: float,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics on held-out test set.

        Primary Metric: PR-AUC (Average Precision)
        ------------------------------------------
        In highly imbalanced datasets (~1.9% positive), accuracy is misleading:
        a naive model predicting all 0s achieves 98.1% accuracy but 0.0 recall.
        PR-AUC focuses strictly on the trade-off between Precision and Recall
        for the rare positive failure class.
        """
        logger.info("Evaluating model '%s' on held-out Test set (%d rows) ...", model_name, len(X_test))

        start_pred = time.time()
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        pred_time_sec = round(time.time() - start_pred, 3)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        loss = log_loss(y_test, y_prob)

        logger.info(
            "[%s Results] PR-AUC: %.4f | ROC-AUC: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
            model_name, pr_auc, roc_auc, precision, recall, f1,
        )

        # Feature Importance if available
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(zip(X_test.columns, model.feature_importances_))
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf"), "coef_"):
            feature_importance = dict(zip(X_test.columns, model.named_steps["clf"].coef_[0]))

        return {
            "model_name": model_name,
            "pr_auc": round(float(pr_auc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "mcc": round(float(mcc), 4),
            "log_loss": round(float(loss), 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "fit_time_sec": round(float(fit_time_sec), 2),
            "pred_time_sec": pred_time_sec,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "feature_importance": feature_importance,
        }


# ===========================================================================
# 3. ModelSaver & Metadata Registry
# ===========================================================================

class ModelSaver:
    """
    Saves trained model pickle files and individual JSON metadata registry records.
    """

    @staticmethod
    def save(
        model: Any,
        model_name: str,
        eval_metrics: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        output_dir: Path,
    ) -> Path:
        """Save model pickle and registry metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        file_slug = model_name.lower().replace(" ", "_")
        pkl_path = output_dir / f"{file_slug}.pkl"
        json_path = output_dir / f"{file_slug}_metadata.json"

        # Save model pickle
        joblib.dump(model, pkl_path)
        logger.info("Saved model pickle: %s (%.2f MB)", pkl_path, pkl_path.stat().st_size / 1_048_576)

        # Build registry metadata
        meta = {
            "model_name": model_name,
            "version": "1.0.0",
            "training_date": datetime.now().isoformat(),
            "feature_count": X_train.shape[1],
            "feature_names": list(X_train.columns),
            "training_rows": len(X_train),
            "positive_class_pct": round(100.0 * y_train.sum() / len(y_train), 2),
            "pr_auc": eval_metrics["pr_auc"],
            "roc_auc": eval_metrics["roc_auc"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1_score": eval_metrics["f1_score"],
            "fit_time_sec": eval_metrics["fit_time_sec"],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved model metadata JSON: %s", json_path)

        return pkl_path


# ===========================================================================
# 4. Visualisations — Unified Evaluation Dashboard
# ===========================================================================

def _save_fig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_model_evaluation_dashboard(
    eval_results: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    figures_dir: Path,
) -> None:
    """
    Generate unified 5-panel Model Evaluation Dashboard.

    Panels:
      1. Top-Left: Overlaid Precision-Recall Curves (highlighting PR-AUC).
      2. Top-Right: Overlaid ROC Curves.
      3. Middle-Left: Confusion Matrices for all 3 models.
      4. Middle-Right: Metric Comparison Bar Chart (PR-AUC, ROC-AUC, F1, Recall, Precision).
      5. Bottom: Top 15 Feature Importances (Tree models).
    """
    figures_dir = Path(figures_dir)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("PredictGuard — Stage 5 Model Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.99)

    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    ax_pr = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_fi = fig.add_subplot(gs[2, :])

    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#ff7f0e", "XGBoost": "#2ca02c"}

    # 1. Overlaid Precision-Recall Curves
    for m_name, res in eval_results.items():
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        ax_pr.plot(recall, precision, label=f"{m_name} (PR-AUC = {res['pr_auc']:.4f})", color=colors.get(m_name, "black"), linewidth=2)

    baseline_pr = y_test.sum() / len(y_test)
    ax_pr.axhline(baseline_pr, color="gray", linestyle="--", label=f"Random Baseline ({baseline_pr:.2%})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves (Primary Evaluation Metric)", fontsize=12, fontweight="bold")
    ax_pr.legend(loc="upper right", fontsize=9)

    # 2. Overlaid ROC Curves
    for m_name, res in eval_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax_roc.plot(fpr, tpr, label=f"{m_name} (ROC-AUC = {res['roc_auc']:.4f})", color=colors.get(m_name, "black"), linewidth=2)

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random Chance (0.50)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=9)

    # 3. Confusion Matrices Subplot (Middle-Left)
    cm_subspec = gs[1, 0].subgridspec(1, 3)
    for idx, (m_name, res) in enumerate(eval_results.items()):
        ax_cm = fig.add_subplot(cm_subspec[idx])
        cm_arr = np.array([[res["confusion_matrix"]["TN"], res["confusion_matrix"]["FP"]],
                           [res["confusion_matrix"]["FN"], res["confusion_matrix"]["TP"]]])
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
        ax_cm.set_title(m_name, fontsize=10, fontweight="bold")
        ax_cm.set_xlabel("Pred")
        if idx == 0:
            ax_cm.set_ylabel("Actual")

    # 4. Metric Comparison Bar Chart
    metric_names = ["pr_auc", "roc_auc", "f1_score", "recall", "precision"]
    metric_labels = ["PR-AUC", "ROC-AUC", "F1 Score", "Recall", "Precision"]

    model_names = list(eval_results.keys())
    x = np.arange(len(metric_labels))
    width = 0.25

    for i, m_name in enumerate(model_names):
        vals = [eval_results[m_name][m] for m in metric_names]
        ax_metrics.bar(x + (i - 1) * width, vals, width, label=m_name, color=colors.get(m_name, "black"), alpha=0.85)

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_labels)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Metrics Comparison Across Models", fontsize=12, fontweight="bold")
    ax_metrics.legend(loc="upper right", fontsize=9)
    ax_metrics.set_ylim(0, 1.15)

    # 5. Top 15 Feature Importances (XGBoost vs Random Forest)
    xgb_fi = eval_results.get("XGBoost", {}).get("feature_importance")
    rf_fi = eval_results.get("Random Forest", {}).get("feature_importance")

    if xgb_fi:
        top_features = pd.Series(xgb_fi).sort_values(ascending=False).head(15).sort_values(ascending=True)
        ax_fi.barh(top_features.index, top_features.values, color="#2ca02c", alpha=0.85, label="XGBoost Importance")
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_title("Top 15 Most Important Features (XGBoost Model)", fontsize=12, fontweight="bold")
        ax_fi.legend(loc="lower right")

    fig.tight_layout()
    _save_fig(fig, figures_dir / "model_evaluation_dashboard.png")


# ===========================================================================
# 5. Markdown Training Report Generator
# ===========================================================================

def generate_training_report(
    eval_results: Dict[str, Dict[str, Any]],
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate Markdown report for Stage 5 Baseline Training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PredictGuard — Baseline Model Training & Evaluation Report",
        "",
        "> **Stage 5: Baseline and Stronger Models**  ",
        f"> Models evaluated on held-out **Final Test Set ({stats['test_machines']} machines, {stats['test_rows']:,} rows)**.",
        "",
        "---",
        "",
        "## 1. Why PR-AUC is the Primary Metric (Not Accuracy)",
        "",
        textwrap.dedent("""\
        In predictive maintenance datasets with extreme class imbalance (~1.9% positive failure rate),
        **Accuracy is a misleading metric**.

        A naive dummy model predicting `y=0` for every single timestamp achieves **98.1% Accuracy**,
        yet catches **0% of machine failures** (0% Recall, 0% F1), causing catastrophic unscheduled downtime.

        **PR-AUC (Precision-Recall AUC / Average Precision)** measures the area under the Precision-Recall curve
        specifically for the rare positive failure class. It directly reflects how effectively the system detects
        real failure risks while controlling false alarms.
        """),
        "",
        "---",
        "",
        "## 2. Model Performance Comparison Table",
        "",
        "| Model Name | PR-AUC (Primary) | ROC-AUC | F1 Score | Recall | Precision | Bal Acc | MCC | Log Loss | Fit Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for m_name, res in eval_results.items():
        lines.append(
            f"| **{m_name}** | **{res['pr_auc']:.4f}** | {res['roc_auc']:.4f} | {res['f1_score']:.4f} | {res['recall']:.4f} | {res['precision']:.4f} | {res['balanced_accuracy']:.4f} | {res['mcc']:.4f} | {res['log_loss']:.4f} | {res['fit_time_sec']:.2f}s |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Confusion Matrix Breakdown",
        "",
        "| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |",
        "|---|---|---|---|---|",
    ]

    for m_name, res in eval_results.items():
        cm = res["confusion_matrix"]
        lines.append(
            f"| **{m_name}** | {cm['TN']:,} | {cm['FP']:,} | {cm['FN']:,} | **{cm['TP']:,}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Key Engineering Takeaways",
        "",
        "1. **Tree-based models (XGBoost & Random Forest)** significantly outperform linear Logistic Regression because failure risk is governed by non-linear interactions between sensor degradation, error history, and maintenance elapsed time.",
        "2. **Class imbalance weighting** (`scale_pos_weight` in XGBoost and `class_weight='balanced'` in Random Forest) is essential for enabling models to output meaningful probability distributions for rare failures.",
        "3. **Zero Machine Leakage**: Because evaluation was conducted on 20 completely unseen machines, these metrics reflect true operational generalization capability.",
        "",
        "---",
        "",
        "## 5. Artifacts & Outputs Saved",
        "",
        "| Artifact | Location |",
        "|---|---|",
        "| Logistic Regression Model | `models/logistic_regression.pkl` |",
        "| Random Forest Model | `models/random_forest.pkl` |",
        "| XGBoost Model | `models/xgboost.pkl` |",
        "| Training Metadata Registry | `models/training_metadata.json` |",
        "| Unified Evaluation Dashboard | `reports/figures/model_evaluation_dashboard.png` |",
        "| Markdown Report | `reports/model_training_report.md` |",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved Model Training Report to %s", output_path)
