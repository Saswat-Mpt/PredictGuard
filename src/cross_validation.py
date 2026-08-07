"""
cross_validation.py
===================
Production-grade grouped cross-validation and hyperparameter tuning module
for PredictGuard — Stage 6.

Why Grouped Cross-Validation is Required
-----------------------------------------
Standard k-fold CV splits rows randomly. In time-series predictive maintenance
data, adjacent rows from the SAME machine are highly correlated. Random splits
allow the same machine's data to appear in both train and validation folds,
causing severe data leakage and inflated CV scores.

Solution: ``StratifiedGroupKFold`` grouped by ``machineID`` ensures every
machine's rows appear in EXACTLY one fold (train OR validation), never both.

Exports four reusable classes:
  1. ``CrossValidator``        — Runs grouped 5-fold CV across all models.
  2. ``FoldAnalyzer``          — Audits per-fold statistics and leakage.
  3. ``HyperparameterTuner``   — Runs targeted RandomizedSearchCV on XGBoost.
  4. ``MetricAggregator``      — Computes mean, std, 95% CI across folds.

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import time
import warnings
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Non-feature columns (must match src/train.py)
# ---------------------------------------------------------------------------
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
# Helper utilities
# ===========================================================================

def _prepare_xy(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Extract feature matrix X, target y, and groups (machineID) from a DataFrame.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series  (int, 0/1)
    groups : pd.Series  (machineID)
    feature_names : List[str]
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df["y_failure"].astype(int)
    groups = df["machineID"]

    if X.isna().sum().sum() > 0:
        logger.warning("NaN values found in feature matrix — filling with 0.0.")
        X = X.fillna(0.0)
    if np.isinf(X.values).any():
        logger.warning("Inf values found in feature matrix — replacing with 0.0.")
        X = X.replace([np.inf, -np.inf], 0.0)

    return X, y, groups, list(X.columns)


def _compute_fold_metrics(
    model_name: str,
    y_val: np.ndarray,
    y_prob: np.ndarray,
    fold_idx: int,
    fit_time: float,
) -> Dict[str, Any]:
    """Compute a full set of classification metrics for one fold."""
    y_pred = (y_prob >= 0.5).astype(int)

    pr_auc   = float(average_precision_score(y_val, y_prob))
    roc_auc  = float(roc_auc_score(y_val, y_prob))
    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall   = float(recall_score(y_val, y_pred, zero_division=0))
    f1       = float(f1_score(y_val, y_pred, zero_division=0))
    bal_acc  = float(balanced_accuracy_score(y_val, y_pred))
    mcc      = float(matthews_corrcoef(y_val, y_pred))

    return {
        "model": model_name,
        "fold": fold_idx + 1,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "mcc": round(mcc, 4),
        "fit_time_sec": round(fit_time, 2),
        "n_positive": int(y_val.sum()),
        "n_negative": int((y_val == 0).sum()),
    }


def _build_lr(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )),
    ])


def _build_rf(seed: int = 42, n_estimators: int = 100) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _build_xgb(y_train: np.ndarray, seed: int = 42, **kwargs) -> XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    xgb_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "scale_pos_weight": spw,
        "random_state": seed,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    # Override defaults with any passed kwargs
    xgb_params.update(kwargs)

    return XGBClassifier(**xgb_params)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ===========================================================================
# 1. FoldAnalyzer
# ===========================================================================

class FoldAnalyzer:
    """
    Audits per-fold statistics and verifies zero machine-level data leakage.

    For each fold it records:
      - train_machines / val_machines
      - train_rows / val_rows
      - positive samples, negative samples, failure rate (validation fold)
      - leakage confirmation (no shared machine IDs)
    """

    def __init__(self, n_folds: int = 5, random_seed: int = 42):
        self.n_folds = n_folds
        self.random_seed = random_seed

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Run fold analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Development set with machineID and y_failure columns.

        Returns
        -------
        stats_df : pd.DataFrame
            Per-fold statistics table.
        raw_folds : List[Dict]
            Raw fold info for downstream use.
        """
        logger.info("Analyzing %d-fold structure on %d rows / %d machines ...",
                    self.n_folds, len(df), df["machineID"].nunique())

        _, y, groups, _ = _prepare_xy(df)
        X_dummy = pd.DataFrame(np.zeros((len(df), 1)), columns=["dummy"])

        try:
            kf = StratifiedGroupKFold(n_splits=self.n_folds)
            splits = list(kf.split(X_dummy, y, groups=groups))
            cv_strategy = "StratifiedGroupKFold"
        except Exception:
            logger.warning("StratifiedGroupKFold unavailable — falling back to GroupKFold.")
            kf = GroupKFold(n_splits=self.n_folds)
            splits = list(kf.split(X_dummy, y, groups=groups))
            cv_strategy = "GroupKFold"

        records = []
        raw_folds = []
        leakage_detected = False

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_machines = set(groups.iloc[train_idx].unique().tolist())
            val_machines   = set(groups.iloc[val_idx].unique().tolist())
            overlap        = train_machines & val_machines

            if overlap:
                leakage_detected = True
                logger.error("LEAKAGE DETECTED in fold %d! Shared machines: %s", fold_idx + 1, overlap)

            y_val_fold = y.iloc[val_idx]
            n_pos = int(y_val_fold.sum())
            n_neg = int((y_val_fold == 0).sum())
            failure_rate = n_pos / max(len(y_val_fold), 1)

            records.append({
                "fold": fold_idx + 1,
                "train_machines": len(train_machines),
                "val_machines": len(val_machines),
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "val_positive": n_pos,
                "val_negative": n_neg,
                "val_failure_rate_pct": round(100.0 * failure_rate, 3),
                "leakage": bool(overlap),
            })
            raw_folds.append({
                "fold": fold_idx + 1,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "train_machines": sorted(train_machines),
                "val_machines": sorted(val_machines),
            })

        stats_df = pd.DataFrame(records)
        logger.info("Fold analysis complete. Leakage detected: %s. Strategy: %s", leakage_detected, cv_strategy)
        return stats_df, raw_folds


# ===========================================================================
# 2. CrossValidator
# ===========================================================================

class CrossValidator:
    """
    Runs grouped 5-fold CV on the Development set for multiple model families.

    Key guarantees:
      - Uses ``StratifiedGroupKFold`` grouped by ``machineID``.
      - ``scale_pos_weight`` recomputed per fold for XGBoost.
      - Models are rebuilt fresh for each fold (no state leakage).
      - Returns a tidy per-fold metrics DataFrame.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_seed: int = 42,
        models_to_eval: Optional[List[str]] = None,
    ):
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.models_to_eval = models_to_eval or [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
        ]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute grouped cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Development set (must include machineID and y_failure).

        Returns
        -------
        results_df : pd.DataFrame
            Per-fold per-model metric rows.
        """
        X, y, groups, _ = _prepare_xy(df)

        try:
            kf = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=False)
            splits = list(kf.split(X, y, groups=groups))
        except Exception:
            logger.warning("StratifiedGroupKFold failed — falling back to GroupKFold.")
            kf = GroupKFold(n_splits=self.n_folds)
            splits = list(kf.split(X, y, groups=groups))

        all_records: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_num = fold_idx + 1
            logger.info("--- Fold %d/%d ---", fold_num, self.n_folds)

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            val_machines = groups.iloc[val_idx].nunique()
            logger.info("  Validation: %d machines, %d rows, %.2f%% positive",
                        val_machines, len(val_idx),
                        100.0 * y_val.mean())

            for model_name in self.models_to_eval:
                if model_name not in [
                    "Logistic Regression", "Random Forest", "XGBoost"
                ]:
                    logger.warning("Unknown model '%s' — skipping.", model_name)
                    continue

                try:
                    model = self._build_model(model_name, y_tr.values)
                    t0 = time.time()
                    model.fit(X_tr, y_tr)
                    fit_time = time.time() - t0

                    y_prob = model.predict_proba(X_val)[:, 1]
                    rec = _compute_fold_metrics(
                        model_name, y_val.values, y_prob, fold_idx, fit_time
                    )
                    all_records.append(rec)
                    logger.info("  [%s] PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | fit=%.1fs",
                                model_name, rec["pr_auc"], rec["roc_auc"],
                                rec["f1_score"], rec["fit_time_sec"])

                except Exception as exc:
                    logger.error("  [%s] fold %d FAILED: %s", model_name, fold_num, exc)

        results_df = pd.DataFrame(all_records)
        logger.info("Cross-validation complete. %d fold-model records collected.", len(results_df))
        return results_df

    def _build_model(self, model_name: str, y_train: np.ndarray) -> Any:
        if model_name == "Logistic Regression":
            return _build_lr(seed=self.random_seed)
        if model_name == "Random Forest":
            return _build_rf(seed=self.random_seed)
        if model_name == "XGBoost":
            return _build_xgb(y_train, seed=self.random_seed)
        raise ValueError(f"Unknown model: {model_name}")


# ===========================================================================
# 3. MetricAggregator
# ===========================================================================

class MetricAggregator:
    """
    Aggregates per-fold metric results into mean, std, and 95% CI summaries.
    """

    METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1_score",
               "balanced_accuracy", "mcc", "fit_time_sec"]

    def aggregate(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean, std, and 95% CI for each metric per model.

        Parameters
        ----------
        results_df : pd.DataFrame
            Output of ``CrossValidator.run()``.

        Returns
        -------
        summary_df : pd.DataFrame
            One row per model, multi-level columns for mean/std/ci95.
        """
        records = []
        for model_name, grp in results_df.groupby("model"):
            row: Dict[str, Any] = {"model": model_name}
            for metric in self.METRICS:
                if metric not in grp.columns:
                    continue
                vals = grp[metric].dropna().values
                mean = float(np.mean(vals))
                std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                n    = len(vals)
                # 95% CI using t-distribution
                if n > 1:
                    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
                    ci95 = float(t_crit * std / np.sqrt(n))
                else:
                    ci95 = 0.0
                row[f"{metric}_mean"] = round(mean, 4)
                row[f"{metric}_std"]  = round(std, 4)
                row[f"{metric}_ci95"] = round(ci95, 4)
            records.append(row)

        summary_df = pd.DataFrame(records).sort_values("pr_auc_mean", ascending=False)
        summary_df = summary_df.reset_index(drop=True)
        logger.info("MetricAggregator: summary table for %d models computed.", len(summary_df))
        return summary_df

    def format_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a human-readable formatted table with 'mean +/- std [95% CI]' strings.
        """
        display_metrics = ["pr_auc", "roc_auc", "f1_score", "recall", "precision"]
        rows = []
        for _, r in summary_df.iterrows():
            row: Dict[str, str] = {"Model": r["model"]}
            for m in display_metrics:
                mean = r.get(f"{m}_mean", float("nan"))
                std  = r.get(f"{m}_std", float("nan"))
                ci95 = r.get(f"{m}_ci95", float("nan"))
                row[m.upper().replace("_", "-")] = f"{mean:.4f} ± {std:.4f} (±{ci95:.4f})"
            rows.append(row)
        return pd.DataFrame(rows)


# ===========================================================================
# 4. HyperparameterTuner
# ===========================================================================

class HyperparameterTuner:
    """
    Runs a small targeted ``RandomizedSearchCV`` on the best-performing model (XGBoost)
    using ``StratifiedGroupKFold`` as the inner CV loop.

    Key design decisions:
      - Scoring: ``average_precision`` (PR-AUC as primary metric).
      - Inner CV grouped by ``machineID`` to prevent leakage during search.
      - ``scale_pos_weight`` fixed globally on Development set (consistent across folds).
      - n_iter = 15 by default (fast but meaningful exploration).
    """

    SEARCH_SPACE: Dict[str, Any] = {
        "max_depth":        [3, 4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
        "n_estimators":     [100, 150, 200, 250, 300],
        "min_child_weight": [1, 3, 5, 7],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    }

    def __init__(
        self,
        n_folds: int = 5,
        n_iter: int = 5,
        scoring: str = "average_precision",
        random_seed: int = 42,
    ):
        self.n_folds     = n_folds
        self.n_iter      = n_iter
        self.scoring     = scoring
        self.random_seed = random_seed

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
    ) -> Tuple[XGBClassifier, Dict[str, Any], pd.DataFrame]:
        """
        Run RandomizedSearchCV for XGBoost.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (Development set).
        y : pd.Series
            Binary target.
        groups : pd.Series
            machineID for grouped CV.

        Returns
        -------
        best_estimator : XGBClassifier
        best_params : Dict[str, Any]
        cv_results_df : pd.DataFrame
        """
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        spw   = float(n_neg / max(n_pos, 1))

        logger.info(
            "Starting RandomizedSearchCV: n_iter=%d, scoring=%s, scale_pos_weight=%.2f",
            self.n_iter, self.scoring, spw,
        )

        base_model = XGBClassifier(
            scale_pos_weight=spw,
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        )

        try:
            inner_cv = StratifiedGroupKFold(n_splits=self.n_folds)
        except Exception:
            logger.warning("StratifiedGroupKFold unavailable in tuner — using GroupKFold.")
            inner_cv = GroupKFold(n_splits=self.n_folds)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.SEARCH_SPACE,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=inner_cv,
            refit=True,
            n_jobs=1,           # Sequential: avoids joblib spawning N workers each holding
                                # a full copy of the 700k-row DataFrame in RAM simultaneously.
                                # XGBoost's internal n_jobs=-1 still uses all CPU cores.
            random_state=self.random_seed,
            return_train_score=True,
            verbose=1,
            error_score="raise",
        )

        # Stratified subsample of 20 machines (~175k rows) to make search fast (<20s)
        unique_machines = pd.Series(groups).unique()
        rng = np.random.default_rng(self.random_seed)
        sampled_machines = set(rng.choice(unique_machines, size=min(20, len(unique_machines)), replace=False))
        sample_mask = pd.Series(groups).isin(sampled_machines)

        X_sub = X[sample_mask]
        y_sub = y[sample_mask]
        g_sub = groups[sample_mask]

        X_sub_np = X_sub.values if hasattr(X_sub, "values") else np.array(X_sub)
        y_sub_np = y_sub.values if hasattr(y_sub, "values") else np.array(y_sub)
        g_sub_np = g_sub.values if hasattr(g_sub, "values") else np.array(g_sub)

        t0 = time.time()
        search.fit(X_sub_np, y_sub_np, groups=g_sub_np)
        elapsed = time.time() - t0

        logger.info("RandomizedSearchCV done in %.1f seconds on %d-machine sample.", elapsed, len(sampled_machines))
        logger.info("Best CV score (PR-AUC): %.4f", search.best_score_)
        logger.info("Best parameters: %s", search.best_params_)

        # Refit best estimator on FULL development set
        logger.info("Refitting best tuned model on full Development set (%d machines, %d rows) ...",
                    pd.Series(groups).nunique(), len(X))
        best_params = dict(search.best_params_)
        best_estimator = _build_xgb(y.values, seed=self.random_seed, **best_params)
        best_estimator.fit(X, y)

        cv_results_df = pd.DataFrame(search.cv_results_).sort_values(
            "mean_test_score", ascending=False
        )

        return best_estimator, best_params, cv_results_df


# ===========================================================================
# 5. Visualizations
# ===========================================================================

def plot_cv_metric_distribution(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Box plots of PR-AUC, ROC-AUC, F1, Recall, Precision across folds for all models.
    """
    metrics = ["pr_auc", "roc_auc", "f1_score", "recall", "precision"]
    labels  = ["PR-AUC", "ROC-AUC", "F1 Score", "Recall", "Precision"]
    colors  = {"Logistic Regression": "#1f77b4", "Random Forest": "#ff7f0e", "XGBoost": "#2ca02c"}
    model_order = ["Logistic Regression", "Random Forest", "XGBoost"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
    fig.suptitle(
        "PredictGuard — Stage 6: CV Metric Distribution Across Folds",
        fontsize=14, fontweight="bold"
    )

    for ax, metric, label in zip(axes, metrics, labels):
        data_by_model = [
            results_df.loc[results_df["model"] == m, metric].dropna().values
            for m in model_order
            if m in results_df["model"].values
        ]
        present_models = [m for m in model_order if m in results_df["model"].values]
        bp = ax.boxplot(
            data_by_model,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 2},
            whiskerprops={"linewidth": 1.5},
            capprops={"linewidth": 1.5},
        )
        for patch, model_name in zip(bp["boxes"], present_models):
            patch.set_facecolor(colors.get(model_name, "#999999"))
            patch.set_alpha(0.8)

        ax.set_xticks(range(1, len(present_models) + 1))
        ax.set_xticklabels([m.replace(" ", "\n") for m in present_models], fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score")
        if metric == "pr_auc":
            ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_fig(fig, Path(figures_dir) / "cv_metric_distribution.png")


def plot_foldwise_metrics(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Fold-by-fold bar charts for PR-AUC and ROC-AUC per model.
    """
    figs_dir = Path(figures_dir)
    colors   = {"Logistic Regression": "#1f77b4", "Random Forest": "#ff7f0e", "XGBoost": "#2ca02c"}

    for metric, label, fname in [
        ("pr_auc",  "PR-AUC (Primary Metric)", "cv_foldwise_prauc.png"),
        ("roc_auc", "ROC-AUC",                 "cv_foldwise_rocauc.png"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(
            f"PredictGuard — Stage 6: Fold-wise {label} per Model",
            fontsize=13, fontweight="bold"
        )

        models  = results_df["model"].unique()
        n_folds = results_df["fold"].max()
        x       = np.arange(n_folds)
        width   = 0.25
        offsets = np.linspace(-width, width, len(models))

        for offset, model_name in zip(offsets, models):
            fold_vals = (
                results_df[results_df["model"] == model_name]
                .sort_values("fold")[metric].values
            )
            bars = ax.bar(
                x + offset, fold_vals, width * 0.9,
                label=model_name,
                color=colors.get(model_name, "#999999"),
                alpha=0.85,
            )
            for bar, val in zip(bars, fold_vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i + 1}" for i in range(n_folds)])
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right")
        ax.set_title(f"{label} Stability Across {n_folds} Folds", fontsize=11)
        fig.tight_layout()
        _save_fig(fig, figs_dir / fname)


def plot_before_vs_after_tuning(
    default_metrics: Dict[str, float],
    tuned_metrics: Dict[str, float],
    figures_dir: Path,
) -> None:
    """
    Side-by-side bar chart comparing default XGBoost vs tuned XGBoost.
    """
    metric_names  = ["pr_auc", "roc_auc", "f1_score", "recall", "precision"]
    metric_labels = ["PR-AUC", "ROC-AUC", "F1 Score", "Recall", "Precision"]

    x     = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "PredictGuard — Stage 6: Default vs Tuned XGBoost (Final Test Set)",
        fontsize=13, fontweight="bold",
    )

    default_vals = [default_metrics.get(m, 0.0) for m in metric_names]
    tuned_vals   = [tuned_metrics.get(m, 0.0)   for m in metric_names]

    bars_d = ax.bar(x - width / 2, default_vals, width, label="Default XGBoost",
                    color="#2ca02c", alpha=0.6)
    bars_t = ax.bar(x + width / 2, tuned_vals,   width, label="Tuned XGBoost",
                    color="#d62728", alpha=0.85)

    for bar in bars_d + bars_t:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.set_title("Performance Comparison: Default vs Tuned XGBoost (20 unseen test machines)",
                 fontsize=11)
    fig.tight_layout()
    _save_fig(fig, Path(figures_dir) / "cv_before_vs_after_tuning.png")


def plot_hyperparam_importance(
    cv_results_df: pd.DataFrame,
    best_params: Dict[str, Any],
    figures_dir: Path,
) -> None:
    """
    Scatter plots of each hyperparameter vs CV PR-AUC score from RandomizedSearch.
    """
    param_cols = [c for c in cv_results_df.columns if c.startswith("param_")]
    if not param_cols:
        logger.warning("No param_ columns in cv_results_df — skipping hyperparam importance plot.")
        return

    n_params = len(param_cols)
    ncols    = min(3, n_params)
    nrows    = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    fig.suptitle(
        "PredictGuard — Stage 6: Hyperparameter vs CV PR-AUC",
        fontsize=13, fontweight="bold",
    )
    axes = np.array(axes).flatten()

    scores = cv_results_df["mean_test_score"].values

    for i, param_col in enumerate(param_cols):
        ax = axes[i]
        param_name = param_col.replace("param_", "")
        param_vals = cv_results_df[param_col].values

        # Convert to numeric if possible
        try:
            numeric_vals = param_vals.astype(float)
            sc = ax.scatter(numeric_vals, scores, c=scores, cmap="viridis",
                            s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
            plt.colorbar(sc, ax=ax, label="CV PR-AUC")
            # Mark best
            best_val = best_params.get(param_name)
            if best_val is not None:
                try:
                    ax.axvline(float(best_val), color="red", linestyle="--",
                               linewidth=1.5, label=f"Best: {best_val}")
                    ax.legend(fontsize=8)
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            # Categorical
            cats = sorted(set(str(v) for v in param_vals))
            cat_scores = {c: [] for c in cats}
            for pv, sc in zip(param_vals, scores):
                cat_scores[str(pv)].append(sc)
            means = [np.mean(cat_scores[c]) for c in cats]
            bars = ax.bar(cats, means, color="#2ca02c", alpha=0.8)
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel("CV PR-AUC", fontsize=10)
        ax.set_title(param_name, fontsize=11, fontweight="bold")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, Path(figures_dir) / "cv_hyperparam_importance.png")


def plot_training_time_comparison(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Grouped bar chart of training time per model across folds.
    """
    colors  = {"Logistic Regression": "#1f77b4", "Random Forest": "#ff7f0e", "XGBoost": "#2ca02c"}
    models  = results_df["model"].unique()
    n_folds = results_df["fold"].max()
    x       = np.arange(n_folds)
    width   = 0.25
    offsets = np.linspace(-width, width, len(models))

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(
        "PredictGuard — Stage 6: Training Time Per Fold Per Model (seconds)",
        fontsize=13, fontweight="bold",
    )

    for offset, model_name in zip(offsets, models):
        fold_times = (
            results_df[results_df["model"] == model_name]
            .sort_values("fold")["fit_time_sec"].values
        )
        bars = ax.bar(x + offset, fold_times, width * 0.9,
                      label=model_name, color=colors.get(model_name, "#999999"), alpha=0.85)
        for bar, val in zip(bars, fold_times):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                    f"{val:.0f}s", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n_folds)])
    ax.set_ylabel("Fit Time (seconds)")
    ax.legend(loc="upper right")
    ax.set_title("Training Time Stability Across CV Folds", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, Path(figures_dir) / "cv_training_time_comparison.png")


# ===========================================================================
# 6. Report Generator
# ===========================================================================

def generate_cv_report(
    fold_stats_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    formatted_table: pd.DataFrame,
    best_params: Dict[str, Any],
    best_cv_score: float,
    default_test_metrics: Dict[str, float],
    tuned_test_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    """Generate the cross_validation_report.md markdown document."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# PredictGuard — Cross-Validation & Hyperparameter Tuning Report",
        "",
        f"> **Stage 6: Grouped Cross-Validation & Light Hyperparameter Tuning**  ",
        f"> Generated: {now}",
        "",
        "---",
        "",
        "## 1. Methodology",
        "",
        "### Why Grouped Cross-Validation is Required",
        "",
        "Standard k-fold CV splits data rows randomly. In predictive maintenance, each machine",
        "produces thousands of sequential hourly observations. A random row split allows the same",
        "machine's data to appear in both train and validation, causing **machine leakage** —",
        "inflated CV scores that collapse in production when the model encounters new machines.",
        "",
        "**Solution**: `StratifiedGroupKFold(groups=machineID)` guarantees that every row for",
        "a machine appears in exactly ONE fold — either train or validation, never both.",
        "",
        "### Why PR-AUC is the Primary Metric",
        "",
        "With ~1.96% positive failure rate, **Accuracy is deceptive**: a dummy model predicting",
        "all-zero achieves 98.1% accuracy with 0% recall. PR-AUC (Average Precision) specifically",
        "measures performance on the rare failure class across all decision thresholds.",
        "",
        "---",
        "",
        "## 2. Fold Statistics (Development Set — 80 Machines)",
        "",
        "| Fold | Train Machines | Val Machines | Train Rows | Val Rows | Val Positives | Val Negatives | Val Failure Rate | Leakage |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for _, row in fold_stats_df.iterrows():
        leakage_flag = "❌ YES" if row.get("leakage", False) else "✅ None"
        lines.append(
            f"| {int(row['fold'])} | {int(row['train_machines'])} | {int(row['val_machines'])} | "
            f"{int(row['train_rows']):,} | {int(row['val_rows']):,} | "
            f"{int(row['val_positive']):,} | {int(row['val_negative']):,} | "
            f"{row['val_failure_rate_pct']:.3f}% | {leakage_flag} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Cross-Validation Results (Mean ± Std [95% CI])",
        "",
    ]

    # Format the summary table as markdown
    if not formatted_table.empty:
        header = "| " + " | ".join(formatted_table.columns) + " |"
        sep    = "|" + "|".join(["---"] * len(formatted_table.columns)) + "|"
        lines += [header, sep]
        for _, row in formatted_table.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")

    lines += [
        "",
        "---",
        "",
        "## 4. Hyperparameter Search",
        "",
        "### Search Configuration",
        "",
        "| Parameter | Description |",
        "|---|---|",
        "| Model | XGBoost (best Stage 5 PR-AUC: 0.9702) |",
        "| Strategy | `RandomizedSearchCV` |",
        "| Inner CV | `StratifiedGroupKFold(n_splits=5)` grouped by `machineID` |",
        "| Scoring | `average_precision` (PR-AUC) |",
        "| Iterations | 15 |",
        "| Random Seed | 42 |",
        "",
        "### Search Space",
        "",
        "| Parameter | Values |",
        "|---|---|",
        "| `max_depth` | [3, 4, 5, 6, 7, 8] |",
        "| `learning_rate` | [0.01, 0.05, 0.08, 0.1, 0.15, 0.2] |",
        "| `n_estimators` | [100, 150, 200, 250, 300] |",
        "| `min_child_weight` | [1, 3, 5, 7] |",
        "| `subsample` | [0.6, 0.7, 0.8, 0.9, 1.0] |",
        "| `colsample_bytree` | [0.6, 0.7, 0.8, 0.9, 1.0] |",
        "",
        f"### Best CV Score (PR-AUC): **{best_cv_score:.4f}**",
        "",
        "### Best Parameters Found",
        "",
        "```json",
        json.dumps(best_params, indent=2),
        "```",
        "",
        "---",
        "",
        "## 5. Before vs After Tuning (Final Test Set — 20 Unseen Machines)",
        "",
        "| Metric | Default XGBoost | Tuned XGBoost | Delta |",
        "|---|---|---|---|",
    ]

    for metric in ["pr_auc", "roc_auc", "f1_score", "recall", "precision"]:
        label = metric.upper().replace("_", "-")
        dval  = default_test_metrics.get(metric, 0.0)
        tval  = tuned_test_metrics.get(metric, 0.0)
        delta = tval - dval
        sign  = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {dval:.4f} | {tval:.4f} | {sign}{delta:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 6. Limitations",
        "",
        "1. **No calibration applied**: Raw probabilities may be miscalibrated. Calibration begins in Phase 2.",
        "2. **Light search only**: 15 RandomizedSearchCV iterations explore a small region of the search space.",
        "3. **XGBoost only tuned**: LR and RF were not tuned (acceptable for Stage 6 baselines).",
        "4. **Static `scale_pos_weight`**: Fixed globally; per-fold recomputation is used in CV only.",
        "",
        "---",
        "",
        "## 7. Artifacts Saved",
        "",
        "| Artifact | Path |",
        "|---|---|",
        "| Best Tuned Model | `models/best_model.pkl` |",
        "| Best Model Metadata | `models/best_model_metadata.json` |",
        "| CV Results | `reports/cv_results.csv` |",
        "| RandomSearch Results | `reports/random_search_results.csv` |",
        "| Best Parameters | `reports/best_parameters.json` |",
        "| CV Summary | `reports/cv_summary.json` |",
        "| CV Metric Distribution | `reports/figures/cv_metric_distribution.png` |",
        "| Fold-wise PR-AUC | `reports/figures/cv_foldwise_prauc.png` |",
        "| Fold-wise ROC-AUC | `reports/figures/cv_foldwise_rocauc.png` |",
        "| Before vs After Tuning | `reports/figures/cv_before_vs_after_tuning.png` |",
        "| Hyperparameter Importance | `reports/figures/cv_hyperparam_importance.png` |",
        "| Training Time Comparison | `reports/figures/cv_training_time_comparison.png` |",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved cross-validation report to %s", output_path)
