"""
explainability.py
=================
PredictGuard — Phase 3 Stage 9: Model Explainability via SHAP.

Provides:
    SHAPExplainer           — TreeExplainer wrapper for local & global SHAP values
    FeatureImportanceAnalyzer — Compare SHAP importance vs native model importance
    PredictionInterpreter   — Human-readable plain-English explanation cards + JSON artifacts

IMPORTANT: SHAP MUST be computed on the RAW (un-calibrated) XGBoost model.
           Calibration changes probabilities but not the underlying tree structure.

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
NON_FEATURE_COLS = {
    "datetime",
    "machineID",
    "y_failure",
    "failure_component",
    "time_to_failure_hours",
}

FIGURE_DPI = 150
FIGURE_STYLE = "darkgrid"
SAMPLE_SIZE_GLOBAL = 5_000   # rows used for global SHAP (memory-safe)
SAMPLE_SIZE_BEESWARM = 3_000
TOP_N_FEATURES = 20          # top features shown in bar/summary plots

sns.set_theme(style=FIGURE_STYLE, rc={"figure.dpi": FIGURE_DPI})


# ===========================================================================
# 1. SHAPExplainer
# ===========================================================================

class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for PredictGuard's XGBoost raw model.

    Computes:
        - Global SHAP values on a representative sample
        - Local SHAP values for individual machine-hour rows
        - Publication-quality SHAP figures
    """

    def __init__(
        self,
        raw_model: Any,
        feature_names: List[str],
        figures_dir: Path,
        random_state: int = 42,
    ) -> None:
        self.raw_model = raw_model
        self.feature_names = feature_names
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self._explainer: Optional[shap.TreeExplainer] = None
        self._global_shap_values: Optional[np.ndarray] = None
        self._global_X_sample: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def _build_explainer(self) -> None:
        if self._explainer is None:
            logger.info("Initialising shap.TreeExplainer (this takes ~10-30s) ...")
            t0 = time.time()
            self._explainer = shap.TreeExplainer(
                self.raw_model,
                feature_perturbation="tree_path_dependent",
            )
            logger.info("TreeExplainer ready in %.1fs.", time.time() - t0)

    # ------------------------------------------------------------------
    def compute_global(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values on a stratified sample of X.

        Returns
        -------
        shap_values : np.ndarray, shape (n_sample, n_features)
        """
        self._build_explainer()
        n = min(SAMPLE_SIZE_GLOBAL, len(X))
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), size=n, replace=False)
        X_sample = X.iloc[idx].reset_index(drop=True)
        logger.info(
            "Computing global SHAP on %d rows (sampled from %d) ...", n, len(X)
        )
        t0 = time.time()
        sv = self._explainer.shap_values(X_sample)
        elapsed = time.time() - t0
        # For binary XGBoost, shap_values may be shape (n, f) or list[(n,f),(n,f)]
        if isinstance(sv, list):
            sv = sv[1]
        logger.info("Global SHAP computed in %.1fs.", elapsed)
        self._global_shap_values = sv
        self._global_X_sample = X_sample
        return sv

    # ------------------------------------------------------------------
    def compute_local(
        self, X_row: pd.DataFrame
    ) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values for a single row (or small batch).

        Returns
        -------
        shap_vals : np.ndarray, shape (n_features,)
        expected_value : float
        """
        self._build_explainer()
        sv = self._explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]
        expected = (
            self._explainer.expected_value[1]
            if isinstance(self._explainer.expected_value, (list, np.ndarray))
            else float(self._explainer.expected_value)
        )
        return sv.squeeze(), expected

    # ------------------------------------------------------------------
    # Figure generators
    # ------------------------------------------------------------------

    def plot_summary(self, X: pd.DataFrame) -> Path:
        """SHAP summary (dot) plot — global."""
        sv = self._global_shap_values
        Xs = self._global_X_sample
        if sv is None:
            self.compute_global(X)
            sv = self._global_shap_values
            Xs = self._global_X_sample
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            sv, Xs, feature_names=self.feature_names,
            max_display=TOP_N_FEATURES, show=False, plot_type="dot"
        )
        plt.title("SHAP Summary Plot — Global Feature Impact", fontsize=15, pad=12)
        plt.tight_layout()
        out = self.figures_dir / "shap_summary.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_beeswarm(self, X: pd.DataFrame) -> Path:
        """SHAP beeswarm plot — global."""
        self._build_explainer()
        n = min(SAMPLE_SIZE_BEESWARM, len(X))
        rng = np.random.RandomState(self.random_state + 1)
        idx = rng.choice(len(X), size=n, replace=False)
        X_bee = X.iloc[idx].reset_index(drop=True)
        sv_bee = self._explainer.shap_values(X_bee)
        if isinstance(sv_bee, list):
            sv_bee = sv_bee[1]
        shap_obj = shap.Explanation(
            values=sv_bee,
            base_values=np.full(n, self._explainer.expected_value
                                if not isinstance(self._explainer.expected_value, (list, np.ndarray))
                                else self._explainer.expected_value[1]),
            data=X_bee.values,
            feature_names=self.feature_names,
        )
        fig, ax = plt.subplots(figsize=(12, 9))
        shap.plots.beeswarm(shap_obj, max_display=TOP_N_FEATURES, show=False)
        plt.title("SHAP Beeswarm — Feature Value vs Impact", fontsize=15, pad=12)
        plt.tight_layout()
        out = self.figures_dir / "shap_beeswarm.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_bar(self, X: pd.DataFrame) -> Path:
        """Global mean |SHAP| bar plot."""
        sv = self._global_shap_values
        if sv is None:
            self.compute_global(X)
            sv = self._global_shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        importance_df = (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=True)
            .tail(TOP_N_FEATURES)
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(
            importance_df["feature"], importance_df["mean_abs_shap"],
            color=sns.color_palette("viridis", len(importance_df))
        )
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title(f"Top {TOP_N_FEATURES} Features by SHAP Importance", fontsize=14)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        plt.tight_layout()
        out = self.figures_dir / "shap_bar.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_waterfall(
        self, X_row: pd.DataFrame, row_label: str = "Sample"
    ) -> Path:
        """SHAP waterfall plot for a single prediction."""
        self._build_explainer()
        sv, base = self.compute_local(X_row)
        exp = shap.Explanation(
            values=sv,
            base_values=base,
            data=X_row.values.squeeze(),
            feature_names=self.feature_names,
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(exp, max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {row_label}", fontsize=14, pad=12)
        plt.tight_layout()
        out = self.figures_dir / "shap_waterfall.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_force(
        self, X_row: pd.DataFrame, row_label: str = "Sample"
    ) -> Path:
        """SHAP force plot saved as PNG (matplotlib backend, no IPython required)."""
        self._build_explainer()
        sv, base = self.compute_local(X_row)
        fig = plt.figure(figsize=(20, 3))
        shap.force_plot(
            base_value=base,
            shap_values=sv,
            features=X_row.values.squeeze(),
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
        )
        out = self.figures_dir / "shap_force.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def get_global_importance_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame of mean |SHAP| per feature, sorted desc."""
        if self._global_shap_values is None:
            raise RuntimeError("Call compute_global() first.")
        mean_abs = np.abs(self._global_shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


# ===========================================================================
# 2. FeatureImportanceAnalyzer
# ===========================================================================

class FeatureImportanceAnalyzer:
    """
    Compare SHAP-derived feature importance vs the model's native
    feature_importances_ attribute (gain-based for XGBoost).

    Produces a combined comparison table and a dual-bar figure.
    """

    def __init__(
        self,
        raw_model: Any,
        feature_names: List[str],
        figures_dir: Path,
    ) -> None:
        self.raw_model = raw_model
        self.feature_names = feature_names
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build_comparison_table(
        self, shap_importance_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build a merged DataFrame with SHAP rank, native rank, and both scores.

        Parameters
        ----------
        shap_importance_df : DataFrame with columns ['feature','mean_abs_shap']
        """
        native_imp = self.raw_model.feature_importances_
        native_df = pd.DataFrame(
            {"feature": self.feature_names, "native_importance": native_imp}
        ).sort_values("native_importance", ascending=False).reset_index(drop=True)
        native_df["native_rank"] = native_df.index + 1

        shap_df = shap_importance_df.copy()
        shap_df["shap_rank"] = shap_df.index + 1
        shap_df["mean_abs_shap_norm"] = (
            shap_df["mean_abs_shap"] / shap_df["mean_abs_shap"].max()
        )
        native_df["native_importance_norm"] = (
            native_df["native_importance"] / native_df["native_importance"].max()
        )

        merged = shap_df.merge(native_df, on="feature", how="inner")
        merged["rank_delta"] = (merged["shap_rank"] - merged["native_rank"]).abs()
        merged = merged.sort_values("shap_rank").reset_index(drop=True)
        return merged

    # ------------------------------------------------------------------
    def plot_comparison(self, comparison_df: pd.DataFrame) -> Path:
        """Dual-bar plot comparing SHAP vs native importance for top features."""
        top = comparison_df.head(TOP_N_FEATURES).copy()
        top = top.sort_values("shap_rank", ascending=False)

        x = np.arange(len(top))
        width = 0.38

        fig, ax = plt.subplots(figsize=(13, 8))
        b1 = ax.barh(x + width / 2, top["mean_abs_shap_norm"], width,
                     label="SHAP Importance (normalised)", color="#4c72b0", alpha=0.85)
        b2 = ax.barh(x - width / 2, top["native_importance_norm"], width,
                     label="Native Gain Importance (normalised)", color="#dd8452", alpha=0.85)

        ax.set_yticks(x)
        ax.set_yticklabels(top["feature"], fontsize=9)
        ax.set_xlabel("Normalised Importance Score", fontsize=11)
        ax.set_title("SHAP vs Native Feature Importance Comparison", fontsize=14)
        ax.legend(fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        out = self.figures_dir / "importance_comparison.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    # ------------------------------------------------------------------
    def plot_native_importance(self) -> Path:
        """Bar plot of native gain-based importance."""
        imp = self.raw_model.feature_importances_
        df = (
            pd.DataFrame({"feature": self.feature_names, "importance": imp})
            .sort_values("importance", ascending=True)
            .tail(TOP_N_FEATURES)
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(df["feature"], df["importance"],
                       color=sns.color_palette("rocket", len(df)))
        ax.set_xlabel("Gain-based Importance", fontsize=11)
        ax.set_title(f"Top {TOP_N_FEATURES} Features — Native XGBoost Importance", fontsize=14)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        plt.tight_layout()
        out = self.figures_dir / "native_importance.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out


# ===========================================================================
# 3. PredictionInterpreter
# ===========================================================================

class PredictionInterpreter:
    """
    Generates human-readable prediction explanation cards (plain English + JSON).

    Given a machine-hour row, raw_prob, and calibrated_prob, produces:
        - A plain-English narrative
        - A structured JSON explanation artifact
    """

    # Feature groupings for narrative generation
    SENSOR_GROUPS: Dict[str, List[str]] = {
        "vibration": ["vibration", "vibration_zscore", "vibration_rolling"],
        "pressure": ["pressure", "pressure_zscore", "pressure_rolling", "pressure_mean"],
        "voltage": ["volt", "voltage", "volt_rolling", "volt_zscore"],
        "rotation": ["rotate", "rotation", "rotate_rolling", "rotate_zscore"],
        "maintenance": ["days_since_last_maintenance", "maintenance_count", "comp_maint"],
        "errors": ["error", "days_since_last_error", "errors_last"],
        "trend": ["delta", "pct_change", "velocity", "acceleration"],
    }

    def __init__(
        self,
        feature_names: List[str],
        reports_dir: Path,
    ) -> None:
        self.feature_names = feature_names
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _group_feature(self, fname: str) -> str:
        for group, keywords in self.SENSOR_GROUPS.items():
            if any(kw in fname for kw in keywords):
                return group
        return "other"

    # ------------------------------------------------------------------
    def build_explanation_card(
        self,
        machine_id: int,
        timestamp: str,
        X_row: pd.DataFrame,
        shap_values: np.ndarray,
        raw_prob: float,
        calibrated_prob: float,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Build a full explanation card for one machine-hour prediction.

        Returns
        -------
        card : dict — structured explanation artifact (JSON-serialisable)
        """
        sv = shap_values.flatten()
        # Sort by absolute SHAP value
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        top_pos = [
            {"feature": self.feature_names[i], "shap": float(sv[i]),
             "value": float(X_row.values.flatten()[i]),
             "group": self._group_feature(self.feature_names[i])}
            for i in sorted_idx if sv[i] > 0
        ][:top_n]
        top_neg = [
            {"feature": self.feature_names[i], "shap": float(sv[i]),
             "value": float(X_row.values.flatten()[i]),
             "group": self._group_feature(self.feature_names[i])}
            for i in sorted_idx if sv[i] < 0
        ][:top_n]

        card: Dict[str, Any] = {
            "machine_id": int(machine_id),
            "timestamp": str(timestamp),
            "raw_score": round(float(raw_prob), 6),
            "calibrated_probability": round(float(calibrated_prob), 6),
            "risk_level": self._risk_level(calibrated_prob),
            "top_positive_features": [f["feature"] for f in top_pos],
            "top_negative_features": [f["feature"] for f in top_neg],
            "top_positive_details": top_pos,
            "top_negative_details": top_neg,
            "plain_english_explanation": self._narrate(
                machine_id, calibrated_prob, top_pos, top_neg
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return card

    # ------------------------------------------------------------------
    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob >= 0.80:
            return "CRITICAL"
        elif prob >= 0.50:
            return "HIGH"
        elif prob >= 0.25:
            return "MEDIUM"
        else:
            return "LOW"

    # ------------------------------------------------------------------
    def _narrate(
        self,
        machine_id: int,
        calibrated_prob: float,
        top_pos: List[Dict],
        top_neg: List[Dict],
    ) -> str:
        """Generate a plain-English explanation narrative."""
        risk = self._risk_level(calibrated_prob)
        lines = [
            f"Machine {machine_id} — Failure Risk {calibrated_prob * 100:.1f}% ({risk})",
            "",
            "Risk increased because:",
        ]
        for f in top_pos:
            fname = f["feature"].replace("_", " ")
            lines.append(f"  • {fname} = {f['value']:.3f} (SHAP +{f['shap']:.4f})")
        lines.append("")
        lines.append("Risk decreased because:")
        for f in top_neg:
            fname = f["feature"].replace("_", " ")
            lines.append(f"  • {fname} = {f['value']:.3f} (SHAP {f['shap']:.4f})")
        if not top_neg:
            lines.append("  • No strong stabilising features identified.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def save_card(self, card: Dict[str, Any], filename: str = "prediction_card.json") -> Path:
        out = self.reports_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, default=str)
        logger.info("Saved prediction card: %s", out)
        return out

    # ------------------------------------------------------------------
    def generate_example_cards(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        explainer: "SHAPExplainer",
        raw_model: Any,
        calibrated_model: Any,
        n_high: int = 3,
        n_low: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Automatically select interesting machine-hours (high + low risk)
        and generate explanation cards for each.
        """
        logger.info("Generating %d high-risk + %d low-risk explanation cards ...",
                    n_high, n_low)
        X_all = df[feature_cols].select_dtypes(include=[np.number]).copy()
        raw_probs = raw_model.predict_proba(X_all)[:, 1]
        cal_probs = calibrated_model.predict_proba(X_all)[:, 1]

        df = df.copy()
        df["_raw_prob"] = raw_probs
        df["_cal_prob"] = cal_probs

        high_rows = df.nlargest(n_high, "_cal_prob")
        low_rows = df.nsmallest(n_low, "_cal_prob")
        selected = pd.concat([high_rows, low_rows])

        cards = []
        for _, row in selected.iterrows():
            machine_id = row.get("machineID", -1)
            timestamp = row.get("datetime", "unknown")
            X_row = pd.DataFrame([row[feature_cols].values], columns=feature_cols)
            X_row = X_row.select_dtypes(include=[np.number])
            sv, _ = explainer.compute_local(X_row)
            card = self.build_explanation_card(
                machine_id=machine_id,
                timestamp=timestamp,
                X_row=X_row,
                shap_values=sv,
                raw_prob=float(row["_raw_prob"]),
                calibrated_prob=float(row["_cal_prob"]),
            )
            cards.append(card)
            logger.info(
                "  Card for Machine %s @ %s — Risk %.1f%% (%s)",
                machine_id, timestamp,
                float(row["_cal_prob"]) * 100,
                card["risk_level"],
            )
        return cards


# ===========================================================================
# 4. ExplainabilityReportWriter
# ===========================================================================

class ExplainabilityReportWriter:
    """Writes the markdown explainability report."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        shap_importance_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        example_cards: List[Dict[str, Any]],
        global_shap_computed_on: int,
    ) -> Path:
        top5 = shap_importance_df.head(5)
        lines = [
            "# PredictGuard — Explainability Report",
            "",
            f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## 1. Overview",
            "",
            "This report documents **why** the PredictGuard XGBoost model predicts failure for any "
            "given machine-hour, using SHAP (SHapley Additive exPlanations). "
            "It covers:",
            "",
            "- **Global explanations**: which features matter most across all predictions",
            "- **Local explanations**: why a specific machine was flagged",
            "- **Native vs SHAP importance**: a critical comparison",
            "",
            "---",
            "",
            "## 2. Why SHAP Instead of Built-in Feature Importance?",
            "",
            "| Property | Native Gain Importance | SHAP |",
            "|---|---|---|",
            "| Handles feature interactions | ❌ No | ✅ Yes |",
            "| Consistent across models | ❌ No | ✅ Yes (Shapley axioms) |",
            "| Per-prediction explanations | ❌ No | ✅ Yes |",
            "| Direction of effect (push up/down) | ❌ No | ✅ Yes |",
            "| Accounts for correlated features | ❌ Partially | ✅ Better |",
            "",
            "**Conclusion**: Native gain importance can over-count high-cardinality features "
            "and ignores feature interactions. SHAP provides a theoretically grounded, "
            "consistent measure guaranteed by the Shapley value axioms (Efficiency, Symmetry, Dummy, Additivity).",
            "",
            "---",
            "",
            "## 3. Important Note on Raw vs Calibrated Model",
            "",
            "> ⚠️ **SHAP values are computed exclusively on the RAW (un-calibrated) XGBoost model.**",
            ">",
            "> Reason: `CalibratedClassifierCV` (Platt Scaling) wraps the tree with an isotonic/sigmoid "
            "> layer that maps raw scores → calibrated probabilities. The SHAP `TreeExplainer` explains "
            "> the **tree structure** — not the post-hoc sigmoid layer. Using it on the calibrated wrapper "
            "> would produce incorrect or misleading SHAP values.",
            "",
            "| Quantity | Source |",
            "|---|---|",
            "| Raw Score (log-odds) | Raw XGBoost model |",
            "| Calibrated Probability | Sigmoid-calibrated wrapper |",
            "| SHAP Values | Raw XGBoost model ✅ |",
            "",
            "---",
            "",
            "## 4. Global Feature Importance (SHAP)",
            "",
            f"> Computed on a stratified random sample of **{global_shap_computed_on:,} rows** "
            f"from the Development set.",
            "",
            "### Top 10 Features by Mean |SHAP|",
            "",
            "| Rank | Feature | Mean |SHAP| |",
            "|---|---|---|",
        ]
        for i, row in shap_importance_df.head(10).iterrows():
            lines.append(f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.6f} |")

        lines += [
            "",
            "---",
            "",
            "## 5. SHAP vs Native Importance Comparison",
            "",
            "### Top 10 Features — Rank Comparison",
            "",
            "| Feature | SHAP Rank | Native Rank | |Rank Delta| |",
            "|---|---|---|---|",
        ]
        for _, row in comparison_df.head(10).iterrows():
            delta = int(row["rank_delta"])
            flag = "⚠️" if delta > 5 else "✅"
            lines.append(
                f"| `{row['feature']}` | {int(row['shap_rank'])} | "
                f"{int(row['native_rank'])} | {delta} {flag} |"
            )

        lines += [
            "",
            "**Discussion:**",
            "Features with large rank deltas (marked ⚠️) indicate cases where native gain importance "
            "is misleading relative to SHAP. This commonly occurs for features that appear frequently "
            "in splits but have low marginal impact (high-frequency, low-gain leaves).",
            "",
            "---",
            "",
            "## 6. Example Local Explanations",
            "",
        ]

        for card in example_cards:
            lines += [
                f"### Machine {card['machine_id']} — {card['timestamp']}",
                "",
                f"| Property | Value |",
                "|---|---|",
                f"| Raw Score | `{card['raw_score']:.4f}` |",
                f"| Calibrated Probability | `{card['calibrated_probability'] * 100:.1f}%` |",
                f"| Risk Level | **{card['risk_level']}** |",
                "",
                "**Plain English:**",
                "",
                f"```",
                card["plain_english_explanation"],
                "```",
                "",
                "**Top Positive Drivers (increase failure risk):**",
                "",
            ]
            for f in card["top_positive_details"][:5]:
                lines.append(
                    f"- `{f['feature']}` = {f['value']:.4f} — SHAP: **+{f['shap']:.4f}**"
                )
            lines += [
                "",
                "**Top Negative Drivers (decrease failure risk):**",
                "",
            ]
            for f in card["top_negative_details"][:5]:
                lines.append(
                    f"- `{f['feature']}` = {f['value']:.4f} — SHAP: **{f['shap']:.4f}**"
                )
            lines.append("")
            lines.append("---")
            lines.append("")

        lines += [
            "## 7. Figures Generated",
            "",
            "| Figure | Description |",
            "|---|---|",
            "| `shap_summary.png` | Global SHAP dot summary — distribution of feature impacts |",
            "| `shap_beeswarm.png` | Beeswarm — feature value vs SHAP value colour-coded |",
            "| `shap_bar.png` | Mean |SHAP| bar chart — top features by global importance |",
            "| `shap_waterfall.png` | Waterfall for highest-risk machine-hour |",
            "| `shap_force.png` | Force plot for highest-risk machine-hour |",
            "| `native_importance.png` | XGBoost native gain-based importance |",
            "| `importance_comparison.png` | SHAP vs native side-by-side comparison |",
            "",
            "---",
            "",
            "## 8. Limitations",
            "",
            "- SHAP values explain **the model's decision**, not the ground truth.",
            "- TreeExplainer with `interventional` perturbation assumes feature independence; "
            "correlated features (e.g., rolling windows at different horizons) may share SHAP mass.",
            "- Local explanations are computed on individual rows; "
            "they do not represent the machine's overall maintenance risk over time.",
            "",
        ]

        out = self.reports_dir / "explainability_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved explainability report: %s", out)
        return out
