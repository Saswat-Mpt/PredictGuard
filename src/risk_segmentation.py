"""
risk_segmentation.py
====================
PredictGuard — Phase 3 Stage 11: Fleet Risk Segmentation.

Converts calibrated probabilities into actionable fleet-level risk intelligence:
    RiskSegmenter   — assigns risk scores & tiers to each machine-hour
    FleetAnalyzer   — aggregates machine-level statistics across the fleet
    FleetReportWriter — generates fleet_segmentation_report.md

Risk Score: RiskScore = 100 × calibrated_probability (0–100 integer scale)

Default risk tiers (configurable):
    LOW      [  0,  20)
    MEDIUM   [ 20,  50)
    HIGH     [ 50,  75)
    CRITICAL [ 75, 100]

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default tier configuration
# ---------------------------------------------------------------------------
DEFAULT_TIER_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "LOW":      (0.0,  20.0),
    "MEDIUM":   (20.0, 50.0),
    "HIGH":     (50.0, 75.0),
    "CRITICAL": (75.0, 100.01),   # inclusive upper bound
}

TIER_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
TIER_COLORS = {
    "LOW":      "#2ecc71",
    "MEDIUM":   "#f39c12",
    "HIGH":     "#e67e22",
    "CRITICAL": "#e74c3c",
}
FIGURE_DPI = 150

sns.set_theme(
    style="darkgrid",
    rc={"figure.dpi": FIGURE_DPI, "axes.titlesize": 13, "axes.labelsize": 11},
)


# ===========================================================================
# 1. RiskSegmenter
# ===========================================================================

class RiskSegmenter:
    """
    Assigns risk scores and risk tiers to each machine-hour prediction.

    RiskScore = round(100 × calibrated_probability)

    Parameters
    ----------
    tier_thresholds : dict mapping tier name → (lower_bound, upper_bound)
                      Bounds are [lower, upper) except CRITICAL which is inclusive.
    """

    def __init__(
        self,
        tier_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.tier_thresholds = tier_thresholds or DEFAULT_TIER_THRESHOLDS

    # ------------------------------------------------------------------
    def assign_risk_scores(
        self,
        df: pd.DataFrame,
        calibrated_model: Any,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Compute calibrated probabilities and risk scores for the full DataFrame.

        Adds columns:
            calibrated_prob  — float in [0, 1]
            risk_score       — integer in [0, 100]
            risk_tier        — one of LOW / MEDIUM / HIGH / CRITICAL

        Returns a copy of df with new columns appended.
        """
        logger.info("Computing calibrated probabilities for %d rows ...", len(df))
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        proba = calibrated_model.predict_proba(X)[:, 1]

        out = df.copy()
        out["calibrated_prob"] = proba
        out["risk_score"] = (proba * 100).round().astype(int).clip(0, 100)
        out["risk_tier"] = out["risk_score"].apply(self._assign_tier)
        logger.info("Risk scores assigned. Tier distribution:\n%s",
                    out["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0).to_string())
        return out

    # ------------------------------------------------------------------
    def _assign_tier(self, score: int) -> str:
        for tier in TIER_ORDER:
            lo, hi = self.tier_thresholds[tier]
            if lo <= score < hi:
                return tier
        return "CRITICAL"   # fallback for score == 100

    # ------------------------------------------------------------------
    def assign_component_predictions(
        self,
        df: pd.DataFrame,
        component_predictor: Any,
    ) -> pd.DataFrame:
        """
        Append predicted_component and component_confidence columns.
        Only fills for rows where calibrated_prob >= 0.05 to save compute.
        """
        logger.info("Predicting failure components for high-risk rows ...")
        feat_cols = component_predictor.feature_cols
        # Predict on all rows (model handles low-risk gracefully)
        available_cols = [c for c in feat_cols if c in df.columns]
        if len(available_cols) < len(feat_cols):
            missing = set(feat_cols) - set(available_cols)
            logger.warning("Missing %d component feature columns — padding with 0", len(missing))
            for mc in missing:
                df = df.copy()
                df[mc] = 0.0
        proba_df = component_predictor.predict_proba_df(df[feat_cols])
        df = df.copy()
        df["predicted_component"] = proba_df["predicted_component"].values
        df["component_confidence"] = proba_df["component_confidence"].values
        return df


# ===========================================================================
# 2. FleetAnalyzer
# ===========================================================================

class FleetAnalyzer:
    """
    Aggregates machine-level statistics from the risk-scored DataFrame.

    Produces:
        - Tier summary (counts, %, avg/max risk, failure rate, component distribution)
        - Machine ranking (top 20 highest-risk and 20 lowest-risk)
        - Per-machine latest-snapshot summary
    """

    def __init__(self, figures_dir: Path, reports_dir: Path) -> None:
        self.figures_dir = Path(figures_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build_machine_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse to one row per machine: latest snapshot + aggregate stats.

        Returns a DataFrame with columns:
            machineID, latest_risk_score, avg_risk_score, max_risk_score,
            risk_tier, calibrated_prob, predicted_component, y_failure_rate
        """
        logger.info("Building per-machine summary from %d rows ...", len(df))
        agg = (
            df.groupby("machineID")
            .agg(
                avg_risk_score=("risk_score", "mean"),
                max_risk_score=("risk_score", "max"),
                avg_calibrated_prob=("calibrated_prob", "mean"),
                max_calibrated_prob=("calibrated_prob", "max"),
                failure_rate=("y_failure", "mean"),
                n_hours=("risk_score", "count"),
            )
            .reset_index()
        )

        # Latest snapshot per machine
        latest = (
            df.sort_values("datetime")
            .groupby("machineID")
            .last()
            .reset_index()[["machineID", "risk_score", "risk_tier",
                             "calibrated_prob", "predicted_component",
                             "component_confidence"]]
            .rename(columns={
                "risk_score": "latest_risk_score",
                "calibrated_prob": "latest_calibrated_prob",
            })
        )
        summary = latest.merge(agg, on="machineID", how="left")
        summary = summary.sort_values("latest_risk_score", ascending=False).reset_index(drop=True)
        logger.info("Machine summary built: %d unique machines", len(summary))
        return summary

    # ------------------------------------------------------------------
    def build_tier_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build per-tier aggregation table."""
        rows = []
        total_machines = df["machineID"].nunique()
        for tier in TIER_ORDER:
            subset = df[df["risk_tier"] == tier]
            n_machines = subset["machineID"].nunique()
            rows.append({
                "tier": tier,
                "n_machine_hours": len(subset),
                "n_unique_machines": n_machines,
                "pct_machines": round(n_machines / total_machines * 100, 2) if total_machines else 0,
                "avg_risk_score": round(subset["risk_score"].mean(), 2) if len(subset) else 0,
                "max_risk_score": int(subset["risk_score"].max()) if len(subset) else 0,
                "avg_calibrated_prob": round(subset["calibrated_prob"].mean(), 4) if len(subset) else 0,
                "failure_rate": round(subset["y_failure"].mean(), 4) if len(subset) else 0,
                "top_component": (
                    subset["predicted_component"].mode().iloc[0]
                    if len(subset) > 0 and subset["predicted_component"].notna().any()
                    else "N/A"
                ),
            })
        tier_df = pd.DataFrame(rows)
        logger.info("Tier summary:\n%s", tier_df.to_string(index=False))
        return tier_df

    # ------------------------------------------------------------------
    def build_top_machines(
        self, machine_summary: pd.DataFrame, top_n: int = 20
    ) -> pd.DataFrame:
        """Return the top_n highest-risk machines."""
        cols = [
            "machineID", "latest_risk_score", "latest_calibrated_prob",
            "risk_tier", "predicted_component", "component_confidence",
            "avg_risk_score", "max_risk_score", "failure_rate",
        ]
        available = [c for c in cols if c in machine_summary.columns]
        top = machine_summary.head(top_n)[available].copy()
        top.index = range(1, len(top) + 1)
        top.index.name = "rank"
        return top

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot_fleet_risk_distribution(self, df: pd.DataFrame) -> Path:
        """Histogram of risk scores across all machine-hours."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: full distribution
        axes[0].hist(df["risk_score"], bins=50, color="#4c72b0", alpha=0.85, edgecolor="white")
        axes[0].set_xlabel("Risk Score (0–100)")
        axes[0].set_ylabel("Machine-Hours")
        axes[0].set_title("Fleet Risk Score Distribution")
        # Add tier shade regions
        for tier, (lo, hi) in DEFAULT_TIER_THRESHOLDS.items():
            axes[0].axvspan(lo, min(hi, 100), alpha=0.07, color=TIER_COLORS[tier], label=tier)
        axes[0].legend(fontsize=9)

        # Right: log scale
        axes[1].hist(df["risk_score"], bins=50, color="#dd8452", alpha=0.85, edgecolor="white",
                     log=True)
        axes[1].set_xlabel("Risk Score (0–100)")
        axes[1].set_ylabel("Machine-Hours (log scale)")
        axes[1].set_title("Fleet Risk Score Distribution (Log Scale)")

        fig.suptitle("PredictGuard — Fleet Risk Score Overview", fontsize=14)
        plt.tight_layout()
        out = self.figures_dir / "fleet_risk_distribution.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_tier_pie(self, tier_df: pd.DataFrame) -> Path:
        """Pie chart of machine-hours per tier."""
        labels = [
            f"{row['tier']}\n{row['n_machine_hours']:,} hrs\n({row['pct_machines']:.1f}% machines)"
            for _, row in tier_df.iterrows()
        ]
        sizes = tier_df["n_machine_hours"].values
        colors = [TIER_COLORS[t] for t in tier_df["tier"]]

        fig, ax = plt.subplots(figsize=(9, 7))
        wedges, texts = ax.pie(
            sizes, labels=labels, colors=colors, startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Fleet Risk Tier Distribution", fontsize=15)
        plt.tight_layout()
        out = self.figures_dir / "fleet_tier_pie.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_tier_bar(self, tier_df: pd.DataFrame) -> Path:
        """Bar chart of tier counts with failure rate overlay."""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        colors = [TIER_COLORS[t] for t in tier_df["tier"]]
        bars = ax1.bar(
            tier_df["tier"], tier_df["n_machine_hours"],
            color=colors, alpha=0.85, edgecolor="white", linewidth=1.5
        )
        ax1.bar_label(bars, fmt="%,.0f", padding=5)
        ax1.set_ylabel("Machine-Hours", fontsize=11)
        ax1.set_title("Risk Tier Distribution with Failure Rate", fontsize=14)

        ax2 = ax1.twinx()
        ax2.plot(
            tier_df["tier"], tier_df["failure_rate"] * 100,
            "D--", color="#2c3e50", markersize=9, linewidth=2, label="Failure Rate %"
        )
        ax2.set_ylabel("Failure Rate (%)", fontsize=11, color="#2c3e50")
        ax2.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        out = self.figures_dir / "fleet_tier_bar.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_machine_ranking(self, top_machines: pd.DataFrame) -> Path:
        """Horizontal bar chart of top 20 machines by risk score."""
        df = top_machines.head(20).copy()
        df = df.sort_values("latest_risk_score", ascending=True)
        colors = [TIER_COLORS.get(t, "#7f8c8d") for t in df["risk_tier"]]

        fig, ax = plt.subplots(figsize=(11, 8))
        bars = ax.barh(
            [f"Machine {m}" for m in df["machineID"]],
            df["latest_risk_score"],
            color=colors, alpha=0.88, edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=9)
        ax.set_xlabel("Risk Score (0–100)")
        ax.set_title("Top Highest-Risk Machines", fontsize=14)
        ax.axvline(75, color="#e74c3c", linestyle="--", linewidth=1.5, label="CRITICAL threshold")
        ax.axvline(50, color="#e67e22", linestyle="--", linewidth=1.2, label="HIGH threshold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = self.figures_dir / "fleet_machine_ranking.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_failure_rate_by_tier(self, tier_df: pd.DataFrame) -> Path:
        """Bar chart of failure rate per tier."""
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [TIER_COLORS[t] for t in tier_df["tier"]]
        bars = ax.bar(
            tier_df["tier"], tier_df["failure_rate"] * 100,
            color=colors, alpha=0.88, edgecolor="white"
        )
        ax.bar_label(bars, fmt="%.2f%%", padding=5, fontsize=10)
        ax.set_ylabel("Actual Failure Rate (%)")
        ax.set_title("Actual Failure Rate by Risk Tier\n(Validates Risk Segmentation Quality)")
        plt.tight_layout()
        out = self.figures_dir / "fleet_failure_rate_by_tier.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_component_by_tier(self, df: pd.DataFrame) -> Path:
        """Stacked bar chart: component distribution per risk tier."""
        pivot = (
            df[df["y_failure"] == 1]
            .groupby(["risk_tier", "predicted_component"])
            .size()
            .unstack(fill_value=0)
            .reindex(TIER_ORDER)
        )
        pivot = pivot.fillna(0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).fillna(0) * 100

        comp_colors = sns.color_palette("Set2", pivot_pct.shape[1])
        ax = pivot_pct.plot(
            kind="bar", figsize=(10, 6), color=comp_colors,
            edgecolor="white", linewidth=1.2
        )
        ax.set_xlabel("Risk Tier")
        ax.set_ylabel("Component Share (%)")
        ax.set_title("Predicted Failure Component Distribution by Risk Tier")
        ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=0)
        plt.tight_layout()
        out = self.figures_dir / "fleet_component_by_tier.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_risk_histogram(self, machine_summary: pd.DataFrame) -> Path:
        """Histogram of per-machine latest risk scores."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            machine_summary["latest_risk_score"], bins=30,
            color="#8e44ad", alpha=0.85, edgecolor="white"
        )
        for tier, (lo, hi) in DEFAULT_TIER_THRESHOLDS.items():
            ax.axvspan(lo, min(hi, 100), alpha=0.08, color=TIER_COLORS[tier], label=tier)
        ax.set_xlabel("Latest Risk Score per Machine")
        ax.set_ylabel("Number of Machines")
        ax.set_title("Per-Machine Risk Score Histogram (Most Recent Snapshot)")
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = self.figures_dir / "fleet_risk_histogram.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out


# ===========================================================================
# 3. FleetReportWriter
# ===========================================================================

class FleetReportWriter:
    """Writes fleet_segmentation_report.md."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        tier_df: pd.DataFrame,
        top_machines: pd.DataFrame,
        machine_summary: pd.DataFrame,
        tier_thresholds: Dict[str, Tuple[float, float]],
    ) -> Path:
        lines = [
            "# PredictGuard — Fleet Risk Segmentation Report",
            "",
            f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## 1. Overview",
            "",
            "Stage 11 converts every machine-hour's calibrated probability into a fleet-wide",
            "**Risk Score** (0–100) and assigns it to one of four **Risk Tiers**.",
            "This transforms individual predictions into a prioritised maintenance queue.",
            "",
            "**Risk Score** = `round(100 × calibrated_probability)`",
            "",
            "---",
            "",
            "## 2. Risk Tier Definitions",
            "",
            "| Tier | Score Range | Recommended Action |",
            "|---|---|---|",
            "| 🟢 LOW | 0–19 | Continue scheduled monitoring |",
            "| 🟡 MEDIUM | 20–49 | Increase monitoring frequency |",
            "| 🟠 HIGH | 50–74 | Schedule maintenance within 48h |",
            "| 🔴 CRITICAL | 75–100 | Dispatch technician immediately |",
            "",
            "> Thresholds are fully configurable in `config.yaml` under `risk_segmentation`.",
            "",
            "---",
            "",
            "## 3. Fleet Tier Summary",
            "",
            "| Tier | Machine-Hours | Unique Machines | % Machines | Avg Risk | Max Risk | Failure Rate | Top Component |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, row in tier_df.iterrows():
            lines.append(
                f"| **{row['tier']}** | {row['n_machine_hours']:,} | {row['n_unique_machines']} | "
                f"{row['pct_machines']:.1f}% | {row['avg_risk_score']:.1f} | "
                f"{row['max_risk_score']} | {row['failure_rate']*100:.2f}% | `{row['top_component']}` |"
            )

        lines += [
            "",
            "---",
            "",
            "## 4. Machine Ranking (Top 20 Highest-Risk)",
            "",
            "| Rank | Machine ID | Risk Score | Tier | Calibrated Prob | Component | Avg Risk | Failure Rate |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for rank, (_, row) in enumerate(top_machines.iterrows(), 1):
            lines.append(
                f"| {rank} | `{int(row['machineID'])}` | **{int(row['latest_risk_score'])}** | "
                f"{row['risk_tier']} | {row.get('latest_calibrated_prob', 0)*100:.1f}% | "
                f"`{row.get('predicted_component', 'N/A')}` | "
                f"{row.get('avg_risk_score', 0):.1f} | {row.get('failure_rate', 0)*100:.2f}% |"
            )

        lines += [
            "",
            "---",
            "",
            "## 5. Methodology",
            "",
            "1. **Calibrated Probability** from Sigmoid-calibrated XGBoost (Phase 2, Stage 8).",
            "2. **Risk Score** = `round(100 × calibrated_prob)` — integer scale for readability.",
            "3. **Tier Assignment** — configurable thresholds, no arbitrary hard-coding.",
            "4. **Machine-Level Aggregation** — each machine's *latest* snapshot determines",
            "   its current tier; historical averages are also tracked.",
            "",
            "---",
            "",
            "## 6. Figures",
            "",
            "| Figure | Description |",
            "|---|---|",
            "| `fleet_risk_distribution.png` | Histogram of risk scores (normal + log scale) |",
            "| `fleet_tier_pie.png` | Pie chart of tier distribution |",
            "| `fleet_tier_bar.png` | Bar chart with failure rate overlay |",
            "| `fleet_machine_ranking.png` | Top 20 machines by risk score |",
            "| `fleet_failure_rate_by_tier.png` | Actual failure rate per tier |",
            "| `fleet_component_by_tier.png` | Component distribution per tier |",
            "| `fleet_risk_histogram.png` | Per-machine latest risk score |",
            "",
        ]

        out = self.reports_dir / "fleet_segmentation_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved fleet segmentation report: %s", out)
        return out
