"""
cost_decision.py
================
PredictGuard — Phase 3 Stage 12: Cost-Sensitive Threshold Optimization & Dispatch Decisions.

Answers: "Should we dispatch a technician?" by minimising expected cost over a
sweep of probability thresholds on the Development set (no test leakage).

Classes:
    ThresholdOptimizer  — sweeps thresholds on dev set, selects minimum-cost point
    CostEvaluator       — computes expected cost & confusion matrix for any threshold
    DecisionEngine      — applies optimal threshold to produce per-machine dispatch decisions
    CostReportWriter    — writes cost_decision_report.md

Cost Model:
    Expected Cost(t) = C_FN × FN(t) + C_FP × FP(t)

    C_FN = cost of missing a real failure (emergency repair + downtime)
    C_FP = cost of a false alarm (unnecessary dispatch + inspection)

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default cost matrix (all values configurable)
# ---------------------------------------------------------------------------
DEFAULT_COST_MATRIX: Dict[str, float] = {
    "C_FN": 10_000.0,   # False Negative: emergency repair + downtime + safety
    "C_FP":    500.0,   # False Positive: technician dispatch + inspection downtime
}

FIGURE_DPI = 150

sns.set_theme(
    style="darkgrid",
    rc={"figure.dpi": FIGURE_DPI, "axes.titlesize": 13, "axes.labelsize": 11},
)


# ===========================================================================
# 1. CostEvaluator
# ===========================================================================

class CostEvaluator:
    """
    Computes confusion matrix, expected cost, and standard metrics for one threshold.
    """

    def __init__(self, cost_matrix: Optional[Dict[str, float]] = None) -> None:
        self.cost_matrix = cost_matrix or DEFAULT_COST_MATRIX

    # ------------------------------------------------------------------
    def evaluate(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        """
        Evaluate at a single threshold.

        Returns
        -------
        dict with keys: threshold, TP, FP, TN, FN, precision, recall, f1,
                        expected_cost, dispatch_rate, cost_per_dispatch
        """
        y_pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        c_fn = self.cost_matrix["C_FN"]
        c_fp = self.cost_matrix["C_FP"]
        expected_cost = c_fn * fn + c_fp * fp

        n = len(y_true)
        dispatch_rate = float((y_pred == 1).sum()) / n if n else 0.0

        return {
            "threshold": round(float(threshold), 4),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "expected_cost": round(float(expected_cost), 2),
            "dispatch_rate": round(dispatch_rate, 4),
            "cost_per_dispatch": round(expected_cost / max((y_pred == 1).sum(), 1), 2),
        }


# ===========================================================================
# 2. ThresholdOptimizer
# ===========================================================================

class ThresholdOptimizer:
    """
    Sweeps probability thresholds on the Development set (NO test data!)
    and selects the threshold that minimises expected cost.

    IMPORTANT: The optimal threshold is chosen exclusively on dev data.
               It is applied to the test set only ONCE for final evaluation.
    """

    def __init__(
        self,
        cost_matrix: Optional[Dict[str, float]] = None,
        threshold_min: float = 0.05,
        threshold_max: float = 0.95,
        threshold_step: float = 0.01,
    ) -> None:
        self.cost_matrix = cost_matrix or DEFAULT_COST_MATRIX
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_step = threshold_step
        self.evaluator = CostEvaluator(cost_matrix=self.cost_matrix)
        self.sweep_df: Optional[pd.DataFrame] = None
        self.optimal_threshold: Optional[float] = None
        self.optimal_metrics: Optional[Dict] = None

    # ------------------------------------------------------------------
    def sweep(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
    ) -> pd.DataFrame:
        """
        Evaluate all thresholds from threshold_min to threshold_max.

        Returns
        -------
        sweep_df : DataFrame with one row per threshold
        """
        thresholds = np.arange(
            self.threshold_min, self.threshold_max + self.threshold_step / 2,
            self.threshold_step
        )
        logger.info(
            "Sweeping %d thresholds on Development set (%d rows, %d failures) ...",
            len(thresholds), len(y_true), int(y_true.sum())
        )
        rows = [self.evaluator.evaluate(y_true, proba, t) for t in thresholds]
        self.sweep_df = pd.DataFrame(rows)
        return self.sweep_df

    # ------------------------------------------------------------------
    def find_optimal(self) -> Tuple[float, Dict]:
        """
        Select the threshold with minimum expected cost from the sweep.

        Returns
        -------
        optimal_threshold : float
        optimal_metrics   : dict of metrics at optimal threshold
        """
        if self.sweep_df is None:
            raise RuntimeError("Call sweep() first.")
        best_row = self.sweep_df.loc[self.sweep_df["expected_cost"].idxmin()]
        self.optimal_threshold = float(best_row["threshold"])
        self.optimal_metrics = best_row.to_dict()
        logger.info(
            "Optimal threshold: %.2f  |  Expected Cost: ${:,.0f}  |  Recall: {:.3f}  |  Precision: {:.3f}".format(
                self.optimal_metrics["expected_cost"],
                self.optimal_metrics["recall"],
                self.optimal_metrics["precision"],
            ),
            self.optimal_threshold,
        )
        return self.optimal_threshold, self.optimal_metrics

    # ------------------------------------------------------------------
    def save_optimal(self, reports_dir: Path) -> Path:
        """Save optimal threshold metadata as JSON."""
        out = Path(reports_dir) / "optimal_threshold.json"
        payload = {
            "optimal_threshold": self.optimal_threshold,
            "cost_matrix": self.cost_matrix,
            "metrics_at_optimal": self.optimal_metrics,
            "sweep_range": {
                "min": self.threshold_min,
                "max": self.threshold_max,
                "step": self.threshold_step,
            },
            "selected_on": "development_set_only",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved optimal threshold JSON: %s", out)
        return out


# ===========================================================================
# 3. DecisionEngine
# ===========================================================================

class DecisionEngine:
    """
    Applies the optimal threshold to generate dispatch decisions for every machine.

    For each machine, produces:
        - DISPATCH or MONITOR decision
        - Risk tier
        - Predicted component
        - Expected cost impact
        - Human-readable maintenance recommendation narrative
    """

    DISPATCH_ACTIONS = {
        "comp1": "Dispatch hydraulic specialist. Inspect pressure seals and pump.",
        "comp2": "Dispatch mechanical engineer. Inspect rotary bearing and lubrication.",
        "comp3": "Dispatch electrician. Check voltage regulators and control circuits.",
        "comp4": "Dispatch vibration analyst. Inspect dampeners and drive belt.",
    }
    MONITOR_ACTION = "Continue scheduled monitoring. No immediate action required."

    def __init__(
        self,
        optimal_threshold: float,
        cost_matrix: Optional[Dict[str, float]] = None,
    ) -> None:
        self.optimal_threshold = optimal_threshold
        self.cost_matrix = cost_matrix or DEFAULT_COST_MATRIX

    # ------------------------------------------------------------------
    def generate_decisions(
        self,
        machine_summary: pd.DataFrame,
        df_scored: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate one dispatch decision per machine (based on latest risk snapshot).

        Returns
        -------
        decisions_df : DataFrame with decision details per machine
        """
        logger.info(
            "Generating dispatch decisions (threshold=%.2f) for %d machines ...",
            self.optimal_threshold, len(machine_summary)
        )
        decisions = machine_summary.copy()
        decisions["decision"] = np.where(
            decisions["max_calibrated_prob"] >= self.optimal_threshold,
            "DISPATCH",
            "MONITOR",
        )
        decisions["decision_reason"] = decisions.apply(
            lambda r: (
                f"Peak risk {r['max_calibrated_prob']*100:.1f}% ≥ "
                f"optimal threshold {self.optimal_threshold*100:.1f}%."
            )
            if r["decision"] == "DISPATCH"
            else (
                f"Peak risk {r['max_calibrated_prob']*100:.1f}% < "
                f"optimal threshold {self.optimal_threshold*100:.1f}%."
            ),
            axis=1,
        )
        decisions["recommended_action"] = decisions.apply(
            lambda r: self.DISPATCH_ACTIONS.get(
                r.get("predicted_component", ""), "Dispatch maintenance team."
            ) if r["decision"] == "DISPATCH" else self.MONITOR_ACTION,
            axis=1,
        )
        n_dispatch = (decisions["decision"] == "DISPATCH").sum()
        logger.info(
            "Decisions: DISPATCH=%d  MONITOR=%d  (%.1f%% dispatch rate)",
            n_dispatch, len(decisions) - n_dispatch,
            n_dispatch / len(decisions) * 100 if len(decisions) else 0
        )
        return decisions

    # ------------------------------------------------------------------
    def build_example_reports(
        self,
        decisions_df: pd.DataFrame,
        n_dispatch: int = 3,
        n_monitor: int = 2,
    ) -> List[Dict]:
        """Build human-readable example decision narratives."""
        dispatch_rows = decisions_df[decisions_df["decision"] == "DISPATCH"].head(n_dispatch)
        monitor_rows = decisions_df[decisions_df["decision"] == "MONITOR"].tail(n_monitor)
        examples = []
        for _, row in pd.concat([dispatch_rows, monitor_rows]).iterrows():
            examples.append({
                "machine_id": int(row["machineID"]),
                "risk_score": int(row.get("latest_risk_score", row.get("max_risk_score", 0))),
                "risk_tier": row.get("risk_tier", "N/A"),
                "calibrated_prob": round(float(row.get("max_calibrated_prob", row.get("latest_calibrated_prob", 0))), 4),
                "predicted_component": row.get("predicted_component", "N/A"),
                "component_confidence": round(float(row.get("component_confidence", 0)), 4),
                "decision": row["decision"],
                "decision_reason": row["decision_reason"],
                "recommended_action": row["recommended_action"],
                "avg_risk_score": round(float(row.get("avg_risk_score", 0)), 2),
                "failure_rate": round(float(row.get("failure_rate", 0)), 4),
            })
        return examples

    # ------------------------------------------------------------------
    def compute_cost_savings(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        baseline_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compare cost at optimal threshold vs naive 0.5 baseline."""
        evaluator = CostEvaluator(self.cost_matrix)
        baseline = evaluator.evaluate(y_true, proba, baseline_threshold)
        optimal = evaluator.evaluate(y_true, proba, self.optimal_threshold)
        savings = baseline["expected_cost"] - optimal["expected_cost"]
        pct_saving = savings / baseline["expected_cost"] * 100 if baseline["expected_cost"] else 0
        return {
            "baseline_threshold": baseline_threshold,
            "baseline_cost": baseline["expected_cost"],
            "optimal_threshold": self.optimal_threshold,
            "optimal_cost": optimal["expected_cost"],
            "cost_savings": round(savings, 2),
            "pct_saving": round(pct_saving, 2),
        }


# ===========================================================================
# 4. CostVisualizer
# ===========================================================================

class CostVisualizer:
    """Generates all Stage 12 publication-quality figures."""

    def __init__(self, figures_dir: Path) -> None:
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def plot_cost_vs_threshold(
        self, sweep_df: pd.DataFrame, optimal_threshold: float
    ) -> Path:
        """Cost curve across all thresholds with optimal marked."""
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(
            sweep_df["threshold"], sweep_df["expected_cost"] / 1000,
            color="#e74c3c", linewidth=2.5, label="Expected Cost ($k)"
        )
        ax1.axvline(
            optimal_threshold, color="#2c3e50", linestyle="--", linewidth=2,
            label=f"Optimal threshold = {optimal_threshold:.2f}"
        )
        opt_cost = sweep_df.loc[
            sweep_df["threshold"].sub(optimal_threshold).abs().idxmin(), "expected_cost"
        ] / 1000
        ax1.scatter([optimal_threshold], [opt_cost], color="#2c3e50", zorder=5, s=100)
        ax1.set_xlabel("Probability Threshold")
        ax1.set_ylabel("Expected Cost ($k)")
        ax1.set_title("Expected Cost vs Probability Threshold\n"
                       f"(C_FN=${DEFAULT_COST_MATRIX['C_FN']:,.0f}, "
                       f"C_FP=${DEFAULT_COST_MATRIX['C_FP']:,.0f})")

        ax2 = ax1.twinx()
        ax2.plot(sweep_df["threshold"], sweep_df["recall"], color="#3498db",
                 linewidth=1.8, linestyle=":", label="Recall", alpha=0.8)
        ax2.plot(sweep_df["threshold"], sweep_df["precision"], color="#27ae60",
                 linewidth=1.8, linestyle="-.", label="Precision", alpha=0.8)
        ax2.set_ylabel("Recall / Precision")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

        plt.tight_layout()
        out = self.figures_dir / "cost_vs_threshold.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_threshold_comparison(
        self, sweep_df: pd.DataFrame, optimal_threshold: float
    ) -> Path:
        """Side-by-side metrics at 0.5 vs optimal threshold."""
        baseline_row = sweep_df.loc[sweep_df["threshold"].sub(0.5).abs().idxmin()]
        optimal_row = sweep_df.loc[sweep_df["threshold"].sub(optimal_threshold).abs().idxmin()]

        metrics = ["recall", "precision", "f1", "dispatch_rate"]
        base_vals = [float(baseline_row[m]) for m in metrics]
        opt_vals = [float(optimal_row[m]) for m in metrics]
        labels = ["Recall", "Precision", "F1", "Dispatch Rate"]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - width / 2, base_vals, width, label="Baseline (t=0.50)",
                    color="#95a5a6", alpha=0.85)
        b2 = ax.bar(x + width / 2, opt_vals, width,
                    label=f"Optimal (t={optimal_threshold:.2f})",
                    color="#e74c3c", alpha=0.85)
        ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
        ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel("Value")
        ax.set_title("Baseline (0.50) vs Optimal Threshold Comparison")
        ax.legend(fontsize=10)
        plt.tight_layout()
        out = self.figures_dir / "threshold_comparison.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_cost_breakdown(
        self,
        sweep_df: pd.DataFrame,
        optimal_threshold: float,
        cost_matrix: Dict[str, float],
    ) -> Path:
        """Stacked area: FP cost vs FN cost across thresholds."""
        fp_cost = sweep_df["FP"] * cost_matrix["C_FP"] / 1000
        fn_cost = sweep_df["FN"] * cost_matrix["C_FN"] / 1000

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.stackplot(
            sweep_df["threshold"], fn_cost, fp_cost,
            labels=["FN Cost ($k) — Missed Failures", "FP Cost ($k) — Unnecessary Dispatches"],
            colors=["#e74c3c", "#f39c12"], alpha=0.75
        )
        ax.axvline(
            optimal_threshold, color="#2c3e50", linestyle="--", linewidth=2,
            label=f"Optimal threshold = {optimal_threshold:.2f}"
        )
        ax.set_xlabel("Probability Threshold")
        ax.set_ylabel("Expected Cost ($k)")
        ax.set_title("Cost Breakdown: False Negative vs False Positive Costs")
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        out = self.figures_dir / "cost_breakdown.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_dispatch_decisions(self, decisions_df: pd.DataFrame) -> Path:
        """Bar chart: DISPATCH vs MONITOR per risk tier."""
        from src.risk_segmentation import TIER_ORDER, TIER_COLORS
        pivot = (
            decisions_df.groupby(["risk_tier", "decision"])
            .size()
            .unstack(fill_value=0)
            .reindex(TIER_ORDER)
        )
        pivot = pivot.fillna(0)
        fig, ax = plt.subplots(figsize=(9, 5))
        bot = np.zeros(len(pivot))
        for col, color in [("DISPATCH", "#e74c3c"), ("MONITOR", "#2ecc71")]:
            if col in pivot.columns:
                vals = pivot[col].values
                bars = ax.bar(pivot.index, vals, bottom=bot, label=col,
                              color=color, alpha=0.85, edgecolor="white")
                bot += vals
        ax.set_xlabel("Risk Tier")
        ax.set_ylabel("Number of Machines")
        ax.set_title(
            f"Dispatch Decisions by Risk Tier\n(Optimal Threshold = {self.optimal_threshold:.2f})"
            if hasattr(self, "optimal_threshold") else "Dispatch Decisions by Risk Tier"
        )
        ax.legend(fontsize=10)
        plt.tight_layout()
        out = self.figures_dir / "dispatch_decisions.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out

    def plot_confusion_matrix_final(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Path:
        """Confusion matrix on final test set at optimal threshold."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Failure"])
        disp.plot(ax=ax, cmap="Reds", colorbar=True, values_format="d")
        ax.set_title("Final Test Set Confusion Matrix\n(at Optimal Threshold)")
        plt.tight_layout()
        out = self.figures_dir / "decision_confusion_matrix.png"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Saved figure: %s", out)
        return out


# ===========================================================================
# 5. CostReportWriter
# ===========================================================================

class CostReportWriter:
    """Writes cost_decision_report.md."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        optimal_threshold: float,
        cost_matrix: Dict[str, float],
        optimal_metrics: Dict,
        test_metrics: Dict,
        cost_savings: Dict,
        example_decisions: List[Dict],
    ) -> Path:
        lines = [
            "# PredictGuard — Cost-Based Dispatch Decision Report",
            "",
            f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
            "## 1. Objective",
            "",
            "Stage 12 answers: **Should we dispatch a technician?**",
            "",
            "Instead of a naive 0.5 default, PredictGuard selects the probability threshold",
            "that **minimises expected business cost** — accounting for the asymmetry between",
            "the cost of missing a real failure vs. the cost of an unnecessary dispatch.",
            "",
            "---",
            "",
            "## 2. Cost Assumptions",
            "",
            "> All cost values are configurable in `config.yaml` under `cost_decision`.",
            "",
            "| Cost Type | Symbol | Value | Description |",
            "|---|---|---|---|",
            f"| False Negative | `C_FN` | ${cost_matrix['C_FN']:,.0f} | Missed failure: emergency repair + downtime + safety impact |",
            f"| False Positive | `C_FP` | ${cost_matrix['C_FP']:,.0f} | False alarm: technician dispatch + inspection overhead |",
            f"| **Ratio** | `C_FN / C_FP` | **{cost_matrix['C_FN']/cost_matrix['C_FP']:.0f}×** | Missing a failure costs {cost_matrix['C_FN']/cost_matrix['C_FP']:.0f}× more than a false alarm |",
            "",
            "**Expected Cost Formula:**",
            "",
            "```",
            "Expected Cost(t) = C_FN × FN(t) + C_FP × FP(t)",
            "```",
            "",
            "---",
            "",
            "## 3. Threshold Optimisation",
            "",
            "> ⚠️ **Optimisation performed on Development set ONLY (80 machines).**",
            "> The Final Test set (20 machines) was never seen during threshold selection.",
            "",
            "| Property | Value |",
            "|---|---|",
            "| Sweep Range | 0.05 → 0.95 (step 0.01) |",
            "| Selection Criterion | Minimum Expected Cost |",
            f"| **Optimal Threshold** | **{optimal_threshold:.4f}** |",
            f"| Expected Cost at Optimal | ${optimal_metrics['expected_cost']:,.0f} |",
            f"| Recall at Optimal | {optimal_metrics['recall']:.4f} |",
            f"| Precision at Optimal | {optimal_metrics['precision']:.4f} |",
            f"| F1 at Optimal | {optimal_metrics['f1']:.4f} |",
            f"| Dispatch Rate at Optimal | {optimal_metrics['dispatch_rate']*100:.2f}% |",
            "",
            "---",
            "",
            "## 4. Final Test Set Evaluation",
            "",
            "> Applied optimal threshold **once** to the Final Test set.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Threshold | **{test_metrics['threshold']:.4f}** |",
            f"| Precision | {test_metrics['precision']:.4f} |",
            f"| Recall | **{test_metrics['recall']:.4f}** |",
            f"| F1 | {test_metrics['f1']:.4f} |",
            f"| Expected Cost | **${test_metrics['expected_cost']:,.0f}** |",
            f"| Dispatch Rate | {test_metrics['dispatch_rate']*100:.2f}% |",
            f"| TP / FP / TN / FN | {test_metrics['TP']} / {test_metrics['FP']} / {test_metrics['TN']} / {test_metrics['FN']} |",
            "",
            "### Cost Savings vs Naive 0.5 Baseline",
            "",
            "| Metric | Baseline (t=0.50) | Optimal | Saving |",
            "|---|---|---|---|",
            f"| Expected Cost | ${cost_savings['baseline_cost']:,.0f} | ${cost_savings['optimal_cost']:,.0f} | **${cost_savings['cost_savings']:,.0f} ({cost_savings['pct_saving']:.1f}%)** |",
            "",
            "---",
            "",
            "## 5. Example Dispatch Decisions",
            "",
        ]

        for ex in example_decisions:
            icon = "🚨 DISPATCH" if ex["decision"] == "DISPATCH" else "👁️ MONITOR"
            lines += [
                f"### Machine {ex['machine_id']} — {icon}",
                "",
                "| Property | Value |",
                "|---|---|",
                f"| Risk Score | **{ex['risk_score']}** / 100 |",
                f"| Risk Tier | **{ex['risk_tier']}** |",
                f"| Calibrated Probability | {ex['calibrated_prob']*100:.1f}% |",
                f"| Predicted Component | `{ex['predicted_component']}` ({ex['component_confidence']*100:.0f}% confidence) |",
                f"| Decision | **{ex['decision']}** |",
                f"| Reason | {ex['decision_reason']} |",
                f"| Action | {ex['recommended_action']} |",
                "",
            ]

        lines += [
            "---",
            "",
            "## 6. Business Interpretation",
            "",
            f"- With C_FN / C_FP = {cost_matrix['C_FN']/cost_matrix['C_FP']:.0f}×, the model correctly learns to **prefer lower thresholds** — "
            "dispatching more technicians to avoid catastrophic missed failures.",
            f"- The optimal threshold of **{optimal_threshold:.2f}** is {'lower' if optimal_threshold < 0.5 else 'higher'} than the naive 0.5 default, "
            "reflecting the asymmetric cost structure.",
            "- **Recall is the primary operational metric**: missing a real failure carries "
            f"{cost_matrix['C_FN']/cost_matrix['C_FP']:.0f}× the cost of a false alarm.",
            "",
            "---",
            "",
            "## 7. Limitations",
            "",
            "- Cost matrix values ($C_{FN}$, $C_{FP}$) are estimates. Sensitivity analysis "
            "on these parameters is recommended before production deployment.",
            "- The threshold is optimised on 80 Development machines; "
            "performance may vary on machine types not represented in training.",
            "- Expected cost assumes independence between failure events, "
            "which may not hold for correlated failure modes.",
            "",
            "---",
            "",
            "## 8. Figures",
            "",
            "| Figure | Description |",
            "|---|---|",
            "| `cost_vs_threshold.png` | Expected cost curve with optimal marked |",
            "| `threshold_comparison.png` | Baseline vs optimal threshold metrics |",
            "| `cost_breakdown.png` | FN vs FP cost stacked area chart |",
            "| `dispatch_decisions.png` | DISPATCH / MONITOR per risk tier |",
            "| `decision_confusion_matrix.png` | Final test confusion matrix at optimal threshold |",
            "",
        ]

        out = self.reports_dir / "cost_decision_report.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved cost decision report: %s", out)
        return out
