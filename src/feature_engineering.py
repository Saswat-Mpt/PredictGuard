"""
feature_engineering.py
======================
Production-grade feature engineering module for PredictGuard — Stage 3.

All feature computations strictly follow temporal and machine-level isolation:
  - Per-machine calculation via `groupby("machineID")`
  - Trailing-only rolling windows (no centered windows)
  - Shifted expanding statistics (observation t excluded from its own expanding mean/std)
  - Backward temporal joins (`pd.merge_asof` with direction='backward') for errors & maintenance
  - Safe mathematical operations (handling div-by-zero & NaN)

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENSOR_COLS: List[str] = ["volt", "rotate", "pressure", "vibration"]
ROLLING_WINDOWS: List[int] = [3, 24]
DELTA_LAGS: List[int] = [1, 3, 6, 12]

sns.set_theme(
    style="darkgrid",
    palette="muted",
    rc={"figure.dpi": 150, "axes.titlesize": 14, "axes.labelsize": 12},
)

# ===========================================================================
# 1. Trailing Rolling Statistics
# ===========================================================================

def compute_rolling_features(
    df: pd.DataFrame,
    sensors: List[str] = SENSOR_COLS,
    windows: List[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Compute trailing rolling mean, std, min, max, median, and range per machine.

    Why trailing windows?
    ---------------------
    Centered rolling windows look into the future (using observations t+1..t+k)
    which introduces catastrophic data leakage. Trailing windows evaluate strictly
    over [t-window+1, t], using only past and current observations.
    """
    logger.info("Computing trailing rolling statistics (windows=%s) ...", windows)
    result = df.copy()

    for w in windows:
        for sensor in sensors:
            # Groupby machineID and apply trailing rolling stats
            rolling = (
                result.groupby("machineID")[sensor]
                .rolling(window=w, min_periods=1)
            )
            
            mean_col = f"{sensor}_rolling_mean_{w}h"
            std_col = f"{sensor}_rolling_std_{w}h"
            min_col = f"{sensor}_rolling_min_{w}h"
            max_col = f"{sensor}_rolling_max_{w}h"
            med_col = f"{sensor}_rolling_median_{w}h"
            rng_col = f"{sensor}_rolling_range_{w}h"

            result[mean_col] = rolling.mean().reset_index(level=0, drop=True)
            result[std_col] = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
            result[min_col] = rolling.min().reset_index(level=0, drop=True)
            result[max_col] = rolling.max().reset_index(level=0, drop=True)
            result[med_col] = rolling.median().reset_index(level=0, drop=True)
            result[rng_col] = result[max_col] - result[min_col]

    return result


# ===========================================================================
# 2. Rate of Change Features
# ===========================================================================

def compute_rate_of_change_features(
    df: pd.DataFrame,
    sensors: List[str] = SENSOR_COLS,
    lags: List[int] = DELTA_LAGS,
) -> pd.DataFrame:
    """
    Compute delta_k, percentage change, rolling velocity, and acceleration per sensor.

    Formulas:
        delta_k = value(t) - value(t-k)
        pct_change = (value(t) - value(t-1)) / (value(t-1) + eps)
        velocity = delta_1h
        acceleration = delta_1h(t) - delta_1h(t-1)
    """
    logger.info("Computing rate of change features (lags=%s) ...", lags)
    result = df.copy()

    for sensor in sensors:
        grp = result.groupby("machineID")[sensor]
        
        # Lags / Deltas
        for k in lags:
            delta_col = f"{sensor}_delta_{k}h"
            result[delta_col] = grp.diff(periods=k).fillna(0.0)

        # Percentage change (1h)
        pct_col = f"{sensor}_pct_change_1h"
        prev_val = grp.shift(1)
        denom = prev_val.abs().replace(0, 1e-6)
        result[pct_col] = ((result[sensor] - prev_val) / denom).fillna(0.0)

        # Velocity & Acceleration
        vel_col = f"{sensor}_velocity"
        acc_col = f"{sensor}_acceleration"
        result[vel_col] = result[f"{sensor}_delta_1h"]
        result[acc_col] = result.groupby("machineID")[vel_col].diff(1).fillna(0.0)

    return result


# ===========================================================================
# 3. Machine Normal Deviation (Expanding Z-score)
# ===========================================================================

def compute_expanding_zscores(
    df: pd.DataFrame,
    sensors: List[str] = SENSOR_COLS,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute expanding mean, expanding std, and expanding z-score per machine.

    Why expanding statistics must be shifted by 1:
    ---------------------------------------------
    If expanding mean includes the current observation value(t), it leaks target
    information if value(t) is an extreme value preceding failure. Shifting by 1
    ensures that statistics at timestamp t are built strictly from [0, t-1].
    """
    logger.info("Computing expanding z-scores per machine (shifted by 1) ...")
    result = df.copy()

    for sensor in sensors:
        # Expanding mean/std shifted by 1 timestep
        exp_grp = result.groupby("machineID")[sensor].expanding(min_periods=1)
        exp_mean = exp_grp.mean().reset_index(level=0, drop=True)
        exp_std = exp_grp.std().reset_index(level=0, drop=True)

        # Shift per machine to exclude current row
        exp_mean_shifted = result.groupby("machineID", group_keys=False).apply(
            lambda g: exp_mean.loc[g.index].shift(1), include_groups=False
        )
        exp_std_shifted = result.groupby("machineID", group_keys=False).apply(
            lambda g: exp_std.loc[g.index].shift(1), include_groups=False
        )

        mean_col = f"{sensor}_exp_mean"
        std_col = f"{sensor}_exp_std"
        z_col = f"{sensor}_zscore"

        result[mean_col] = exp_mean_shifted.fillna(result[sensor])
        result[std_col] = exp_std_shifted.fillna(0.0)

        diff = result[sensor] - result[mean_col]
        denom = result[std_col].replace(0, eps)
        result[z_col] = (diff / denom).fillna(0.0)

    return result


# ===========================================================================
# 4. Interaction Features
# ===========================================================================

def compute_interaction_features(
    df: pd.DataFrame,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute domain-inspired sensor interaction terms with safe division.

    Interactions:
        - pressure_x_vibration: Mechanical load & fatigue product
        - rotation_div_voltage: Operational efficiency / electrical strain ratio
        - pressure_div_rotation: Pressure per rotation unit
        - vibration_div_pressure: Structural instability per pressure unit
        - voltage_x_rotation: Total electrical-mechanical power proxy
    """
    logger.info("Computing domain interaction features ...")
    result = df.copy()

    result["pressure_x_vibration"] = result["pressure"] * result["vibration"]
    result["voltage_x_rotation"] = result["volt"] * result["rotate"]

    result["rotation_div_voltage"] = result["rotate"] / (result["volt"].abs() + eps)
    result["pressure_div_rotation"] = result["pressure"] / (result["rotate"].abs() + eps)
    result["vibration_div_pressure"] = result["vibration"] / (result["pressure"].abs() + eps)

    return result


# ===========================================================================
# 5. Error Features
# ===========================================================================

def compute_error_features(
    df: pd.DataFrame,
    errors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract error history features using leakage-safe temporal matching.

    Features:
        - errors_last_24h: Count of any error in trailing 24h
        - errors_last_72h: Count of any error in trailing 72h
        - days_since_last_error: Days elapsed since the most recent error
        - rolling_error_rate_24h: Errors per hour in last 24h
        - error1_count_24h .. error5_count_24h: Counts by error type
    """
    logger.info("Computing error history features ...")
    result = df.copy()

    # One-hot encode error types
    err_encoded = pd.get_dummies(errors_df, columns=["errorID"], prefix="err")
    error_cols = [c for c in err_encoded.columns if c.startswith("err_")]

    # Merge asof backward
    errors_sorted = errors_df.sort_values("datetime")
    result_sorted = result.sort_values("datetime")

    # Time since last error
    errors_right = errors_sorted[["machineID", "datetime"]].assign(error_ts=errors_sorted["datetime"])
    merged_last = pd.merge_asof(
        result_sorted[["machineID", "datetime"]],
        errors_right,
        on="datetime",
        by="machineID",
        direction="backward",
    )
    # Ensure index alignment with result
    merged_last = merged_last.sort_index()
    time_diff_days = (merged_last["datetime"] - merged_last["error_ts"]).dt.total_seconds() / 86400.0
    result["days_since_last_error"] = time_diff_days.values
    result["days_since_last_error"] = result["days_since_last_error"].fillna(365.0)

    # Compute trailing counts for errors_24h and errors_72h
    # Build hour-level counts grid
    err_counts = (
        errors_df.assign(error_count=1)
        .groupby(["machineID", pd.Grouper(key="datetime", freq="h")])["error_count"]
        .sum()
        .unstack(level=0, fill_value=0)
    )

    # Reindex to full telemetry grid
    tel_grid = result.set_index(["datetime", "machineID"])
    
    # Calculate per-machine 24h & 72h error counts
    err_24h_series = []
    err_72h_series = []

    # Map error count events onto telemetry rows
    err_merged = pd.merge_asof(
        result_sorted[["machineID", "datetime"]].reset_index(),
        errors_df.assign(err_flag=1).sort_values("datetime"),
        on="datetime",
        by="machineID",
        direction="backward"
    )

    # Count errors in window via list comprehension / rolling per machine
    err_counts_24h = np.zeros(len(result), dtype=np.float32)
    err_counts_72h = np.zeros(len(result), dtype=np.float32)

    # Fast numpy searchsorted per machine for error windows
    for mid in result["machineID"].unique():
        m_idx = result[result["machineID"] == mid].index
        m_tel_ts = result.loc[m_idx, "datetime"].values.astype("datetime64[ns]").astype(np.int64)
        m_err = errors_df[errors_df["machineID"] == mid].sort_values("datetime")

        if m_err.empty:
            continue

        m_err_ts = m_err["datetime"].values.astype("datetime64[ns]").astype(np.int64)
        h24_ns = 24 * 3600 * 1_000_000_000
        h72_ns = 72 * 3600 * 1_000_000_000

        lo_24 = np.searchsorted(m_err_ts, m_tel_ts - h24_ns, side="left")
        hi_24 = np.searchsorted(m_err_ts, m_tel_ts, side="right")
        err_counts_24h[m_idx] = hi_24 - lo_24

        lo_72 = np.searchsorted(m_err_ts, m_tel_ts - h72_ns, side="left")
        hi_72 = np.searchsorted(m_err_ts, m_tel_ts, side="right")
        err_counts_72h[m_idx] = hi_72 - lo_72

        # Error type counts in trailing 24h
        for err_type in ["error1", "error2", "error3", "error4", "error5"]:
            col_name = f"{err_type}_count_24h"
            if col_name not in result.columns:
                result[col_name] = 0
            
            sub_err = m_err[m_err["errorID"] == err_type]
            if sub_err.empty:
                continue
            sub_ts = sub_err["datetime"].values.astype("datetime64[ns]").astype(np.int64)
            sub_lo = np.searchsorted(sub_ts, m_tel_ts - h24_ns, side="left")
            sub_hi = np.searchsorted(sub_ts, m_tel_ts, side="right")
            result.loc[m_idx, col_name] = sub_hi - sub_lo

    result["errors_last_24h"] = err_counts_24h
    result["errors_last_72h"] = err_counts_72h
    result["rolling_error_rate_24h"] = result["errors_last_24h"] / 24.0

    return result


# ===========================================================================
# 6. Maintenance Features
# ===========================================================================

def compute_maintenance_features(
    df: pd.DataFrame,
    maint_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract maintenance history features using temporal joins (pd.merge_asof).

    Features:
        - days_since_last_maintenance: Days elapsed since most recent maintenance
        - maintenance_count_last_30_days: Total replacements in trailing 30d
        - maintenance_count_last_90_days: Total replacements in trailing 90d
        - maintenance_count_last_1_year: Total replacements in trailing 365d
        - comp1_maint_count_90d .. comp4_maint_count_90d: Per-component counts
    """
    logger.info("Computing maintenance history features ...")
    result = df.copy()

    # Time since last maintenance
    maint_sorted = maint_df.sort_values("datetime")
    result_sorted = result.sort_values("datetime")

    maint_right = maint_sorted[["machineID", "datetime"]].assign(maint_ts=maint_sorted["datetime"])
    merged_last = pd.merge_asof(
        result_sorted[["machineID", "datetime"]],
        maint_right,
        on="datetime",
        by="machineID",
        direction="backward",
    )
    merged_last = merged_last.sort_index()
    time_diff_days = (merged_last["datetime"] - merged_last["maint_ts"]).dt.total_seconds() / 86400.0
    result["days_since_last_maintenance"] = time_diff_days.values
    result["days_since_last_maintenance"] = result["days_since_last_maintenance"].fillna(365.0)

    # Trailing maintenance counts using searchsorted
    cnt_30d = np.zeros(len(result), dtype=np.float32)
    cnt_90d = np.zeros(len(result), dtype=np.float32)
    cnt_1y = np.zeros(len(result), dtype=np.float32)

    for mid in result["machineID"].unique():
        m_idx = result[result["machineID"] == mid].index
        m_tel_ts = result.loc[m_idx, "datetime"].values.astype("datetime64[ns]").astype(np.int64)
        m_maint = maint_df[maint_df["machineID"] == mid].sort_values("datetime")

        if m_maint.empty:
            continue

        m_maint_ts = m_maint["datetime"].values.astype("datetime64[ns]").astype(np.int64)
        ns_30d = 30 * 86400 * 1_000_000_000
        ns_90d = 90 * 86400 * 1_000_000_000
        ns_1y = 365 * 86400 * 1_000_000_000

        lo_30 = np.searchsorted(m_maint_ts, m_tel_ts - ns_30d, side="left")
        hi_30 = np.searchsorted(m_maint_ts, m_tel_ts, side="right")
        cnt_30d[m_idx] = hi_30 - lo_30

        lo_90 = np.searchsorted(m_maint_ts, m_tel_ts - ns_90d, side="left")
        hi_90 = np.searchsorted(m_maint_ts, m_tel_ts, side="right")
        cnt_90d[m_idx] = hi_90 - lo_90

        lo_1y = np.searchsorted(m_maint_ts, m_tel_ts - ns_1y, side="left")
        hi_1y = np.searchsorted(m_maint_ts, m_tel_ts, side="right")
        cnt_1y[m_idx] = hi_1y - lo_1y

        # Per-component maintenance counts (90d)
        for comp in ["comp1", "comp2", "comp3", "comp4"]:
            col_name = f"{comp}_maint_count_90d"
            if col_name not in result.columns:
                result[col_name] = 0
            sub_maint = m_maint[m_maint["comp"] == comp]
            if sub_maint.empty:
                continue
            sub_ts = sub_maint["datetime"].values.astype("datetime64[ns]").astype(np.int64)
            sub_lo = np.searchsorted(sub_ts, m_tel_ts - ns_90d, side="left")
            sub_hi = np.searchsorted(sub_ts, m_tel_ts, side="right")
            result.loc[m_idx, col_name] = sub_hi - sub_lo

    result["maintenance_count_last_30_days"] = cnt_30d
    result["maintenance_count_last_90_days"] = cnt_90d
    result["maintenance_count_last_year"] = cnt_1y

    return result


# ===========================================================================
# 7. Static Machine Features
# ===========================================================================

def compute_static_machine_features(
    df: pd.DataFrame,
    machines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge machine age and model (one-hot encoded) onto telemetry.

    Why static features do not get rolling statistics:
    ---------------------------------------------------
    Machine age and model are constant for a given machine across the dataset span.
    Rolling statistics on constant values yield 0 std and identical min/max/mean,
    creating redundant constant features.
    """
    logger.info("Merging static machine metadata (age, model) ...")
    result = df.copy()

    # One-hot encode model
    model_dummies = pd.get_dummies(machines_df["model"], prefix="model", dtype=int)
    mac_encoded = pd.concat([machines_df[["machineID", "age"]], model_dummies], axis=1)

    result = result.merge(mac_encoded, on="machineID", how="left")
    return result


# ===========================================================================
# 8. Feature Validation & Diagnostics
# ===========================================================================

def validate_engineered_features(
    df: pd.DataFrame,
    target_col: str = "y_failure",
) -> Dict[str, Any]:
    """
    Exhaustive feature validation check.

    Checks:
        1. Missing values per feature
        2. Infinite / NaN values
        3. Constant columns (zero variance)
        4. Highly correlated duplicate features (|corr| > 0.995)
    """
    logger.info("Validating engineered features ...")
    
    # Feature columns (exclude datetime, machineID, targets)
    non_feature_cols = {"datetime", "machineID", "y_failure", "failure_component", "time_to_failure_hours"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # Missing counts
    missing = df[feature_cols].isna().sum()
    missing_cols = missing[missing > 0].to_dict()

    # Infinite counts
    infs = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum()
    inf_cols = infs[infs > 0].to_dict()

    # Constant columns
    stds = df[feature_cols].select_dtypes(include=[np.number]).std()
    constant_cols = stds[stds == 0].index.tolist()

    # Highly correlated pairs
    numeric_df = df[feature_cols].select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [
        (col1, col2, round(upper.loc[col1, col2], 4))
        for col1 in upper.columns
        for col2 in upper.index
        if upper.loc[col1, col2] > 0.995
    ]

    report = {
        "total_features": len(feature_cols),
        "missing_features": missing_cols,
        "infinite_features": inf_cols,
        "constant_features": constant_cols,
        "high_correlation_pairs": high_corr_pairs[:10],  # top 10
        "validation_passed": (len(missing_cols) == 0 and len(inf_cols) == 0 and len(constant_cols) == 0),
    }

    if report["validation_passed"]:
        logger.info("Feature validation PASSED — all %d features clean.", len(feature_cols))
    else:
        logger.warning(
            "Feature validation found issues: missing=%d, infs=%d, constants=%d",
            len(missing_cols), len(inf_cols), len(constant_cols)
        )

    return report


# ===========================================================================
# 9. Feature Importance Preview
# ===========================================================================

def compute_feature_importance_preview(
    df: pd.DataFrame,
    target_col: str = "y_failure",
    n_sample: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute correlation with target, mutual information score, variance, and missing % for ranking features.
    """
    logger.info("Computing feature importance preview (correlation & mutual information) ...")
    
    non_feature_cols = {"datetime", "machineID", "y_failure", "failure_component", "time_to_failure_hours"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in non_feature_cols]

    # Sample for fast mutual information computation
    sample_df = df.sample(min(n_sample, len(df)), random_state=seed) if len(df) > n_sample else df

    X_sample = sample_df[feature_cols].fillna(0.0)
    y_sample = sample_df[target_col]

    # Calculate Pearson correlation with target
    correlations = df[feature_cols].apply(lambda c: c.corr(df[target_col])).abs()

    # Calculate Mutual Information
    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=seed)

    # Variance and missing %
    variances = df[feature_cols].var()
    missing_pcts = (df[feature_cols].isna().sum() / len(df)) * 100.0

    preview_df = pd.DataFrame({
        "feature_name": feature_cols,
        "abs_target_corr": correlations.values.round(4),
        "mutual_info_score": mi_scores.round(4),
        "variance": variances.values.round(4),
        "missing_pct": missing_pcts.values.round(2),
    }).sort_values("mutual_info_score", ascending=False).reset_index(drop=True)

    logger.info("Top 5 features by Mutual Information score:")
    for _, row in preview_df.head(5).iterrows():
        logger.info("  - %-30s | MI=%.4f | corr=%.4f", row["feature_name"], row["mutual_info_score"], row["abs_target_corr"])

    return preview_df


# ===========================================================================
# 10. Feature Dictionary Builder
# ===========================================================================

def generate_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate metadata DataFrame for all engineered features.
    """
    non_feature_cols = {"datetime", "machineID", "y_failure", "failure_component", "time_to_failure_hours"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    dict_records = []
    for col in feature_cols:
        if "rolling_mean" in col:
            desc = "Trailing rolling mean over window"
            source = "PdM_telemetry.csv"
            window = col.split("_")[-1]
            formula = f"mean({col.split('_')[0]}) over {window}"
        elif "rolling_std" in col:
            desc = "Trailing rolling std dev over window"
            source = "PdM_telemetry.csv"
            window = col.split("_")[-1]
            formula = f"std({col.split('_')[0]}) over {window}"
        elif "delta" in col:
            desc = "Rate of change over lag period"
            source = "PdM_telemetry.csv"
            window = col.split("_")[-1]
            formula = f"value(t) - value(t-{window})"
        elif "zscore" in col:
            desc = "Machine expanding z-score (shifted by 1)"
            source = "PdM_telemetry.csv"
            window = "Expanding"
            formula = "(val - exp_mean) / exp_std"
        elif "errors_last" in col or "err_" in col or "error" in col:
            desc = "Error history metric in trailing window"
            source = "PdM_errors.csv"
            window = "24h / 72h"
            formula = "Count / rate of errors in trailing window"
        elif "maintenance" in col or "maint" in col:
            desc = "Maintenance history metric"
            source = "PdM_maint.csv"
            window = "30d / 90d / 1y"
            formula = "Time since / count of maintenance events"
        elif "age" in col or "model" in col:
            desc = "Static machine metadata"
            source = "PdM_machines.csv"
            window = "Static"
            formula = "Raw attribute / One-hot"
        else:
            desc = "Interaction / Derived feature"
            source = "PdM_telemetry.csv"
            window = "1h"
            formula = "Mathematical combination"

        dict_records.append({
            "feature_name": col,
            "description": desc,
            "formula": formula,
            "source_table": source,
            "window_size": window,
            "leakage_safe": "Yes",
        })

    return pd.DataFrame(dict_records)


# ===========================================================================
# 11. Visualisation Functions
# ===========================================================================

def _save_fig(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_rolling_stats_example(df: pd.DataFrame, figures_dir: Path, machine_id: int = 1) -> None:
    """Plot trailing rolling mean & std for sensor voltage on Machine 1."""
    sub = df[df["machineID"] == machine_id].head(168).copy()  # First 1 week
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub["datetime"], sub["volt"], label="Raw Volt", alpha=0.5, color="gray", linestyle=":")
    ax.plot(sub["datetime"], sub["volt_rolling_mean_3h"], label="3h Rolling Mean", color="#1f77b4", linewidth=1.5)
    ax.plot(sub["datetime"], sub["volt_rolling_mean_24h"], label="24h Rolling Mean", color="#ff7f0e", linewidth=2)
    
    ax.set_title(f"Trailing Rolling Statistics — Machine {machine_id} (Volt)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Voltage")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "feature_rolling_stats_example.png")


def plot_expanding_zscore_example(df: pd.DataFrame, figures_dir: Path, machine_id: int = 1) -> None:
    """Plot expanding mean & z-score on Machine 1."""
    sub = df[df["machineID"] == machine_id].head(336).copy()  # 2 weeks

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    axes[0].plot(sub["datetime"], sub["rotate"], label="Raw Rotate", alpha=0.6, color="teal")
    axes[0].plot(sub["datetime"], sub["rotate_exp_mean"], label="Shifted Expanding Mean", color="crimson", linewidth=2)
    axes[0].set_ylabel("Rotation")
    axes[0].set_title(f"Shifted Expanding Mean & Z-Score — Machine {machine_id} (Rotate)", fontsize=14, fontweight="bold")
    axes[0].legend()

    axes[1].plot(sub["datetime"], sub["rotate_zscore"], label="Rotate Z-Score", color="purple", linewidth=1.5)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(3, color="red", linestyle=":", label="+3 Std Dev")
    axes[1].axhline(-3, color="red", linestyle=":", label="-3 Std Dev")
    axes[1].set_ylabel("Z-Score")
    axes[1].set_xlabel("Date")
    axes[1].legend()

    fig.tight_layout()
    _save_fig(fig, figures_dir / "feature_expanding_zscore_example.png")


def plot_rate_of_change(df: pd.DataFrame, figures_dir: Path, machine_id: int = 1) -> None:
    """Plot delta rate of change features."""
    sub = df[df["machineID"] == machine_id].head(168).copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub["datetime"], sub["pressure_delta_1h"], label="Delta 1h", alpha=0.7)
    ax.plot(sub["datetime"], sub["pressure_delta_6h"], label="Delta 6h", alpha=0.7)
    ax.plot(sub["datetime"], sub["pressure_delta_12h"], label="Delta 12h", linewidth=2)

    ax.set_title(f"Rate of Change Deltas — Machine {machine_id} (Pressure)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Delta Pressure")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, figures_dir / "feature_rate_of_change.png")


def plot_maintenance_timeline(maint_df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot maintenance timeline per component."""
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.countplot(data=maint_df, x="comp", hue="comp", palette="Set2", legend=False, ax=ax)
    ax.set_title("Maintenance Events Frequency per Component", fontsize=14, fontweight="bold")
    ax.set_xlabel("Component Replaced")
    ax.set_ylabel("Total Maintenance Events")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "feature_maintenance_timeline.png")


def plot_error_timeline(errors_df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot error timeline distribution."""
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.countplot(data=errors_df, x="errorID", hue="errorID", palette="Set1", legend=False, ax=ax)
    ax.set_title("Error Log Occurrences per Error Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Total Error Occurrences")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "feature_error_timeline.png")


def plot_feature_correlation_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot Pearson correlation heatmap of key engineered features."""
    sample_cols = [
        "volt_rolling_mean_24h", "rotate_rolling_mean_24h", "pressure_rolling_mean_24h", "vibration_rolling_mean_24h",
        "volt_zscore", "rotate_zscore", "pressure_zscore", "vibration_zscore",
        "pressure_x_vibration", "days_since_last_error", "days_since_last_maintenance", "y_failure"
    ]
    corr = df[sample_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Engineered Features Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "feature_correlation_heatmap.png")


def plot_top_feature_importance(preview_df: pd.DataFrame, figures_dir: Path) -> None:
    """Bar chart ranking top 15 features by Mutual Information score."""
    top15 = preview_df.head(15).sort_values("mutual_info_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top15["feature_name"], top15["mutual_info_score"], color="#2ca02c", alpha=0.85)
    ax.set_xlabel("Mutual Information Score with y_failure")
    ax.set_title("Top 15 Engineered Features (Mutual Information Ranking)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "feature_importance_top15.png")
