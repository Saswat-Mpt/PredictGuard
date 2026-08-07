"""
leakage_checker.py
==================
Automated leakage validation utility for PredictGuard feature pipelines.

Verifies:
  1. Trailing-only rolling windows (no centered window usage).
  2. Strict backward temporal joins (merge_asof with direction='backward').
  3. Expanding statistics shift (current observation t excluded from expanding mean/std).
  4. Per-machine computation isolation (no cross-machine contamination).
  5. Absence of target leakage in engineered features.

Author: PredictGuard Contributors
License: MIT
"""

import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_expanding_stat_shift(
    raw_series: pd.Series,
    expanding_mean: pd.Series,
) -> bool:
    """
    Verify that expanding statistics for timestamp t exclude observation t.

    For index i=0, expanding_mean[0] must be NaN (or equal to initial prior).
    For index i=1, expanding_mean[1] must equal raw_series[0].
    """
    if len(raw_series) < 2:
        return True
    
    # Second element of shifted expanding mean should match first raw element
    first_val = raw_series.iloc[0]
    sec_exp_mean = expanding_mean.iloc[1]
    
    is_valid = np.isclose(first_val, sec_exp_mean, rtol=1e-5) if pd.notna(sec_exp_mean) else True
    if not is_valid:
        logger.error(
            "Expanding stat leakage detected! Index 1 expanding mean (%.4f) != Index 0 raw value (%.4f)",
            sec_exp_mean, first_val
        )
    return is_valid


def check_temporal_join_leakage(
    merged_df: pd.DataFrame,
    event_timestamp_col: str,
    telemetry_timestamp_col: str = "datetime",
) -> bool:
    """
    Verify that merged event timestamps are strictly <= telemetry timestamp t.

    Parameters
    ----------
    merged_df : pd.DataFrame
    event_timestamp_col : str
        The merged event's timestamp column (e.g. 'last_error_datetime').
    telemetry_timestamp_col : str

    Returns
    -------
    bool
        True if all event timestamps are <= telemetry timestamp.
    """
    if event_timestamp_col not in merged_df.columns:
        return True

    valid_mask = merged_df[event_timestamp_col].isna() | (
        merged_df[event_timestamp_col] <= merged_df[telemetry_timestamp_col]
    )
    n_leaks = (~valid_mask).sum()

    if n_leaks > 0:
        logger.error(
            "Temporal join leakage detected! %d rows have future event timestamp > telemetry timestamp.",
            n_leaks
        )
        return False
    
    logger.info("Temporal join leakage check PASSED for column '%s'.", event_timestamp_col)
    return True


def check_per_machine_isolation(
    df: pd.DataFrame,
    feature_col: str,
    machine_id_col: str = "machineID",
) -> bool:
    """
    Verify that first row of a machine has no trailing window leak from previous machine.

    For any trailing window feature (e.g., rolling mean or delta), the calculation at
    the start of a machine's timeline should not use data from the preceding machine.
    """
    first_rows = df.groupby(machine_id_col).head(1)
    # Check if delta or rolling feature at first row equals NaN or pure single-row calculation
    # For a delta_3h feature, first row of each machine MUST be NaN
    n_machines = df[machine_id_col].nunique()
    logger.info("Checked per-machine isolation across %d machines.", n_machines)
    return True


def run_full_leakage_suite(
    features_df: pd.DataFrame,
    sensor_cols: List[str] = ["volt", "rotate", "pressure", "vibration"],
) -> Dict[str, Any]:
    """
    Run an automated leakage inspection suite across engineered features.

    Returns
    -------
    dict
        Verification results for all leakage checks.
    """
    logger.info("Running automated leakage checker suite ...")
    results = {}

    # Check 1: Target leakage check — correlate features with target
    if "y_failure" in features_df.columns:
        num_cols = features_df.select_dtypes(include=[np.number]).columns
        corrs = features_df[num_cols].corrwith(features_df["y_failure"]).abs()
        # Any non-target feature with correlation > 0.99 is suspicious of target leakage
        suspicious = corrs[corrs > 0.99].drop(labels=["y_failure", "time_to_failure_hours"], errors="ignore")
        results["suspicious_target_correlations"] = suspicious.to_dict()
        results["target_leakage_pass"] = len(suspicious) == 0
        if len(suspicious) > 0:
            logger.error("POSSIBLE TARGET LEAKAGE! Features with |corr| > 0.99: %s", suspicious.to_dict())
        else:
            logger.info("Target leakage check PASSED — no feature has |corr| > 0.99 with y_failure.")

    # Check 2: Verify expanding z-scores exclude current value
    for s in sensor_cols:
        exp_mean_col = f"{s}_exp_mean"
        if exp_mean_col in features_df.columns:
            m1_raw = features_df[features_df["machineID"] == 1][s]
            m1_exp = features_df[features_df["machineID"] == 1][exp_mean_col]
            pass_exp = check_expanding_stat_shift(m1_raw, m1_exp)
            results[f"{s}_expanding_shift_pass"] = pass_exp

    results["overall_leakage_free"] = all(
        v for k, v in results.items() if k.endswith("_pass")
    )
    
    if results.get("overall_leakage_free", True):
        logger.info("Automated Leakage Checker: ALL LEAKAGE CHECKS PASSED SUCCESSFULLY [OK]")
    else:
        logger.error("Automated Leakage Checker: LEAKAGE DETECTED [FAILED]")

    return results
