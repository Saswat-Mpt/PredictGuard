"""
data_validation.py
==================
Production-quality data validation utilities for PredictGuard — Phase 1.

All public functions are:
  - Fully typed (PEP 484 type hints)
  - Documented with NumPy-style docstrings
  - Logging-based (no bare print statements)
  - Reusable across notebooks and scripts

Author: PredictGuard Contributors
License: MIT
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend — safe for CI/headless runs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

#: Expected column → dtype mapping for every table.
EXPECTED_SCHEMAS: Dict[str, Dict[str, str]] = {
    "telemetry": {
        "datetime": "datetime64[ns]",
        "machineID": "int64",
        "volt": "float64",
        "rotate": "float64",
        "pressure": "float64",
        "vibration": "float64",
    },
    "errors": {
        "datetime": "datetime64[ns]",
        "machineID": "int64",
        "errorID": "object",
    },
    "maint": {
        "datetime": "datetime64[ns]",
        "machineID": "int64",
        "comp": "object",
    },
    "failures": {
        "datetime": "datetime64[ns]",
        "machineID": "int64",
        "failure": "object",
    },
    "machines": {
        "machineID": "int64",
        "model": "object",
        "age": "int64",
    },
}

#: Names of raw sensor columns in the telemetry table.
SENSOR_COLUMNS: List[str] = ["volt", "rotate", "pressure", "vibration"]

#: Inclusive sensor value bounds ``(min, max)`` used for range validation.
#: These are empirically derived from domain knowledge and dataset inspection.
#: Adjust here without touching any downstream code.
SENSOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "volt": (0.0, 300.0),
    "rotate": (0.0, 600.0),
    "pressure": (0.0, 200.0),
    "vibration": (0.0, 100.0),
}

#: Default rolling window size (number of rows) for frozen-sensor detection.
DEFAULT_FROZEN_WINDOW: int = 10

#: Default gap threshold in hours for telemetry gap detection.
DEFAULT_GAP_HOURS: float = 3.0

# ---------------------------------------------------------------------------
# Matplotlib / Seaborn global style
# ---------------------------------------------------------------------------

sns.set_theme(
    style="darkgrid",
    palette="muted",
    font="DejaVu Sans",
    rc={
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    },
)


# ===========================================================================
# Section 1 — Data Loading
# ===========================================================================


def load_all_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all five PdM CSV files and return them as a named dictionary.

    Timestamp columns are parsed automatically. The function logs
    row/column counts and raises ``FileNotFoundError`` with a descriptive
    message if any file is missing.

    Parameters
    ----------
    data_dir : Path
        Directory containing the five CSV files.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Keys: ``'telemetry'``, ``'errors'``, ``'maint'``,
        ``'failures'``, ``'machines'``.

    Raises
    ------
    FileNotFoundError
        If any expected CSV file is not found in *data_dir*.
    """
    file_map: Dict[str, str] = {
        "telemetry": "PdM_telemetry.csv",
        "errors": "PdM_errors.csv",
        "maint": "PdM_maint.csv",
        "failures": "PdM_failures.csv",
        "machines": "PdM_machines.csv",
    }

    # Tables that contain a timestamp column
    timestamp_tables: set = {"telemetry", "errors", "maint", "failures"}

    datasets: Dict[str, pd.DataFrame] = {}
    for name, filename in file_map.items():
        fpath = Path(data_dir) / filename
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected dataset file not found: {fpath}\n"
                f"Place all five PdM CSV files in '{data_dir}'."
            )
        logger.info("Loading %s from %s …", name, fpath)
        df = pd.read_csv(fpath, low_memory=False)

        # Parse datetime column where applicable
        if name in timestamp_tables and "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=False)

        datasets[name] = df
        logger.info(
            "  Loaded %-10s | rows=%d  cols=%d",
            name,
            len(df),
            df.shape[1],
        )

    return datasets


# ===========================================================================
# Section 2 — Schema Validation
# ===========================================================================


def validate_schemas(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, List[str]]:
    """
    Validate that every table contains the expected columns.

    This function checks *column presence* only — it does not coerce dtypes.
    Dtype mismatches are surfaced as warnings so that downstream code can
    decide whether to cast or raise.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dataset dictionary returned by :func:`load_all_datasets`.

    Returns
    -------
    Dict[str, List[str]]
        Mapping from table name to a list of issue strings.
        An empty list means the table passed validation.
    """
    issues: Dict[str, List[str]] = {}

    for table_name, expected in EXPECTED_SCHEMAS.items():
        df = datasets.get(table_name)
        if df is None:
            issues[table_name] = [f"Table '{table_name}' is missing entirely."]
            continue

        table_issues: List[str] = []

        # Column presence check
        missing_cols = set(expected.keys()) - set(df.columns)
        for col in sorted(missing_cols):
            table_issues.append(f"Missing column: '{col}'")

        # Dtype mismatch check (for columns that are present)
        for col, expected_dtype in expected.items():
            if col not in df.columns:
                continue  # already flagged above
            actual_dtype = str(df[col].dtype)
            # Normalise datetime check
            if "datetime" in expected_dtype and "datetime" not in actual_dtype:
                table_issues.append(
                    f"Column '{col}': expected datetime, got '{actual_dtype}'"
                )
            elif "datetime" not in expected_dtype:
                # pandas 2.x may report 'str' or 'string' for object columns
                expected_norm = expected_dtype.lower()
                actual_norm = actual_dtype.lower()
                # Treat 'str', 'string', 'object' as equivalent
                object_aliases = {"object", "str", "string"}
                if expected_norm in object_aliases and actual_norm in object_aliases:
                    pass  # compatible
                elif actual_norm != expected_norm:
                    table_issues.append(
                        f"Column '{col}': expected '{expected_dtype}', "
                        f"got '{actual_dtype}'"
                    )

        issues[table_name] = table_issues
        if table_issues:
            for issue in table_issues:
                logger.warning("[%s] Schema issue — %s", table_name, issue)
        else:
            logger.info("[%s] Schema OK.", table_name)

    return issues


# ===========================================================================
# Section 3 — Missing Values & Duplicates
# ===========================================================================


def check_missing_values(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Count and compute the percentage of missing values per column per table.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
        Dataset dictionary.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        ``['table', 'column', 'missing_count', 'missing_pct']``.
    """
    records: List[Dict[str, Any]] = []
    for table_name, df in datasets.items():
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            pct = round(100.0 * n_missing / max(len(df), 1), 4)
            records.append(
                {
                    "table": table_name,
                    "column": col,
                    "missing_count": n_missing,
                    "missing_pct": pct,
                }
            )
    result = pd.DataFrame(records)
    total_missing = result["missing_count"].sum()
    logger.info(
        "Missing values — total cells with NaN across all tables: %d",
        total_missing,
    )
    return result


def check_duplicate_rows(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Count fully duplicated rows in every table.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]

    Returns
    -------
    pd.DataFrame
        Columns: ``['table', 'total_rows', 'duplicate_rows', 'duplicate_pct']``.
    """
    records: List[Dict[str, Any]] = []
    for table_name, df in datasets.items():
        n_dup = int(df.duplicated().sum())
        pct = round(100.0 * n_dup / max(len(df), 1), 4)
        records.append(
            {
                "table": table_name,
                "total_rows": len(df),
                "duplicate_rows": n_dup,
                "duplicate_pct": pct,
            }
        )
        if n_dup > 0:
            logger.warning(
                "[%s] Found %d duplicate rows (%.2f%%)", table_name, n_dup, pct
            )
        else:
            logger.info("[%s] No duplicate rows.", table_name)
    return pd.DataFrame(records)


# ===========================================================================
# Section 4 — Cross-Table Machine Consistency
# ===========================================================================


def verify_machine_consistency(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Verify that machine IDs are consistent across all five tables.

    Checks:
      - Every machine in the telemetry appears in the machines table.
      - Orphan records in errors / maint / failures that have no matching
        telemetry.
      - Machines with no events whatsoever (errors, maint, failures).

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]

    Returns
    -------
    dict
        Keys:
        ``'machine_ids_per_table'``, ``'orphan_errors'``,
        ``'orphan_maint'``, ``'orphan_failures'``,
        ``'machines_with_no_telemetry'``,
        ``'machines_with_no_events'``.
    """
    machines_set = set(datasets["machines"]["machineID"].unique())
    telemetry_set = set(datasets["telemetry"]["machineID"].unique())
    errors_set = set(datasets["errors"]["machineID"].unique())
    maint_set = set(datasets["maint"]["machineID"].unique())
    failures_set = set(datasets["failures"]["machineID"].unique())

    machine_ids_per_table: Dict[str, int] = {
        "machines": len(machines_set),
        "telemetry": len(telemetry_set),
        "errors": len(errors_set),
        "maint": len(maint_set),
        "failures": len(failures_set),
    }

    # Machines in events/telemetry but NOT in machines table
    orphan_errors = errors_set - machines_set
    orphan_maint = maint_set - machines_set
    orphan_failures = failures_set - machines_set
    machines_with_no_telemetry = machines_set - telemetry_set

    # Machines with no failures, no errors, and no maintenance
    event_machines = errors_set | maint_set | failures_set
    machines_with_no_events = machines_set - event_machines

    def _log_orphans(label: str, ids: set) -> None:
        if ids:
            logger.warning(
                "Orphan machineIDs in %s (not in machines table): %s",
                label,
                sorted(ids),
            )
        else:
            logger.info("%s — all machine IDs match.", label)

    _log_orphans("errors", orphan_errors)
    _log_orphans("maint", orphan_maint)
    _log_orphans("failures", orphan_failures)

    if machines_with_no_telemetry:
        logger.warning(
            "Machines with NO telemetry: %s", sorted(machines_with_no_telemetry)
        )
    if machines_with_no_events:
        logger.info(
            "Machines with no recorded events: %d machines",
            len(machines_with_no_events),
        )

    return {
        "machine_ids_per_table": machine_ids_per_table,
        "orphan_errors": sorted(orphan_errors),
        "orphan_maint": sorted(orphan_maint),
        "orphan_failures": sorted(orphan_failures),
        "machines_with_no_telemetry": sorted(machines_with_no_telemetry),
        "machines_with_no_events": sorted(machines_with_no_events),
    }


# ===========================================================================
# Section 5 — Telemetry Summary Statistics
# ===========================================================================


def compute_telemetry_summary(
    telemetry: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-machine summary statistics from telemetry data.

    Returns
    -------
    pd.DataFrame
        Index: machineID.
        Columns: ``['row_count', 'min_ts', 'max_ts', 'span_days',
        'model', 'age']``.
    """
    logger.info("Computing per-machine telemetry summary …")

    summary = (
        telemetry.groupby("machineID")["datetime"]
        .agg(
            row_count="count",
            min_ts="min",
            max_ts="max",
        )
        .copy()
    )
    summary["span_days"] = (summary["max_ts"] - summary["min_ts"]).dt.days

    # Join machine metadata
    meta = machines.set_index("machineID")[["model", "age"]]
    summary = summary.join(meta, how="left")

    logger.info(
        "Telemetry summary -- %d machines | total rows: %d | "
        "overall span: %s to %s",
        len(summary),
        summary["row_count"].sum(),
        summary["min_ts"].min(),
        summary["max_ts"].max(),
    )
    return summary


# ===========================================================================
# Section 6 — Telemetry Gap Detection
# ===========================================================================


def detect_telemetry_gaps(
    telemetry: pd.DataFrame,
    gap_threshold_hours: float = DEFAULT_GAP_HOURS,
) -> pd.DataFrame:
    """
    Detect time gaps larger than *gap_threshold_hours* in each machine's
    telemetry stream.

    Algorithm:
      1. Sort each machine's records by timestamp.
      2. Compute pairwise time differences.
      3. Flag differences exceeding the threshold.

    Parameters
    ----------
    telemetry : pd.DataFrame
        Must contain columns ``['machineID', 'datetime']``.
    gap_threshold_hours : float
        Minimum gap size (in hours) to report. Default is 3.0.

    Returns
    -------
    pd.DataFrame
        Tidy gap DataFrame with columns:
        ``['machineID', 'gap_start', 'gap_end', 'gap_hours']``,
        sorted by ``gap_hours`` descending.
    """
    logger.info(
        "Detecting telemetry gaps > %.1f hours per machine …",
        gap_threshold_hours,
    )
    threshold = pd.Timedelta(hours=gap_threshold_hours)
    gap_records: List[Dict[str, Any]] = []

    for machine_id, grp in telemetry.groupby("machineID"):
        ts = grp["datetime"].sort_values().reset_index(drop=True)
        diffs = ts.diff()
        gap_mask = diffs > threshold
        for idx in diffs[gap_mask].index:
            gap_records.append(
                {
                    "machineID": machine_id,
                    "gap_start": ts.iloc[idx - 1],
                    "gap_end": ts.iloc[idx],
                    "gap_hours": round(diffs.iloc[idx].total_seconds() / 3600, 2),
                }
            )

    gaps_df = pd.DataFrame(gap_records)
    if not gaps_df.empty:
        gaps_df = gaps_df.sort_values("gap_hours", ascending=False).reset_index(
            drop=True
        )
    logger.info(
        "Found %d gaps > %.1f hours across all machines.",
        len(gaps_df),
        gap_threshold_hours,
    )
    return gaps_df


# ===========================================================================
# Section 7 — Frozen Sensor Detection
# ===========================================================================


def detect_frozen_sensors(
    telemetry: pd.DataFrame,
    window: int = DEFAULT_FROZEN_WINDOW,
) -> pd.DataFrame:
    """
    Detect intervals where a sensor appears "stuck" (zero rolling std dev).

    For every ``(machineID, sensor)`` combination the function computes a
    rolling standard deviation over *window* consecutive readings.  Any
    window with std == 0 indicates the sensor reported an identical value
    for *window* consecutive hours — a strong signal of sensor failure.

    Parameters
    ----------
    telemetry : pd.DataFrame
        Must contain columns ``['machineID', 'datetime'] + SENSOR_COLUMNS``.
    window : int
        Rolling window size in number of rows (default: 10 readings = 10 h).

    Returns
    -------
    pd.DataFrame
        Columns: ``['machineID', 'sensor', 'window_start', 'window_end',
        'frozen_value']``, one row per detected frozen interval.
    """
    logger.info(
        "Detecting frozen sensors (rolling window=%d rows) …", window
    )
    frozen_records: List[Dict[str, Any]] = []

    for machine_id, grp in telemetry.groupby("machineID"):
        grp_sorted = grp.sort_values("datetime").reset_index(drop=True)

        for sensor in SENSOR_COLUMNS:
            rolling_std = (
                grp_sorted[sensor]
                .rolling(window=window, min_periods=window)
                .std()
            )
            frozen_mask = rolling_std == 0.0

            if not frozen_mask.any():
                continue

            # Collapse consecutive frozen rows into contiguous intervals
            in_frozen = False
            start_idx: Optional[int] = None
            for idx, is_frozen in enumerate(frozen_mask):
                if is_frozen and not in_frozen:
                    in_frozen = True
                    start_idx = idx - (window - 1)
                elif not is_frozen and in_frozen:
                    in_frozen = False
                    frozen_records.append(
                        {
                            "machineID": machine_id,
                            "sensor": sensor,
                            "window_start": grp_sorted["datetime"].iloc[
                                start_idx
                            ],
                            "window_end": grp_sorted["datetime"].iloc[
                                idx - 1
                            ],
                            "frozen_value": grp_sorted[sensor].iloc[start_idx],
                        }
                    )
            # Handle frozen interval that extends to the end of the series
            if in_frozen and start_idx is not None:
                frozen_records.append(
                    {
                        "machineID": machine_id,
                        "sensor": sensor,
                        "window_start": grp_sorted["datetime"].iloc[start_idx],
                        "window_end": grp_sorted["datetime"].iloc[-1],
                        "frozen_value": grp_sorted[sensor].iloc[start_idx],
                    }
                )

    frozen_df = pd.DataFrame(frozen_records)
    if not frozen_df.empty:
        frozen_df = frozen_df.sort_values(
            ["machineID", "sensor", "window_start"]
        ).reset_index(drop=True)

    logger.info("Found %d frozen sensor interval(s).", len(frozen_df))
    return frozen_df


# ===========================================================================
# Section 8 — Sensor Range Validation
# ===========================================================================


def validate_sensor_ranges(
    telemetry: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]] = SENSOR_BOUNDS,
) -> pd.DataFrame:
    """
    Flag sensor readings that fall outside expected bounds.

    **Does NOT remove or modify the original data** — only adds boolean
    flag columns of the form ``'<sensor>_out_of_range'``.

    Parameters
    ----------
    telemetry : pd.DataFrame
    bounds : dict
        Mapping ``{sensor_name: (min_val, max_val)}``.
        Defaults to :data:`SENSOR_BOUNDS`.

    Returns
    -------
    pd.DataFrame
        Original telemetry with extra boolean flag columns appended.
        A ``True`` value means the reading is *outside* the valid range.
    """
    result = telemetry.copy()
    for sensor, (lo, hi) in bounds.items():
        if sensor not in result.columns:
            logger.warning("Sensor '%s' not found in telemetry — skipping.", sensor)
            continue
        flag_col = f"{sensor}_out_of_range"
        result[flag_col] = (result[sensor] < lo) | (result[sensor] > hi)
        n_flagged = int(result[flag_col].sum())
        pct = 100.0 * n_flagged / max(len(result), 1)
        logger.info(
            "Sensor '%s' out-of-range: %d rows (%.3f%%)", sensor, n_flagged, pct
        )
    return result


# ===========================================================================
# Section 9 — Cross-Table Integrity Check
# ===========================================================================


def verify_cross_table_integrity(
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Verify that every failure, maintenance, and error record has a
    matching telemetry observation within the same hour.

    Strategy: for each event timestamp, check that the machine's telemetry
    contains at least one record in the same calendar hour.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]

    Returns
    -------
    dict
        Keys: ``'unmatched_failures'``, ``'unmatched_maint'``,
        ``'unmatched_errors'``.
        Values are DataFrames of unmatched records.
    """
    logger.info("Verifying cross-table record integrity …")
    telemetry = datasets["telemetry"].copy()
    telemetry["hour_key"] = (
        telemetry["machineID"].astype(str)
        + "_"
        + telemetry["datetime"].dt.floor("h").astype(str)
    )
    telemetry_keys = set(telemetry["hour_key"])

    results: Dict[str, pd.DataFrame] = {}
    for table_name in ("failures", "maint", "errors"):
        df = datasets[table_name].copy()
        df["hour_key"] = (
            df["machineID"].astype(str)
            + "_"
            + df["datetime"].dt.floor("h").astype(str)
        )
        unmatched = df[~df["hour_key"].isin(telemetry_keys)].drop(
            columns=["hour_key"]
        )
        results[f"unmatched_{table_name}"] = unmatched
        logger.info(
            "[%s] Unmatched records (no telemetry in same hour): %d",
            table_name,
            len(unmatched),
        )

    return results


# ===========================================================================
# Section 10 — Data Quality Report
# ===========================================================================


def build_quality_report(
    datasets: Dict[str, pd.DataFrame],
    missing_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    consistency: Dict[str, Any],
    gaps_df: pd.DataFrame,
    frozen_df: pd.DataFrame,
    telemetry_flagged: pd.DataFrame,
    integrity: Dict[str, pd.DataFrame],
    telemetry_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate all validation results into a single tidy quality report.

    Parameters
    ----------
    datasets : dict
        Raw loaded datasets.
    missing_df : pd.DataFrame
        Output of :func:`check_missing_values`.
    duplicates_df : pd.DataFrame
        Output of :func:`check_duplicate_rows`.
    consistency : dict
        Output of :func:`verify_machine_consistency`.
    gaps_df : pd.DataFrame
        Output of :func:`detect_telemetry_gaps`.
    frozen_df : pd.DataFrame
        Output of :func:`detect_frozen_sensors`.
    telemetry_flagged : pd.DataFrame
        Output of :func:`validate_sensor_ranges`.
    integrity : dict
        Output of :func:`verify_cross_table_integrity`.
    telemetry_summary : pd.DataFrame
        Output of :func:`compute_telemetry_summary`.

    Returns
    -------
    pd.DataFrame
        Columns: ``['check', 'metric', 'value']`` — suitable for CSV export.
    """
    logger.info("Building consolidated data quality report …")
    records: List[Dict[str, Any]] = []

    def _add(check: str, metric: str, value: Any) -> None:
        records.append({"check": check, "metric": metric, "value": str(value)})

    # --- Table sizes ---
    for name, df in datasets.items():
        _add("table_size", f"{name}_rows", len(df))
        _add("table_size", f"{name}_cols", df.shape[1])

    # --- Missing values ---
    total_missing = missing_df["missing_count"].sum()
    _add("missing_values", "total_missing_cells", total_missing)
    tables_with_missing = missing_df[missing_df["missing_count"] > 0]["table"].nunique()
    _add("missing_values", "tables_with_missing", tables_with_missing)

    # --- Duplicates ---
    total_dups = duplicates_df["duplicate_rows"].sum()
    _add("duplicates", "total_duplicate_rows", total_dups)

    # --- Machine consistency ---
    _add("consistency", "unique_machines", consistency["machine_ids_per_table"]["machines"])
    _add("consistency", "orphan_errors", len(consistency["orphan_errors"]))
    _add("consistency", "orphan_maint", len(consistency["orphan_maint"]))
    _add("consistency", "orphan_failures", len(consistency["orphan_failures"]))
    _add(
        "consistency",
        "machines_with_no_telemetry",
        len(consistency["machines_with_no_telemetry"]),
    )
    _add(
        "consistency",
        "machines_with_no_events",
        len(consistency["machines_with_no_events"]),
    )

    # --- Telemetry summary ---
    _add("telemetry_summary", "total_rows", int(telemetry_summary["row_count"].sum()))
    _add("telemetry_summary", "min_timestamp", str(telemetry_summary["min_ts"].min()))
    _add("telemetry_summary", "max_timestamp", str(telemetry_summary["max_ts"].max()))
    _add("telemetry_summary", "avg_rows_per_machine", round(telemetry_summary["row_count"].mean(), 1))
    short_machines = int((telemetry_summary["span_days"] < 30).sum())
    _add("telemetry_summary", "machines_with_short_history_lt30d", short_machines)

    # --- Gaps ---
    _add("telemetry_gaps", "total_gaps_detected", len(gaps_df))
    if not gaps_df.empty:
        _add("telemetry_gaps", "max_gap_hours", gaps_df["gap_hours"].max())
        _add("telemetry_gaps", "mean_gap_hours", round(gaps_df["gap_hours"].mean(), 2))
        _add("telemetry_gaps", "machines_with_gaps", gaps_df["machineID"].nunique())

    # --- Frozen sensors ---
    _add("frozen_sensors", "total_frozen_intervals", len(frozen_df))
    if not frozen_df.empty:
        _add("frozen_sensors", "sensors_affected", frozen_df["sensor"].nunique())
        _add(
            "frozen_sensors",
            "machines_affected",
            frozen_df["machineID"].nunique(),
        )

    # --- Out-of-range sensors ---
    flag_cols = [c for c in telemetry_flagged.columns if c.endswith("_out_of_range")]
    for col in flag_cols:
        sensor = col.replace("_out_of_range", "")
        _add("sensor_range", f"{sensor}_out_of_range_count", int(telemetry_flagged[col].sum()))

    # --- Cross-table integrity ---
    for key, df in integrity.items():
        _add("cross_table_integrity", f"{key}_count", len(df))

    report_df = pd.DataFrame(records)
    logger.info("Quality report built — %d entries.", len(report_df))
    return report_df


def save_report_csv(report_df: pd.DataFrame, output_path: Path) -> None:
    """
    Persist the quality report DataFrame to a CSV file.

    Parameters
    ----------
    report_df : pd.DataFrame
    output_path : Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    logger.info("Quality report (CSV) saved to %s", output_path)


def save_report_md(
    report_df: pd.DataFrame,
    datasets: Dict[str, pd.DataFrame],
    missing_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    frozen_df: pd.DataFrame,
    telemetry_flagged: pd.DataFrame,
    integrity: Dict[str, pd.DataFrame],
    telemetry_summary: pd.DataFrame,
    consistency: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Write a human-readable Markdown data quality report.

    Parameters
    ----------
    report_df : pd.DataFrame
        Output of :func:`build_quality_report`.
    output_path : Path
        Destination ``.md`` file path.
    (All other parameters are forwarded for inline table rendering.)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _section(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    lines = [
        "# PredictGuard — Data Quality Report",
        "",
        "> **Stage 1: Data Understanding & Validation**  ",
        f"> Generated automatically by `src/data_validation.py`.",
        "",
        "---",
        "",
        "## 1. Table Sizes",
        "",
    ]

    # Table sizes
    size_rows = [
        f"| {name} | {len(df):,} | {df.shape[1]} |"
        for name, df in datasets.items()
    ]
    lines += [
        "| Table | Rows | Columns |",
        "|---|---|---|",
    ] + size_rows + [""]

    # Missing values
    lines += [
        "---",
        "",
        "## 2. Missing Values",
        "",
        _section(missing_df[missing_df["missing_count"] > 0]) or "_No missing values detected._",
        "",
    ]

    # Duplicates
    lines += [
        "---",
        "",
        "## 3. Duplicate Rows",
        "",
        _section(duplicates_df),
        "",
    ]

    # Machine consistency
    lines += [
        "---",
        "",
        "## 4. Machine ID Consistency",
        "",
        f"| Table | Unique machineIDs |",
        f"|---|---|",
    ]
    for t, cnt in consistency["machine_ids_per_table"].items():
        lines.append(f"| {t} | {cnt} |")
    lines += [
        "",
        f"- **Orphan errors**: {consistency['orphan_errors']}",
        f"- **Orphan maint**: {consistency['orphan_maint']}",
        f"- **Orphan failures**: {consistency['orphan_failures']}",
        f"- **Machines with no telemetry**: {consistency['machines_with_no_telemetry']}",
        f"- **Machines with no events**: {len(consistency['machines_with_no_events'])} machines",
        "",
    ]

    # Telemetry summary
    lines += [
        "---",
        "",
        "## 5. Telemetry Summary",
        "",
        _section(telemetry_summary.reset_index()),
        "",
    ]

    # Gaps
    lines += [
        "---",
        "",
        "## 6. Telemetry Gaps (> 3 hours)",
        "",
        f"**Total gaps detected**: {len(gaps_df)}",
        "",
    ]
    if not gaps_df.empty:
        lines += [_section(gaps_df.head(20)), ""]

    # Frozen sensors
    lines += [
        "---",
        "",
        "## 7. Frozen Sensors",
        "",
        f"**Total frozen intervals**: {len(frozen_df)}",
        "",
    ]
    if not frozen_df.empty:
        lines += [_section(frozen_df.head(20)), ""]

    # Out-of-range sensors
    flag_cols = [c for c in telemetry_flagged.columns if c.endswith("_out_of_range")]
    lines += [
        "---",
        "",
        "## 8. Sensor Range Violations",
        "",
        "| Sensor | Out-of-range count | % of rows |",
        "|---|---|---|",
    ]
    for col in flag_cols:
        sensor = col.replace("_out_of_range", "")
        n = int(telemetry_flagged[col].sum())
        pct = round(100.0 * n / max(len(telemetry_flagged), 1), 4)
        lines.append(f"| {sensor} | {n:,} | {pct}% |")
    lines.append("")

    # Cross-table integrity
    lines += [
        "---",
        "",
        "## 9. Cross-Table Integrity",
        "",
        "| Event Table | Unmatched Records |",
        "|---|---|",
    ]
    for key, df in integrity.items():
        lines.append(f"| {key} | {len(df):,} |")
    lines += [""]

    # Full summary table
    lines += [
        "---",
        "",
        "## 10. Full Metrics Table",
        "",
        _section(report_df),
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Quality report (Markdown) saved to %s", output_path)


# ===========================================================================
# Section 11 — Visualisations
# ===========================================================================


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save a figure to *path* (creating parent dirs) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_missing_heatmap(
    datasets: Dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """
    Save a missing-value heatmap for each table using ``missingno``.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
    figures_dir : Path
        Directory where figures are saved.
    """
    figures_dir = Path(figures_dir)
    for name, df in datasets.items():
        if df.isna().sum().sum() == 0:
            logger.info("[%s] No missing values — skipping heatmap.", name)
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        msno.matrix(df, ax=ax, sparkline=False, color=(0.25, 0.45, 0.70))
        ax.set_title(f"Missing Value Matrix — {name}", fontsize=14, fontweight="bold")
        _save_fig(fig, figures_dir / f"missing_heatmap_{name}.png")

    # Also save a combined nullity matrix if any table has nulls
    tables_with_nulls = [
        df.add_prefix(f"{name}_")
        for name, df in datasets.items()
        if df.isna().sum().sum() > 0
    ]
    if tables_with_nulls:
        combined = pd.concat(tables_with_nulls, axis=1)
        fig, ax = plt.subplots(figsize=(14, 6))
        msno.matrix(combined, ax=ax, sparkline=False)
        ax.set_title("Combined Missing Value Matrix", fontsize=14, fontweight="bold")
        _save_fig(fig, figures_dir / "missing_heatmap_combined.png")
    else:
        logger.info("No missing values in any table -- skipping combined heatmap.")
        # Save a placeholder "all clean" figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5,
            "No missing values detected in any table.",
            ha="center", va="center", fontsize=14, color="#2e7d32",
            transform=ax.transAxes,
        )
        ax.axis("off")
        ax.set_title("Missing Value Check -- All Tables Clean", fontsize=14, fontweight="bold")
        _save_fig(fig, figures_dir / "missing_heatmap_combined.png")


def plot_sensor_histograms(
    telemetry: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a 2×2 grid of sensor value histograms.

    Parameters
    ----------
    telemetry : pd.DataFrame
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sensor Value Distributions", fontsize=16, fontweight="bold", y=1.01)

    colors = sns.color_palette("muted", 4)
    for ax, sensor, color in zip(axes.flatten(), SENSOR_COLUMNS, colors):
        ax.hist(
            telemetry[sensor].dropna(),
            bins=80,
            color=color,
            edgecolor="none",
            alpha=0.85,
        )
        ax.axvline(
            telemetry[sensor].mean(),
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {telemetry[sensor].mean():.2f}",
        )
        ax.set_title(sensor.capitalize(), fontsize=13)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save_fig(fig, figures_dir / "sensor_histograms.png")


def plot_boxplots(
    telemetry: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a 2×2 grid of sensor boxplots per machine model.

    Parameters
    ----------
    telemetry : pd.DataFrame
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Sensor Boxplots by Machine", fontsize=16, fontweight="bold", y=1.01)

    palette = sns.color_palette("Set2", 4)
    for ax, sensor, color in zip(axes.flatten(), SENSOR_COLUMNS, palette):
        # Sample to keep the plot tractable
        sample = telemetry.sample(min(20_000, len(telemetry)), random_state=42)
        sns.boxplot(
            data=sample,
            x="machineID",
            y=sensor,
            ax=ax,
            color=color,
            flierprops={"marker": ".", "alpha": 0.3, "markersize": 2},
            linewidth=0.7,
        )
        ax.set_title(sensor.capitalize(), fontsize=13)
        ax.set_xlabel("Machine ID")
        ax.set_ylabel("Value")
        # Thin out x-axis labels if too many machines
        n_machines = telemetry["machineID"].nunique()
        if n_machines > 20:
            visible_ticks = list(range(0, n_machines, max(1, n_machines // 10)))
            ax.set_xticks(visible_ticks)

    fig.tight_layout()
    _save_fig(fig, figures_dir / "sensor_boxplots.png")


def plot_machine_timeline(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    figures_dir: Path,
    max_machines: int = 20,
) -> None:
    """
    Visualise the telemetry time span and failure events for each machine.

    Each machine gets a horizontal bar representing its observation window,
    with failure events overlaid as red markers.

    Parameters
    ----------
    telemetry : pd.DataFrame
    failures : pd.DataFrame
    figures_dir : Path
    max_machines : int
        Maximum number of machines to display (default: 20).
    """
    figures_dir = Path(figures_dir)

    summary = (
        telemetry.groupby("machineID")["datetime"]
        .agg(min_ts="min", max_ts="max")
        .reset_index()
        .sort_values("machineID")
    )
    summary = summary.head(max_machines)

    fig, ax = plt.subplots(figsize=(16, max(6, len(summary) * 0.45)))
    y_positions = range(len(summary))

    for y, (_, row) in zip(y_positions, summary.iterrows()):
        ax.barh(
            y,
            left=row["min_ts"],
            width=row["max_ts"] - row["min_ts"],
            height=0.6,
            color="#4e9af1",
            alpha=0.7,
        )

    # Overlay failure events
    machine_to_y = {
        int(row["machineID"]): y
        for y, (_, row) in zip(y_positions, summary.iterrows())
    }
    for _, row in failures.iterrows():
        mid = int(row["machineID"])
        if mid in machine_to_y:
            ax.scatter(
                row["datetime"],
                machine_to_y[mid],
                color="crimson",
                marker="x",
                s=60,
                zorder=5,
                linewidths=1.5,
            )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([f"Machine {int(r['machineID'])}" for _, r in summary.iterrows()])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Date")
    ax.set_title(
        f"Machine Telemetry Timeline (first {max_machines} machines)\n"
        "Blue = observation window | ✗ = failure event",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, figures_dir / "machine_timeline.png")


def plot_records_per_machine(
    datasets: Dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """
    Save a grouped bar chart showing record counts per machine for each table.

    Parameters
    ----------
    datasets : Dict[str, pd.DataFrame]
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    event_tables = {
        k: v for k, v in datasets.items() if k in ("errors", "maint", "failures")
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle("Records per Machine", fontsize=15, fontweight="bold")

    palette = sns.color_palette("muted", 3)
    for ax, (name, df), color in zip(axes, event_tables.items(), palette):
        counts = df["machineID"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=color, alpha=0.85)
        ax.set_title(name.capitalize(), fontsize=13)
        ax.set_xlabel("Machine ID")
        ax.set_ylabel("Record Count")
        n_ticks = len(counts)
        if n_ticks > 20:
            step = max(1, n_ticks // 10)
            ax.set_xticks(ax.get_xticks()[::step])

    fig.tight_layout()
    _save_fig(fig, figures_dir / "records_per_machine.png")


def plot_gap_distribution(
    gaps_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Plot the distribution of telemetry gap sizes in hours.

    Parameters
    ----------
    gaps_df : pd.DataFrame
        Output of :func:`detect_telemetry_gaps`.
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Telemetry Gap Distribution", fontsize=14, fontweight="bold")

    if gaps_df.empty:
        for ax in axes:
            ax.text(
                0.5, 0.5,
                "No gaps > 3 hours detected.",
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes,
            )
            ax.axis("off")
    else:
        # Histogram
        axes[0].hist(
            gaps_df["gap_hours"], bins=40, color="#5c6bc0", edgecolor="none", alpha=0.85
        )
        axes[0].set_xlabel("Gap Duration (hours)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Histogram of Gap Durations")

        # Per-machine gap count
        per_machine = gaps_df["machineID"].value_counts().sort_index()
        axes[1].bar(
            per_machine.index.astype(str),
            per_machine.values,
            color="#ef6c00",
            alpha=0.85,
        )
        axes[1].set_xlabel("Machine ID")
        axes[1].set_ylabel("Number of Gaps")
        axes[1].set_title("Gaps per Machine")
        if len(per_machine) > 20:
            step = max(1, len(per_machine) // 10)
            axes[1].set_xticks(axes[1].get_xticks()[::step])

    fig.tight_layout()
    _save_fig(fig, figures_dir / "gap_distribution.png")


def plot_sensor_correlation(
    telemetry: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a heatmap of the Pearson correlation matrix for sensor columns.

    Parameters
    ----------
    telemetry : pd.DataFrame
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    corr = telemetry[SENSOR_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Sensor Pearson Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "sensor_correlation_heatmap.png")


def plot_failure_counts(
    failures: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a bar chart of failure counts per component type.

    Parameters
    ----------
    failures : pd.DataFrame
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    counts = failures["failure"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=sns.color_palette("Set1", len(counts)),
        alpha=0.88,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Failure Count per Component", fontsize=14, fontweight="bold")
    ax.set_xlabel("Component")
    ax.set_ylabel("Failure Count")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "failure_count_per_component.png")


def plot_maintenance_frequency(
    maint: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a bar chart of maintenance event frequency per component.

    Parameters
    ----------
    maint : pd.DataFrame
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    counts = maint["comp"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=sns.color_palette("Set2", len(counts)),
        alpha=0.88,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Maintenance Event Frequency per Component", fontsize=14, fontweight="bold")
    ax.set_xlabel("Component")
    ax.set_ylabel("Maintenance Count")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "maintenance_frequency.png")


def plot_error_frequency(
    errors: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a bar chart of error frequency per error type.

    Parameters
    ----------
    errors : pd.DataFrame
    figures_dir : Path
    """
    figures_dir = Path(figures_dir)
    counts = errors["errorID"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=sns.color_palette("tab10", len(counts)),
        alpha=0.88,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_title("Error Frequency per Error Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Error Count")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "error_frequency.png")
