"""
target_creation.py
==================
Leakage-safe prediction target construction for PredictGuard — Stage 2.

Prediction problem
------------------
"Given machine telemetry at hour t, predict whether the machine will fail
within the next 24 hours."

Leakage prevention strategy
----------------------------
Every computation is scoped strictly to a **single machine**.

- Telemetry is sorted by (machineID, datetime) before any operation.
- Future windows are computed per machine via groupby(machineID).
- No global shifts, no cross-machine joins on future data.
- Failure timestamps from machine A cannot influence labels for machine B.

Target definitions
------------------
y_failure (binary)
    1  if at least one failure occurs in the half-open window (t, t + 24h]
    0  otherwise

failure_component (multiclass)
    The component that fails *first* in the prediction window when y_failure=1.
    NaN when y_failure=0.

time_to_failure_hours (auxiliary)
    Hours from timestamp t to the first failure in the window (positive labels only).
    NaN when y_failure=0.

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
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (overridden at runtime via config.yaml)
# ---------------------------------------------------------------------------

#: Default prediction horizon in hours.
DEFAULT_HORIZON_HOURS: int = 24

#: Nanoseconds per hour — used for numpy datetime64 arithmetic.
_NS_PER_HOUR: int = 3_600 * 1_000_000_000

#: Components present in the dataset.
FAILURE_COMPONENTS: List[str] = ["comp1", "comp2", "comp3", "comp4"]

# ---------------------------------------------------------------------------
# Seaborn / Matplotlib style
# ---------------------------------------------------------------------------
sns.set_theme(
    style="darkgrid",
    palette="muted",
    rc={"figure.dpi": 150, "axes.titlesize": 14, "axes.labelsize": 12},
)


# ===========================================================================
# Section 1 — Data Loading
# ===========================================================================


def load_stage1_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load the five raw PdM CSV files produced/validated in Stage 1.

    Parameters
    ----------
    data_dir : Path
        Directory containing the five CSV files.

    Returns
    -------
    dict
        Keys: ``'telemetry'``, ``'failures'``, ``'errors'``,
        ``'maint'``, ``'machines'``.

    Raises
    ------
    FileNotFoundError
        If any expected file is absent.
    """
    file_map = {
        "telemetry": "PdM_telemetry.csv",
        "failures": "PdM_failures.csv",
        "errors": "PdM_errors.csv",
        "maint": "PdM_maint.csv",
        "machines": "PdM_machines.csv",
    }
    timestamp_tables = {"telemetry", "failures", "errors", "maint"}
    datasets: Dict[str, pd.DataFrame] = {}

    for name, fname in file_map.items():
        fpath = Path(data_dir) / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Expected file not found: {fpath}. "
                "Ensure Stage 1 data is present."
            )
        df = pd.read_csv(fpath, low_memory=False)
        if name in timestamp_tables:
            df["datetime"] = pd.to_datetime(df["datetime"])
        datasets[name] = df
        logger.info("Loaded %-10s | rows=%d", name, len(df))

    return datasets


# ===========================================================================
# Section 2 — Input Validation
# ===========================================================================


def validate_inputs(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
) -> List[str]:
    """
    Validate the schema and basic integrity of the two primary tables.

    Parameters
    ----------
    telemetry : pd.DataFrame
    failures : pd.DataFrame

    Returns
    -------
    List[str]
        List of issue strings. Empty list means validation passed.
    """
    issues: List[str] = []

    # Required columns
    for col in ["machineID", "datetime"]:
        if col not in telemetry.columns:
            issues.append(f"telemetry: missing column '{col}'")
        if col not in failures.columns:
            issues.append(f"failures: missing column '{col}'")

    if "failure" not in failures.columns:
        issues.append("failures: missing column 'failure'")

    # Datetime types
    if "datetime" in telemetry.columns and not pd.api.types.is_datetime64_any_dtype(
        telemetry["datetime"]
    ):
        issues.append("telemetry.datetime is not datetime64")
    if "datetime" in failures.columns and not pd.api.types.is_datetime64_any_dtype(
        failures["datetime"]
    ):
        issues.append("failures.datetime is not datetime64")

    # Check for simultaneous failure timestamps per machine (e.g., cascade failures)
    dup_count = failures.duplicated(subset=["machineID", "datetime"]).sum()
    if dup_count > 0:
        logger.info(
            "Found %d simultaneous failure events (multiple components failing at the exact same timestamp). "
            "Handled cleanly by deterministic component prioritization.",
            dup_count,
        )

    # Component values
    if "failure" in failures.columns:
        unknown = set(failures["failure"].unique()) - set(FAILURE_COMPONENTS)
        if unknown:
            issues.append(f"failures: unknown components: {unknown}")

    for issue in issues:
        logger.error("[validate_inputs] %s", issue)

    if not issues:
        logger.info("Input validation passed.")

    return issues


# ===========================================================================
# Section 3 — Binary Target Construction  (leakage-safe core)
# ===========================================================================


def _compute_machine_binary_target(
    machine_tel: pd.DataFrame,
    machine_fail: pd.DataFrame,
    horizon_ns: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ``y_failure`` and ``time_to_failure_hours`` for a single machine.

    Uses ``numpy.searchsorted`` for O(n log m) performance — no row-level
    Python loops.

    Parameters
    ----------
    machine_tel : pd.DataFrame
        Telemetry for ONE machine, sorted ascending by ``datetime``.
    machine_fail : pd.DataFrame
        Failures for the SAME machine, sorted ascending by ``datetime``.
    horizon_ns : int
        Prediction window width in nanoseconds.

    Returns
    -------
    y_failure : np.ndarray of int  (shape: [len(machine_tel)])
    time_to_failure_hours : np.ndarray of float  (NaN for negatives)
    """
    n = len(machine_tel)
    y_failure = np.zeros(n, dtype=np.int8)
    time_to_failure = np.full(n, np.nan, dtype=np.float64)

    if machine_fail.empty:
        return y_failure, time_to_failure

    # Convert to int64 nanoseconds for fast arithmetic (ensuring ns resolution)
    tel_ns = machine_tel["datetime"].values.astype("datetime64[ns]").astype(np.int64)
    fail_ns = np.sort(machine_fail["datetime"].values.astype("datetime64[ns]").astype(np.int64))

    # For each telemetry timestamp t:
    #   lo = first failure index with time  >  t
    #   hi = first failure index with time  >  t + horizon
    # → failure exists in (t, t+horizon] iff lo < hi
    lo = np.searchsorted(fail_ns, tel_ns, side="right")
    hi = np.searchsorted(fail_ns, tel_ns + horizon_ns, side="right")

    has_failure = lo < hi
    y_failure[has_failure] = 1

    # Time to the FIRST failure in the window (for positive labels)
    pos_idx = np.where(has_failure)[0]
    if pos_idx.size > 0:
        first_fail_ns = fail_ns[lo[pos_idx]]
        time_to_failure[pos_idx] = (
            (first_fail_ns - tel_ns[pos_idx]) / _NS_PER_HOUR
        )

    return y_failure, time_to_failure


def compute_binary_target(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> pd.DataFrame:
    """
    Construct the leakage-safe binary prediction target.

    Iterates over machines independently. For each telemetry row at time t,
    labels ``y_failure = 1`` if the machine fails in ``(t, t + horizon_hours]``.

    Parameters
    ----------
    telemetry : pd.DataFrame
        Must contain ``['machineID', 'datetime']`` plus sensor columns.
    failures : pd.DataFrame
        Must contain ``['machineID', 'datetime', 'failure']``.
    horizon_hours : int
        Prediction window width in hours. Default: 24.

    Returns
    -------
    pd.DataFrame
        Input telemetry extended with columns:
        ``['y_failure', 'time_to_failure_hours']``.
        Rows are sorted by ``(machineID, datetime)``.
    """
    logger.info(
        "Building binary target — horizon=%dh | machines=%d | rows=%d",
        horizon_hours,
        telemetry["machineID"].nunique(),
        len(telemetry),
    )

    horizon_ns = horizon_hours * _NS_PER_HOUR
    results: List[pd.DataFrame] = []

    machine_ids = sorted(telemetry["machineID"].unique())
    for machine_id in machine_ids:
        tel_grp = (
            telemetry[telemetry["machineID"] == machine_id]
            .sort_values("datetime")
            .copy()
        )
        fail_grp = failures[failures["machineID"] == machine_id]

        y_fail, time_to_fail = _compute_machine_binary_target(
            tel_grp, fail_grp, horizon_ns
        )
        tel_grp["y_failure"] = y_fail
        tel_grp["time_to_failure_hours"] = time_to_fail
        results.append(tel_grp)

    df = pd.concat(results, ignore_index=True)

    pos = int(df["y_failure"].sum())
    neg = len(df) - pos
    pct = 100.0 * pos / max(len(df), 1)
    logger.info(
        "Binary target done — positives=%d (%.2f%%) | negatives=%d",
        pos, pct, neg,
    )
    return df


# ===========================================================================
# Section 4 — Component Target Assignment
# ===========================================================================


def assign_failure_component(
    df: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> pd.DataFrame:
    """
    Add ``failure_component`` column to the target DataFrame.

    For each positive label (``y_failure == 1``), the column is set to the
    component that fails **first** in the prediction window.  For negative
    labels the value is ``pd.NA``.

    No component labels are invented.  Values are drawn directly from
    ``PdM_failures.csv``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_binary_target`.
    failures : pd.DataFrame
    horizon_hours : int

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with ``'failure_component'``.
    """
    logger.info("Assigning failure components to positive labels …")

    result = df.copy()
    result["failure_component"] = pd.NA

    horizon_ns = horizon_hours * _NS_PER_HOUR

    for machine_id in sorted(result["machineID"].unique()):
        mask_tel = result["machineID"] == machine_id
        tel_grp = result.loc[mask_tel]
        pos_mask = tel_grp["y_failure"] == 1

        if not pos_mask.any():
            continue

        fail_grp = (
            failures[failures["machineID"] == machine_id]
            .sort_values(["datetime", "failure"])
        )
        if fail_grp.empty:
            # Should not happen if y_failure=1 — guard anyway
            logger.warning(
                "Machine %d has positive labels but no failure records!", machine_id
            )
            continue

        fail_ns = fail_grp["datetime"].values.astype("datetime64[ns]").astype(np.int64)
        fail_components = fail_grp["failure"].values

        pos_tel = tel_grp.loc[pos_mask]
        pos_ns = pos_tel["datetime"].values.astype("datetime64[ns]").astype(np.int64)

        # First failure in window: lo = first index with fail > t
        lo = np.searchsorted(fail_ns, pos_ns, side="right")

        # Clamp to valid range (should always be valid given y_failure=1)
        lo = np.clip(lo, 0, len(fail_ns) - 1)
        components = fail_components[lo]

        result.loc[pos_tel.index, "failure_component"] = components

    assigned = result["failure_component"].notna().sum()
    logger.info(
        "Component assignment done — %d positive rows assigned components.", assigned
    )
    return result


# ===========================================================================
# Section 5 — Leakage-Safe Validation
# ===========================================================================


def validate_targets(
    df: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    spot_check_n: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Exhaustive row-level validation that every label is correct.

    Validation checks
    -----------------
    1. **Positive completeness**: every ``y_failure=1`` row has a real failure
       in its machine's ``(t, t+horizon]`` window.
    2. **Negative purity**: every ``y_failure=0`` row has NO failure in window.
    3. **Component consistency**: every assigned ``failure_component`` is a real
       component that fails first in the window.
    4. **NaN consistency**: ``failure_component`` is NaN iff ``y_failure=0``.
    5. **Spot-check report**: logs a human-readable per-machine summary for
       *n* randomly chosen machines.

    Parameters
    ----------
    df : pd.DataFrame
    failures : pd.DataFrame
    horizon_hours : int
    spot_check_n : int
    random_seed : int

    Returns
    -------
    dict
        Keys: ``'positive_check_pass'``, ``'negative_check_pass'``,
        ``'component_check_pass'``, ``'nan_consistency_pass'``,
        ``'errors'`` (list of strings).
    """
    logger.info("Running exhaustive target validation …")

    horizon_ns = horizon_hours * _NS_PER_HOUR
    errors: List[str] = []
    pos_violations = 0
    neg_violations = 0
    comp_violations = 0

    for machine_id in sorted(df["machineID"].unique()):
        tel_grp = df[df["machineID"] == machine_id].sort_values("datetime")
        fail_grp = failures[failures["machineID"] == machine_id].sort_values(["datetime", "failure"])
        fail_ns = (
            np.sort(fail_grp["datetime"].values.astype("datetime64[ns]").astype(np.int64))
            if not fail_grp.empty else np.array([], dtype=np.int64)
        )
        fail_comp_arr = (
            fail_grp.sort_values(["datetime", "failure"])["failure"].values
            if not fail_grp.empty else np.array([], dtype=object)
        )

        tel_ns = tel_grp["datetime"].values.astype("datetime64[ns]").astype(np.int64)
        lo = np.searchsorted(fail_ns, tel_ns, side="right")
        hi = np.searchsorted(fail_ns, tel_ns + horizon_ns, side="right")
        expected_y = (hi > lo).astype(int)

        actual_y = tel_grp["y_failure"].values.astype(int)
        mismatch = expected_y != actual_y

        if mismatch.any():
            n_pos_err = ((actual_y == 1) & mismatch).sum()
            n_neg_err = ((actual_y == 0) & mismatch).sum()
            pos_violations += n_pos_err
            neg_violations += n_neg_err
            if n_pos_err:
                errors.append(
                    f"Machine {machine_id}: {n_pos_err} false-positive labels"
                )
            if n_neg_err:
                errors.append(
                    f"Machine {machine_id}: {n_neg_err} false-negative labels"
                )

        # Component consistency check for positive labels
        pos_rows = tel_grp[tel_grp["y_failure"] == 1]
        if not pos_rows.empty and fail_comp_arr.size > 0:
            pos_ns_arr = pos_rows["datetime"].values.astype("datetime64[ns]").astype(np.int64)
            lo_pos = np.searchsorted(fail_ns, pos_ns_arr, side="right")
            lo_pos_clamped = np.clip(lo_pos, 0, len(fail_comp_arr) - 1)
            expected_comp = fail_comp_arr[lo_pos_clamped]
            actual_comp = pos_rows["failure_component"].values.astype(str)
            comp_mismatch = expected_comp != actual_comp
            if comp_mismatch.any():
                comp_violations += int(comp_mismatch.sum())
                errors.append(
                    f"Machine {machine_id}: {comp_mismatch.sum()} component mismatches"
                )

    # NaN consistency
    nan_wrong = (
        (df["y_failure"] == 1) & (df["failure_component"].isna())
    ).sum() + (
        (df["y_failure"] == 0) & (df["failure_component"].notna())
    ).sum()

    if nan_wrong > 0:
        errors.append(f"NaN consistency violated: {nan_wrong} rows affected")

    # --- Spot-check report ---
    rng = np.random.default_rng(random_seed)
    sample_ids = rng.choice(
        sorted(df["machineID"].unique()),
        size=min(spot_check_n, df["machineID"].nunique()),
        replace=False,
    )
    logger.info("Spot-check report for machines: %s", sorted(sample_ids.tolist()))
    for mid in sorted(sample_ids):
        sub = df[df["machineID"] == mid]
        n_pos = int((sub["y_failure"] == 1).sum())
        n_fail = int(
            (failures["machineID"] == mid).sum()
        )
        comps = (
            sub["failure_component"].dropna().value_counts().to_dict()
        )
        logger.info(
            "  Machine %3d | rows=%4d | positives=%3d | actual_failures=%2d | comps=%s",
            mid, len(sub), n_pos, n_fail, comps,
        )

    # Summary
    positive_ok = pos_violations == 0
    negative_ok = neg_violations == 0
    component_ok = comp_violations == 0
    nan_ok = nan_wrong == 0

    if positive_ok and negative_ok and component_ok and nan_ok:
        logger.info("Validation PASSED — all labels are correct.")
    else:
        logger.error(
            "Validation FAILED — pos_violations=%d neg_violations=%d "
            "comp_violations=%d nan_inconsistencies=%d",
            pos_violations, neg_violations, comp_violations, nan_wrong,
        )

    return {
        "positive_check_pass": positive_ok,
        "negative_check_pass": negative_ok,
        "component_check_pass": component_ok,
        "nan_consistency_pass": nan_ok,
        "pos_violations": pos_violations,
        "neg_violations": neg_violations,
        "comp_violations": comp_violations,
        "nan_inconsistencies": int(nan_wrong),
        "errors": errors,
    }


# ===========================================================================
# Section 6 — Target Statistics
# ===========================================================================


def compute_target_statistics(
    df: pd.DataFrame,
    failures: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about the constructed targets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``y_failure`` and ``failure_component``.
    failures : pd.DataFrame

    Returns
    -------
    dict
        Rich statistics dictionary used by the report and notebook.
    """
    logger.info("Computing target statistics …")

    n_total = len(df)
    n_pos = int(df["y_failure"].sum())
    n_neg = n_total - n_pos
    pos_pct = round(100.0 * n_pos / max(n_total, 1), 4)
    imbalance_ratio = round(n_neg / max(n_pos, 1), 2)

    component_dist = (
        df["failure_component"].value_counts().to_dict()
    )

    failures_per_machine = (
        failures.groupby("machineID").size()
        .rename("failure_count")
        .reset_index()
    )

    positives_per_machine = (
        df.groupby("machineID")["y_failure"]
        .sum()
        .rename("positive_count")
        .reset_index()
    )

    # Maximum consecutive positive labels per machine
    max_consecutive = 0
    for _, grp in df.sort_values(["machineID", "datetime"]).groupby("machineID"):
        y = grp["y_failure"].values
        if y.max() == 0:
            continue
        # Run-length encoding
        changes = np.diff(y, prepend=0, append=0)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        lengths = ends - starts
        if lengths.size > 0:
            max_consecutive = max(max_consecutive, int(lengths.max()))

    # Average time to failure for positive labels
    pos_df = df[df["y_failure"] == 1]
    avg_time_to_failure = (
        round(pos_df["time_to_failure_hours"].mean(), 2)
        if not pos_df.empty else 0.0
    )

    # Machines with zero failures
    all_machines = set(df["machineID"].unique())
    failed_machines = set(failures["machineID"].unique())
    machines_no_failure = sorted(all_machines - failed_machines)

    stats = {
        "total_rows": n_total,
        "positive_samples": n_pos,
        "negative_samples": n_neg,
        "positive_pct": pos_pct,
        "imbalance_ratio": imbalance_ratio,
        "component_distribution": component_dist,
        "failures_per_machine": failures_per_machine,
        "positives_per_machine": positives_per_machine,
        "max_consecutive_positives": max_consecutive,
        "avg_time_to_failure_hours": avg_time_to_failure,
        "machines_with_no_failure": machines_no_failure,
        "n_machines_no_failure": len(machines_no_failure),
        "n_unique_failure_events": len(failures),
    }

    logger.info(
        "Statistics — rows=%d | pos=%d (%.2f%%) | neg=%d | ratio=1:%.1f",
        n_total, n_pos, pos_pct, n_neg, imbalance_ratio,
    )
    logger.info("Component distribution: %s", component_dist)
    logger.info(
        "Machines with no failure: %d | max_consecutive_pos: %d",
        len(machines_no_failure), max_consecutive,
    )
    return stats


# ===========================================================================
# Section 7 — Persistence
# ===========================================================================


def save_targets(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Persist the labelled DataFrame to Parquet and CSV.

    Parquet is the primary format (type-safe, compact, fast).
    CSV is provided for human inspection and compatibility.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "telemetry_with_targets.parquet"
    csv_path = output_dir / "telemetry_with_targets.csv"

    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved parquet: %s  (%s MB)", parquet_path, _file_mb(parquet_path))

    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV    : %s  (%s MB)", csv_path, _file_mb(csv_path))


def _file_mb(path: Path) -> str:
    """Return file size in MB as a formatted string."""
    size = path.stat().st_size / 1_048_576
    return f"{size:.1f}"


# ===========================================================================
# Section 8 — Markdown Report
# ===========================================================================


def generate_target_report(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    validation: Dict[str, Any],
    output_path: Path,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> None:
    """
    Write a comprehensive Markdown report for Stage 2.

    Parameters
    ----------
    df : pd.DataFrame
    stats : dict
        Output of :func:`compute_target_statistics`.
    validation : dict
        Output of :func:`validate_targets`.
    output_path : Path
    horizon_hours : int
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    val_icon = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731

    lines = [
        "# PredictGuard — Target Construction Report",
        "",
        "> **Stage 2: Target Construction**  ",
        f"> Prediction horizon: **{horizon_hours} hours**",
        "",
        "---",
        "",
        "## 1. Prediction Problem",
        "",
        textwrap.dedent(f"""\
        Given machine sensor telemetry at hour **t**, predict whether the machine
        will experience a failure within the next **{horizon_hours} hours**.

        **Formal target definition:**

        ```
        y_failure(t) = 1   if  failure_time in (t, t + {horizon_hours}h]  for the same machine
                     = 0   otherwise
        ```

        **Prediction window:** half-open interval `(t, t + {horizon_hours}h]`
        — the current timestamp is excluded (no simultaneous failure counts),
        and the endpoint is included (a failure at exactly t + {horizon_hours}h is a positive).
        """),
        "",
        "---",
        "",
        "## 2. Leakage Prevention Strategy",
        "",
        textwrap.dedent("""\
        Strict leakage prevention is applied at every step:

        | Rule | Implementation |
        |---|---|
        | Per-machine computation | `groupby(machineID)` — never mix machines |
        | No global shifting | `y_failure` uses forward-looking windows, not `.shift()` |
        | No future sensor data | Labels use only failure timestamps, not future telemetry |
        | No cross-machine join | Future windows are computed against the SAME machine's failures |
        | Sorted before labelling | Telemetry sorted by `(machineID, datetime)` before any operation |

        > **Why per-machine?**  Rolling or shifting across the full DataFrame would silently
        > mix timestamps from different machines whenever machine boundaries occur
        > consecutively in memory, corrupting labels for boundary rows.
        """),
        "",
        "---",
        "",
        "## 3. Target Statistics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total telemetry rows | {stats['total_rows']:,} |",
        f"| Positive samples (y=1) | {stats['positive_samples']:,} |",
        f"| Negative samples (y=0) | {stats['negative_samples']:,} |",
        f"| Positive class % | {stats['positive_pct']:.2f}% |",
        f"| Class imbalance ratio | 1 : {stats['imbalance_ratio']} |",
        f"| Avg time to failure (pos labels) | {stats['avg_time_to_failure_hours']:.1f} hours |",
        f"| Max consecutive positives | {stats['max_consecutive_positives']} |",
        f"| Machines with no failures | {stats['n_machines_no_failure']} |",
        f"| Total failure events | {stats['n_unique_failure_events']} |",
        "",
        "---",
        "",
        "## 4. Component Distribution",
        "",
        "| Component | Positive Label Count |",
        "|---|---|",
    ]
    for comp, cnt in sorted(stats["component_distribution"].items()):
        lines.append(f"| {comp} | {cnt:,} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Validation Results",
        "",
        f"| Check | Status |",
        f"|---|---|",
        f"| Positive labels valid (all y=1 have real future failure) | **{val_icon(validation['positive_check_pass'])}** |",
        f"| Negative labels pure (all y=0 have no future failure) | **{val_icon(validation['negative_check_pass'])}** |",
        f"| Component labels consistent | **{val_icon(validation['component_check_pass'])}** |",
        f"| NaN consistency (comp is NaN iff y=0) | **{val_icon(validation['nan_consistency_pass'])}** |",
        "",
    ]
    if validation["errors"]:
        lines += ["**Validation errors:**", ""]
        for err in validation["errors"]:
            lines.append(f"- {err}")
        lines.append("")
    else:
        lines += ["**All validation checks passed. Zero label errors detected.**", ""]

    lines += [
        "---",
        "",
        "## 6. Edge Cases Handled",
        "",
        "| Edge Case | Handling Strategy |",
        "|---|---|",
        "| Machine with no failures | All rows labelled y=0; failure_component=NaN |",
        "| Multiple failures in 24h window | y=1; component = component of FIRST failure |",
        "| Failure exactly at t + 24h | Included (window end is inclusive) |",
        "| Last telemetry rows | Labelled normally; they can still be y=1 if a real future failure exists |",
        "| Duplicate failure timestamps | Handled naturally by searchsorted; no special case needed |",
        "| Missing failure records | Treated as no-failure machine |",
        "",
        "---",
        "",
        "## 7. Machines With No Failures",
        "",
        f"The following **{stats['n_machines_no_failure']}** machine(s) have "
        f"no recorded failure events. All their telemetry rows are negative (y=0):",
        "",
        str(stats["machines_with_no_failure"]),
        "",
        "---",
        "",
        "## 8. Outputs",
        "",
        "| File | Description |",
        "|---|---|",
        "| `data/processed/telemetry_with_targets.parquet` | Primary output — typed, compact |",
        "| `data/processed/telemetry_with_targets.csv` | Human-readable copy |",
        "| `reports/target_construction_report.md` | This report |",
        "| `reports/figures/target_*.png` | 7 publication-quality visualisations |",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Target report saved: %s", output_path)


# ===========================================================================
# Section 9 — Visualisations
# ===========================================================================


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure and close to free memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_class_distribution(
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Save a two-panel class distribution chart (bar + pie).

    Parameters
    ----------
    df : pd.DataFrame
    figures_dir : Path
    """
    counts = df["y_failure"].value_counts().sort_index()
    labels = ["Negative (y=0)\nNo failure", "Positive (y=1)\nFailure within 24h"]
    colors = ["#4e9af1", "#ef5350"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Target Class Distribution", fontsize=16, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, alpha=0.88, width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            f"{val:,}\n({100*val/len(df):.2f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    axes[0].set_ylabel("Row Count")
    axes[0].set_title("Absolute Count")
    axes[0].set_ylim(0, counts.max() * 1.15)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct="%1.2f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    axes[1].set_title("Proportion")

    fig.tight_layout()
    _save_fig(fig, figures_dir / "target_class_distribution.png")


def plot_failure_timeline(
    failures: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Scatter plot of all failure events over time, coloured by component.

    Parameters
    ----------
    failures : pd.DataFrame
    figures_dir : Path
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    palette = sns.color_palette("Set1", len(FAILURE_COMPONENTS))
    comp_colors = dict(zip(FAILURE_COMPONENTS, palette))

    for comp in FAILURE_COMPONENTS:
        sub = failures[failures["failure"] == comp]
        ax.scatter(
            sub["datetime"],
            sub["machineID"],
            c=[comp_colors[comp]],
            label=comp,
            s=40,
            alpha=0.75,
            edgecolors="none",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Machine ID")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_title("Machine Failure Timeline by Component", fontsize=14, fontweight="bold")
    ax.legend(title="Component", loc="upper left", framealpha=0.85)
    ax.set_yticks(range(1, failures["machineID"].max() + 1, 5))
    fig.tight_layout()
    _save_fig(fig, figures_dir / "target_failure_timeline.png")


def plot_component_frequency(
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Bar chart of positive-label count per failure component.

    Parameters
    ----------
    df : pd.DataFrame
    figures_dir : Path
    """
    comp_counts = (
        df[df["y_failure"] == 1]["failure_component"]
        .value_counts()
        .reindex(FAILURE_COMPONENTS, fill_value=0)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    palette = sns.color_palette("Set1", len(comp_counts))
    bars = ax.bar(comp_counts.index, comp_counts.values, color=palette, alpha=0.88)

    for bar, val in zip(bars, comp_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + comp_counts.max() * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xlabel("Component")
    ax.set_ylabel("Positive Label Count")
    ax.set_title("Positive Labels per Failure Component", fontsize=14, fontweight="bold")
    ax.set_ylim(0, comp_counts.max() * 1.15)
    fig.tight_layout()
    _save_fig(fig, figures_dir / "target_component_frequency.png")


def plot_positives_per_machine(
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Horizontal bar chart of positive label count per machine.

    Parameters
    ----------
    df : pd.DataFrame
    figures_dir : Path
    """
    per_machine = (
        df.groupby("machineID")["y_failure"]
        .sum()
        .sort_values(ascending=True)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(per_machine) * 0.22)))
    colors = [
        "#ef5350" if v > 0 else "#78909c" for v in per_machine.values
    ]
    bars = ax.barh(
        [f"Machine {m}" for m in per_machine.index],
        per_machine.values,
        color=colors,
        alpha=0.85,
    )
    ax.set_xlabel("Positive Label Count (y=1)")
    ax.set_title(
        "Positive Labels per Machine\n(red = has failures, grey = no failures)",
        fontsize=13, fontweight="bold",
    )
    ax.axvline(per_machine.mean(), color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean: {per_machine.mean():.0f}")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, figures_dir / "target_positives_per_machine.png")


def plot_time_to_failure_histogram(
    df: pd.DataFrame,
    figures_dir: Path,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> None:
    """
    Histogram of ``time_to_failure_hours`` for positive labels.

    Reveals whether labels are uniformly distributed across the prediction
    window or concentrated near the boundary.

    Parameters
    ----------
    df : pd.DataFrame
    figures_dir : Path
    horizon_hours : int
    """
    pos = df[df["y_failure"] == 1]["time_to_failure_hours"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Time to Failure Distribution (Positive Labels Only)",
        fontsize=14, fontweight="bold",
    )

    # Main histogram
    axes[0].hist(
        pos, bins=horizon_hours, range=(0, horizon_hours),
        color="#ef5350", edgecolor="none", alpha=0.82,
    )
    axes[0].set_xlabel(f"Hours until first failure (0–{horizon_hours}h)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Raw Count")
    axes[0].axvline(pos.mean(), color="navy", linestyle="--",
                    label=f"Mean: {pos.mean():.1f}h")
    axes[0].axvline(pos.median(), color="green", linestyle=":",
                    label=f"Median: {pos.median():.1f}h")
    axes[0].legend()

    # CDF
    sorted_vals = np.sort(pos)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1].plot(sorted_vals, cdf, color="#ef5350", linewidth=2)
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1, label="50th pct")
    axes[1].axhline(0.9, color="grey", linestyle=":", linewidth=1, label="90th pct")
    axes[1].set_xlabel(f"Hours until failure")
    axes[1].set_ylabel("Cumulative Probability")
    axes[1].set_title("CDF")
    axes[1].legend()

    fig.tight_layout()
    _save_fig(fig, figures_dir / "target_time_to_failure_hist.png")


def plot_positive_windows_sample(
    df: pd.DataFrame,
    failures: pd.DataFrame,
    figures_dir: Path,
    n_machines: int = 4,
    random_seed: int = 42,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> None:
    """
    Timeline plot showing positive label windows for sample machines.

    For each selected machine, shows:
    - Grey background: full observation period
    - Red shading: 24h windows before each failure
    - Red markers: actual failure events
    - Blue line: y_failure signal (0/1)

    Parameters
    ----------
    df : pd.DataFrame
    failures : pd.DataFrame
    figures_dir : Path
    n_machines : int
    random_seed : int
    horizon_hours : int
    """
    rng = np.random.default_rng(random_seed)
    # Pick machines that actually have failures for a meaningful plot
    failed_machines = sorted(failures["machineID"].unique())
    sample_ids = rng.choice(
        failed_machines,
        size=min(n_machines, len(failed_machines)),
        replace=False,
    )

    fig, axes = plt.subplots(
        len(sample_ids), 1,
        figsize=(16, 3.5 * len(sample_ids)),
        sharex=False,
    )
    if len(sample_ids) == 1:
        axes = [axes]

    fig.suptitle(
        f"Positive Label Windows — Sample Machines\n"
        f"(red shading = {horizon_hours}h prediction window before failure)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    horizon = pd.Timedelta(hours=horizon_hours)

    for ax, mid in zip(axes, sorted(sample_ids)):
        tel = df[df["machineID"] == mid].sort_values("datetime")
        fail = failures[failures["machineID"] == mid].sort_values("datetime")

        # Plot y_failure as a filled area
        ax.fill_between(
            tel["datetime"], tel["y_failure"],
            step="post", color="#ef5350", alpha=0.35, label="y_failure=1",
        )
        ax.plot(
            tel["datetime"], tel["y_failure"],
            color="#ef5350", linewidth=0.8, alpha=0.6,
        )

        # Shade 24h pre-failure windows
        for _, frow in fail.iterrows():
            window_start = frow["datetime"] - horizon
            ax.axvspan(window_start, frow["datetime"],
                       color="#ef5350", alpha=0.12)
            ax.axvline(frow["datetime"], color="darkred",
                       linewidth=1.5, linestyle="--", alpha=0.8)

        ax.set_title(f"Machine {mid}", fontsize=12)
        ax.set_ylabel("y_failure")
        ax.set_ylim(-0.1, 1.35)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save_fig(fig, figures_dir / "target_positive_windows_sample.png")


def plot_failure_heatmap(
    failures: pd.DataFrame,
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Heatmap of failure counts per machine per month.

    Rows = machines, columns = calendar months.
    Color intensity = failure count.

    Parameters
    ----------
    failures : pd.DataFrame
    df : pd.DataFrame
        Used to get all machine IDs (including those with 0 failures).
    figures_dir : Path
    """
    fail_copy = failures.copy()
    fail_copy["month"] = fail_copy["datetime"].dt.to_period("M")

    pivot = (
        fail_copy.groupby(["machineID", "month"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all machines present
    all_machines = sorted(df["machineID"].unique())
    pivot = pivot.reindex(all_machines, fill_value=0)

    fig, ax = plt.subplots(figsize=(18, max(8, len(all_machines) * 0.22)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Failure Count", "shrink": 0.6},
        xticklabels=True,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Machine ID")
    ax.set_title(
        "Failure Heatmap — Machines x Months",
        fontsize=14, fontweight="bold",
    )
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save_fig(fig, figures_dir / "target_failure_heatmap.png")
