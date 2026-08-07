# PredictGuard — Target Construction Report

> **Stage 2: Target Construction**  
> Prediction horizon: **24 hours**

---

## 1. Prediction Problem

Given machine sensor telemetry at hour **t**, predict whether the machine
will experience a failure within the next **24 hours**.

**Formal target definition:**

```
y_failure(t) = 1   if  failure_time in (t, t + 24h]  for the same machine
             = 0   otherwise
```

**Prediction window:** half-open interval `(t, t + 24h]`
— the current timestamp is excluded (no simultaneous failure counts),
and the endpoint is included (a failure at exactly t + 24h is a positive).


---

## 2. Leakage Prevention Strategy

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


---

## 3. Target Statistics

| Metric | Value |
|---|---|
| Total telemetry rows | 876,100 |
| Positive samples (y=1) | 17,184 |
| Negative samples (y=0) | 858,916 |
| Positive class % | 1.96% |
| Class imbalance ratio | 1 : 49.98 |
| Avg time to failure (pos labels) | 12.5 hours |
| Max consecutive positives | 24 |
| Machines with no failures | 2 |
| Total failure events | 761 |

---

## 4. Component Distribution

| Component | Positive Label Count |
|---|---|
| comp1 | 4,581 |
| comp2 | 5,991 |
| comp3 | 2,901 |
| comp4 | 3,711 |

---

## 5. Validation Results

| Check | Status |
|---|---|
| Positive labels valid (all y=1 have real future failure) | **PASS** |
| Negative labels pure (all y=0 have no future failure) | **PASS** |
| Component labels consistent | **PASS** |
| NaN consistency (comp is NaN iff y=0) | **PASS** |

**All validation checks passed. Zero label errors detected.**

---

## 6. Edge Cases Handled

| Edge Case | Handling Strategy |
|---|---|
| Machine with no failures | All rows labelled y=0; failure_component=NaN |
| Multiple failures in 24h window | y=1; component = component of FIRST failure |
| Failure exactly at t + 24h | Included (window end is inclusive) |
| Last telemetry rows | Labelled normally; they can still be y=1 if a real future failure exists |
| Duplicate failure timestamps | Handled naturally by searchsorted; no special case needed |
| Missing failure records | Treated as no-failure machine |

---

## 7. Machines With No Failures

The following **2** machine(s) have no recorded failure events. All their telemetry rows are negative (y=0):

[np.int64(6), np.int64(77)]

---

## 8. Outputs

| File | Description |
|---|---|
| `data/processed/telemetry_with_targets.parquet` | Primary output — typed, compact |
| `data/processed/telemetry_with_targets.csv` | Human-readable copy |
| `reports/target_construction_report.md` | This report |
| `reports/figures/target_*.png` | 7 publication-quality visualisations |
