# PredictGuard — Fleet Risk Segmentation Report

> Generated: 2026-08-07 22:23 UTC

---

## 1. Overview

Stage 11 converts every machine-hour's calibrated probability into a fleet-wide
**Risk Score** (0–100) and assigns it to one of four **Risk Tiers**.
This transforms individual predictions into a prioritised maintenance queue.

**Risk Score** = `round(100 × calibrated_probability)`

---

## 2. Risk Tier Definitions

| Tier | Score Range | Recommended Action |
|---|---|---|
| 🟢 LOW | 0–19 | Continue scheduled monitoring |
| 🟡 MEDIUM | 20–49 | Increase monitoring frequency |
| 🟠 HIGH | 50–74 | Schedule maintenance within 48h |
| 🔴 CRITICAL | 75–100 | Dispatch technician immediately |

> Thresholds are fully configurable in `config.yaml` under `risk_segmentation`.

---

## 3. Fleet Tier Summary

| Tier | Machine-Hours | Unique Machines | % Machines | Avg Risk | Max Risk | Failure Rate | Top Component |
|---|---|---|---|---|---|---|---|
| **LOW** | 171,111 | 20 | 100.0% | 0.0 | 19 | 0.04% | `comp2` |
| **MEDIUM** | 121 | 17 | 85.0% | 33.6 | 49 | 19.01% | `comp3` |
| **HIGH** | 65 | 16 | 80.0% | 64.0 | 74 | 43.08% | `comp4` |
| **CRITICAL** | 3,923 | 20 | 100.0% | 93.5 | 94 | 93.98% | `comp2` |

---

## 4. Machine Ranking (Top 20 Highest-Risk)

| Rank | Machine ID | Risk Score | Tier | Calibrated Prob | Component | Avg Risk | Failure Rate |
|---|---|---|---|---|---|---|---|
| 1 | `2` | **0** | LOW | 0.0% | `comp2` | 0.8 | 0.82% |
| 2 | `3` | **0** | LOW | 0.0% | `comp2` | 1.2 | 1.37% |
| 3 | `16` | **0** | LOW | 0.0% | `comp4` | 3.0 | 2.98% |
| 4 | `22` | **0** | LOW | 0.0% | `comp1` | 3.4 | 3.53% |
| 5 | `23` | **0** | LOW | 0.1% | `comp3` | 3.2 | 3.29% |
| 6 | `25` | **0** | LOW | 0.0% | `comp4` | 1.8 | 1.64% |
| 7 | `31` | **0** | LOW | 0.0% | `comp3` | 1.6 | 1.64% |
| 8 | `39` | **0** | LOW | 0.0% | `comp2` | 1.1 | 1.10% |
| 9 | `53` | **0** | LOW | 0.0% | `comp2` | 1.8 | 1.64% |
| 10 | `54` | **0** | LOW | 0.0% | `comp2` | 1.3 | 1.37% |
| 11 | `62` | **0** | LOW | 0.0% | `comp1` | 1.9 | 1.92% |
| 12 | `65` | **0** | LOW | 0.0% | `comp4` | 1.3 | 1.37% |
| 13 | `73` | **0** | LOW | 0.0% | `comp4` | 2.9 | 3.01% |
| 14 | `76` | **0** | LOW | 0.0% | `comp1` | 2.5 | 2.47% |
| 15 | `85` | **0** | LOW | 0.0% | `comp1` | 3.2 | 3.01% |
| 16 | `89` | **0** | LOW | 0.0% | `comp1` | 1.9 | 1.92% |
| 17 | `90` | **0** | LOW | 0.0% | `comp2` | 3.1 | 3.01% |
| 18 | `92` | **0** | LOW | 0.0% | `comp1` | 2.2 | 2.19% |
| 19 | `94` | **0** | LOW | 0.0% | `comp2` | 2.5 | 2.71% |
| 20 | `95` | **0** | LOW | 0.0% | `comp1` | 2.4 | 2.47% |

---

## 5. Methodology

1. **Calibrated Probability** from Sigmoid-calibrated XGBoost (Phase 2, Stage 8).
2. **Risk Score** = `round(100 × calibrated_prob)` — integer scale for readability.
3. **Tier Assignment** — configurable thresholds, no arbitrary hard-coding.
4. **Machine-Level Aggregation** — each machine's *latest* snapshot determines
   its current tier; historical averages are also tracked.

---

## 6. Figures

| Figure | Description |
|---|---|
| `fleet_risk_distribution.png` | Histogram of risk scores (normal + log scale) |
| `fleet_tier_pie.png` | Pie chart of tier distribution |
| `fleet_tier_bar.png` | Bar chart with failure rate overlay |
| `fleet_machine_ranking.png` | Top 20 machines by risk score |
| `fleet_failure_rate_by_tier.png` | Actual failure rate per tier |
| `fleet_component_by_tier.png` | Component distribution per tier |
| `fleet_risk_histogram.png` | Per-machine latest risk score |
