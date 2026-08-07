# PredictGuard — Cost-Based Dispatch Decision Report

> Generated: 2026-08-07 22:24 UTC

---

## 1. Objective

Stage 12 answers: **Should we dispatch a technician?**

Instead of a naive 0.5 default, PredictGuard selects the probability threshold
that **minimises expected business cost** — accounting for the asymmetry between
the cost of missing a real failure vs. the cost of an unnecessary dispatch.

---

## 2. Cost Assumptions

> All cost values are configurable in `config.yaml` under `cost_decision`.

| Cost Type | Symbol | Value | Description |
|---|---|---|---|
| False Negative | `C_FN` | $10,000 | Missed failure: emergency repair + downtime + safety impact |
| False Positive | `C_FP` | $500 | False alarm: technician dispatch + inspection overhead |
| **Ratio** | `C_FN / C_FP` | **20×** | Missing a failure costs 20× more than a false alarm |

**Expected Cost Formula:**

```
Expected Cost(t) = C_FN × FN(t) + C_FP × FP(t)
```

---

## 3. Threshold Optimisation

> ⚠️ **Optimisation performed on Development set ONLY (80 machines).**
> The Final Test set (20 machines) was never seen during threshold selection.

| Property | Value |
|---|---|
| Sweep Range | 0.05 → 0.95 (step 0.01) |
| Selection Criterion | Minimum Expected Cost |
| **Optimal Threshold** | **0.6800** |
| Expected Cost at Optimal | $337,500 |
| Recall at Optimal | 0.9998 |
| Precision at Optimal | 0.9560 |
| F1 at Optimal | 0.9774 |
| Dispatch Rate at Optimal | 2.00% |

---

## 4. Final Test Set Evaluation

> Applied optimal threshold **once** to the Final Test set.

| Metric | Value |
|---|---|
| Threshold | **0.6800** |
| Precision | 0.9371 |
| Recall | **0.9711** |
| F1 | 0.9538 |
| Expected Cost | **$1,224,000** |
| Dispatch Rate | 2.25% |
| TP / FP / TN / FN | 3697 / 248 / 171165 / 110 |

### Cost Savings vs Naive 0.5 Baseline

| Metric | Baseline (t=0.50) | Optimal | Saving |
|---|---|---|---|
| Expected Cost | $1,056,500 | $1,224,000 | **$-167,500 (-15.8%)** |

---

## 5. Example Dispatch Decisions

### Machine 2 — 🚨 DISPATCH

| Property | Value |
|---|---|
| Risk Score | **0** / 100 |
| Risk Tier | **LOW** |
| Calibrated Probability | 94.4% |
| Predicted Component | `comp2` (29% confidence) |
| Decision | **DISPATCH** |
| Reason | Peak risk 94.4% ≥ optimal threshold 68.0%. |
| Action | Dispatch mechanical engineer. Inspect rotary bearing and lubrication. |

### Machine 3 — 🚨 DISPATCH

| Property | Value |
|---|---|
| Risk Score | **0** / 100 |
| Risk Tier | **LOW** |
| Calibrated Probability | 94.4% |
| Predicted Component | `comp2` (31% confidence) |
| Decision | **DISPATCH** |
| Reason | Peak risk 94.4% ≥ optimal threshold 68.0%. |
| Action | Dispatch mechanical engineer. Inspect rotary bearing and lubrication. |

### Machine 16 — 🚨 DISPATCH

| Property | Value |
|---|---|
| Risk Score | **0** / 100 |
| Risk Tier | **LOW** |
| Calibrated Probability | 94.4% |
| Predicted Component | `comp4` (95% confidence) |
| Decision | **DISPATCH** |
| Reason | Peak risk 94.4% ≥ optimal threshold 68.0%. |
| Action | Dispatch vibration analyst. Inspect dampeners and drive belt. |

---

## 6. Business Interpretation

- With C_FN / C_FP = 20×, the model correctly learns to **prefer lower thresholds** — dispatching more technicians to avoid catastrophic missed failures.
- The optimal threshold of **0.68** is higher than the naive 0.5 default, reflecting the asymmetric cost structure.
- **Recall is the primary operational metric**: missing a real failure carries 20× the cost of a false alarm.

---

## 7. Limitations

- Cost matrix values ($C_{FN}$, $C_{FP}$) are estimates. Sensitivity analysis on these parameters is recommended before production deployment.
- The threshold is optimised on 80 Development machines; performance may vary on machine types not represented in training.
- Expected cost assumes independence between failure events, which may not hold for correlated failure modes.

---

## 8. Figures

| Figure | Description |
|---|---|
| `cost_vs_threshold.png` | Expected cost curve with optimal marked |
| `threshold_comparison.png` | Baseline vs optimal threshold metrics |
| `cost_breakdown.png` | FN vs FP cost stacked area chart |
| `dispatch_decisions.png` | DISPATCH / MONITOR per risk tier |
| `decision_confusion_matrix.png` | Final test confusion matrix at optimal threshold |
