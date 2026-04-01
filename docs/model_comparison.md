# Fire Model: Experiment Comparison

## Run History

| Run | Date | Model | Description | AUC-A | AUC-B | Delta | Notes |
|-----|------|-------|-------------|-------|-------|-------|-------|
| v1-baseline | 2026-04-01 | LogReg | All features, 300-acre min, balanced weights | 0.9226 | 0.9207 | +0.0019 | Initial baseline. Emulator operationally viable. |
| v2-area-threshold | 2026-04-01 | LogReg | Same model, area-calibrated threshold (0.130) | 0.9226 | 0.9207 | +0.0019 | Post-hoc threshold only. Burned area ratio improves from 1.57x/1.94x to 0.71x/0.81x. |

## Key Findings

### v1-baseline
- **Overall:** Delta AUC +0.0019 — emulator degrades fire skill by <0.2%
- **Fire season (Jun-Nov):** Track A 0.8583, Track B 0.8573 (Delta +0.0011)
- **WY2021 (Dixie Fire):** Emulator AUC 0.9228 vs BCMv8 0.8989 — emulator captures megadrought signal better (-0.0238 delta in emulator's favor)
- **WY2022:** Emulator outperforms (0.8551 vs 0.8380, -0.0171 delta)
- **WY2024:** Emulator outperforms (0.9267 vs 0.9214, -0.0053 delta)
- **Top features by permutation importance:** tmax (+0.393), vpd (+0.057), dist_airbase (+0.043), sws (+0.043), tsf_years (+0.032), tmin (+0.032)
- **Top features by |coefficient|:** tmax (+4.10), vpd (-1.70), tmin (-1.31), sws (-1.21), dist_airbase (-1.16), kbdi (-0.76)
- **Conclusion:** Emulator is operationally viable for fire prediction

### v2-area-threshold (Experiment 2: Area-calibrated threshold)
- **Same model as v1** — no retraining, only threshold change
- **Method:** Generated spatial maps for WY2018-2019 (calib period), searched thresholds 0.01-0.50 to minimize MSE between predicted and actual burned area
- **Area-calibrated threshold:** 0.130 (vs rate-matching 0.118)
- **Calib fit:** WY2018 ratio=0.96 (near perfect), WY2019 ratio=0.55 (underpredicts — low-fire year)
- **Test period burned area:**

| Threshold | Method | BCMv8 ratio | Emulator ratio |
|:---------:|--------|:-----------:|:--------------:|
| 0.118 | Rate-matching | 1.57x | 1.94x |
| 0.130 | Area-calibrated | 0.71x | 0.81x |

- **Per-WY burned area — rate-matching threshold (0.118):**

| WY | Actual (km²) | BCMv8 (km²) | Emulator (km²) | BCMv8 ratio | Emulator ratio |
|:--:|:------:|:---------:|:---------:|:-----:|:-----:|
| 2020 | 21,000 | 15,017 | 15,001 | 0.72 | 0.71 |
| 2021 | 12,301 | 26,453 | 30,898 | 2.15 | 2.51 |
| 2022 | 1,807 | 13,810 | 15,762 | 7.64 | 8.72 |
| 2023 | 2,045 | 2,552 | 7,429 | 1.25 | 3.63 |
| 2024 | 5,274 | 8,958 | 13,043 | 1.70 | 2.47 |
| **Total** | **42,427** | **66,790** | **82,133** | **1.57** | **1.94** |

- **Per-WY burned area — area-calibrated threshold (0.130):**

| WY | Actual (km²) | BCMv8 (km²) | Emulator (km²) | BCMv8 ratio | Emulator ratio |
|:--:|:------:|:---------:|:---------:|:-----:|:-----:|
| 2020 | 21,000 | 4,691 | 5,593 | 0.22 | 0.27 |
| 2021 | 12,301 | 14,175 | 13,744 | 1.15 | 1.12 |
| 2022 | 1,807 | 7,240 | 7,326 | 4.01 | 4.05 |
| 2023 | 2,045 | 1,354 | 2,306 | 0.66 | 1.13 |
| 2024 | 5,274 | 2,598 | 5,270 | 0.49 | 1.00 |
| **Total** | **42,427** | **30,058** | **34,239** | **0.71** | **0.81** |

- **Interpretation:** The overprediction was primarily a threshold calibration issue caused by the rate-matching approach operating on the subsampled panel (inflated positive rate). The area-calibrated threshold reduces total overprediction substantially (BCMv8: 1.57x to 0.71x, Emulator: 1.94x to 0.81x). However, per-year variance remains high — the model cannot capture the 12x range in annual burned area (1,807 to 21,000 km²) with a static threshold. WY2020 (record fire year) is severely underpredicted at both thresholds, while WY2022 (low fire year) is overpredicted at both. This reflects the fundamental limitation of converting pixel-level ignition probabilities to area predictions via a fixed threshold.
- **Notable:** The emulator (Track B) achieves near-perfect area prediction for WY2024 (ratio 1.00) and WY2021 (ratio 1.12) with the area-calibrated threshold, outperforming BCMv8 in area accuracy for those years.
- **Conclusion:** Burned area overprediction is largely a threshold calibration problem, not an information gap. The existing model's probability calibration is good (Brier score 0.031), but translating probabilities to binary area predictions requires area-aware threshold selection.
