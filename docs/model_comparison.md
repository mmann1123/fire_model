# Fire Model: Experiment Comparison

## Run History

| Run | Date | Model | Sampling | Panel Size | AUC-A | AUC-B | Delta | Notes |
|-----|------|-------|----------|-----------|-------|-------|-------|-------|
| v1-baseline | 2026-04-01 | LogReg | grid_thin (5px) | 9.6M (1:64) | 0.9226 | 0.9207 | +0.0019 | Initial baseline. Emulator operationally viable. |
| v2-area-threshold | 2026-04-01 | LogReg | grid_thin (5px) | 9.6M (1:64) | 0.9226 | 0.9207 | +0.0019 | Post-hoc threshold only. Burned area ratio improves from 1.57x/1.94x to 0.71x/0.81x. |
| v3-matched-ratio | 2026-04-01 | LogReg | matched_ratio (10:1) | 1.6M (1:10) | 0.9234 | 0.9218 | +0.0017 | 6x smaller panel, AUC slightly improved. Fastest iteration. |
| v4-temporal-thin | 2026-04-01 | LogReg | temporal_thin (5px, 3mo) | 3.3M (1:21) | 0.9253 | 0.9158 | +0.0095 | Highest AUC-A but largest A-B gap. |
| v5-random-subsample | 2026-04-01 | LogReg | random_subsample (1%) | 2.5M (1:15) | 0.9221 | 0.9204 | +0.0017 | Spatially unbiased negatives, tight A-B gap. |
| threshold-model | 2026-04-01 | Ridge | — (post-hoc on v3) | 36 WYs | — | — | — | Year-specific thresholds from climate indices. LOO R²=0.329. Informative negative: threshold nearly constant (~0.357±0.02), cannot capture 12x burned area range. |

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

### v3-matched-ratio (Sampling experiment: 10:1 matched ratio)
- **Strategy:** Random subsample of negatives to achieve exactly 10 negatives per positive (1,473,530 neg / 147,353 pos). Panel is 6x smaller than v1/v2 baseline.
- **Overall:** AUC-A 0.9234, AUC-B 0.9218, Delta +0.0017 — tightest A-B gap of all runs
- **Fire season:** Track A 0.8599, Track B 0.8593 (Delta +0.0006)
- **Per-WY AUC:**

| WY | AUC-A | AUC-B | Delta |
|:--:|:-----:|:-----:|:-----:|
| 2020 | 0.9288 | 0.9221 | +0.0067 |
| 2021 | 0.8985 | 0.9226 | -0.0241 |
| 2022 | 0.8383 | 0.8546 | -0.0163 |
| 2023 | 0.8952 | 0.8890 | +0.0062 |
| 2024 | 0.9229 | 0.9290 | -0.0061 |

- **Area-calibrated threshold:** 0.470 (much higher than v2's 0.130 due to higher positive prevalence in the panel)
- **Per-WY burned area (area-calibrated, threshold 0.470):**

| WY | Actual (km²) | BCMv8 (km²) | Emulator (km²) | BCMv8 ratio | Emulator ratio |
|:--:|:------:|:---------:|:---------:|:-----:|:-----:|
| 2020 | 21,000 | 4,822 | 4,994 | 0.23 | 0.24 |
| 2021 | 12,301 | 14,574 | 11,363 | 1.18 | 0.92 |
| 2022 | 1,807 | 7,920 | 7,256 | 4.38 | 4.01 |
| 2023 | 2,045 | 1,192 | 1,656 | 0.58 | 0.81 |
| 2024 | 5,274 | 3,496 | 3,830 | 0.66 | 0.73 |
| **Total** | **42,427** | **32,004** | **29,099** | **0.75** | **0.69** |

- **Conclusion:** Matched ratio sampling produces the most compact panel with slightly improved AUC and the tightest emulator gap. Best candidate for rapid experimentation. Area predictions comparable to v2.
- **Operational threshold:** Use the area-calibrated fixed threshold of **0.470** for any burned area predictions from the v3-matched-ratio model. **This threshold is specific to v3-matched-ratio** — it is a property of this model's probability scale (driven by the 10:1 neg:pos sampling ratio), not a transferable constant. If the model is retrained with different sampling, features, or calibration period, the threshold must be recomputed against calibration-period spatial maps (WY2017–2019).

### v4-temporal-thin (Sampling experiment: spatial + temporal thinning)
- **Strategy:** Every 5th pixel spatially, every 3rd month temporally for negatives (3,136,450 neg / 147,353 pos, ratio 1:21).
- **Overall:** AUC-A 0.9253 (highest of all runs), AUC-B 0.9158, Delta +0.0095
- **Fire season:** Track A 0.8626, Track B 0.8441 (Delta +0.0184 — largest fire-season gap)
- **Per-WY AUC:**

| WY | AUC-A | AUC-B | Delta |
|:--:|:-----:|:-----:|:-----:|
| 2020 | 0.9491 | 0.9164 | +0.0327 |
| 2021 | 0.8976 | 0.9247 | -0.0272 |
| 2022 | 0.8491 | 0.8680 | -0.0188 |
| 2023 | 0.8652 | 0.8725 | -0.0073 |
| 2024 | 0.9147 | 0.9121 | +0.0026 |

- **Area-calibrated threshold:** 0.295
- **Per-WY burned area (area-calibrated, threshold 0.295):**

| WY | Actual (km²) | BCMv8 (km²) | Emulator (km²) | BCMv8 ratio | Emulator ratio |
|:--:|:------:|:---------:|:---------:|:-----:|:-----:|
| 2020 | 21,000 | 6,192 | 8,506 | 0.29 | 0.41 |
| 2021 | 12,301 | 16,141 | 14,819 | 1.31 | 1.20 |
| 2022 | 1,807 | 5,634 | 5,957 | 3.12 | 3.30 |
| 2023 | 2,045 | 1,349 | 2,207 | 0.66 | 1.08 |
| 2024 | 5,274 | 3,340 | 5,660 | 0.63 | 1.07 |
| **Total** | **42,427** | **32,656** | **37,149** | **0.77** | **0.88** |

- **Concern:** Temporal thinning removes autocorrelation from negatives, which may inflate BCMv8 AUC while the emulator (which relies on temporal patterns) loses more signal. The +0.0327 WY2020 gap is the largest single-year A-B divergence across all runs.
- **Conclusion:** Highest Track A AUC but at the cost of the largest emulator gap. Not recommended if emulator fidelity is the priority.

### v5-random-subsample (Sampling experiment: 1% random negatives)
- **Strategy:** Each valid negative pixel-month kept with 1% probability (2,353,790 neg / 147,353 pos, ratio ~1:15). Spatially unbiased — no grid artifacts.
- **Overall:** AUC-A 0.9221, AUC-B 0.9204, Delta +0.0017
- **Fire season:** Track A 0.8576, Track B 0.8570 (Delta +0.0006)
- **Per-WY AUC:**

| WY | AUC-A | AUC-B | Delta |
|:--:|:-----:|:-----:|:-----:|
| 2020 | 0.9266 | 0.9203 | +0.0063 |
| 2021 | 0.8991 | 0.9227 | -0.0236 |
| 2022 | 0.8354 | 0.8526 | -0.0172 |
| 2023 | 0.8935 | 0.8873 | +0.0062 |
| 2024 | 0.9213 | 0.9268 | -0.0055 |

- **Area-calibrated threshold:** 0.365
- **Per-WY burned area (area-calibrated, threshold 0.365):**

| WY | Actual (km²) | BCMv8 (km²) | Emulator (km²) | BCMv8 ratio | Emulator ratio |
|:--:|:------:|:---------:|:---------:|:-----:|:-----:|
| 2020 | 21,000 | 4,788 | 5,460 | 0.23 | 0.26 |
| 2021 | 12,301 | 15,396 | 12,599 | 1.25 | 1.02 |
| 2022 | 1,807 | 8,191 | 7,573 | 4.53 | 4.19 |
| 2023 | 2,045 | 1,303 | 1,891 | 0.64 | 0.92 |
| 2024 | 5,274 | 3,092 | 4,543 | 0.59 | 0.86 |
| **Total** | **42,427** | **32,770** | **32,066** | **0.77** | **0.76** |

- **Conclusion:** Very similar to v3 in AUC performance. Spatially unbiased sampling avoids grid artifacts. Emulator burned area totals nearly match BCMv8 (0.76x vs 0.77x). Good default if spatial bias is a concern.

### Year-Specific Threshold Model (Experiment: climate-driven threshold prediction)

**Goal:** Predict the optimal burned-area threshold for each water year from antecedent climate indices, addressing the 12x range in annual burned area (1,703–20,670 km²) that no fixed threshold can capture.

**Method:** (1) Generate full-grid spatial probability maps (GeoTIFFs) for all 36 training years (WY1984–2019) using `save_spatial_maps()` with Track A. (2) For each year, search thresholds 0.595→0.01 (high-to-low, preferring conservative thresholds) to find the threshold that minimizes the ratio error between predicted and actual burned pixels. (3) Fit Ridge regression models predicting optimal threshold from climate indices. (4) Evaluate on WY2020–2024 test period.

**Bug fix (v2 of script):** The original `compute_optimal_thresholds()` used the spatially-thinned panel, where sampled pixel counts ≠ real burned area. Replaced with full-grid spatial probability maps for all training years.

**Training threshold distribution (36 years, WY1984–2019):**
- Range: 0.310–0.385 (extremely narrow, span = 0.075)
- Mean: 0.357, Std: 0.020
- Correlation(actual_area, optimal_threshold) = **-0.099** (weakly negative, nearly flat)

The narrow range is the central finding: the fire probability model already encodes year-to-year climate variability in the probabilities themselves, so the optimal decision threshold barely changes across years.

**Ridge regression models (LOO cross-validation):**

| Model | Features | LOO R² | LOO MAE |
|-------|----------|:------:|:-------:|
| causal | cwd_anom, sws_may, pck_april_anom, tmax_jun_jul_anom | 0.329 | 0.0122 |
| cwd_only | cwd_anom_wateryear | 0.330 | 0.0129 |
| insseason | causal + kbdi_july | 0.288 | 0.0125 |

`cwd_only` matches the full causal model (R² 0.330 vs 0.329), meaning additional features provide no incremental predictive power. Adding KBDI (in-season) actually degrades fit due to overfitting with the small sample (n=36).

**Coefficient signs (causal model):**

| Feature | Std Coef | Expected | Actual | Match |
|---------|:--------:|----------|--------|:-----:|
| cwd_anom_wateryear | +0.018 | negative | positive | REVERSED |
| sws_may | +0.011 | positive | positive | OK |
| pck_april_anom | -0.003 | positive | negative | REVERSED |
| tmax_jun_jul_anom | +0.002 | negative | positive | REVERSED |

Three of four coefficient signs are reversed from physical expectations, though all magnitudes are very small. The positive CWD coefficient is particularly counterintuitive: it suggests drier antecedent conditions → higher (more conservative) threshold → less predicted burned area. This may reflect that the probability model has already absorbed the drought signal, and in drought years the entire probability surface shifts up uniformly, so a slightly higher threshold still captures the burned area.

**Test period burned area (WY2020–2024):**

| WY | Actual (km²) | Year-specific (km²) | Ratio | Threshold |
|:--:|:------:|:--------:|:-----:|:---------:|
| 2020 | 20,670 | 7,335 | 0.35 | 0.356 |
| 2021 | 12,023 | 16,017 | 1.33 | 0.363 |
| 2022 | 1,703 | 11,392 | 6.69 | 0.353 |
| 2023 | 2,071 | 6,209 | 3.00 | 0.323 |
| 2024 | 5,250 | 8,227 | 1.57 | 0.344 |
| **Total** | **41,717** | **49,180** | **1.18** | — |

The year-specific thresholds cluster in a narrow band (0.323–0.363), producing predicted areas of 6,200–16,000 km² regardless of actual area (1,703–20,670 km²). The model predicts ~10,000 km²/year ± 5,000 for every year — it cannot distinguish record fire years from quiet ones.

**Comparison against v3 area-calibrated threshold (0.470):** The fixed thresholds tested in the script (0.118, 0.130) are from v1/v2 and are inapplicable to v3's probability maps. The comparison showing 15–162x overprediction with fixed 0.118 reflects a model mismatch, not a meaningful baseline. The correct fixed comparison is v3's area-calibrated threshold of 0.470, which produces total ratio 0.69x (see v3-matched-ratio results above). The year-specific approach achieves 1.18x total ratio — closer to 1.0, but with extreme per-year variance (0.35x–6.69x) that is worse than the fixed 0.470's per-year range.

**Conclusion — informative negative result:**
1. The optimal threshold is nearly constant (~0.357 ± 0.02) across 36 years with a 12x range in burned area. This means the fire probability model already captures climate-driven fire risk variation in the probabilities, and the threshold is not where the year-to-year burned area signal lives.
2. Climate indices explain ~33% of threshold variance (R² = 0.33), but the variance itself is tiny (std = 0.02) — the model is fitting noise in a narrow band.
3. Reversed coefficient signs and identical R² between the 1-feature and 4-feature models confirm the signal is marginal.
4. **The fundamental limitation is not the threshold but the probability-to-area translation.** A pixel-level ignition probability model cannot capture the stochastic fire spread process that determines whether 1,700 or 20,000 km² burns in a given year. The 12x range in annual area is driven by fire weather episodes, ignition timing, and suppression capacity — factors not in the probability model's feature set.
5. **Recommendation:** Use the v3 area-calibrated fixed threshold (0.470) for operational burned area estimates. The year-specific approach does not improve reliability. Note: this threshold is specific to v3-matched-ratio — if the model is retrained with different sampling or features, recompute the threshold against calibration-period spatial maps (WY2017–2019). Further threshold tuning is unlikely to improve burned area predictions.

## Sampling Strategy Comparison

| Strategy | Panel Size | Neg:Pos | AUC-A | AUC-B | Delta | Brier | Area ratio (emulator) | Train time |
|----------|-----------|---------|-------|-------|-------|-------|-----------------------|------------|
| grid_thin (v1/v2) | 9.6M | 1:64 | 0.9226 | 0.9207 | +0.0019 | 0.031 | 0.81x | ~6 min |
| matched_ratio (v3) | 1.6M | 1:10 | 0.9234 | 0.9218 | +0.0017 | 0.096 | 0.69x | ~2 min |
| temporal_thin (v4) | 3.3M | 1:21 | 0.9253 | 0.9158 | +0.0095 | 0.066 | 0.88x | ~3 min |
| random_subsample (v5) | 2.5M | 1:15 | 0.9221 | 0.9204 | +0.0017 | 0.079 | 0.76x | ~4 min |

**Takeaways:**
- **AUC is robust to sampling strategy.** All runs yield AUC > 0.92 and Delta < 0.01. The logistic regression model converges to similar decision boundaries regardless of how negatives are sampled.
- **Emulator fidelity (Delta AUC) varies more.** `matched_ratio` and `random_subsample` both achieve +0.0017 — the tightest gap. `temporal_thin` at +0.0095 is notably worse for the emulator, likely because removing temporal autocorrelation from negatives penalizes the emulator's temporal features more than BCMv8's.
- **Brier score depends on prevalence.** Higher positive prevalence (matched_ratio at 10%) produces higher Brier scores — this is an artifact of calibration against different base rates, not worse probability estimates.
- **Burned area predictions are consistent across strategies.** All area-calibrated runs produce total emulator ratios in the 0.69x–0.88x range, confirming that area prediction is threshold-dominated, not sampling-dominated.
- **Recommendation:** Use `matched_ratio` (v3) for fast iteration (6x smaller panel, same AUC) and `grid_thin` (v1) for final production runs where Brier calibration against the true base rate matters.
