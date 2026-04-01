# Fire Probability Predictions: BCM Emulator Validation

## Summary

A baseline logistic regression model predicting monthly wildfire ignition probability at 1 km resolution across California demonstrates that replacing BCMv8 ground-truth hydrology with BCM emulator predictions has negligible impact on downstream fire prediction skill. The overall ROC-AUC degradation is **+0.0008** (0.9139 vs 0.9131), well below the 0.02 operational viability threshold. During the fire season (Jun-Nov), the emulator slightly *outperforms* BCMv8 (AUC 0.8454 vs 0.8432). These results validate the BCM emulator for operational wildfire risk forecasting.

## Experimental Design

### Tracks

- **Track A (BCMv8):** Uses BCMv8 modeled outputs (CWD, AET, PET) as predictors. Represents the theoretical performance ceiling — what fire prediction could achieve with perfect climate water balance inputs.
- **Track B (Emulator):** Uses v19a BCM emulator predictions as predictors for the test period (Oct 2019 – Sep 2024). Represents operational end-to-end performance when the emulator replaces BCMv8.

Both tracks use identical training data (BCMv8 targets for 1984–2016). They differ only in the source of hydrologic features during the test period.

### Fire Data

**Source:** FRAP fire perimeter database (`fire24_1.gdb`, layer `firep24_1`), 22,810 perimeters.

**Filters applied:**
- California only (`STATE = CA`)
- Suppression fires (`OBJECTIVE = 1`)
- Minimum 300 acres (1.2 km² — sub-pixel at 1 km resolution excluded)
- Valid alarm date, 1984–2024

**After filtering:** 3,004 fires retained.

### Panel Construction

- **Positive samples:** 147,353 pixel-months where a fire perimeter touches the pixel in the alarm month (rasterized with `all_touched=True`)
- **Negative samples:** 9,410,338 pixel-months from a spatially thinned grid (every 5th pixel, ~19,000 grid points) across all non-fire months
- **Class imbalance:** Handled by `class_weight="balanced"` in the logistic regression, not by downsampling

### Temporal Splits

| Split | Period | Purpose |
|-------|--------|---------|
| Train | Jan 1984 – Dec 2016 | Model fitting (87,018 fires, 7.6M negatives) |
| Calibration | Jan 2017 – Sep 2019 | Isotonic probability calibration (17,386 fires) |
| Test | Oct 2019 – Sep 2024 | Out-of-sample evaluation (42,949 fires) |

The calibration set is held out from training to prevent overfit probability estimates. The test period matches the BCM emulator's out-of-sample evaluation window.

## Features

### Hydroclimatic Features (Track-Specific)

These features come from BCMv8 targets (Track A) or emulator predictions (Track B) during the test period. During training, both tracks use BCMv8 targets.

| Feature | Description |
|---------|-------------|
| CWD anomaly | CWD minus pixel-month climatological mean (1984–2016) |
| AET anomaly | AET minus pixel-month climatological mean |
| PET anomaly | PET minus pixel-month climatological mean |
| CWD 3-month cumulative anomaly | Trailing 3-month sum of CWD anomalies |
| CWD 6-month cumulative anomaly | Trailing 6-month sum of CWD anomalies |

### Climate Inputs (Shared)

| Feature | Description |
|---------|-------------|
| PPT | Monthly precipitation (mm) |
| Tmin, Tmax | Monthly min/max temperature (°C) |
| VPD | Vapor pressure deficit (kPa) |
| SRAD | Solar radiation (W/m²) |
| KBDI | Keetch-Byram Drought Index |
| SWS | Soil water storage (mm) |
| VPD rolling 6-month std | Climate variability signal |

### Time Since Fire

| Feature | Description |
|---------|-------------|
| TSF years | Years since last fire at pixel, capped at 50 |
| TSF log | log(1 + TSF years) — captures diminishing fuel accumulation |

TSF is initialized from the FRAP-derived `timeSinceFire_1983.tif` raster (years since fire as of end of 1983) and updated monthly by forward-iterating through the fire raster. Causal ordering: TSF at month *t* is recorded before that month's fires are processed.

### Static and Seasonal

| Feature | Description |
|---------|-------------|
| Elevation | Meters above sea level |
| Aridity index | BCMv8 aridity |
| Windward index | Orographic exposure |
| FVEG (forest, shrub, herb) | Vegetation broad category dummies (from FRAP CWHR LIFEFORM) |
| Month sin/cos | Seasonal cycle encoding |
| Fire season flag | 1 if Jun–Nov |

## Results

### Overall Test Period (Oct 2019 – Sep 2024)

| Metric | Track A (BCMv8) | Track B (Emulator) | Delta |
|--------|----------------:|-------------------:|------:|
| **ROC-AUC** | **0.9139** | **0.9131** | **+0.0008** |
| Average Precision | 0.2345 | 0.2292 | +0.0053 |
| Brier Score | 0.0312 | 0.0311 | +0.0001 |
| Fire Season ROC-AUC | 0.8432 | 0.8454 | -0.0022 |

The AUC delta of +0.0008 means the emulator's hydrologic prediction errors contribute less than 0.1% degradation to fire prediction skill.

### By Water Year

| Water Year | Track A AUC | Track B AUC | Delta | Test Fires | Prevalence |
|:----------:|:-----------:|:-----------:|:-----:|:----------:|:----------:|
| WY2020 | 0.922 | 0.915 | +0.007 | 21,066 | 8.4% |
| WY2021 | 0.878 | 0.908 | -0.030 | 12,560 | 5.2% |
| WY2022 | 0.845 | 0.859 | -0.014 | 1,896 | 0.8% |
| WY2023 | 0.916 | 0.904 | +0.012 | 2,053 | 0.9% |
| WY2024 | 0.902 | 0.913 | -0.011 | 5,374 | 2.3% |

The emulator outperforms BCMv8 in three of five water years (WY2021, WY2022, WY2024). WY2021, the year with the largest delta (-0.030 in the emulator's favor), includes the Dixie Fire — the largest single fire in California history. The emulator's drought signal may better capture the cumulative moisture deficit leading up to that event.

### By Month

Peak fire months (Jul–Sep) show strong discrimination in both tracks:

| Month | Track A AUC | Fires | Prevalence |
|:-----:|:-----------:|:-----:|:----------:|
| Jul | 0.741 | 14,466 | 13.1% |
| Aug | 0.809 | 18,279 | 16.1% |
| Sep | 0.834 | 6,961 | 6.8% |
| Oct | 0.774 | 1,089 | 1.1% |

Winter months (Nov–Apr) have near-zero fire prevalence and AUC values are not meaningful.

### Feature Importance

Top 10 features by absolute logistic regression coefficient (Track A):

| Rank | Feature | Coefficient | Odds Ratio | Interpretation |
|:----:|---------|:----------:|:----------:|----------------|
| 1 | Tmax | +4.78 | 118.7 | Hotter months strongly increase fire probability |
| 2 | VPD | -1.95 | 0.14 | High VPD correlated with tmax; negative after controlling for tmax |
| 3 | Tmin | -1.63 | 0.20 | Cool nights reduce fire spread (higher humidity recovery) |
| 4 | SWS | -1.24 | 0.29 | Wetter soil = less fire |
| 5 | KBDI | -1.17 | 0.31 | Collinear with tmax/vpd; sign reflects partial correlation |
| 6 | TSF years | -0.84 | 0.43 | Combined with TSF log: non-monotonic fuel effect |
| 7 | Month sin | -0.82 | 0.44 | Seasonal pattern encoding |
| 8 | Month cos | -0.57 | 0.57 | Seasonal pattern encoding |
| 9 | TSF log | +0.49 | 1.63 | Fuel accumulation increases fire risk (log-diminishing) |
| 10 | SRAD | -0.48 | 0.62 | After controlling for temperature, radiation is collinear |

The TSF features jointly model a non-monotonic relationship: fire risk increases with fuel accumulation but at a diminishing rate. The negative linear term (TSF years) combined with the positive log term (TSF log) creates an inverted-U shape that peaks around 15-20 years.

CWD anomaly features (ranks 18-21) have modest coefficients (+0.04 to +0.14), reflecting that the drought signal is largely captured by the collinear KBDI and SWS features. The 6-month cumulative CWD anomaly (+0.14) is larger than the single-month anomaly (+0.04), consistent with multi-month drought accumulation driving fire risk.

## Interpretation

### Why the Delta is So Small

The emulator's v19a-extended evaluation showed AET pbias +11.6% and CWD NSE 0.919 on the test period — meaningful hydrologic prediction errors. Yet these errors produce negligible fire model degradation because:

1. **Feature redundancy:** CWD anomalies rank 18th-21st in feature importance. The dominant fire predictors (temperature, VPD, soil moisture, KBDI) come from climate inputs shared between tracks. The model does not rely heavily on emulated hydrology.

2. **Anomaly formulation:** Using anomalies rather than absolute values reduces sensitivity to systematic bias. If the emulator consistently overpredicts AET by 12%, the anomaly (deviation from the pixel-month mean) partially cancels this bias.

3. **Spatial coherence:** The emulator's errors are spatially smooth (NSE > 0.9 everywhere except desert margins). Fire prediction cares more about spatial patterns (which pixels are anomalously dry) than absolute magnitudes.

### Emulator Outperformance in WY2021

The emulator achieves AUC 0.908 vs BCMv8's 0.878 in WY2021 (Dixie Fire year). This suggests the emulator's drought signal — which captures the cumulative effect of the 2020-2021 hot drought through its learned representations — may provide a more informative predictor than BCMv8's process-based calculations for extreme multi-year drought events that push vegetation beyond normal operating ranges.

### Model Limitations

- **Logistic regression baseline:** A gradient-boosted model or spatial model would likely improve absolute AUC, but the Track A vs B delta would remain similar since the feature information content is the same.
- **Monthly resolution:** Sub-monthly fire weather (Santa Ana winds, heat waves) is not captured.
- **1 km pixel scale:** Fire spread dynamics are implicit in perimeter-based labels; no explicit spread modeling.
- **300-acre minimum:** Excludes small fires that collectively burn significant area.

## BCM Emulator Out-of-Sample Validation Summary

The BCM emulator (v19a) was trained on 1980–2019 and evaluated on 2019–2024:

| Variable | Training NSE | Test NSE | Degradation |
|----------|:-----------:|:--------:|:-----------:|
| CWD | 0.925 | 0.919 | -0.006 |
| PCK | 0.948 | 0.965 | +0.017 |
| PET | 0.825 | 0.863 | +0.038 |
| AET | 0.858 | 0.824 | -0.034 |

CWD — the primary fire-relevant variable — shows only 0.006 NSE degradation across a 5-year out-of-sample period that included the 2020-2024 California megadrought. PCK (snowpack) actually *improved* out-of-sample, achieving the best NSE in the project's 23-run history.

## Outputs

All outputs are in `outputs/`:

| File | Description |
|------|-------------|
| `manifest.json` | Run metadata, sample counts, AUC scores |
| `evaluation/comparison_summary.csv` | Head-to-head comparison table |
| `evaluation/roc_comparison.png` | Overlaid ROC curves |
| `evaluation/trackA/metrics_overall.json` | Track A detailed metrics |
| `evaluation/trackA/metrics_by_month.csv` | Monthly breakdown |
| `evaluation/trackA/metrics_by_water_year.csv` | Water year breakdown |
| `evaluation/trackA/metrics_by_ecoregion.csv` | Ecoregion breakdown |
| `evaluation/trackA/calibration_curve.png` | Reliability diagram |
| `evaluation/trackB/` | Same structure for Track B |
| `model/trackA/coefficients.csv` | Feature coefficients and odds ratios |
| `model/trackA/lr_calibrated.pkl` | Calibrated model (pickle) |
| `predictions/test_predictions_trackA.parquet` | Pixel-level predictions for post-hoc analysis |
| `spatial_maps/trackA/fire_prob_WY*_fire_season.tif` | GeoTIFF probability maps (EPSG:3310) |
| `tsf_state_2024-09.npy` | Terminal TSF state for forward forecasting |
| `climatology_1984_2016.npz` | Monthly pixel climatologies |

## Reproduction

```bash
# First, export emulator predictions from bcm_emulator project:
cd /home/mmann1123/extra_space/bcm_emulator
conda run -n deep_field python scripts/fire_model/00_export_predictions.py \
    --output-dir /home/mmann1123/extra_space/fire_model/data/predictions

# Then run fire model pipeline:
cd /home/mmann1123/extra_space/fire_model
conda run -n deep_field python scripts/01_build_panel.py
conda run -n deep_field python scripts/02_train_model.py
conda run -n deep_field python scripts/03_evaluate.py
```

All scripts are idempotent (skip existing outputs unless `--force` is passed). Random seed = 42 throughout.
