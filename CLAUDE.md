# CLAUDE.md — Fire Probability Model

This file provides guidance to Claude Code when working with code in this repository.

## Environment

Always use the `deep_field` conda environment:

```bash
conda run -n deep_field python <script.py>
```

## PREREQUISITE: Export Emulator Predictions

**Before running any fire model scripts**, you must export emulator predictions from the BCM emulator project. This step runs the emulator in inference mode and produces the `.npy` files that Track B consumes.

```bash
cd /home/mmann1123/extra_space/bcm_emulator
conda run -n deep_field python scripts/fire_model/00_export_predictions.py \
    --output-dir /home/mmann1123/extra_space/fire_model/data/predictions
```

This produces `data/predictions/{pet,pck,aet,cwd}.npy` + `time_index.npy`. Without these files, `01_build_panel.py` will fail. Re-run whenever the emulator is retrained or the zarr is rebuilt.

## Run Sequence

```bash
cd /home/mmann1123/extra_space/fire_model

# 1. Build panel — rasterize FRAP fires, compute TSF, extract features (~5 min)
conda run -n deep_field python scripts/01_build_panel.py

# 2. Train model for both tracks
#    Default (logistic regression, ~30 sec):
conda run -n deep_field python scripts/02_train_model.py
#    With model type override:
conda run -n deep_field python scripts/02_train_model.py --model-type lightgbm
#    With Optuna tuning:
conda run -n deep_field python scripts/02_train_model.py --tune --n-trials 100
#    Full example:
conda run -n deep_field python scripts/02_train_model.py --model-type random_forest --tune --n-trials 100 --cv-folds 3 --tune-subsample 0.2

# 3. Evaluate — metrics, spatial maps, comparison figures (~5 min)
conda run -n deep_field python scripts/03_evaluate.py --run-id v6-random-forest --notes "RF with Optuna tuning"

# 4. Snapshot — freeze outputs for experiment tracking
conda run -n deep_field python scripts/create_snapshot.py --run-id v6-random-forest \
    --notes "RandomForest with Optuna tuning, matched_ratio sampling"
```

All scripts read paths from `config.yaml`. Use `--force` to overwrite existing outputs.

## Architecture

**Modular fire probability model** predicting monthly wildfire ignition at 1km California pixels (EPSG:3310, 1209x941). Supports 6 model types via `src/fire_model/models.py`.

### Data Flow

```
bcm_emulator/scripts/fire_model/00_export_predictions.py (in bcm_emulator project)
    → data/predictions/{pet,pck,aet,cwd,time_index}.npy

scripts/01_build_panel.py
    → outputs/fire_raster.npy, outputs/panel/fire_panel.parquet
    → outputs/climatology_1984_2016.npz, outputs/tsf_state_*.npy

scripts/02_train_model.py
    → outputs/model/track{A,B}/lr_calibrated.pkl

scripts/03_evaluate.py
    → outputs/evaluation/, outputs/spatial_maps/, outputs/comparison/

scripts/create_snapshot.py
    → snapshots/{run-id}/ (frozen config + model + metrics)
```

### Two-Track Comparison

- **Track A (BCMv8):** Uses BCMv8 target values as predictors — theoretical upper bound.
- **Track B (Emulator):** Uses emulator predictions as predictors — operational configuration.

Both tracks share all features except hydrology anomalies (CWD, AET, PET), which differ only during the test period.

### Features (34 total)

**Common (29):** ppt, tmin, tmax, vpd, srad, kbdi, sws, vpd_roll6_std, month_sin, month_cos, fire_season, elev, aridity_index, windward_index, fveg_forest, fveg_shrub, fveg_herb, tsf_years, tsf_log, tst_broadcast_years, tst_mechanical_years, any_treatment_5yr, dist_campground_km, dist_transmission_km, dist_airbase_km, dist_firestation_km, dist_road_km, housing_density, log_housing_density

**Track-specific (5):** cwd_anom, aet_anom, pet_anom, cwd_cum3_anom, cwd_cum6_anom

### Forward Forecasting (src/fire_model/forecast.py)

`FireProbabilityForecaster` generates monthly fire probability maps from BCM emulator outputs. Maintains TSF state (deterministic — no fire resets in forecast mode). Supports SERGOM housing density projections through 2099.

## Model Types

All models are wrapped in sklearn Pipelines so `model.predict_proba(X)[:, 1]` works identically in evaluation and forecasting. `FeatureTransformer` (for interaction features) is serialized inside the pipeline pickle.

| Type | Description | Pipeline |
|------|-------------|----------|
| `logistic_regression` | Baseline LogReg | FeatureTransformer → StandardScaler → LogisticRegression |
| `elasticnet_logreg` | ElasticNet with interaction features | FeatureTransformer → StandardScaler → SGDClassifier(elasticnet) |
| `random_forest` | Random Forest | FeatureTransformer → RandomForestClassifier |
| `lightgbm` | LightGBM (GPU) | FeatureTransformer → LGBMClassifier |
| `tabnet` | TabNet (GPU) | FeatureTransformer → StandardScaler → TabNetWrapper |
| `ecoregion_logreg` | Per-ecoregion LogReg | EcoregionClassifier (fits separate LogReg per L3 ecoregion) |

Set model type in `config.yaml` under `model.type`, or override via `--model-type` CLI arg.

### Optuna Tuning (src/fire_model/tuning.py)

`--tune` enables Optuna hyperparameter search. Uses 3-fold stratified CV on the training split (1984-2016). Tree/neural models use a 20% stratified subsample for speed. The calibration split (2017-2019) is reserved for isotonic calibration only. Best params saved to `best_params.json`.

### EcoregionClassifier

Fits separate LogReg per L3 ecoregion. Ecoregions with <`min_pos_per_eco` positives fall back to the global model. Requires `pixel_indices` kwarg during predict_proba for spatial maps — handled automatically by `03_evaluate.py`.

## Configuration

All settings live in `config.yaml`. Key sections: `paths` (data locations), `infrastructure` (distance rasters), `grid` (EPSG:3310 reference), `temporal` (train/test split dates), `sampling` (panel construction), `model` (model type + hyperparameters), `features` (feature lists).

## Snapshot System

Each `--run-id` creates `snapshots/{id}/` containing:
- `manifest.json` — git hash, run date, AUC scores, feature list, notes
- `config.yaml` — frozen copy of config at run time
- `model/trackA/lr_calibrated.pkl`, `model/trackB/lr_calibrated.pkl` — trained models
- `model/trackA/coefficients.csv`, `model/trackB/coefficients.csv`
- `evaluation/` — all metrics CSVs, comparison_summary, plots
- `spatial_maps/` — GeoTIFFs

Use `docs/model_comparison.md` to track experiments across runs.

## Current Status

Best run: **v1-baseline** — Logistic regression, all features, 300-acre min.
- Overall AUC: Track A 0.9226, Track B 0.9207, Delta +0.0019
- Fire Season AUC: Track A 0.8583, Track B 0.8573, Delta +0.0011
- Emulator outperforms BCMv8 in WY2021 (-0.024), WY2022 (-0.017), WY2024 (-0.005)
- Top features: tmax (+0.393 perm imp), vpd (+0.057), dist_airbase (+0.043), sws (+0.043)
- Conclusion: Emulator is operationally viable for fire prediction

## Development Practices

- Always tag runs with `--run-id` and `--notes` for reproducibility.
- Feature order must match exactly between `02_train_model.py`, `03_evaluate.py`, and `forecast.py`.
- When adding new features: update `config.yaml` feature lists, all three scripts, and `forecast.py`.
- The panel stores raw feature values. Scaling is handled by `StandardScaler` inside the sklearn pipeline.
- When adding new model types: add to `build_model()` in `models.py`, add search space in `tuning.py`, add config block in `config.yaml`.
- **Primary success metric:** Track A-B Delta AUC, not raw AUC. Delta < 0.02 = emulator viable; 0.02-0.03 = marginal; > 0.03 = not viable for that model type.

## Dependencies on BCM Emulator

This project depends on the BCM emulator (`/home/mmann1123/extra_space/bcm_emulator/`) for:
1. **Zarr store** (`data/bcm_dataset.zarr`) — climate inputs, static features, BCMv8 targets
2. **Emulator predictions** — exported via `00_export_predictions.py` (lives in bcm_emulator)
3. **FVEG class map** — `data/fveg/fveg_class_map.json`

No Python imports from bcm_emulator are needed — all data is consumed as files.
