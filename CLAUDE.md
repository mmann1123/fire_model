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

# 2. Train logistic regression for both tracks (~30 sec)
conda run -n deep_field python scripts/02_train_model.py

# 3. Evaluate — metrics, spatial maps, comparison figures (~5 min)
conda run -n deep_field python scripts/03_evaluate.py

# 4. Snapshot — freeze outputs for experiment tracking
conda run -n deep_field python scripts/create_snapshot.py --run-id v1-baseline \
    --notes "Logistic regression baseline, all features, 300-acre min"
```

All scripts read paths from `config.yaml`. Use `--force` to overwrite existing outputs.

## Architecture

**Baseline logistic regression** predicting monthly wildfire ignition at 1km California pixels (EPSG:3310, 1209x941).

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

## Configuration

All settings live in `config.yaml`. Key sections: `paths` (data locations), `infrastructure` (distance rasters), `grid` (EPSG:3310 reference), `temporal` (train/test split dates), `sampling` (panel construction), `model` (logistic regression hyperparameters), `features` (feature lists).

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
- Overall AUC: Track A 0.9139, Track B 0.9131, Delta +0.0008
- Fire Season AUC: Track B outperforms (+0.0022)
- Conclusion: Emulator is operationally viable for fire prediction

## Development Practices

- Always tag runs with `--run-id` and `--notes` for reproducibility.
- Feature order must match exactly between `02_train_model.py`, `03_evaluate.py`, and `forecast.py`.
- When adding new features: update `config.yaml` feature lists, all three scripts, and `forecast.py`.
- The panel stores raw feature values. Scaling is handled by `StandardScaler` inside the sklearn pipeline.

## Dependencies on BCM Emulator

This project depends on the BCM emulator (`/home/mmann1123/extra_space/bcm_emulator/`) for:
1. **Zarr store** (`data/bcm_dataset.zarr`) — climate inputs, static features, BCMv8 targets
2. **Emulator predictions** — exported via `00_export_predictions.py` (lives in bcm_emulator)
3. **FVEG class map** — `data/fveg/fveg_class_map.json`

No Python imports from bcm_emulator are needed — all data is consumed as files.
