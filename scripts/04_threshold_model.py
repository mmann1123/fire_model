"""Year-specific threshold prediction model for fire probability maps.

Fits a secondary regression model that predicts the optimal burned-area threshold
for each water year from antecedent climate indices. This sits on top of the
existing fire probability model — no retraining of the logistic regression needed.

Usage:
    conda run -n deep_field python scripts/04_threshold_model.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
import zarr
from rasterio.transform import Affine
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Load configuration ----
with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

ZARR_PATH = cfg["paths"]["zarr_store"]
OUTPUT_DIR = Path((PROJECT_ROOT / cfg["paths"]["output_dir"]).resolve())
PANEL_PATH = OUTPUT_DIR / "panel" / "fire_panel.parquet"
THRESH_DIR = OUTPUT_DIR / "threshold_model"
MAPS_DIR = OUTPUT_DIR / "spatial_maps"

H = cfg["grid"]["height"]
W = cfg["grid"]["width"]
tx = cfg["grid"]["transform"]
TRANSFORM = Affine(tx[0], tx[1], tx[2], tx[3], tx[4], tx[5])

# Feature lists from config (must match training)
COMMON_FEATURES = cfg["features"]["common"]
TRACK_A_FEATURES = COMMON_FEATURES + [
    f"{feat}_a" for feat in cfg["features"]["track_specific"]
]
TRACK_B_FEATURES = COMMON_FEATURES + [
    f"{feat}_b" for feat in cfg["features"]["track_specific"]
]

CAUSAL_FEATURES = [
    "cwd_anom_wateryear",
    "sws_may",
    "pck_april_anom",
    "tmax_jun_jul_anom",
]

ALL_FEATURES = CAUSAL_FEATURES + ["kbdi_july", "cwd_anom_fireseason"]


# =========================================================================
# Step 1: Compute annual climate indices from zarr
# =========================================================================
def compute_annual_climate_indices(zarr_path, valid_mask, time_index):
    """Compute statewide mean climate indices per water year.

    Water year Y = Oct(Y-1) through Sep(Y).
    Antecedent period = Oct(Y-1) through May(Y).
    Fire season = Jun through Nov of year Y.
    """
    store = zarr.open_group(zarr_path, mode="r")
    dyn = store["inputs/dynamic"]  # (T, 15, H, W) — raw values
    cwd_raw = store["targets/cwd"]  # (T, H, W) — raw mm

    # Load climatology for anomalies
    clim = np.load(str(OUTPUT_DIR / "climatology_1984_2016.npz"))
    cwd_clim = clim["cwd"]  # (12, H, W)

    # PCK target — raw
    pck_raw = store["targets/pck"]  # (T, H, W)
    # Compute PCK climatology from training period (1984-01 to 2016-12)
    train_start_idx = np.searchsorted(time_index, "1984-01")
    train_end_idx = np.searchsorted(time_index, "2017-01")  # exclusive
    pck_clim = np.zeros((12, H, W), dtype=np.float32)
    for m in range(12):
        month_indices = [
            i for i in range(train_start_idx, train_end_idx)
            if int(time_index[i][5:7]) == m + 1
        ]
        if month_indices:
            pck_clim[m] = np.mean(
                [pck_raw[i] for i in month_indices], axis=0
            )

    # Dynamic channel indices (raw values, no denormalization needed — zarr stores raw)
    # Actually wait — CLAUDE.md says zarr stores raw. Let me verify by reading directly.
    # Channel indices: SWS=11, KBDI=10, Tmax=2
    CH_SWS = 11
    CH_KBDI = 10
    CH_TMAX = 2

    # Tmax climatology from training period
    tmax_clim = np.zeros((12, H, W), dtype=np.float32)
    for m in range(12):
        month_indices = [
            i for i in range(train_start_idx, train_end_idx)
            if int(time_index[i][5:7]) == m + 1
        ]
        if month_indices:
            tmax_clim[m] = np.mean(
                [dyn[i, CH_TMAX] for i in month_indices], axis=0
            )

    # Determine water year range
    first_wy = int(time_index[0][:4]) + 1  # first complete WY
    last_wy = int(time_index[-1][:4])
    if int(time_index[-1][5:7]) < 9:
        last_wy = last_wy  # partial WY still usable for antecedent
    logger.info(f"Computing climate indices for WY{first_wy}-WY{last_wy}")

    rows = []
    for wy in range(first_wy, last_wy + 1):
        # Antecedent months: Oct(wy-1) through May(wy)
        ante_yms = [f"{wy-1:04d}-{m:02d}" for m in [10, 11, 12]] + \
                   [f"{wy:04d}-{m:02d}" for m in [1, 2, 3, 4, 5]]
        ante_indices = [
            np.searchsorted(time_index, ym)
            for ym in ante_yms
            if ym in time_index
        ]
        ante_indices = [i for i in ante_indices if i < len(time_index) and time_index[i] in ante_yms]

        # Fire season months: Jun-Nov of year wy
        fire_yms = [f"{wy:04d}-{m:02d}" for m in [6, 7, 8, 9, 10, 11]]
        fire_indices = [
            np.searchsorted(time_index, ym)
            for ym in fire_yms
            if ym in time_index
        ]
        fire_indices = [i for i in fire_indices if i < len(time_index) and time_index[i] in fire_yms]

        if len(ante_indices) < 4:
            continue

        # CWD anomaly antecedent (Oct-May)
        cwd_anoms = []
        for i in ante_indices:
            m = int(time_index[i][5:7]) - 1  # 0-indexed month
            anom = np.array(cwd_raw[i]) - cwd_clim[m]
            cwd_anoms.append(np.nanmean(anom[valid_mask]))
        cwd_anom_wy = np.mean(cwd_anoms)

        # SWS in May
        may_ym = f"{wy:04d}-05"
        may_idx = np.searchsorted(time_index, may_ym)
        if may_idx < len(time_index) and time_index[may_idx] == may_ym:
            sws_may = np.nanmean(np.array(dyn[may_idx, CH_SWS])[valid_mask])
        else:
            sws_may = np.nan

        # PCK anomaly in April
        apr_ym = f"{wy:04d}-04"
        apr_idx = np.searchsorted(time_index, apr_ym)
        if apr_idx < len(time_index) and time_index[apr_idx] == apr_ym:
            pck_apr = np.array(pck_raw[apr_idx])
            pck_anom = np.nanmean((pck_apr - pck_clim[3])[valid_mask])  # April = month index 3
        else:
            pck_anom = np.nan

        # Tmax anomaly Jun-Jul
        tmax_anoms = []
        for m in [6, 7]:
            ym = f"{wy:04d}-{m:02d}"
            idx = np.searchsorted(time_index, ym)
            if idx < len(time_index) and time_index[idx] == ym:
                tmax_val = np.array(dyn[idx, CH_TMAX])
                tmax_anoms.append(
                    np.nanmean((tmax_val - tmax_clim[m - 1])[valid_mask])
                )
        tmax_jun_jul_anom = np.mean(tmax_anoms) if tmax_anoms else np.nan

        # KBDI in July (in-season predictor)
        jul_ym = f"{wy:04d}-07"
        jul_idx = np.searchsorted(time_index, jul_ym)
        if jul_idx < len(time_index) and time_index[jul_idx] == jul_ym:
            kbdi_july = np.nanmean(np.array(dyn[jul_idx, CH_KBDI])[valid_mask])
        else:
            kbdi_july = np.nan

        # CWD anomaly fire season (Jun-Nov) — non-causal, for comparison
        cwd_fire_anoms = []
        for i in fire_indices:
            m = int(time_index[i][5:7]) - 1
            anom = np.array(cwd_raw[i]) - cwd_clim[m]
            cwd_fire_anoms.append(np.nanmean(anom[valid_mask]))
        cwd_anom_fs = np.mean(cwd_fire_anoms) if cwd_fire_anoms else np.nan

        rows.append({
            "wy": wy,
            "cwd_anom_wateryear": cwd_anom_wy,
            "sws_may": sws_may,
            "pck_april_anom": pck_anom,
            "tmax_jun_jul_anom": tmax_jun_jul_anom,
            "kbdi_july": kbdi_july,
            "cwd_anom_fireseason": cwd_anom_fs,
        })

    return pd.DataFrame(rows)


# =========================================================================
# Step 1b: Generate spatial probability maps for training years
# =========================================================================
def generate_training_spatial_maps(model_path, feature_cols, maps_dir, train_years):
    """Generate full-grid spatial probability GeoTIFFs for training years.

    Imports save_spatial_maps from 03_evaluate.py and calls it with
    track_suffix="_a" so it uses BCMv8 targets (not emulator predictions).
    """
    track_dir = maps_dir / "trackA"

    # Check which years already have maps
    missing_years = []
    for wy in train_years:
        tif_path = track_dir / f"fire_prob_WY{wy}_fire_season.tif"
        if not tif_path.exists():
            missing_years.append(wy)

    if not missing_years:
        logger.info(f"All {len(train_years)} training year maps already exist, skipping generation")
        return

    logger.info(f"Generating spatial maps for {len(missing_years)} training years "
                f"(WY{min(missing_years)}-WY{max(missing_years)})...")

    # Import save_spatial_maps from 03_evaluate.py via importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluate", PROJECT_ROOT / "scripts" / "03_evaluate.py"
    )
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    eval_mod.save_spatial_maps(
        str(model_path),
        feature_cols,
        "_a",
        track_dir,
        wy_range=missing_years,
    )
    logger.info(f"Training spatial maps saved to {track_dir}")


# =========================================================================
# Step 2: Compute optimal threshold per training year
# =========================================================================
def compute_optimal_thresholds(maps_dir, fire_raster, valid_mask,
                               time_index, train_years):
    """Find optimal burned-area threshold for each training year.

    Uses full-grid spatial probability maps (GeoTIFFs) for all years.
    Searches thresholds high-to-low to prefer conservative (higher) thresholds
    when multiple thresholds achieve similar ratio error.
    """
    store = zarr.open_group(ZARR_PATH, mode="r")
    time_index_zarr = np.array(store["meta/time"])

    results = []
    for wy in train_years:
        # Actual burned area from fire_raster (full grid, fire season months)
        actual_pixels = 0
        for m in [6, 7, 8, 9, 10, 11]:
            ym = f"{wy:04d}-{m:02d}"
            idx = np.searchsorted(time_index_zarr, ym)
            if idx < len(time_index_zarr) and time_index_zarr[idx] == ym:
                actual_pixels += int(
                    ((fire_raster[idx] == 1) & valid_mask).sum()
                )

        if actual_pixels == 0:
            logger.debug(f"WY{wy}: no fires in fire season, skipping")
            continue

        tif_path = maps_dir / "trackA" / f"fire_prob_WY{wy}_fire_season.tif"
        if not tif_path.exists():
            logger.warning(f"WY{wy}: spatial map not found at {tif_path}, skipping")
            continue

        with rasterio.open(str(tif_path)) as src:
            prob_map = src.read(1)
        prob_map[prob_map == -9999.0] = 0.0

        # Search HIGH-to-LOW: prefer conservative (higher) threshold when
        # multiple thresholds achieve the same ratio error. This prevents
        # implausibly low thresholds for quiet fire years.
        best_thresh, best_err = None, float("inf")
        for thresh in np.arange(0.595, 0.005, -0.005):
            pred_pixels = int(((prob_map >= thresh) & valid_mask).sum())
            if pred_pixels == 0:
                continue
            ratio_err = abs(pred_pixels / actual_pixels - 1.0)
            if ratio_err < best_err:
                best_err = ratio_err
                best_thresh = thresh

        if best_thresh is None:
            logger.warning(f"WY{wy}: no valid threshold found, skipping")
            continue

        pred_at_opt = int(((prob_map >= best_thresh) & valid_mask).sum())

        results.append({
            "wy": wy,
            "optimal_threshold": best_thresh,
            "actual_area_km2": actual_pixels,
            "pred_area_at_optimal": pred_at_opt,
            "ratio_at_optimal": pred_at_opt / actual_pixels if actual_pixels > 0 else np.nan,
            "ratio_error": best_err,
        })

    return pd.DataFrame(results)


# =========================================================================
# Step 3: Fit threshold prediction model
# =========================================================================
def fit_threshold_models(train_data, output_dir):
    """Fit Ridge regression models predicting optimal threshold from climate indices."""
    results = {}

    model_specs = {
        "causal": CAUSAL_FEATURES,
        "insseason": CAUSAL_FEATURES + ["kbdi_july"],
        "cwd_only": ["cwd_anom_wateryear"],
    }

    for name, features in model_specs.items():
        logger.info(f"Fitting threshold model: {name} ({features})")

        X = train_data[features].values
        y = train_data["optimal_threshold"].values

        # Check for NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]
        n = len(y)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])

        # Leave-one-out CV
        loo = LeaveOneOut()
        y_pred_loo = np.zeros(n)
        for train_idx, test_idx in loo.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_pred_loo[test_idx] = model.predict(X[test_idx])

        loo_r2 = r2_score(y, y_pred_loo)
        loo_mae = mean_absolute_error(y, y_pred_loo)
        logger.info(f"  LOO R²: {loo_r2:.3f}, LOO MAE: {loo_mae:.4f}")

        # Fit final model
        model.fit(X, y)

        # Coefficients
        scaler = model.named_steps["scaler"]
        ridge = model.named_steps["ridge"]
        coefs = ridge.coef_
        std_coefs = coefs  # already standardized since scaler is in pipeline
        raw_coefs = coefs / scaler.scale_

        coef_table = []
        for i, feat in enumerate(features):
            coef_table.append({
                "feature": feat,
                "coefficient_raw": float(raw_coefs[i]),
                "coefficient_std": float(std_coefs[i]),
            })

        # Save model
        model_file = f"threshold_prediction_model.pkl" if name == "causal" else f"threshold_model_{name}.pkl"
        with open(output_dir / model_file, "wb") as f:
            pickle.dump(model, f)

        results[name] = {
            "features": features,
            "n_train": int(n),
            "loo_r2": float(loo_r2),
            "loo_mae": float(loo_mae),
            "intercept": float(ridge.intercept_),
            "coefficients": coef_table,
            "y_actual": y.tolist(),
            "y_pred_loo": y_pred_loo.tolist(),
            "wy_used": train_data.loc[mask, "wy"].tolist() if "wy" in train_data.columns else [],
        }

    return results


# =========================================================================
# Step 4 & 5: Predict thresholds and evaluate burned area
# =========================================================================
def evaluate_threshold_approaches(model_results, climate_indices, fire_raster,
                                  time_index_full, valid_mask, maps_dir,
                                  output_dir):
    """Compare fixed vs year-specific thresholds on test period."""
    test_years = [2020, 2021, 2022, 2023, 2024]

    # Load threshold models
    with open(output_dir / "threshold_prediction_model.pkl", "rb") as f:
        model_causal = pickle.load(f)
    with open(output_dir / "threshold_model_insseason.pkl", "rb") as f:
        model_insseason = pickle.load(f)

    # Predict thresholds for test years
    thresh_causal = {}
    thresh_insseason = {}
    for wy in test_years:
        row = climate_indices[climate_indices["wy"] == wy]
        if len(row) == 0:
            continue
        t_c = float(model_causal.predict(row[CAUSAL_FEATURES].values)[0])
        t_c = np.clip(t_c, 0.05, 0.50)
        thresh_causal[wy] = t_c

        t_i = float(model_insseason.predict(row[CAUSAL_FEATURES + ["kbdi_july"]].values)[0])
        t_i = np.clip(t_i, 0.05, 0.50)
        thresh_insseason[wy] = t_i

        logger.info(f"WY{wy}: causal={t_c:.4f}, insseason={t_i:.4f}")

    # Define threshold approaches
    approaches = {
        "fixed_rate_match": {wy: 0.118 for wy in test_years},
        "fixed_area_calib": {wy: 0.130 for wy in test_years},
        "year_specific_causal": thresh_causal,
        "year_specific_insseason": thresh_insseason,
    }

    store = zarr.open_group(ZARR_PATH, mode="r")
    time_index_zarr = np.array(store["meta/time"])

    # Compute actual burned area per test WY
    actual_area = {}
    for wy in test_years:
        burned = 0
        for m in [6, 7, 8, 9, 10, 11]:
            ym = f"{wy:04d}-{m:02d}"
            idx = np.searchsorted(time_index_zarr, ym)
            if idx < len(time_index_zarr) and time_index_zarr[idx] == ym:
                burned += int(((fire_raster[idx] == 1) & valid_mask).sum())
        actual_area[wy] = burned

    # Evaluate each approach using spatial maps (Track A)
    comparison_rows = []
    for wy in test_years:
        tif_path = maps_dir / "trackA" / f"fire_prob_WY{wy}_fire_season.tif"
        if not tif_path.exists():
            continue
        with rasterio.open(str(tif_path)) as src:
            prob_map = src.read(1)
        prob_map[prob_map == -9999.0] = 0.0

        row = {"wy": wy, "actual_km2": actual_area[wy]}
        for approach_name, thresholds in approaches.items():
            if wy not in thresholds:
                continue
            thresh = thresholds[wy]
            pred = int(((prob_map >= thresh) & valid_mask).sum())
            ratio = pred / max(actual_area[wy], 1)
            row[f"pred_{approach_name}"] = pred
            row[f"ratio_{approach_name}"] = round(ratio, 3)
            row[f"thresh_{approach_name}"] = round(thresh, 4)
        comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)

    # Save predicted thresholds
    thresh_rows = []
    for wy in test_years:
        thresh_rows.append({
            "wy": wy,
            "threshold_causal": thresh_causal.get(wy),
            "threshold_insseason": thresh_insseason.get(wy),
            "actual_area_km2": actual_area[wy],
        })
    pd.DataFrame(thresh_rows).to_csv(
        output_dir / "predicted_thresholds_test.csv", index=False
    )

    return comp_df, approaches, actual_area


# =========================================================================
# Step 6: Diagnostics and plots
# =========================================================================
def make_diagnostic_plots(model_results, optimal_thresholds, climate_indices,
                          comp_df, actual_area, output_dir):
    """Generate all diagnostic plots."""

    # 1. Threshold vs severity scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ot = optimal_thresholds
    ax.scatter(ot["actual_area_km2"], ot["optimal_threshold"], s=40, alpha=0.7)
    ax.set_xlabel("Actual Burned Area (km²)")
    ax.set_ylabel("Optimal Threshold")
    ax.set_title("Optimal Threshold vs Fire Severity (Training Years)")
    # Add trend line
    from numpy.polynomial.polynomial import polyfit
    valid = ~ot["optimal_threshold"].isna()
    if valid.sum() > 2:
        c = polyfit(ot.loc[valid, "actual_area_km2"], ot.loc[valid, "optimal_threshold"], 1)
        x_line = np.linspace(ot["actual_area_km2"].min(), ot["actual_area_km2"].max(), 100)
        ax.plot(x_line, c[0] + c[1] * x_line, "r--", alpha=0.7, label="Linear fit")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_vs_severity.png", dpi=150)
    plt.close()

    # 2. LOO fit plot (causal model)
    causal = model_results["causal"]
    fig, ax = plt.subplots(figsize=(7, 6))
    wys = causal["wy_used"]
    y_act = causal["y_actual"]
    y_pred = causal["y_pred_loo"]
    ax.scatter(y_act, y_pred, s=40, alpha=0.7)
    mn, mx = min(min(y_act), min(y_pred)), max(max(y_act), max(y_pred))
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="1:1")
    ax.set_xlabel("Actual Optimal Threshold")
    ax.set_ylabel("Predicted (LOO)")
    ax.set_title(f"Threshold Prediction LOO (R²={causal['loo_r2']:.3f}, MAE={causal['loo_mae']:.4f})")
    # Annotate select years
    for i, wy in enumerate(wys):
        if wy in [1984, 2017, 2018, 2019, 2020]:
            ax.annotate(str(wy), (y_act[i], y_pred[i]), fontsize=7, alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loo_fit.png", dpi=150)
    plt.close()

    # 3. Burned area scatter — all approaches
    test_years = sorted(comp_df["wy"].tolist())
    fig, ax = plt.subplots(figsize=(8, 6))

    approach_styles = {
        "fixed_rate_match": ("s", "gray", "Fixed rate-match (0.118)"),
        "fixed_area_calib": ("D", "blue", "Fixed area-calib (0.130)"),
        "year_specific_causal": ("o", "green", "Year-specific (causal)"),
        "year_specific_insseason": ("^", "orange", "Year-specific (in-season)"),
    }
    for approach, (marker, color, label) in approach_styles.items():
        col = f"pred_{approach}"
        if col in comp_df.columns:
            ax.scatter(comp_df["actual_km2"], comp_df[col],
                       marker=marker, color=color, s=60, label=label, alpha=0.8)

    mn = 0
    mx = max(comp_df["actual_km2"].max(), max(
        comp_df[[c for c in comp_df.columns if c.startswith("pred_")]].max()
    )) * 1.1
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.4, label="1:1")
    ax.set_xlabel("Actual Burned Area (km²)")
    ax.set_ylabel("Predicted Burned Area (km²)")
    ax.set_title("Burned Area: Predicted vs Actual (Test WY2020-2024)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "burned_area_scatter.png", dpi=150)
    plt.close()

    # 4. CWD anomaly vs threshold diagnostic
    merged = optimal_thresholds.merge(climate_indices, on="wy")
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = ~merged["optimal_threshold"].isna() & ~merged["cwd_anom_wateryear"].isna()
    ax.scatter(merged.loc[valid, "cwd_anom_wateryear"],
               merged.loc[valid, "optimal_threshold"], s=40, alpha=0.7)
    ax.set_xlabel("CWD Anomaly (Antecedent Oct-May, mm)")
    ax.set_ylabel("Optimal Threshold")
    ax.set_title("CWD Anomaly vs Optimal Threshold")
    if valid.sum() > 2:
        c = polyfit(merged.loc[valid, "cwd_anom_wateryear"],
                    merged.loc[valid, "optimal_threshold"], 1)
        x_line = np.linspace(merged["cwd_anom_wateryear"].min(),
                             merged["cwd_anom_wateryear"].max(), 100)
        ax.plot(x_line, c[0] + c[1] * x_line, "r--", alpha=0.7)
        slope_sign = "negative" if c[1] < 0 else "POSITIVE (unexpected!)"
        ax.text(0.05, 0.95, f"Slope: {c[1]:.4f} ({slope_sign})",
                transform=ax.transAxes, fontsize=9, va="top")
    plt.tight_layout()
    plt.savefig(output_dir / "cwd_vs_threshold.png", dpi=150)
    plt.close()


def print_comparison_table(comp_df, actual_area):
    """Print formatted comparison table."""
    print("\n" + "=" * 110)
    print("BURNED AREA COMPARISON: FIXED vs YEAR-SPECIFIC THRESHOLDS (Track A)")
    print("=" * 110)
    header = f"{'WY':<6} {'Actual':>8}"
    for approach in ["fixed_rate_match", "fixed_area_calib",
                     "year_specific_causal", "year_specific_insseason"]:
        short = {
            "fixed_rate_match": "Fixed(0.118)",
            "fixed_area_calib": "Fixed(0.130)",
            "year_specific_causal": "YearSpec(causal)",
            "year_specific_insseason": "YearSpec(inssn)",
        }[approach]
        header += f"  {short:>22}"
    print(header)
    print("-" * 110)

    totals = {"actual": 0}
    for approach in ["fixed_rate_match", "fixed_area_calib",
                     "year_specific_causal", "year_specific_insseason"]:
        totals[approach] = 0

    for _, row in comp_df.iterrows():
        line = f"WY{int(row['wy']):<4} {int(row['actual_km2']):>8,}"
        totals["actual"] += row["actual_km2"]
        for approach in ["fixed_rate_match", "fixed_area_calib",
                         "year_specific_causal", "year_specific_insseason"]:
            pred_col = f"pred_{approach}"
            ratio_col = f"ratio_{approach}"
            thresh_col = f"thresh_{approach}"
            if pred_col in row and not pd.isna(row[pred_col]):
                pred = int(row[pred_col])
                ratio = row[ratio_col]
                thresh = row[thresh_col]
                totals[approach] += pred
                line += f"  {pred:>10,}({ratio:.2f})t={thresh:.3f}"
            else:
                line += f"  {'N/A':>22}"
        print(line)

    print("-" * 110)
    line = f"{'Total':<6} {int(totals['actual']):>8,}"
    for approach in ["fixed_rate_match", "fixed_area_calib",
                     "year_specific_causal", "year_specific_insseason"]:
        pred = totals[approach]
        ratio = pred / max(totals["actual"], 1)
        line += f"  {int(pred):>10,}({ratio:.2f}){'':>10}"

    print(line)
    print("=" * 110)

    # RMSE
    for approach in ["fixed_rate_match", "fixed_area_calib",
                     "year_specific_causal", "year_specific_insseason"]:
        pred_col = f"pred_{approach}"
        if pred_col in comp_df.columns:
            errors = comp_df[pred_col] - comp_df["actual_km2"]
            rmse = np.sqrt((errors ** 2).mean())
            max_ratio_err = comp_df[f"ratio_{approach}"].apply(lambda r: abs(r - 1.0)).max()
            print(f"  {approach}: RMSE={rmse:,.0f} km², max ratio error={max_ratio_err:.2f}")


# =========================================================================
# Main
# =========================================================================
def main():
    THRESH_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared data
    store = zarr.open_group(ZARR_PATH, mode="r")
    valid_mask = np.array(store["meta/valid_mask"])
    time_index = np.array(store["meta/time"])
    fire_raster = np.load(str(OUTPUT_DIR / "fire_raster.npy"), mmap_mode="r")

    # ---- Step 1: Climate indices ----
    logger.info("Step 1: Computing annual climate indices...")
    climate_csv = THRESH_DIR / "annual_climate_indices.csv"
    climate_indices = compute_annual_climate_indices(ZARR_PATH, valid_mask, time_index)
    climate_indices.to_csv(climate_csv, index=False)
    logger.info(f"Saved {len(climate_indices)} years to {climate_csv}")
    logger.info(f"Columns: {list(climate_indices.columns)}")
    logger.info(f"WY range: {climate_indices['wy'].min()}-{climate_indices['wy'].max()}")

    # ---- Step 1b: Generate spatial maps for training years ----
    train_years = list(range(1984, 2020))
    logger.info("Step 1b: Generating spatial probability maps for training years...")
    generate_training_spatial_maps(
        OUTPUT_DIR / "model" / "trackA" / "lr_calibrated.pkl",
        TRACK_A_FEATURES,
        MAPS_DIR,
        train_years,
    )

    # ---- Step 2: Optimal thresholds for training years ----
    logger.info("Step 2: Computing optimal thresholds per training year...")
    optimal_thresholds = compute_optimal_thresholds(
        MAPS_DIR, fire_raster, valid_mask, time_index, train_years,
    )
    optimal_thresholds.to_csv(
        THRESH_DIR / "optimal_thresholds_training.csv", index=False
    )
    logger.info(f"Optimal thresholds for {len(optimal_thresholds)} years")
    logger.info(f"Threshold range: {optimal_thresholds['optimal_threshold'].min():.3f} - "
                f"{optimal_thresholds['optimal_threshold'].max():.3f}")

    # Diagnostic: check monotonicity
    corr = optimal_thresholds[["actual_area_km2", "optimal_threshold"]].corr().iloc[0, 1]
    logger.info(f"Correlation(actual_area, optimal_threshold) = {corr:.3f}")
    if corr > 0:
        logger.warning("POSITIVE correlation — severe fire years have HIGHER thresholds. "
                        "Expected negative. Investigate before trusting model.")

    # ---- Step 3: Fit threshold models ----
    logger.info("Step 3: Fitting threshold prediction models...")
    train_data = optimal_thresholds.merge(climate_indices, on="wy")
    train_data = train_data[train_data["wy"] <= 2019]

    model_results = fit_threshold_models(train_data, THRESH_DIR)

    # Check coefficient signs
    causal_coefs = model_results["causal"]["coefficients"]
    expected_signs = {
        "cwd_anom_wateryear": "negative",
        "sws_may": "positive",
        "pck_april_anom": "positive",
        "tmax_jun_jul_anom": "negative",
    }
    print("\nCoefficient Sign Check (causal model):")
    for entry in causal_coefs:
        feat = entry["feature"]
        coef = entry["coefficient_std"]
        expected = expected_signs.get(feat, "unknown")
        actual = "negative" if coef < 0 else "positive"
        match = "OK" if actual == expected else "REVERSED"
        print(f"  {feat:<25s}: std_coef={coef:+.4f} (expected {expected}, got {actual}) [{match}]")

    # Save model summary
    summary = {}
    for name, res in model_results.items():
        summary[name] = {
            "features": res["features"],
            "n_train": res["n_train"],
            "loo_r2": res["loo_r2"],
            "loo_mae": res["loo_mae"],
            "intercept": res["intercept"],
            "coefficients": res["coefficients"],
        }
    with open(THRESH_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Step 4 & 5: Evaluate ----
    logger.info("Step 4-5: Predicting test thresholds and evaluating...")
    comp_df, approaches, actual_area = evaluate_threshold_approaches(
        model_results, climate_indices, fire_raster,
        time_index, valid_mask, MAPS_DIR, THRESH_DIR,
    )
    comp_df.to_csv(THRESH_DIR / "burned_area_comparison.csv", index=False)

    print_comparison_table(comp_df, actual_area)

    # ---- Step 6: Diagnostics ----
    logger.info("Step 6: Generating diagnostic plots...")
    make_diagnostic_plots(
        model_results, optimal_thresholds, climate_indices,
        comp_df, actual_area, THRESH_DIR,
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("THRESHOLD MODEL SUMMARY")
    print("=" * 60)
    for name, res in model_results.items():
        print(f"  {name}: LOO R²={res['loo_r2']:.3f}, LOO MAE={res['loo_mae']:.4f}")

    causal_r2 = model_results["causal"]["loo_r2"]
    if causal_r2 > 0.3:
        print(f"\nCausal model LOO R² = {causal_r2:.3f} > 0.3 — SUCCESS")
        print("Climate indices explain meaningful variance in optimal thresholds.")
    else:
        print(f"\nCausal model LOO R² = {causal_r2:.3f} < 0.3 — INSUFFICIENT")
        print("Climate indices do not reliably predict threshold.")
        print("Fixed area-calibrated threshold (0.130) remains best available.")

    logger.info(f"All outputs saved to {THRESH_DIR}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
