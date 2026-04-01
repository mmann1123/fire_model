"""Evaluate fire models: Track A vs Track B comparison, metrics, spatial maps.

Usage:
    conda run -n deep_field python scripts/03_evaluate.py
    conda run -n deep_field python scripts/03_evaluate.py --run-id v3-matched-ratio --notes "10:1 neg ratio"
"""

import argparse
import json
import logging
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
import zarr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Load configuration ----
with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR = Path((PROJECT_ROOT / cfg["paths"]["output_dir"]).resolve())
PANEL_PATH = OUTPUT_DIR / "panel" / "fire_panel.parquet"
ECOREGION_TIF = cfg["paths"]["ecoregion_tif"]
ZARR_PATH = cfg["paths"]["zarr_store"]
FVEG_MAP_PATH = cfg["paths"]["fveg_map"]
FVEG_VAT_PATH = cfg["paths"]["fveg_vat"]
SERGOM_DIR = cfg["paths"]["sergom_dir"]
PREDICTIONS_DIR = str((PROJECT_ROOT / cfg["paths"]["predictions_dir"]).resolve())

# BCM grid
H = cfg["grid"]["height"]
W = cfg["grid"]["width"]
from rasterio.transform import Affine
tx = cfg["grid"]["transform"]
TRANSFORM = Affine(tx[0], tx[1], tx[2], tx[3], tx[4], tx[5])

# Build feature lists from config
COMMON_FEATURES = cfg["features"]["common"]
TRACK_A_FEATURES = COMMON_FEATURES + [
    f"{feat}_a" for feat in cfg["features"]["track_specific"]
]
TRACK_B_FEATURES = COMMON_FEATURES + [
    f"{feat}_b" for feat in cfg["features"]["track_specific"]
]

# Infrastructure rasters from config
INFRA_RASTERS = {}
for key, info in cfg["infrastructure"].items():
    feat_name = f"{key}_km"
    INFRA_RASTERS[feat_name] = (info["path"], info["scale"])


def compute_metrics(y_true, y_prob):
    """Compute classification metrics. Returns dict or None if insufficient data."""
    if len(y_true) < 10 or y_true.sum() < 2 or y_true.sum() == len(y_true):
        return None
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
    }


def load_ecoregions():
    """Load ecoregion raster, resample to BCM grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(ECOREGION_TIF) as src:
        eco_data = np.full((H, W), 0, dtype=np.int32)
        reproject(
            source=rasterio.band(src, 1),
            destination=eco_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=TRANSFORM,
            dst_crs="EPSG:3310",
            resampling=Resampling.nearest,
        )
    return eco_data


def evaluate_track(df_test, features, model_path, track_name):
    """Run evaluation for one track. Returns predictions and metrics."""
    logger.info(f"Evaluating {track_name}...")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = df_test[features].values
    y_test = df_test["fire"].values
    # Support ecoregion models that need pixel indices
    if hasattr(model, "needs_pixel_indices") and "row" in df_test.columns:
        y_prob = model.predict_proba(
            X_test, pixel_indices=(df_test["row"].values, df_test["col"].values)
        )[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    # Overall metrics
    overall = compute_metrics(y_test, y_prob)
    logger.info(f"  Overall: AUC={overall['roc_auc']:.4f}, AP={overall['avg_precision']:.4f}, "
                f"Brier={overall['brier_score']:.4f}")

    # Monthly metrics
    monthly = {}
    for m in range(1, 13):
        mask = df_test["month"].values == m
        if mask.sum() > 0:
            met = compute_metrics(y_test[mask], y_prob[mask])
            if met:
                monthly[m] = met

    # Quarterly metrics
    quarterly = {}
    q_map = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12],
             "fire_season": [6, 7, 8, 9, 10, 11]}
    for qname, months in q_map.items():
        mask = np.isin(df_test["month"].values, months)
        if mask.sum() > 0:
            met = compute_metrics(y_test[mask], y_prob[mask])
            if met:
                quarterly[qname] = met

    # Water year metrics
    wy_metrics = {}
    years = df_test["year"].values
    months = df_test["month"].values
    wy = np.where(months >= 10, years + 1, years)
    for wy_val in sorted(np.unique(wy)):
        if wy_val < 2020 or wy_val > 2024:
            continue
        mask = wy == wy_val
        if mask.sum() > 0:
            met = compute_metrics(y_test[mask], y_prob[mask])
            if met:
                wy_metrics[int(wy_val)] = met

    # Ecoregion metrics
    eco_metrics = {}
    try:
        eco_raster = load_ecoregions()
        eco_vals = eco_raster[df_test["row"].values, df_test["col"].values]
        for eco_id in np.unique(eco_vals):
            if eco_id == 0:
                continue
            mask = eco_vals == eco_id
            if mask.sum() > 0:
                met = compute_metrics(y_test[mask], y_prob[mask])
                if met and met["n_positive"] >= 100:
                    eco_metrics[int(eco_id)] = met
    except Exception as e:
        logger.warning(f"  Ecoregion metrics failed: {e}")

    results = {
        "overall": overall,
        "monthly": monthly,
        "quarterly": quarterly,
        "water_year": wy_metrics,
        "ecoregion": eco_metrics,
    }

    return y_prob, results


def save_track_metrics(results, track_dir):
    """Save metrics for one track."""
    track_dir.mkdir(parents=True, exist_ok=True)

    # Overall
    with open(track_dir / "metrics_overall.json", "w") as f:
        json.dump(results["overall"], f, indent=2)

    # Monthly
    if results["monthly"]:
        rows = [{"month": m, **v} for m, v in sorted(results["monthly"].items())]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_month.csv", index=False)

    # Quarterly
    if results["quarterly"]:
        rows = [{"quarter": q, **v} for q, v in results["quarterly"].items()]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_quarter.csv", index=False)

    # Water year
    if results["water_year"]:
        rows = [{"water_year": wy, **v} for wy, v in sorted(results["water_year"].items())]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_water_year.csv", index=False)

    # Ecoregion
    if results["ecoregion"]:
        rows = [{"ecoregion_id": eid, **v} for eid, v in sorted(results["ecoregion"].items())]
        pd.DataFrame(rows).to_csv(track_dir / "metrics_by_ecoregion.csv", index=False)


def save_calibration_curve(y_true, y_prob, path, track_name, n_bins=10):
    """Save reliability diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_prob[mask].mean())
            bin_means.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_centers, bin_means, "o-", label=track_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {track_name}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_roc_comparison(y_true, prob_a, prob_b, path):
    """Save overlaid ROC curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr_a, tpr_a, _ = roc_curve(y_true, prob_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, prob_b)
    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr_a, tpr_a, label=f"Track A (BCMv8) AUC={auc_a:.4f}", linewidth=2)
    ax.plot(fpr_b, tpr_b, label=f"Track B (Emulator) AUC={auc_b:.4f}", linewidth=2, linestyle="--")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: BCMv8 vs Emulator")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_spatial_maps(model_path, features_list, track_suffix, track_dir,
                      wy_range=range(2020, 2025)):
    """Save full-grid fire probability GeoTIFFs for each water year fire season.

    Predicts on ALL valid pixels (not just sampled panel), so the output
    covers the entire California domain without gaps.
    """
    track_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load zarr data
    store = zarr.open_group(ZARR_PATH, mode="r")
    valid_mask = np.array(store["meta/valid_mask"])
    time_index = np.array(store["meta/time"])
    static = np.array(store["inputs/static"])
    dyn = store["inputs/dynamic"]

    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)

    # Load climatology
    clim = np.load(str(OUTPUT_DIR / "climatology_1984_2016.npz"))
    cwd_clim = clim["cwd"]
    aet_clim = clim["aet"]
    pet_clim = clim["pet"]

    # Load TSF (full zarr range)
    fire_raster = np.load(str(OUTPUT_DIR / "fire_raster.npy"), mmap_mode="r")
    tsf_init_path = OUTPUT_DIR / "tsf_state_1984-01.npy"
    initial_tsf = np.load(str(tsf_init_path)) if tsf_init_path.exists() else None

    # Compute TSF inline (same logic as 01_build_panel.py)
    T_fire = fire_raster.shape[0]
    if initial_tsf is not None:
        tsf_state = initial_tsf.astype(np.float32).copy()
    else:
        tsf_state = np.full((H, W), 600.0, dtype=np.float32)
    tsf_state[~valid_mask] = 0.0
    tsf_full = np.zeros((T_fire, H, W), dtype=np.float32)
    for t in range(T_fire):
        tsf_full[t] = tsf_state
        tsf_state[valid_mask] += 1.0
        tsf_state = np.minimum(tsf_state, 600.0)
        burned = (fire_raster[t] == 1) & valid_mask
        tsf_state[burned] = 0.0

    # Compute time since treatment inline
    def _compute_tst(raster_path, cap_years):
        if not Path(raster_path).exists():
            return np.full((T_fire, H, W), cap_years * 12, dtype=np.float32)
        tr = np.load(str(raster_path), mmap_mode="r")
        max_m = int(cap_years * 12)
        st = np.full((H, W), float(max_m), dtype=np.float32)
        st[~valid_mask] = 0.0
        tst = np.zeros((T_fire, H, W), dtype=np.float32)
        for t in range(T_fire):
            tst[t] = st
            st[valid_mask] += 1.0
            st = np.minimum(st, max_m)
            treated = (tr[t] == 1) & valid_mask
            st[treated] = 0.0
        return tst

    tst_broadcast_full = _compute_tst(OUTPUT_DIR / "broadcast_raster.npy", 7)
    tst_mechanical_full = _compute_tst(OUTPUT_DIR / "mechanical_raster.npy", 5)

    # Load hydrology source based on track
    if track_suffix == "_b":
        pred_dir = Path(PREDICTIONS_DIR)
        pred_cwd = np.load(str(pred_dir / "cwd.npy"), mmap_mode="r")
        pred_aet = np.load(str(pred_dir / "aet.npy"), mmap_mode="r")
        pred_pet = np.load(str(pred_dir / "pet.npy"), mmap_mode="r")
        pred_time = np.load(str(pred_dir / "time_index.npy"), allow_pickle=True)
        pred_ym_to_idx = {str(ym): i for i, ym in enumerate(pred_time)}
    else:
        pred_ym_to_idx = {}

    # CWD targets for cumulative anomaly computation
    cwd_targets = np.array(store["targets/cwd"])
    aet_targets = np.array(store["targets/aet"])
    pet_targets = np.array(store["targets/pet"])

    # FVEG broad categories
    with open(FVEG_MAP_PATH) as f:
        fveg_map = json.load(f)
    vat = pd.read_csv(FVEG_VAT_PATH)
    whrnum_to_lf = vat.drop_duplicates("WHRNUM").set_index("WHRNUM")["LIFEFORM"].to_dict()
    fveg_ids = static[13].astype(int)
    fveg_forest = np.zeros((H, W), dtype=np.float32)
    fveg_shrub = np.zeros((H, W), dtype=np.float32)
    fveg_herb = np.zeros((H, W), dtype=np.float32)
    for cid_str, info in fveg_map["id_to_info"].items():
        lf = whrnum_to_lf.get(info["whrnum"], "OTHER")
        m = fveg_ids == int(cid_str)
        if lf in ("CONIFER", "HARDWOOD"):
            fveg_forest[m] = 1.0
        elif lf == "SHRUB":
            fveg_shrub[m] = 1.0
        elif lf == "HERBACEOUS":
            fveg_herb[m] = 1.0

    # Static features for all valid pixels
    elev_v = static[0, valid_rows, valid_cols]
    aridity_v = static[8, valid_rows, valid_cols]
    windward_v = static[12, valid_rows, valid_cols]
    forest_v = fveg_forest[valid_rows, valid_cols]
    shrub_v = fveg_shrub[valid_rows, valid_cols]
    herb_v = fveg_herb[valid_rows, valid_cols]

    # Infrastructure distance features
    infra_v = {}
    for feat_name, (ipath, scale) in INFRA_RASTERS.items():
        with rasterio.open(ipath) as src:
            d = src.read(1).astype(np.float32)
            nd = src.nodata
        if nd is not None:
            d[d == nd] = 0.0
        d = np.maximum(d * scale, 0.0)
        infra_v[feat_name] = d[valid_rows, valid_cols]

    # SERGOM housing density (load all needed years)
    sergom_cache = {}
    def _get_housing(year):
        if year not in sergom_cache:
            p = Path(SERGOM_DIR) / f"bhc{year}.tif"
            if p.exists():
                with rasterio.open(str(p)) as src:
                    sergom_cache[year] = np.maximum(src.read(1).astype(np.float32), 0.0)
            else:
                sergom_cache[year] = np.zeros((H, W), dtype=np.float32)
        return sergom_cache[year]

    profile = {
        "driver": "GTiff", "dtype": "float32", "width": W, "height": H,
        "count": 1, "crs": "EPSG:3310", "transform": TRANSFORM,
        "nodata": -9999.0, "compress": "lzw",
    }

    # Fire season months: Jun(6) through Nov(11)
    fire_season_months = [6, 7, 8, 9, 10, 11]

    ym_to_zarr_idx = {ym: i for i, ym in enumerate(time_index)}

    for wy_val in tqdm(wy_range, desc="Spatial maps (full grid)"):
        # Months in this WY's fire season
        wy_months = []
        for m in [10, 11]:  # Oct, Nov of prior year
            ym = f"{wy_val-1:04d}-{m:02d}"
            if ym in ym_to_zarr_idx:
                wy_months.append((ym, wy_val - 1, m))
        for m in [6, 7, 8, 9]:  # Jun-Sep of WY year
            ym = f"{wy_val:04d}-{m:02d}"
            if ym in ym_to_zarr_idx:
                wy_months.append((ym, wy_val, m))

        if not wy_months:
            continue

        prob_sum = np.zeros(n_valid, dtype=np.float64)
        n_months = 0

        for ym, year, month in wy_months:
            zarr_t = ym_to_zarr_idx[ym]
            m_idx = month - 1

            # Dynamic features
            dyn_slice = np.array(dyn[zarr_t, :, :, :])  # (15, H, W)
            ppt_v = dyn_slice[0, valid_rows, valid_cols]
            tmin_v = dyn_slice[1, valid_rows, valid_cols]
            tmax_v = dyn_slice[2, valid_rows, valid_cols]
            vpd_v = dyn_slice[9, valid_rows, valid_cols]
            srad_v = dyn_slice[5, valid_rows, valid_cols]
            kbdi_v = dyn_slice[10, valid_rows, valid_cols]
            sws_v = dyn_slice[11, valid_rows, valid_cols]
            vpd_std_v = dyn_slice[12, valid_rows, valid_cols]

            # Hydrology source
            if track_suffix == "_b" and ym in pred_ym_to_idx:
                pt = pred_ym_to_idx[ym]
                cwd_v = pred_cwd[pt, valid_rows, valid_cols]
                aet_v = pred_aet[pt, valid_rows, valid_cols]
                pet_v = pred_pet[pt, valid_rows, valid_cols]
            else:
                cwd_v = cwd_targets[zarr_t, valid_rows, valid_cols]
                aet_v = aet_targets[zarr_t, valid_rows, valid_cols]
                pet_v = pet_targets[zarr_t, valid_rows, valid_cols]

            # Anomalies
            cwd_anom_v = cwd_v - cwd_clim[m_idx, valid_rows, valid_cols]
            aet_anom_v = aet_v - aet_clim[m_idx, valid_rows, valid_cols]
            pet_anom_v = pet_v - pet_clim[m_idx, valid_rows, valid_cols]

            # Cumulative CWD anomalies
            cwd_cum3 = np.zeros(n_valid, dtype=np.float32)
            cwd_cum6 = np.zeros(n_valid, dtype=np.float32)
            for w in range(6):
                t_back = zarr_t - w
                if t_back < 0:
                    continue
                m_back = int(time_index[t_back][5:7]) - 1
                if track_suffix == "_b" and str(time_index[t_back]) in pred_ym_to_idx:
                    pt_b = pred_ym_to_idx[str(time_index[t_back])]
                    cwd_back = pred_cwd[pt_b, valid_rows, valid_cols]
                else:
                    cwd_back = cwd_targets[t_back, valid_rows, valid_cols]
                anom_back = cwd_back - cwd_clim[m_back, valid_rows, valid_cols]
                cwd_cum6 += anom_back
                if w < 3:
                    cwd_cum3 += anom_back

            # TSF
            tsf_months = tsf_full[zarr_t, valid_rows, valid_cols]
            tsf_years_v = tsf_months / 12.0
            tsf_log_v = np.log1p(tsf_years_v)

            # Seasonal
            month_sin = np.float32(np.sin(2 * np.pi * month / 12))
            month_cos = np.float32(np.cos(2 * np.pi * month / 12))
            fire_season = np.float32(1.0 if month in fire_season_months else 0.0)

            # Assemble features (must match TRACK_A/B_FEATURES order)
            X = np.column_stack([
                ppt_v, tmin_v, tmax_v, vpd_v, srad_v, kbdi_v, sws_v, vpd_std_v,
                np.full(n_valid, month_sin),
                np.full(n_valid, month_cos),
                np.full(n_valid, fire_season),
                elev_v, aridity_v, windward_v,
                forest_v, shrub_v, herb_v,
                tsf_years_v, tsf_log_v,
                tst_broadcast_full[zarr_t, valid_rows, valid_cols] / 12.0,
                tst_mechanical_full[zarr_t, valid_rows, valid_cols] / 12.0,
                ((tst_broadcast_full[zarr_t, valid_rows, valid_cols] <= 60) |
                 (tst_mechanical_full[zarr_t, valid_rows, valid_cols] <= 60)).astype(np.float32),
                infra_v["dist_campground_km"],
                infra_v["dist_transmission_km"],
                infra_v["dist_airbase_km"],
                infra_v["dist_firestation_km"],
                infra_v["dist_road_km"],
                _get_housing(year)[valid_rows, valid_cols],
                np.log1p(_get_housing(year)[valid_rows, valid_cols]),
                cwd_anom_v, aet_anom_v, pet_anom_v,
                cwd_cum3, cwd_cum6,
            ])

            # Support ecoregion models that need pixel indices
            if hasattr(model, "needs_pixel_indices"):
                probs = model.predict_proba(
                    X, pixel_indices=(valid_rows, valid_cols)
                )[:, 1]
            else:
                probs = model.predict_proba(X)[:, 1]
            prob_sum += probs
            n_months += 1

        # Average across fire season months
        prob_mean_valid = (prob_sum / n_months).astype(np.float32)
        prob_map = np.full((H, W), -9999.0, dtype=np.float32)
        prob_map[valid_rows, valid_cols] = prob_mean_valid

        out_path = track_dir / f"fire_prob_WY{wy_val}_fire_season.tif"
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(prob_map[np.newaxis, :])

    logger.info(f"  Spatial maps saved to {track_dir}")


def compute_permutation_importance(model_path, features, df_test, track_name,
                                    out_dir, n_repeats=5, subsample=100000):
    """Compute and save permutation importance (model-agnostic)."""
    from sklearn.inspection import permutation_importance

    logger.info(f"  Computing permutation importance for {track_name} "
                f"(n_repeats={n_repeats}, subsample={subsample})...")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Subsample for speed (full test set is >1M rows)
    rng = np.random.RandomState(42)
    if len(df_test) > subsample:
        idx = rng.choice(len(df_test), subsample, replace=False)
        df_sub = df_test.iloc[idx]
    else:
        df_sub = df_test

    X = df_sub[features].values
    y = df_sub["fire"].values

    result = permutation_importance(
        model, X, y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp_df.to_csv(out_dir / "permutation_importance.csv", index=False)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(6, len(features) * 0.3)))
    imp_sorted = imp_df.sort_values("importance_mean", ascending=True)
    colors = ["#c0392b" if v > 0.001 else "#7f8c8d" for v in imp_sorted["importance_mean"]]
    ax.barh(range(len(imp_sorted)), imp_sorted["importance_mean"],
            xerr=imp_sorted["importance_std"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(imp_sorted)))
    ax.set_yticklabels(imp_sorted["feature"], fontsize=9)
    ax.set_xlabel("Mean AUC decrease when feature is shuffled", fontsize=10)
    ax.set_title(f"Permutation Importance — {track_name}\n(test set, {n_repeats} repeats)", fontsize=11)
    ax.axvline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(out_dir / "permutation_importance.png"), dpi=150)
    plt.close(fig)

    # Print top features
    logger.info(f"  Top features by permutation importance ({track_name}):")
    for _, row in imp_df.head(10).iterrows():
        logger.info(f"    {row['feature']:25s} {row['importance_mean']:+.4f} "
                    f"(+/- {row['importance_std']:.4f})")

    return imp_df


def _make_importance_comparison(imp_a, imp_b, out_path):
    """Side-by-side permutation importance for Track A vs Track B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Use common feature names (strip _a/_b suffixes for alignment)
    def normalize_name(n):
        for suffix in ("_a", "_b"):
            if n.endswith(suffix):
                return n[:-len(suffix)]
        return n

    imp_a = imp_a.copy()
    imp_b = imp_b.copy()
    imp_a["feat_norm"] = imp_a["feature"].apply(normalize_name)
    imp_b["feat_norm"] = imp_b["feature"].apply(normalize_name)

    merged = imp_a[["feat_norm", "importance_mean"]].merge(
        imp_b[["feat_norm", "importance_mean"]],
        on="feat_norm", suffixes=("_A", "_B"),
    )
    merged = merged.sort_values("importance_mean_A", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(merged) * 0.35)))
    y = np.arange(len(merged))
    height = 0.35
    ax.barh(y - height / 2, merged["importance_mean_A"], height,
            label="Track A (BCMv8)", color="#e67e22", alpha=0.85)
    ax.barh(y + height / 2, merged["importance_mean_B"], height,
            label="Track B (Emulator)", color="#2980b9", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(merged["feat_norm"], fontsize=9)
    ax.set_xlabel("Mean AUC decrease (permutation importance)", fontsize=10)
    ax.set_title("Feature Importance Comparison: BCMv8 vs Emulator", fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved importance comparison: {out_path.name}")


def _make_comparison_figure(wy, prob_map, burned_mask, valid_mask, threshold, track_name, out_path):
    """3-panel figure: predicted probability, overlap map, actual fires."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pred_bool = (prob_map >= threshold) & valid_mask
    actual_bool = burned_mask & valid_mask

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Panel 1: probability
    prob_display = np.ma.masked_where(~valid_mask, prob_map)
    im = axes[0].imshow(prob_display, cmap="YlOrRd", vmin=0, vmax=0.15, origin="upper")
    axes[0].set_title(f"Predicted Fire Probability\n{track_name} — WY{wy}", fontsize=11)
    plt.colorbar(im, ax=axes[0], shrink=0.6, label="Probability")

    # Panel 2: overlap
    overlay = np.full((H, W, 3), 0.9, dtype=np.float32)
    overlay[~valid_mask] = 1.0
    pred_only = pred_bool & ~actual_bool
    actual_only = actual_bool & ~pred_bool
    both = pred_bool & actual_bool
    overlay[pred_only] = [1.0, 0.6, 0.0]
    overlay[actual_only] = [0.2, 0.4, 0.8]
    overlay[both] = [0.8, 0.0, 0.0]
    axes[1].imshow(overlay, origin="upper")
    axes[1].set_title(f"Predicted vs Actual\nWY{wy} (threshold={threshold:.3f})", fontsize=11)
    axes[1].legend(handles=[
        Patch(facecolor=(1.0, 0.6, 0.0), label=f"Predicted only ({pred_only.sum():,} km\u00b2)"),
        Patch(facecolor=(0.2, 0.4, 0.8), label=f"Actual only ({actual_only.sum():,} km\u00b2)"),
        Patch(facecolor=(0.8, 0.0, 0.0), label=f"Overlap ({both.sum():,} km\u00b2)"),
    ], loc="lower left", fontsize=9)

    # Panel 3: actual
    actual_display = np.full((H, W, 3), 0.9, dtype=np.float32)
    actual_display[~valid_mask] = 1.0
    actual_display[actual_bool] = [0.8, 0.0, 0.0]
    axes[2].imshow(actual_display, origin="upper")
    axes[2].set_title(f"Actual Fires\nWY{wy} ({actual_bool.sum():,} km\u00b2)", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_area_chart(area_data, threshold, out_path):
    """Bar chart of actual vs predicted burned area by water year."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wys = sorted(area_data.keys())
    x = np.arange(len(wys))
    width = 0.25
    actual = [area_data[wy]["actual_km2"] for wy in wys]
    pred_a = [area_data[wy].get("predicted_km2_trackA", 0) for wy in wys]
    pred_b = [area_data[wy].get("predicted_km2_trackB", 0) for wy in wys]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, actual, width, label="Actual Burned", color="#c0392b", alpha=0.85)
    ax.bar(x, pred_a, width, label="Predicted (BCMv8)", color="#e67e22", alpha=0.85)
    ax.bar(x + width, pred_b, width, label="Predicted (Emulator)", color="#2980b9", alpha=0.85)
    ax.set_xlabel("Water Year", fontsize=12)
    ax.set_ylabel("Burned Area (km\u00b2)", fontsize=12)
    ax.set_title(f"Actual vs Predicted Burned Area — Fire Season\n(threshold={threshold:.3f})", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"WY{wy}" for wy in wys])
    ax.legend(fontsize=11)
    for bars in ax.containers:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., h + 50,
                        f"{int(h):,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(max(actual), max(pred_a), max(pred_b)) * 1.15)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def select_area_calibrated_threshold(maps_dir, fire_raster, time_index, valid_mask,
                                      wy_range=range(2018, 2020), track="trackA"):
    """Select threshold that minimizes MSE between predicted and actual burned area.

    Uses calibration-period water years to find the threshold where predicted
    burned area (pixels above threshold) best matches actual burned area from
    the fire raster, averaged across fire seasons.

    Parameters
    ----------
    maps_dir : Path
        Directory containing spatial_maps/track{A,B}/ GeoTIFFs.
    fire_raster : array
        Full fire raster (T, H, W).
    time_index : array
        Year-month strings for each timestep.
    valid_mask : array
        (H, W) boolean mask of valid pixels.
    wy_range : range
        Water years to use for calibration.
    track : str
        Track subdirectory ("trackA" or "trackB").

    Returns
    -------
    threshold : float
        Area-calibrated threshold.
    diagnostics : dict
        Per-threshold and per-WY diagnostics.
    """
    # Compute actual burned area per WY fire season
    wy_actual = {}
    for wy in wy_range:
        burned = np.zeros((H, W), dtype=bool)
        for m in [10, 11]:
            ym = f"{wy-1:04d}-{m:02d}"
            idx = np.searchsorted(time_index, ym)
            if idx < len(time_index) and time_index[idx] == ym:
                burned |= (fire_raster[idx] == 1)
        for m in [6, 7, 8, 9]:
            ym = f"{wy:04d}-{m:02d}"
            idx = np.searchsorted(time_index, ym)
            if idx < len(time_index) and time_index[idx] == ym:
                burned |= (fire_raster[idx] == 1)
        burned &= valid_mask
        wy_actual[wy] = int(burned.sum())

    # Load probability maps for calib WYs
    wy_prob_maps = {}
    for wy in wy_range:
        tif_path = maps_dir / track / f"fire_prob_WY{wy}_fire_season.tif"
        if tif_path.exists():
            with rasterio.open(str(tif_path)) as src:
                prob = src.read(1)
            prob[prob == -9999.0] = 0.0
            wy_prob_maps[wy] = prob

    if not wy_prob_maps:
        logger.warning("No calib-period spatial maps found for area threshold calibration")
        return 0.05, {}

    # Search for threshold minimizing area ratio error across calib WYs
    thresholds = np.arange(0.01, 0.50, 0.005)
    best_thresh = 0.05
    best_mse = float("inf")
    diagnostics = {"per_threshold": [], "per_wy_actual": wy_actual}

    for thresh in thresholds:
        ratios = []
        squared_errors = []
        for wy in sorted(wy_prob_maps.keys()):
            pred_area = int(((wy_prob_maps[wy] >= thresh) & valid_mask).sum())
            actual_area = wy_actual[wy]
            if actual_area > 0:
                ratio = pred_area / actual_area
                ratios.append(ratio)
                squared_errors.append((pred_area - actual_area) ** 2)

        if ratios:
            mean_ratio = np.mean(ratios)
            mse = np.mean(squared_errors)
            diagnostics["per_threshold"].append({
                "threshold": float(thresh),
                "mean_ratio": float(mean_ratio),
                "mse": float(mse),
            })
            if mse < best_mse:
                best_mse = mse
                best_thresh = float(thresh)

    logger.info(f"  Area-calibrated threshold: {best_thresh:.3f}")
    for wy in sorted(wy_prob_maps.keys()):
        pred = int(((wy_prob_maps[wy] >= best_thresh) & valid_mask).sum())
        act = wy_actual[wy]
        ratio = pred / act if act > 0 else float("inf")
        logger.info(f"    WY{wy}: actual={act:,} km², predicted={pred:,} km², ratio={ratio:.2f}")

    return best_thresh, diagnostics


SNAPSHOT_DIR = Path((PROJECT_ROOT / cfg["paths"]["snapshot_dir"]).resolve())


def create_snapshot(run_id, notes, git_hash):
    """Create a frozen snapshot of current outputs."""
    snap_dir = SNAPSHOT_DIR / run_id
    if snap_dir.exists():
        logger.warning(f"Overwriting existing snapshot: {snap_dir}")
        shutil.rmtree(snap_dir)

    snap_dir.mkdir(parents=True)
    logger.info(f"Creating snapshot: {snap_dir}")

    # Copy config
    shutil.copy2(PROJECT_ROOT / "config.yaml", snap_dir / "config.yaml")

    # Copy models
    for track in ["trackA", "trackB"]:
        model_src = OUTPUT_DIR / "model" / track
        if model_src.exists():
            model_dst = snap_dir / "model" / track
            model_dst.mkdir(parents=True)
            for f in model_src.iterdir():
                shutil.copy2(f, model_dst / f.name)

    # Copy evaluation, comparison, spatial_maps
    for subdir in ["evaluation", "comparison", "spatial_maps"]:
        src = OUTPUT_DIR / subdir
        if src.exists():
            shutil.copytree(src, snap_dir / subdir)

    # Copy and augment manifest
    manifest_src = OUTPUT_DIR / "manifest.json"
    manifest = {}
    if manifest_src.exists():
        with open(manifest_src) as f:
            manifest = json.load(f)

    manifest["snapshot"] = {
        "run_id": run_id,
        "notes": notes,
        "git_hash": git_hash,
    }
    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    n_files = sum(1 for _ in snap_dir.rglob("*") if _.is_file())
    total_mb = sum(f.stat().st_size for f in snap_dir.rglob("*") if f.is_file()) / 1e6
    logger.info(f"Snapshot created: {n_files} files, {total_mb:.1f} MB")
    logger.info(f"  Run ID: {run_id}")
    if "trackA_overall_auc" in manifest:
        logger.info(f"  AUC-A: {manifest['trackA_overall_auc']:.4f}, "
                     f"AUC-B: {manifest['trackB_overall_auc']:.4f}, "
                     f"Delta: {manifest['auc_delta']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fire models")
    parser.add_argument("--run-id", default=None,
                        help="If provided, auto-create snapshot with this ID after evaluation")
    parser.add_argument("--notes", default="", help="Notes for snapshot")
    args = parser.parse_args()

    logger.info("Loading panel...")
    df = pd.read_parquet(PANEL_PATH)
    df_test = df[df["split"] == "test"].copy()
    logger.info(f"Test set: {len(df_test)} rows, {df_test['fire'].sum()} fires")

    eval_dir = OUTPUT_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate both tracks
    model_a_path = OUTPUT_DIR / "model" / "trackA" / "lr_calibrated.pkl"
    model_b_path = OUTPUT_DIR / "model" / "trackB" / "lr_calibrated.pkl"

    prob_a, results_a = evaluate_track(df_test, TRACK_A_FEATURES, model_a_path, "Track A (BCMv8)")
    prob_b, results_b = evaluate_track(df_test, TRACK_B_FEATURES, model_b_path, "Track B (Emulator)")

    # Save per-track metrics
    save_track_metrics(results_a, eval_dir / "trackA")
    save_track_metrics(results_b, eval_dir / "trackB")

    # Save test predictions
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for track_name, probs in [("trackA", prob_a), ("trackB", prob_b)]:
        pred_df = df_test[["year", "month", "row", "col", "fire"]].copy()
        pred_df["prob"] = probs
        pred_df.to_parquet(pred_dir / f"test_predictions_{track_name}.parquet", index=False)

    # Calibration curves
    y_test = df_test["fire"].values
    save_calibration_curve(y_test, prob_a, eval_dir / "trackA" / "calibration_curve.png", "Track A (BCMv8)")
    save_calibration_curve(y_test, prob_b, eval_dir / "trackB" / "calibration_curve.png", "Track B (Emulator)")

    # ROC comparison
    save_roc_comparison(y_test, prob_a, prob_b, eval_dir / "roc_comparison.png")

    # Permutation importance (model-agnostic)
    logger.info("Computing feature importance...")
    imp_a = compute_permutation_importance(
        model_a_path, TRACK_A_FEATURES, df_test, "Track A (BCMv8)", eval_dir / "trackA")
    imp_b = compute_permutation_importance(
        model_b_path, TRACK_B_FEATURES, df_test, "Track B (Emulator)", eval_dir / "trackB")

    # Combined importance comparison plot
    _make_importance_comparison(imp_a, imp_b, eval_dir / "importance_comparison.png")

    # Spatial maps (full grid prediction, not just panel samples)
    # Generate for both test (WY2020-2024) and calib (WY2018-2019) periods
    maps_dir = OUTPUT_DIR / "spatial_maps"
    save_spatial_maps(model_a_path, TRACK_A_FEATURES, "_a", maps_dir / "trackA",
                      wy_range=range(2018, 2025))
    save_spatial_maps(model_b_path, TRACK_B_FEATURES, "_b", maps_dir / "trackB",
                      wy_range=range(2018, 2025))

    # ---- Comparison summary ----
    summary_rows = []

    def add_row(metric, val_a, val_b):
        summary_rows.append({
            "metric": metric,
            "trackA_bcmv8": round(val_a, 4) if val_a is not None else None,
            "trackB_emulator": round(val_b, 4) if val_b is not None else None,
            "delta": round(val_a - val_b, 4) if val_a is not None and val_b is not None else None,
        })

    add_row("Overall ROC-AUC", results_a["overall"]["roc_auc"], results_b["overall"]["roc_auc"])
    add_row("Overall Avg Precision", results_a["overall"]["avg_precision"], results_b["overall"]["avg_precision"])
    add_row("Overall Brier Score", results_a["overall"]["brier_score"], results_b["overall"]["brier_score"])

    if "fire_season" in results_a["quarterly"] and "fire_season" in results_b["quarterly"]:
        add_row("Fire Season ROC-AUC",
                results_a["quarterly"]["fire_season"]["roc_auc"],
                results_b["quarterly"]["fire_season"]["roc_auc"])

    for wy in range(2020, 2025):
        if wy in results_a["water_year"] and wy in results_b["water_year"]:
            add_row(f"WY{wy} ROC-AUC",
                    results_a["water_year"][wy]["roc_auc"],
                    results_b["water_year"][wy]["roc_auc"])

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(eval_dir / "comparison_summary.csv", index=False)

    # Print comparison
    print("\n" + "=" * 72)
    print("FIRE MODEL: TRACK A (BCMv8) vs TRACK B (EMULATOR)")
    print("=" * 72)
    print(f"{'Metric':<30s} {'Track A':>10s} {'Track B':>10s} {'Delta':>10s}")
    print("-" * 72)
    for _, row in summary_df.iterrows():
        a_str = f"{row['trackA_bcmv8']:.4f}" if row['trackA_bcmv8'] is not None else "N/A"
        b_str = f"{row['trackB_emulator']:.4f}" if row['trackB_emulator'] is not None else "N/A"
        d_str = f"{row['delta']:+.4f}" if row['delta'] is not None else "N/A"
        print(f"{row['metric']:<30s} {a_str:>10s} {b_str:>10s} {d_str:>10s}")
    print("=" * 72)

    delta_auc = results_a["overall"]["roc_auc"] - results_b["overall"]["roc_auc"]
    if abs(delta_auc) < 0.02:
        print(f"\nDelta AUC = {delta_auc:+.4f} (< 0.02) — EMULATOR IS OPERATIONALLY VIABLE")
    else:
        print(f"\nDelta AUC = {delta_auc:+.4f} (>= 0.02) — emulator degrades fire skill")

    # ---- Comparison maps: predicted vs actual burned area ----
    logger.info("Generating comparison maps...")
    comp_dir = OUTPUT_DIR / "comparison"
    comp_dir.mkdir(exist_ok=True)

    # Load actual fire data
    fire_raster = np.load(str(OUTPUT_DIR / "fire_raster.npy"), mmap_mode="r")
    store_meta = zarr.open_group(ZARR_PATH, mode="r")
    time_index_full = np.array(store_meta["meta/time"])
    valid_mask = np.array(store_meta["meta/valid_mask"])

    # ---- Threshold 1: Rate-matching (original approach) ----
    df_calib = df[df["split"] == "calib"]
    X_calib_a = df_calib[TRACK_A_FEATURES].values
    y_calib = df_calib["fire"].values
    with open(model_a_path, "rb") as f:
        model_for_thresh = pickle.load(f)
    y_calib_prob = model_for_thresh.predict_proba(X_calib_a)[:, 1]
    actual_rate = y_calib.mean()
    threshold_rate = 0.05
    best_diff = float("inf")
    for t in np.arange(0.001, 0.20, 0.001):
        if abs((y_calib_prob >= t).mean() - actual_rate) < best_diff:
            best_diff = abs((y_calib_prob >= t).mean() - actual_rate)
            threshold_rate = t
    logger.info(f"  Rate-matching threshold: {threshold_rate:.3f}")

    # ---- Threshold 2: Area-calibrated (Experiment 2) ----
    logger.info("  Computing area-calibrated threshold from WY2018-2019 spatial maps...")
    threshold_area, area_diag = select_area_calibrated_threshold(
        maps_dir, fire_raster, time_index_full, valid_mask,
        wy_range=range(2018, 2020), track="trackA",
    )

    # Compute actual burned area for test WYs
    wy_burned = {}
    for wy in range(2020, 2025):
        burned = np.zeros((H, W), dtype=bool)
        for m in [10, 11]:
            ym = f"{wy-1:04d}-{m:02d}"
            idx = np.searchsorted(time_index_full, ym)
            if idx < len(time_index_full) and time_index_full[idx] == ym:
                burned |= (fire_raster[idx] == 1)
        for m in [6, 7, 8, 9]:
            ym = f"{wy:04d}-{m:02d}"
            idx = np.searchsorted(time_index_full, ym)
            if idx < len(time_index_full) and time_index_full[idx] == ym:
                burned |= (fire_raster[idx] == 1)
        burned &= valid_mask
        wy_burned[wy] = burned

    # Generate comparison data for BOTH thresholds
    for thresh_name, threshold in [("rate_match", threshold_rate),
                                    ("area_calib", threshold_area)]:
        area_data = {}
        for track, track_label in [("trackA", "Track A (BCMv8)"),
                                    ("trackB", "Track B (Emulator)")]:
            for wy in range(2020, 2025):
                tif_path = maps_dir / track / f"fire_prob_WY{wy}_fire_season.tif"
                if not tif_path.exists():
                    continue
                with rasterio.open(str(tif_path)) as src:
                    prob_map = src.read(1)
                prob_map[prob_map == -9999.0] = 0.0

                burned_mask = wy_burned[wy]
                actual_km2 = int(burned_mask.sum())
                predicted_km2 = int(((prob_map >= threshold) & valid_mask).sum())

                if wy not in area_data:
                    area_data[wy] = {"actual_km2": actual_km2}
                area_data[wy][f"predicted_km2_{track}"] = predicted_km2

                # Side-by-side comparison figure
                _make_comparison_figure(wy, prob_map, burned_mask, valid_mask,
                                       threshold, track_label,
                                       comp_dir / f"{track}_WY{wy}_{thresh_name}.png")

        # Burned area bar chart
        _make_area_chart(area_data, threshold,
                         comp_dir / f"burned_area_{thresh_name}.png")

        # Save area CSV
        area_rows = [{"water_year": wy, **area_data[wy]}
                     for wy in sorted(area_data.keys())]
        pd.DataFrame(area_rows).to_csv(
            comp_dir / f"burned_area_{thresh_name}.csv", index=False)

        # Print burned area table
        label = "RATE-MATCHING" if thresh_name == "rate_match" else "AREA-CALIBRATED"
        print(f"\nBURNED AREA — {label} THRESHOLD ({threshold:.3f})")
        print(f"{'WY':<8} {'Actual':>10} {'BCMv8':>10} {'Emulator':>10}")
        print("-" * 42)
        total_actual = total_pred_a = total_pred_b = 0
        for wy in sorted(area_data.keys()):
            d = area_data[wy]
            a = d['actual_km2']
            pa = d.get('predicted_km2_trackA', 0)
            pb = d.get('predicted_km2_trackB', 0)
            total_actual += a
            total_pred_a += pa
            total_pred_b += pb
            print(f"WY{wy}  {a:>10,} {pa:>10,} {pb:>10,}")
        print("-" * 42)
        print(f"{'Total':<8} {total_actual:>10,} {total_pred_a:>10,} {total_pred_b:>10,}")
        if total_actual > 0:
            ratio_a = total_pred_a / total_actual
            ratio_b = total_pred_b / total_actual
            print(f"{'Ratio':<8} {'1.00':>10s} {ratio_a:>10.2f} {ratio_b:>10.2f}")

    # Save threshold comparison summary
    thresh_summary = {
        "rate_match_threshold": float(threshold_rate),
        "area_calib_threshold": float(threshold_area),
        "area_calib_wy_range": "WY2018-WY2019",
    }
    with open(comp_dir / "threshold_comparison.json", "w") as f:
        json.dump(thresh_summary, f, indent=2)

    # ---- Manifest ----
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:12]
    except Exception:
        pass

    # Load TSF coefficients
    tsf_coefs = {}
    for track in ["trackA", "trackB"]:
        coef_path = OUTPUT_DIR / "model" / track / "coefficients.csv"
        if coef_path.exists():
            cdf = pd.read_csv(coef_path)
            for feat in ["tsf_log", "tsf_years"]:
                row = cdf[cdf["feature"] == feat]
                if len(row):
                    tsf_coefs[f"{feat}_coef_{track}"] = float(row.iloc[0]["coefficient"])

    manifest = {
        "run_date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "frap_filter": {"state": "CA", "objective": 1, "min_acres": cfg["sampling"]["min_fire_acres"]},
        "train_period": f"{cfg['temporal']['fire_start']} to {cfg['temporal']['train_end']}",
        "calib_period": f"{cfg['temporal']['train_end'][:5]}01 to {cfg['temporal']['calib_end']}",
        "test_period": f"{cfg['temporal']['calib_end'][:5]}10 to 2024-09",
        "sampling": cfg["sampling"],
        "feature_cols": TRACK_A_FEATURES,
        "n_train_pos": int(df[df["split"] == "train"]["fire"].sum()),
        "n_train_neg": int((df["split"] == "train").sum() - df[df["split"] == "train"]["fire"].sum()),
        "n_calib_pos": int(df[df["split"] == "calib"]["fire"].sum()),
        "n_test_pos": int(df_test["fire"].sum()),
        "n_test_neg": int(len(df_test) - df_test["fire"].sum()),
        "trackA_overall_auc": results_a["overall"]["roc_auc"],
        "trackB_overall_auc": results_b["overall"]["roc_auc"],
        "auc_delta": delta_auc,
        "threshold_rate_match": float(threshold_rate),
        "threshold_area_calib": float(threshold_area),
        **tsf_coefs,
    }

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved to {OUTPUT_DIR / 'manifest.json'}")

    # Auto-create snapshot if --run-id provided
    if args.run_id:
        create_snapshot(args.run_id, args.notes, git_hash)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
