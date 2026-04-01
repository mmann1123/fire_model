"""Build fire-ignition panel dataset from FRAP fires and BCM/emulator features.

Rasterizes FRAP fire perimeters, computes time since last fire, samples
positive/negative pixel-months, and extracts hydroclimatic features for
both Track A (BCMv8 targets) and Track B (emulator predictions).

Usage:
    conda run -n deep_field python scripts/01_build_panel.py
    conda run -n deep_field python scripts/01_build_panel.py --force
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import yaml
import zarr
from rasterio.transform import Affine
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

ZARR_PATH = cfg["paths"]["zarr_store"]
FRAP_GDB = cfg["paths"]["frap_gdb"]
FRAP_LAYER = cfg["paths"]["frap_layer"]
RXBURN_LAYER = cfg["paths"]["rxburn_layer"]
ECOREGION_TIF = cfg["paths"]["ecoregion_tif"]
FVEG_MAP_PATH = cfg["paths"]["fveg_map"]
FVEG_VAT_PATH = cfg["paths"]["fveg_vat"]
TSF_RASTER_DIR = cfg["paths"]["tsf_raster_dir"]
SERGOM_DIR = cfg["paths"]["sergom_dir"]

# Resolve relative paths against project root
PREDICTIONS_DIR = str((PROJECT_ROOT / cfg["paths"]["predictions_dir"]).resolve())
OUTPUT_DIR = str((PROJECT_ROOT / cfg["paths"]["output_dir"]).resolve())

# Infrastructure distance rasters
INFRA_RASTERS = {}
for key, info in cfg["infrastructure"].items():
    feat_name = f"{key}_km"
    INFRA_RASTERS[feat_name] = (info["path"], info["scale"])

# BCM grid
H = cfg["grid"]["height"]
W = cfg["grid"]["width"]
tx = cfg["grid"]["transform"]
TRANSFORM = Affine(tx[0], tx[1], tx[2], tx[3], tx[4], tx[5])

# Sampling
MIN_ACRES = cfg["sampling"]["min_fire_acres"]
SEED = cfg["sampling"]["seed"]

# Time periods
TRAIN_END_YM = cfg["temporal"]["train_end"]
CALIB_END_YM = cfg["temporal"]["calib_end"]


def load_frap_fires():
    """Load and filter FRAP fire perimeters."""
    logger.info("Loading FRAP fire perimeters...")
    gdf = gpd.read_file(FRAP_GDB, layer=FRAP_LAYER)
    n_raw = len(gdf)

    gdf = gdf[
        (gdf["STATE"] == "CA")
        & (gdf["OBJECTIVE"] == 1)
        & (gdf["GIS_ACRES"] >= MIN_ACRES)
        & (gdf["ALARM_DATE"].notna())
    ].copy()

    gdf["alarm_dt"] = pd.to_datetime(gdf["ALARM_DATE"], errors="coerce")
    gdf = gdf[gdf["alarm_dt"].notna()].copy()
    gdf["year"] = gdf["alarm_dt"].dt.year
    gdf["month"] = gdf["alarm_dt"].dt.month
    gdf = gdf[(gdf["year"] >= 1984) & (gdf["year"] <= 2024)].copy()
    gdf = gdf.to_crs("EPSG:3310")

    gdf["ym"] = gdf["year"].astype(str).str.zfill(4) + "-" + gdf["month"].astype(str).str.zfill(2)

    logger.info(f"FRAP: {n_raw} raw -> {len(gdf)} filtered (CA, objective=1, >={MIN_ACRES} acres, 1984-2024)")
    logger.info(f"  Year range: {gdf['year'].min()}-{gdf['year'].max()}")
    logger.info(f"  Unique fire-months: {gdf['ym'].nunique()}")
    return gdf


def _rasterize_month(args):
    """Worker: rasterize geometries for one month. Returns (time_idx, raster)."""
    time_idx, geom_wkbs = args
    from shapely import wkb
    shapes = [(wkb.loads(g), 1) for g in geom_wkbs]
    raster = rasterio.features.rasterize(
        shapes, out_shape=(H, W), transform=TRANSFORM,
        fill=0, dtype="uint8", all_touched=True,
    )
    return time_idx, raster


def build_fire_raster(gdf, time_index):
    """Rasterize FRAP fires to monthly binary grid."""
    fire_raster_path = Path(OUTPUT_DIR) / "fire_raster.npy"

    T = len(time_index)
    ym_to_idx = {ym: i for i, ym in enumerate(time_index)}

    # Group fires by month
    fire_months = {}
    for ym, group in gdf.groupby("ym"):
        if ym in ym_to_idx:
            # Serialize geometries for multiprocessing
            geom_wkbs = [g.wkb for g in group.geometry]
            fire_months[ym] = (ym_to_idx[ym], geom_wkbs)

    logger.info(f"Rasterizing {len(fire_months)} fire-months (of {T} total)...")

    fire_raster = np.zeros((T, H, W), dtype=np.uint8)

    # Parallel rasterization
    tasks = list(fire_months.values())
    with ProcessPoolExecutor() as executor:
        for time_idx, raster in tqdm(
            executor.map(_rasterize_month, tasks),
            total=len(tasks), desc="Rasterizing fires"
        ):
            fire_raster[time_idx] = raster

    n_burned = int(fire_raster.sum())
    n_months_with_fire = int((fire_raster.sum(axis=(1, 2)) > 0).sum())
    logger.info(f"Fire raster: {n_burned} burned pixel-months across {n_months_with_fire} months")

    np.save(str(fire_raster_path), fire_raster)
    logger.info(f"Saved fire_raster.npy ({fire_raster.nbytes / 1e6:.0f} MB)")
    return fire_raster


def compute_time_since_fire(fire_raster, valid_mask, initial_tsf_months=None, max_months=600):
    """Compute TSF for every pixel-month (causal: recorded BEFORE that month's fires)."""
    T = fire_raster.shape[0]

    if initial_tsf_months is not None:
        state = initial_tsf_months.astype(np.float32).copy()
    else:
        state = np.full((H, W), float(max_months), dtype=np.float32)
    state[~valid_mask] = 0.0

    tsf = np.zeros((T, H, W), dtype=np.float32)

    for t in tqdm(range(T), desc="Computing TSF"):
        tsf[t] = state
        state[valid_mask] += 1.0
        state = np.minimum(state, max_months)
        burned = (fire_raster[t] == 1) & valid_mask
        state[burned] = 0.0

    return tsf, state


def load_tsf_initial():
    """Load TSF initialization from timeSinceFire_1983.tif."""
    tsf_path = Path(TSF_RASTER_DIR) / "timeSinceFire_1983.tif"
    if not tsf_path.exists():
        logger.warning(f"TSF raster not found: {tsf_path}. Using 50-year default.")
        return None, "50yr_default"

    with rasterio.open(str(tsf_path)) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0

    # Same grid shape (1209x941, 1km) but CRS is EPSG:10917 — read directly
    if data.shape != (H, W):
        logger.warning(f"TSF raster shape {data.shape} != BCM grid ({H},{W}). Using default.")
        return None, "50yr_default"

    # Convert years to months, handle nodata
    data[data == nodata] = 50.0  # default for nodata pixels
    data = np.clip(data * 12.0, 0, 600)

    logger.info(f"TSF init from timeSinceFire_1983.tif: mean={data[data>0].mean():.0f} months")
    return data, "TimeSinceFire_1983"


def load_rxburn():
    """Load and filter FRAP prescribed burn records."""
    logger.info("Loading FRAP prescribed burn records...")
    gdf = gpd.read_file(FRAP_GDB, layer=RXBURN_LAYER)
    n_raw = len(gdf)

    gdf = gdf[
        (gdf["STATE"] == "CA")
        & (gdf["START_DATE"].notna())
        & (gdf["TREATED_AC"] >= 10)
    ].copy()

    gdf["start_dt"] = pd.to_datetime(gdf["START_DATE"], errors="coerce")
    gdf = gdf[
        gdf["start_dt"].notna()
        & (gdf["start_dt"].dt.year >= 1984)
        & (gdf["start_dt"].dt.year <= 2024)
    ].copy()

    gdf["year"] = gdf["start_dt"].dt.year
    gdf["month"] = gdf["start_dt"].dt.month
    gdf = gdf.to_crs("EPSG:3310")

    gdf["is_broadcast"] = gdf["TREATMENT_TYPE"].isin([1, 2])
    gdf["is_mechanical"] = gdf["TREATMENT_TYPE"].isin([3, 4, 5])

    logger.info(f"Rxburn: {n_raw} raw -> {len(gdf)} filtered (CA, >=10 acres, 1984-2024)")
    logger.info(f"  Broadcast/fire-use: {gdf['is_broadcast'].sum()}")
    logger.info(f"  Mechanical: {gdf['is_mechanical'].sum()}")

    # Coverage by decade
    decades = (gdf["year"] // 10) * 10
    for d in sorted(decades.unique()):
        logger.info(f"  {d}s: {(decades == d).sum()} records")

    return gdf


def build_treatment_rasters(rxburn_gdf, time_index):
    """Build binary treatment rasters for broadcast and mechanical treatments."""
    T = len(time_index)
    ym_to_idx = {}
    for i, ym in enumerate(time_index):
        y, m = int(ym[:4]), int(ym[5:7])
        ym_to_idx[(y, m)] = i

    broadcast_raster = np.zeros((T, H, W), dtype=np.uint8)
    mechanical_raster = np.zeros((T, H, W), dtype=np.uint8)

    # Group by year-month
    broadcast_by_ym = {}
    mechanical_by_ym = {}
    for _, row in rxburn_gdf.iterrows():
        if row.geometry is None:
            continue
        ym = (int(row["year"]), int(row["month"]))
        if row["is_broadcast"]:
            broadcast_by_ym.setdefault(ym, []).append(row.geometry.wkb)
        if row["is_mechanical"]:
            mechanical_by_ym.setdefault(ym, []).append(row.geometry.wkb)

    from shapely import wkb

    def rasterize_geoms(geom_wkbs):
        shapes = [(wkb.loads(g), 1) for g in geom_wkbs]
        return rasterio.features.rasterize(
            shapes, out_shape=(H, W), transform=TRANSFORM,
            fill=0, dtype="uint8", all_touched=True,
        )

    logger.info(f"Rasterizing {len(broadcast_by_ym)} broadcast treatment-months...")
    for ym, geom_wkbs in tqdm(broadcast_by_ym.items(), desc="Broadcast treatments"):
        t = ym_to_idx.get(ym)
        if t is not None:
            broadcast_raster[t] = rasterize_geoms(geom_wkbs)

    logger.info(f"Rasterizing {len(mechanical_by_ym)} mechanical treatment-months...")
    for ym, geom_wkbs in tqdm(mechanical_by_ym.items(), desc="Mechanical treatments"):
        t = ym_to_idx.get(ym)
        if t is not None:
            mechanical_raster[t] = rasterize_geoms(geom_wkbs)

    n_b = int(broadcast_raster.sum())
    n_m = int(mechanical_raster.sum())
    logger.info(f"Treatment rasters: {n_b} broadcast pixel-months, {n_m} mechanical pixel-months")

    np.save(str(Path(OUTPUT_DIR) / "broadcast_raster.npy"), broadcast_raster)
    np.save(str(Path(OUTPUT_DIR) / "mechanical_raster.npy"), mechanical_raster)

    return broadcast_raster, mechanical_raster


def compute_time_since_treatment(treatment_raster, valid_mask, cap_years=7):
    """Compute time since last treatment (same logic as compute_time_since_fire)."""
    max_months = int(cap_years * 12)
    T = treatment_raster.shape[0]

    state = np.full((H, W), float(max_months), dtype=np.float32)
    state[~valid_mask] = 0.0

    tst = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        tst[t] = state  # record BEFORE this month's treatments
        state[valid_mask] += 1.0
        state = np.minimum(state, max_months)
        treated = (treatment_raster[t] == 1) & valid_mask
        state[treated] = 0.0

    return tst, state


def load_sergom_housing(years_needed):
    """Load SERGOM housing density rasters for needed years.

    Returns dict {year: (H, W) float32 array} of housing units/km².
    For years beyond available data, uses the last available year.
    """
    sergom_dir = Path(SERGOM_DIR)
    housing = {}
    last_available = None

    for year in sorted(set(years_needed)):
        path = sergom_dir / f"bhc{year}.tif"
        if path.exists():
            with rasterio.open(str(path)) as src:
                data = src.read(1).astype(np.float32)
            data = np.maximum(data, 0.0)
            housing[year] = data
            last_available = data
        elif last_available is not None:
            housing[year] = last_available
        else:
            housing[year] = np.zeros((H, W), dtype=np.float32)

    years_loaded = [y for y in housing if (sergom_dir / f"bhc{y}.tif").exists()]
    logger.info(f"SERGOM housing: {len(housing)} years loaded "
                f"({min(years_loaded)}-{max(years_loaded)} from files)")
    sample_year = max(years_loaded)
    v = housing[sample_year]
    logger.info(f"  {sample_year}: mean={v[v>0].mean():.1f} units/km², "
                f"max={v.max():.0f}, {(v>0).sum()} pixels with housing")
    return housing


def load_infra_rasters():
    """Load infrastructure distance rasters, convert to km."""
    infra = {}
    for feat_name, (path, scale) in INFRA_RASTERS.items():
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = 0.0
        data = np.maximum(data * scale, 0.0)  # convert units, clamp
        infra[feat_name] = data
        logger.info(f"  {feat_name}: range [{data[data>0].min():.1f}, {data.max():.1f}] km")
    return infra


def build_fveg_broad(store):
    """Map FVEG class IDs to broad categories via LIFEFORM."""
    with open(FVEG_MAP_PATH) as f:
        fveg_map = json.load(f)

    # Build whrnum -> lifeform from VAT
    vat = pd.read_csv(FVEG_VAT_PATH)
    whrnum_to_lifeform = vat.drop_duplicates("WHRNUM").set_index("WHRNUM")["LIFEFORM"].to_dict()

    # class_id -> broad category
    classid_to_broad = {}
    for cid_str, info in fveg_map["id_to_info"].items():
        whrnum = info["whrnum"]
        lifeform = whrnum_to_lifeform.get(whrnum, "OTHER")
        if lifeform in ("CONIFER", "HARDWOOD"):
            classid_to_broad[int(cid_str)] = "forest"
        elif lifeform == "SHRUB":
            classid_to_broad[int(cid_str)] = "shrub"
        elif lifeform == "HERBACEOUS":
            classid_to_broad[int(cid_str)] = "herb"
        else:
            classid_to_broad[int(cid_str)] = "other"

    # Get FVEG raster from zarr static channel 13
    static = np.array(store["inputs/static"])
    fveg_ids = static[13].astype(int)

    # Map to broad
    fveg_broad = np.full((H, W), "other", dtype="U6")
    for cid, cat in classid_to_broad.items():
        fveg_broad[fveg_ids == cid] = cat

    logger.info(f"FVEG broad: forest={np.sum(fveg_broad=='forest')}, "
                f"shrub={np.sum(fveg_broad=='shrub')}, "
                f"herb={np.sum(fveg_broad=='herb')}, "
                f"other={np.sum(fveg_broad=='other')}")
    return fveg_broad


def compute_climatology(data, time_index, clim_end="2016-12"):
    """Compute pixel-month climatology over training period (1984-2016)."""
    months = np.array([int(ym[5:7]) for ym in time_index])
    years = np.array([int(ym[:4]) for ym in time_index])
    clim_mask = (years >= 1984) & (np.array(time_index) <= clim_end)

    clim = np.zeros((12, H, W), dtype=np.float32)
    for m in range(1, 13):
        mask = clim_mask & (months == m)
        if mask.sum() > 0:
            clim[m - 1] = np.nanmean(data[mask], axis=0)
    return clim


def main():
    parser = argparse.ArgumentParser(description="Build fire-ignition panel")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panel"
    panel_dir.mkdir(exist_ok=True)

    panel_path = panel_dir / "fire_panel.parquet"
    if panel_path.exists() and not args.force:
        logger.info("Panel already exists. Use --force to rebuild.")
        return

    # Open zarr
    store = zarr.open_group(ZARR_PATH, mode="r")
    time_index = np.array(store["meta/time"])
    valid_mask = np.array(store["meta/valid_mask"])
    T = len(time_index)

    # Filter to 1984+ (zarr starts at 1980-01, so idx 48 = 1984-01)
    start_idx = np.searchsorted(time_index, "1984-01")
    time_index_fire = time_index[start_idx:]
    T_fire = len(time_index_fire)
    logger.info(f"Fire analysis period: {time_index_fire[0]} to {time_index_fire[-1]} ({T_fire} months)")

    # ---- 1a. Fire rasterization ----
    fire_raster_path = out_dir / "fire_raster.npy"
    if fire_raster_path.exists() and not args.force:
        logger.info("Loading existing fire_raster.npy")
        fire_raster_full = np.load(str(fire_raster_path))
    else:
        gdf = load_frap_fires()
        # Build full-zarr fire raster (537 months) for TSF computation
        fire_raster_full = build_fire_raster(gdf, time_index)

    # Slice to 1984+
    fire_raster = fire_raster_full[start_idx:]

    # ---- 1b. Time since fire ----
    initial_tsf, tsf_init_method = load_tsf_initial()
    logger.info(f"TSF initialization: {tsf_init_method}")

    # Compute TSF on full zarr range (1980-2024) for proper state tracking
    tsf_full, tsf_terminal_state = compute_time_since_fire(
        fire_raster_full, valid_mask, initial_tsf_months=initial_tsf
    )
    tsf = tsf_full[start_idx:]  # slice to 1984+
    tsf_years = tsf / 12.0
    tsf_log = np.log1p(tsf_years)

    # Save TSF states
    np.save(str(out_dir / "tsf_state_2024-09.npy"), tsf_terminal_state)
    if initial_tsf is not None:
        np.save(str(out_dir / "tsf_state_1984-01.npy"), initial_tsf)

    tsf_meta = {
        "reference_date": "2024-09",
        "units": "months",
        "max_value": 600,
        "initialization": tsf_init_method,
        "mean_tsf_months": float(tsf_terminal_state[valid_mask].mean()),
        "pct_never_burned": float((tsf_terminal_state[valid_mask] >= 600).mean()),
    }
    with open(out_dir / "tsf_state_metadata.json", "w") as f:
        json.dump(tsf_meta, f, indent=2)
    logger.info(f"TSF terminal state: mean={tsf_meta['mean_tsf_months']:.0f} months, "
                f"{tsf_meta['pct_never_burned']*100:.1f}% never burned")

    # ---- 1b2. Prescribed burn treatments ----
    broadcast_path = out_dir / "broadcast_raster.npy"
    mechanical_path = out_dir / "mechanical_raster.npy"
    if broadcast_path.exists() and mechanical_path.exists() and not args.force:
        logger.info("Loading existing treatment rasters")
        broadcast_raster = np.load(str(broadcast_path))
        mechanical_raster = np.load(str(mechanical_path))
    else:
        rxburn_gdf = load_rxburn()
        broadcast_raster, mechanical_raster = build_treatment_rasters(rxburn_gdf, time_index)

    logger.info("Computing time since treatment...")
    tst_broadcast_full, tst_b_terminal = compute_time_since_treatment(
        broadcast_raster, valid_mask, cap_years=7
    )
    tst_mechanical_full, tst_m_terminal = compute_time_since_treatment(
        mechanical_raster, valid_mask, cap_years=5
    )

    # Slice to 1984+ and derive features
    tst_broadcast = tst_broadcast_full[start_idx:]
    tst_mechanical = tst_mechanical_full[start_idx:]
    tst_broadcast_years = tst_broadcast / 12.0  # max=7.0
    tst_mechanical_years = tst_mechanical / 12.0  # max=5.0
    any_treatment_5yr = (
        (tst_broadcast <= 60) | (tst_mechanical <= 60)
    ).astype(np.float32)

    # Save terminal states for forecasting
    np.save(str(out_dir / "tst_broadcast_state_2024-09.npy"), tst_b_terminal)
    np.save(str(out_dir / "tst_mechanical_state_2024-09.npy"), tst_m_terminal)
    logger.info(f"Treatment states saved. Broadcast never-treated: "
                f"{(tst_b_terminal[valid_mask] >= 84).mean()*100:.1f}%, "
                f"Mechanical never-treated: {(tst_m_terminal[valid_mask] >= 60).mean()*100:.1f}%")

    # ---- 1c. Sampling ----
    logger.info("Sampling pixel-months...")

    from src.fire_model.sampling import sample_panel

    all_t, all_r, all_c, all_fire = sample_panel(fire_raster, valid_mask, cfg["sampling"])
    n_pos = int(all_fire.sum())
    n_neg = len(all_fire) - n_pos

    # Convert local t (within fire period) to year/month
    all_ym = time_index_fire[all_t]
    all_year = np.array([int(ym[:4]) for ym in all_ym])
    all_month = np.array([int(ym[5:7]) for ym in all_ym])

    logger.info(f"Total panel: {len(all_t)} rows ({n_pos} pos, {n_neg} neg, ratio 1:{n_neg//max(n_pos,1)})")

    # ---- 1d. Feature extraction ----
    logger.info("Extracting features...")

    # Dynamic inputs (raw from zarr)
    dyn = store["inputs/dynamic"]  # (T_zarr, 15, H, W)
    static = np.array(store["inputs/static"])  # (14, H, W)

    # Map local fire-period indices to zarr indices
    zarr_t = all_t + start_idx

    # Extract dynamic features in chunks to manage memory
    chunk_size = 500000
    n_samples = len(all_t)

    features = {}

    # Climate inputs — extract via zarr fancy indexing (chunk by chunk)
    dyn_channels = {"ppt": 0, "tmin": 1, "tmax": 2, "vpd": 9, "srad": 5,
                    "kbdi": 10, "sws": 11, "vpd_roll6_std": 12}

    for feat_name, ch_idx in tqdm(dyn_channels.items(), desc="Dynamic features"):
        arr = np.zeros(n_samples, dtype=np.float32)
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            t_chunk = zarr_t[i:end]
            r_chunk = all_r[i:end]
            c_chunk = all_c[i:end]
            # Read full slices then index (faster than per-pixel)
            t_unique = np.unique(t_chunk)
            data_block = np.array(dyn[t_unique.min():t_unique.max()+1, ch_idx, :, :])
            for j in range(i, end):
                local_t = zarr_t[j] - t_unique.min()
                arr[j] = data_block[local_t, all_r[j], all_c[j]]
        features[feat_name] = arr

    # Targets (Track A) — raw from zarr
    target_vars = {"pet": "pet", "aet": "aet", "cwd": "cwd"}
    for feat_name, tgt_name in tqdm(target_vars.items(), desc="Track A targets"):
        tgt = store[f"targets/{tgt_name}"]
        arr = np.zeros(n_samples, dtype=np.float32)
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            t_chunk = zarr_t[i:end]
            t_unique = np.unique(t_chunk)
            data_block = np.array(tgt[t_unique.min():t_unique.max()+1, :, :])
            for j in range(i, end):
                local_t = zarr_t[j] - t_unique.min()
                arr[j] = data_block[local_t, all_r[j], all_c[j]]
        features[f"{feat_name}_a"] = arr

    # Track B predictions — only for test period, BCMv8 for train/calib
    pred_time = np.load(str(Path(PREDICTIONS_DIR) / "time_index.npy"), allow_pickle=True)
    pred_start_ym = str(pred_time[0])
    pred_ym_to_idx = {str(ym): i for i, ym in enumerate(pred_time)}

    for feat_name, tgt_name in tqdm(target_vars.items(), desc="Track B predictions"):
        pred_data = np.load(str(Path(PREDICTIONS_DIR) / f"{tgt_name}.npy"), mmap_mode="r")
        arr_b = features[f"{feat_name}_a"].copy()  # start with Track A values

        # Overwrite test period with emulator predictions
        for j in range(n_samples):
            ym = all_ym[j]
            if ym in pred_ym_to_idx:
                pred_t = pred_ym_to_idx[ym]
                arr_b[j] = pred_data[pred_t, all_r[j], all_c[j]]

        features[f"{feat_name}_b"] = arr_b

    # Static features
    features["elev"] = static[0, all_r, all_c]
    features["aridity_index"] = static[8, all_r, all_c]
    features["windward_index"] = static[12, all_r, all_c]

    # Infrastructure distance features
    logger.info("Loading infrastructure distance rasters...")
    infra = load_infra_rasters()
    for feat_name, data in infra.items():
        features[feat_name] = data[all_r, all_c]

    # SERGOM housing density (time-varying by year)
    logger.info("Loading SERGOM housing density...")
    unique_years = sorted(set(all_year))
    housing = load_sergom_housing(unique_years)
    housing_arr = np.zeros(n_samples, dtype=np.float32)
    for yr in unique_years:
        yr_mask = all_year == yr
        housing_arr[yr_mask] = housing[yr][all_r[yr_mask], all_c[yr_mask]]
    features["housing_density"] = housing_arr
    features["log_housing_density"] = np.log1p(housing_arr)

    # FVEG broad categories
    fveg_broad = build_fveg_broad(store)
    fveg_vals = fveg_broad[all_r, all_c]
    features["fveg_forest"] = (fveg_vals == "forest").astype(np.float32)
    features["fveg_shrub"] = (fveg_vals == "shrub").astype(np.float32)
    features["fveg_herb"] = (fveg_vals == "herb").astype(np.float32)

    # TSF features
    features["tsf_years"] = tsf_years[all_t, all_r, all_c]
    features["tsf_log"] = tsf_log[all_t, all_r, all_c]

    # Treatment features
    features["tst_broadcast_years"] = tst_broadcast_years[all_t, all_r, all_c]
    features["tst_mechanical_years"] = tst_mechanical_years[all_t, all_r, all_c]
    features["any_treatment_5yr"] = any_treatment_5yr[all_t, all_r, all_c]

    # Seasonal features
    features["month_sin"] = np.sin(2 * np.pi * all_month / 12).astype(np.float32)
    features["month_cos"] = np.cos(2 * np.pi * all_month / 12).astype(np.float32)
    features["fire_season"] = np.isin(all_month, [6, 7, 8, 9, 10, 11]).astype(np.float32)

    # ---- Anomaly features ----
    logger.info("Computing climatologies and anomalies...")

    # CWD climatology from targets (1984-2016)
    cwd_full = np.array(store["targets/cwd"])  # (T_zarr, H, W)
    cwd_clim = compute_climatology(cwd_full, time_index, clim_end=TRAIN_END_YM)
    aet_full = np.array(store["targets/aet"])
    aet_clim = compute_climatology(aet_full, time_index, clim_end=TRAIN_END_YM)
    pet_full = np.array(store["targets/pet"])
    pet_clim = compute_climatology(pet_full, time_index, clim_end=TRAIN_END_YM)

    # Save climatology for forecaster
    np.savez(str(out_dir / "climatology_1984_2016.npz"),
             cwd=cwd_clim, aet=aet_clim, pet=pet_clim)

    # Track A anomalies
    features["cwd_anom_a"] = features["cwd_a"] - cwd_clim[all_month - 1, all_r, all_c]
    features["aet_anom_a"] = features["aet_a"] - aet_clim[all_month - 1, all_r, all_c]
    features["pet_anom_a"] = features["pet_a"] - pet_clim[all_month - 1, all_r, all_c]

    # Track B anomalies (use same climatology — it's from BCMv8 targets)
    features["cwd_anom_b"] = features["cwd_b"] - cwd_clim[all_month - 1, all_r, all_c]
    features["aet_anom_b"] = features["aet_b"] - aet_clim[all_month - 1, all_r, all_c]
    features["pet_anom_b"] = features["pet_b"] - pet_clim[all_month - 1, all_r, all_c]

    # Cumulative CWD anomalies (3-month and 6-month trailing)
    logger.info("Computing cumulative CWD anomalies...")
    cwd_anom_full_a = cwd_full - cwd_clim[
        np.array([int(ym[5:7]) - 1 for ym in time_index]), :, :
    ]

    for suffix, source_full in [("_a", cwd_anom_full_a)]:
        for window, name in [(3, "cwd_cum3_anom"), (6, "cwd_cum6_anom")]:
            cum = np.zeros_like(source_full)
            for w in range(window):
                shifted = np.roll(source_full, w, axis=0)
                shifted[:w] = 0
                cum += shifted
            arr = cum[zarr_t, all_r, all_c]
            features[f"{name}{suffix}"] = arr

    # Track B cumulative — for test period use predictions, otherwise same as A
    pred_cwd = np.load(str(Path(PREDICTIONS_DIR) / "cwd.npy"), mmap_mode="r")
    cwd_anom_full_b = cwd_anom_full_a.copy()
    # Overwrite test period in full array
    test_start_idx = np.searchsorted(time_index, pred_start_ym)
    for t_pred in range(len(pred_time)):
        t_zarr = test_start_idx + t_pred
        m = int(time_index[t_zarr][5:7]) - 1
        cwd_anom_full_b[t_zarr] = pred_cwd[t_pred] - cwd_clim[m]

    for window, name in [(3, "cwd_cum3_anom"), (6, "cwd_cum6_anom")]:
        cum = np.zeros_like(cwd_anom_full_b)
        for w in range(window):
            shifted = np.roll(cwd_anom_full_b, w, axis=0)
            shifted[:w] = 0
            cum += shifted
        features[f"{name}_b"] = cum[zarr_t, all_r, all_c]

    # ---- Build DataFrame ----
    logger.info("Building DataFrame...")

    # Assign splits
    split = np.where(
        np.array(all_ym) <= TRAIN_END_YM, "train",
        np.where(np.array(all_ym) <= CALIB_END_YM, "calib", "test")
    )

    df = pd.DataFrame({
        "year": all_year,
        "month": all_month,
        "row": all_r,
        "col": all_c,
        "fire": all_fire,
        "split": split,
    })
    for feat_name, arr in features.items():
        df[feat_name] = arr

    # Drop raw target columns (keep anomalies and derived)
    for v in ["pet_a", "aet_a", "cwd_a", "pet_b", "aet_b", "cwd_b"]:
        if v in df.columns:
            pass  # keep them — useful for debugging

    logger.info(f"Panel shape: {df.shape}")
    logger.info(f"Split counts:\n{df['split'].value_counts()}")
    logger.info(f"Fire counts by split:\n{df.groupby('split')['fire'].sum()}")
    logger.info(f"NaN check: {df.isna().sum().sum()} total NaNs")

    # Save
    df.to_parquet(str(panel_path), index=False)
    logger.info(f"Saved panel to {panel_path} ({panel_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
