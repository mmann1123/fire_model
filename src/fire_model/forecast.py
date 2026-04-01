"""Forward forecasting module for fire probability.

Maintains time-since-fire state and generates monthly fire probability maps
from BCM emulator outputs using a pre-trained logistic regression model.

Key distinction:
- In 01_build_panel.py (historical): fires reset TSF because they occurred.
- In forecast mode (this module): TSF accumulates without resets.
  This is a probability forecast, not a fire occurrence simulation.

Usage:
    from src.fire_model.forecast import FireProbabilityForecaster

    forecaster = FireProbabilityForecaster(
        model_path="outputs/model/trackB/lr_calibrated.pkl",
        tsf_init_path="outputs/tsf_state_2024-09.npy",
        climatology_path="outputs/climatology_1984_2016.npz",
        zarr_path="/path/to/bcm_dataset.zarr",
    )

    # Each month, pass BCM outputs:
    prob_map = forecaster.step(bcm_outputs={"cwd": ..., "aet": ..., "pet": ...,
                                            "sws": ..., "kbdi": ..., "vpd_roll6_std": ...},
                               month=10, year=2024)
"""

import json
import logging
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import rasterio
import yaml
import zarr

logger = logging.getLogger(__name__)

# Load config for paths
_FORECAST_ROOT = Path(__file__).resolve().parent.parent.parent
_CFG_PATH = _FORECAST_ROOT / "config.yaml"
if _CFG_PATH.exists():
    with open(_CFG_PATH) as _f:
        _CFG = yaml.safe_load(_f)
else:
    _CFG = None


class TimeSinceFireState:
    """Maintains time-since-fire state for deterministic forward forecasting.

    Time accumulates each month without fire resets (deterministic mode).
    """

    def __init__(self, initial_state_months, valid_mask, max_months=600):
        self.state = initial_state_months.astype(np.float32).copy()
        self.state[~valid_mask] = 0.0
        self.valid_mask = valid_mask
        self.max_months = max_months

    def step(self):
        """Return tsf_years and tsf_log for current month BEFORE incrementing.

        Matches causal ordering used in historical computation.
        """
        tsf_years = self.state / 12.0
        tsf_log = np.log1p(tsf_years)

        # Advance state — no fire resets in deterministic mode
        self.state[self.valid_mask] += 1.0
        self.state = np.minimum(self.state, self.max_months)

        return tsf_years, tsf_log

    def get_state(self):
        return self.state.copy()


class FireProbabilityForecaster:
    """Generates monthly fire probability maps from BCM emulator outputs.

    Parameters
    ----------
    model_path : str
        Path to calibrated logistic regression pickle.
    tsf_init_path : str
        Path to TSF state .npy file (months since fire at start of forecast).
    climatology_path : str
        Path to climatology .npz (cwd, aet, pet monthly means from 1984-2016).
    zarr_path : str
        Path to BCM zarr store (for valid_mask and static features).
    """

    # Feature order must match 02_train_model.py TRACK_B_FEATURES
    FEATURE_NAMES = [
        "ppt", "tmin", "tmax", "vpd", "srad", "kbdi", "sws", "vpd_roll6_std",
        "month_sin", "month_cos", "fire_season",
        "elev", "aridity_index", "windward_index",
        "fveg_forest", "fveg_shrub", "fveg_herb",
        "tsf_years", "tsf_log",
        "tst_broadcast_years", "tst_mechanical_years", "any_treatment_5yr",
        "dist_campground_km", "dist_transmission_km", "dist_airbase_km",
        "dist_firestation_km", "dist_road_km",
        "housing_density", "log_housing_density",
        "cwd_anom_b", "aet_anom_b", "pet_anom_b",
        "cwd_cum3_anom_b", "cwd_cum6_anom_b",
    ]

    # Infrastructure distance rasters (static) — loaded from config if available
    if _CFG:
        INFRA_RASTERS = {
            f"{key}_km": (info["path"], info["scale"])
            for key, info in _CFG["infrastructure"].items()
        }
        SERGOM_DIR = _CFG["paths"]["sergom_dir"]
    else:
        INFRA_RASTERS = {
            "dist_campground_km": ("/home/mmann1123/extra_space/Campgrounds/dist_campground.tif", 0.001),
            "dist_transmission_km": ("/home/mmann1123/extra_space/Electrical/transmissionLines_Dist.tif", 1.0),
            "dist_airbase_km": ("/home/mmann1123/extra_space/Fire Stations/AirBaseDist_Meters.tif", 0.001),
            "dist_firestation_km": ("/home/mmann1123/extra_space/Fire Stations/FireStatDist_Meters.tif", 0.001),
            "dist_road_km": ("/home/mmann1123/extra_space/Roads/PrimSecRoads_Dist_km.tif", 1.0),
        }
        SERGOM_DIR = "/home/mmann1123/extra_space/SERGOM_Housing/Interpolated_New"

    def __init__(self, model_path, tsf_init_path, climatology_path, zarr_path,
                 tst_broadcast_init_path=None, tst_mechanical_init_path=None):
        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Verify feature names match (model-agnostic)
        base_model = self.model.estimator if hasattr(self.model, "estimator") else self.model
        if hasattr(base_model, "n_features_in_"):
            n_features = base_model.n_features_in_
        elif hasattr(base_model, "named_steps"):
            # Pipeline — check last step
            clf = list(base_model.named_steps.values())[-1]
            if hasattr(clf, "coef_"):
                n_features = clf.coef_.shape[1]
            elif hasattr(clf, "n_features_in_"):
                n_features = clf.n_features_in_
            else:
                n_features = len(self.FEATURE_NAMES)
        else:
            n_features = len(self.FEATURE_NAMES)
        assert n_features == len(self.FEATURE_NAMES), (
            f"Model expects {n_features} features but forecaster has {len(self.FEATURE_NAMES)}. "
            f"Feature list mismatch between forecast.py and 02_train_model.py."
        )

        # Load zarr metadata
        store = zarr.open_group(zarr_path, mode="r")
        self.valid_mask = np.array(store["meta/valid_mask"])
        self.H, self.W = self.valid_mask.shape
        self.n_valid = self.valid_mask.sum()

        # Static features (pre-extract for valid pixels)
        static = np.array(store["inputs/static"])
        self.elev = static[0]
        self.aridity_index = static[8]
        self.windward_index = static[12]

        # FVEG broad categories
        fveg_map_path = Path(zarr_path).parent / "fveg" / "fveg_class_map.json"
        fveg_vat_path = Path(zarr_path).parent.parent / "fveg" / "fveg_vat.csv"
        # Use config paths if available
        if _CFG:
            fveg_map_path = Path(_CFG["paths"]["fveg_map"])
            fveg_vat_path = Path(_CFG["paths"]["fveg_vat"])
        self.fveg_forest, self.fveg_shrub, self.fveg_herb = self._build_fveg(
            static[13].astype(int), fveg_map_path, fveg_vat_path
        )

        # TSF state
        tsf_init = np.load(tsf_init_path)
        self.tsf_state = TimeSinceFireState(tsf_init, self.valid_mask)

        # Treatment states (broadcast cap=7yr=84mo, mechanical cap=5yr=60mo)
        if tst_broadcast_init_path and Path(tst_broadcast_init_path).exists():
            self.tst_broadcast = TimeSinceFireState(
                np.load(tst_broadcast_init_path), self.valid_mask, max_months=84)
        else:
            # No treatment data — initialize at cap (never treated)
            self.tst_broadcast = TimeSinceFireState(
                np.full((self.H, self.W), 84.0, dtype=np.float32), self.valid_mask, max_months=84)

        if tst_mechanical_init_path and Path(tst_mechanical_init_path).exists():
            self.tst_mechanical = TimeSinceFireState(
                np.load(tst_mechanical_init_path), self.valid_mask, max_months=60)
        else:
            self.tst_mechanical = TimeSinceFireState(
                np.full((self.H, self.W), 60.0, dtype=np.float32), self.valid_mask, max_months=60)

        # Infrastructure distances (static, pre-extract for valid pixels)
        self.infra = {}
        for feat_name, (ipath, scale) in self.INFRA_RASTERS.items():
            with rasterio.open(ipath) as src:
                d = src.read(1).astype(np.float32)
                nd = src.nodata
            if nd is not None:
                d[d == nd] = 0.0
            self.infra[feat_name] = np.maximum(d * scale, 0.0)

        # Climatology
        clim = np.load(climatology_path)
        self.cwd_clim = clim["cwd"]  # (12, H, W)
        self.aet_clim = clim["aet"]
        self.pet_clim = clim["pet"]

        # SERGOM housing density cache
        self._housing_cache = {}

        # Rolling buffer for cumulative CWD anomalies
        self.cwd_anom_buffer = deque(maxlen=6)

        logger.info(f"FireProbabilityForecaster initialized: {self.n_valid} valid pixels")

    @staticmethod
    def _build_fveg(fveg_ids, fveg_map_path, fveg_vat_path):
        """Build FVEG broad category arrays."""
        import json
        import pandas as pd

        with open(fveg_map_path) as f:
            fveg_map = json.load(f)
        vat = pd.read_csv(fveg_vat_path)
        whrnum_to_lf = vat.drop_duplicates("WHRNUM").set_index("WHRNUM")["LIFEFORM"].to_dict()

        H, W = fveg_ids.shape
        forest = np.zeros((H, W), dtype=np.float32)
        shrub = np.zeros((H, W), dtype=np.float32)
        herb = np.zeros((H, W), dtype=np.float32)

        for cid_str, info in fveg_map["id_to_info"].items():
            lf = whrnum_to_lf.get(info["whrnum"], "OTHER")
            mask = fveg_ids == int(cid_str)
            if lf in ("CONIFER", "HARDWOOD"):
                forest[mask] = 1.0
            elif lf == "SHRUB":
                shrub[mask] = 1.0
            elif lf == "HERBACEOUS":
                herb[mask] = 1.0

        return forest, shrub, herb

    def step(self, bcm_outputs, month, year):
        """Generate fire probability map for one month.

        Parameters
        ----------
        bcm_outputs : dict
            Must contain keys: "cwd", "aet", "pet", "ppt", "tmin", "tmax",
            "vpd", "srad", "kbdi", "sws", "vpd_roll6_std".
            Each value is a (H, W) float32 array of raw (denormalized) values.
        month : int
            Calendar month (1-12).
        year : int
            Calendar year.

        Returns
        -------
        prob_map : (H, W) float32
            Calibrated fire probability. Invalid pixels = -9999.0.
        """
        m_idx = month - 1  # 0-based month index

        # TSF features (before incrementing)
        tsf_years, tsf_log = self.tsf_state.step()

        # Housing density for this year (from SERGOM projections)
        if year not in self._housing_cache:
            p = Path(self.SERGOM_DIR) / f"bhc{year}.tif"
            if p.exists():
                with rasterio.open(str(p)) as src:
                    self._housing_cache[year] = np.maximum(src.read(1).astype(np.float32), 0.0)
            else:
                logger.warning(f"SERGOM housing not found for {year}, using zeros")
                self._housing_cache[year] = np.zeros((self.H, self.W), dtype=np.float32)
        housing = self._housing_cache[year]

        # Treatment features (before incrementing — deterministic, no resets)
        tst_b_years, _ = self.tst_broadcast.step()
        tst_m_years, _ = self.tst_mechanical.step()
        any_treat = ((tst_b_years <= 5) | (tst_m_years <= 5)).astype(np.float32)

        # Anomalies
        cwd_anom = bcm_outputs["cwd"] - self.cwd_clim[m_idx]
        aet_anom = bcm_outputs["aet"] - self.aet_clim[m_idx]
        pet_anom = bcm_outputs["pet"] - self.pet_clim[m_idx]

        # Update CWD anomaly buffer and compute cumulative
        self.cwd_anom_buffer.append(cwd_anom.copy())
        cwd_cum3 = sum(list(self.cwd_anom_buffer)[-3:])
        cwd_cum6 = sum(list(self.cwd_anom_buffer))

        # Seasonal
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        fire_season = 1.0 if month in (6, 7, 8, 9, 10, 11) else 0.0

        # Assemble feature array for valid pixels
        valid_idx = np.where(self.valid_mask)
        n = self.n_valid

        X = np.column_stack([
            bcm_outputs["ppt"][valid_idx],
            bcm_outputs["tmin"][valid_idx],
            bcm_outputs["tmax"][valid_idx],
            bcm_outputs["vpd"][valid_idx],
            bcm_outputs["srad"][valid_idx],
            bcm_outputs["kbdi"][valid_idx],
            bcm_outputs["sws"][valid_idx],
            bcm_outputs["vpd_roll6_std"][valid_idx],
            np.full(n, month_sin),
            np.full(n, month_cos),
            np.full(n, fire_season),
            self.elev[valid_idx],
            self.aridity_index[valid_idx],
            self.windward_index[valid_idx],
            self.fveg_forest[valid_idx],
            self.fveg_shrub[valid_idx],
            self.fveg_herb[valid_idx],
            tsf_years[valid_idx],
            tsf_log[valid_idx],
            tst_b_years[valid_idx],
            tst_m_years[valid_idx],
            any_treat[valid_idx],
            self.infra["dist_campground_km"][valid_idx],
            self.infra["dist_transmission_km"][valid_idx],
            self.infra["dist_airbase_km"][valid_idx],
            self.infra["dist_firestation_km"][valid_idx],
            self.infra["dist_road_km"][valid_idx],
            housing[valid_idx],
            np.log1p(housing[valid_idx]),
            cwd_anom[valid_idx],
            aet_anom[valid_idx],
            pet_anom[valid_idx],
            cwd_cum3[valid_idx],
            cwd_cum6[valid_idx],
        ])

        # Predict
        probs = self.model.predict_proba(X)[:, 1]

        # Fill output map
        prob_map = np.full((self.H, self.W), -9999.0, dtype=np.float32)
        prob_map[valid_idx] = probs

        return prob_map

    def save_state(self, path):
        """Save current TSF state for resuming forecasts."""
        np.save(path, self.tsf_state.get_state())
