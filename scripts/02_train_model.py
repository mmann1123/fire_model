"""Train fire probability models for Track A (BCMv8) and Track B (emulator).

Supports multiple model types via build_model() factory, optional Optuna tuning,
and isotonic calibration on a held-out period.

Usage:
    conda run -n deep_field python scripts/02_train_model.py
    conda run -n deep_field python scripts/02_train_model.py --model-type lightgbm --tune --n-trials 100
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fire_model.models import (
    EcoregionClassifier,
    CalibratedWithCoords,
    build_model,
    extract_feature_importance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Load configuration ----
with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR = Path((PROJECT_ROOT / cfg["paths"]["output_dir"]).resolve())
PANEL_PATH = OUTPUT_DIR / "panel" / "fire_panel.parquet"

# Build feature lists from config
COMMON_FEATURES = cfg["features"]["common"]

TRACK_A_FEATURES = COMMON_FEATURES + [
    f"{feat}_a" for feat in cfg["features"]["track_specific"]
]

TRACK_B_FEATURES = COMMON_FEATURES + [
    f"{feat}_b" for feat in cfg["features"]["track_specific"]
]


def train_track(df_train, df_calib, features, track_name, cfg_model, seed,
                tune=False, n_trials=100, cv_folds=3, tune_subsample=0.2):
    """Train and calibrate a fire probability model."""
    logger.info(f"Training {track_name} (model: {cfg_model.get('type', 'logistic_regression')})...")

    X_train = df_train[features].values
    y_train = df_train["fire"].values
    X_calib = df_calib[features].values
    y_calib = df_calib["fire"].values

    logger.info(f"  Train: {len(X_train)} samples, {y_train.sum()} positive ({100*y_train.mean():.2f}%)")
    logger.info(f"  Calib: {len(X_calib)} samples, {y_calib.sum()} positive ({100*y_calib.mean():.2f}%)")

    model_type = cfg_model.get("type", "logistic_regression")

    # ---- Optional Optuna tuning ----
    best_params = None
    if tune:
        from src.fire_model.tuning import tune_model
        best_params = tune_model(
            model_type, X_train, y_train, features, cfg_model,
            n_trials=n_trials, cv_folds=cv_folds,
            tune_subsample=tune_subsample, seed=seed,
        )
        # Merge tuned params into config
        tuned_cfg = {**cfg_model, **best_params}
    else:
        tuned_cfg = cfg_model

    # ---- Build and train model ----
    is_ecoregion = model_type == "ecoregion_logreg"

    if is_ecoregion:
        pipeline = build_model(tuned_cfg, features, seed=seed)
        # EcoregionClassifier needs row/col for ecoregion assignment
        rows = df_train["row"].values if "row" in df_train.columns else None
        cols = df_train["col"].values if "col" in df_train.columns else None
        pipeline.fit(X_train, y_train, rows=rows, cols=cols)
        logger.info(f"  EcoregionClassifier fitted")
    else:
        pipeline = build_model(tuned_cfg, features, seed=seed)
        pipeline.fit(X_train, y_train)

        # Log convergence info for linear models
        clf_name = list(pipeline.named_steps.keys())[-1]
        clf = pipeline.named_steps[clf_name]
        if hasattr(clf, "n_iter_"):
            iters = clf.n_iter_[0] if hasattr(clf.n_iter_, '__len__') else clf.n_iter_
            logger.info(f"  Base model fitted (converged in {iters} iterations)")
        else:
            logger.info(f"  Base model fitted")

    # ---- Calibrate on held-out period ----
    if is_ecoregion:
        calibrated = CalibratedWithCoords(pipeline, method="isotonic", cv="prefit")
        calibrated.fit(X_calib, y_calib)
    else:
        calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv="prefit")
        calibrated.fit(X_calib, y_calib)
    logger.info(f"  Calibration fitted on {len(X_calib)} samples")

    # ---- Save model ----
    model_dir = OUTPUT_DIR / "model" / track_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Keep filename as lr_calibrated.pkl for backward compatibility
    with open(model_dir / "lr_calibrated.pkl", "wb") as f:
        pickle.dump(calibrated, f)

    # ---- Save best params if tuning was used ----
    if best_params is not None:
        with open(model_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"  Saved best_params.json")

    # ---- Extract and save feature importance ----
    importance_df = extract_feature_importance(pipeline, features)
    importance_df.to_csv(model_dir / "coefficients.csv", index=False)

    logger.info(f"  Saved model and coefficients to {model_dir}")

    # Print top features
    logger.info(f"  Top 10 features by importance:")
    for _, row in importance_df.head(10).iterrows():
        coef_str = ""
        if "odds_ratio" in importance_df.columns:
            sign = "+" if row["coefficient"] > 0 else ""
            coef_str = f" {sign}{row['coefficient']:.4f}  (OR={row['odds_ratio']:.3f})"
        else:
            coef_str = f" {row['importance']:.4f}"
        logger.info(f"    {row['feature']:25s}{coef_str}")

    # Verify TSF coefficients (for linear models)
    if "coefficient" in importance_df.columns:
        for tsf_feat in ["tsf_years", "tsf_log"]:
            match = importance_df[importance_df["feature"] == tsf_feat]
            if len(match):
                c = match.iloc[0]["coefficient"]
                if c < 0:
                    logger.warning(f"  ⚠ {tsf_feat} has NEGATIVE coefficient ({c:.4f}) — unexpected!")
                else:
                    logger.info(f"  ✓ {tsf_feat} coefficient is positive ({c:.4f})")

        # Verify treatment feature signs
        for feat, expected_sign in [("tst_broadcast_years", "+"), ("tst_mechanical_years", "+"),
                                     ("any_treatment_5yr", "-")]:
            match = importance_df[importance_df["feature"] == feat]
            if len(match):
                c = match.iloc[0]["coefficient"]
                sign_ok = (c > 0 and expected_sign == "+") or (c < 0 and expected_sign == "-")
                if sign_ok:
                    logger.info(f"  ✓ {feat} coefficient sign correct ({c:+.4f})")
                else:
                    logger.warning(f"  ⚠ {feat} has unexpected sign ({c:+.4f}, expected {expected_sign})")

    return calibrated, importance_df


def main():
    parser = argparse.ArgumentParser(description="Train fire probability models")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Override config model type")
    parser.add_argument("--tune", action="store_true",
                        help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials (default: 100)")
    parser.add_argument("--cv-folds", type=int, default=3,
                        help="CV folds for tuning (default: 3)")
    parser.add_argument("--tune-subsample", type=float, default=0.2,
                        help="Fraction of training data for tuning (default: 0.2)")
    args = parser.parse_args()

    cfg_model = dict(cfg["model"])
    if args.model_type:
        cfg_model["type"] = args.model_type
    seed = cfg["sampling"]["seed"]

    logger.info(f"Model type: {cfg_model.get('type', 'logistic_regression')}")
    if args.tune:
        logger.info(f"Optuna tuning: {args.n_trials} trials, {args.cv_folds}-fold CV, "
                     f"{args.tune_subsample:.0%} subsample")

    logger.info(f"Loading panel from {PANEL_PATH}...")
    df = pd.read_parquet(PANEL_PATH)
    logger.info(f"Panel: {len(df)} rows, {df.columns.tolist()}")

    # Verify required columns exist
    for col in TRACK_A_FEATURES + TRACK_B_FEATURES + ["fire", "split"]:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            sys.exit(1)

    df_train = df[df["split"] == "train"]
    df_calib = df[df["split"] == "calib"]
    df_test = df[df["split"] == "test"]

    logger.info(f"Splits: train={len(df_train)}, calib={len(df_calib)}, test={len(df_test)}")

    # Train Track A
    model_a, coefs_a = train_track(
        df_train, df_calib, TRACK_A_FEATURES, "trackA", cfg_model, seed,
        tune=args.tune, n_trials=args.n_trials, cv_folds=args.cv_folds,
        tune_subsample=args.tune_subsample,
    )

    # Train Track B
    model_b, coefs_b = train_track(
        df_train, df_calib, TRACK_B_FEATURES, "trackB", cfg_model, seed,
        tune=args.tune, n_trials=args.n_trials, cv_folds=args.cv_folds,
        tune_subsample=args.tune_subsample,
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
