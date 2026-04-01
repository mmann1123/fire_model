"""Train logistic regression fire models for Track A (BCMv8) and Track B (emulator).

Usage:
    conda run -n deep_field python scripts/02_train_model.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
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


def train_track(df_train, df_calib, features, track_name):
    """Train and calibrate a logistic regression model."""
    logger.info(f"Training {track_name}...")

    X_train = df_train[features].values
    y_train = df_train["fire"].values
    X_calib = df_calib[features].values
    y_calib = df_calib["fire"].values

    logger.info(f"  Train: {len(X_train)} samples, {y_train.sum()} positive ({100*y_train.mean():.2f}%)")
    logger.info(f"  Calib: {len(X_calib)} samples, {y_calib.sum()} positive ({100*y_calib.mean():.2f}%)")

    # Base pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=cfg["model"]["C"],
            class_weight=cfg["model"]["class_weight"],
            max_iter=cfg["model"]["max_iter"],
            solver=cfg["model"]["solver"],
            random_state=cfg["sampling"]["seed"],
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)
    logger.info(f"  Base model fitted (converged in {pipeline['lr'].n_iter_[0]} iterations)")

    # Calibrate on held-out period
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv="prefit")
    calibrated.fit(X_calib, y_calib)
    logger.info(f"  Calibration fitted on {len(X_calib)} samples")

    # Save model
    model_dir = OUTPUT_DIR / "model" / track_name
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "lr_calibrated.pkl", "wb") as f:
        pickle.dump(calibrated, f)

    # Save coefficients
    coefs = pipeline["lr"].coef_[0]
    intercept = pipeline["lr"].intercept_[0]
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    })
    coef_df = coef_df.sort_values("coefficient", ascending=False, key=abs)
    coef_df.to_csv(model_dir / "coefficients.csv", index=False)

    logger.info(f"  Saved model and coefficients to {model_dir}")

    # Print top coefficients
    logger.info(f"  Top 10 features by |coefficient|:")
    for _, row in coef_df.head(10).iterrows():
        sign = "+" if row["coefficient"] > 0 else ""
        logger.info(f"    {row['feature']:25s} {sign}{row['coefficient']:.4f}  (OR={row['odds_ratio']:.3f})")

    # Verify TSF coefficients
    for tsf_feat in ["tsf_years", "tsf_log"]:
        if tsf_feat in features:
            c = coefs[features.index(tsf_feat)]
            if c < 0:
                logger.warning(f"  ⚠ {tsf_feat} has NEGATIVE coefficient ({c:.4f}) — unexpected!")
            else:
                logger.info(f"  ✓ {tsf_feat} coefficient is positive ({c:.4f})")

    # Verify treatment coefficients
    for feat, expected_sign in [("tst_broadcast_years", "+"), ("tst_mechanical_years", "+"),
                                 ("any_treatment_5yr", "-")]:
        if feat in features:
            c = coefs[features.index(feat)]
            sign_ok = (c > 0 and expected_sign == "+") or (c < 0 and expected_sign == "-")
            if sign_ok:
                logger.info(f"  ✓ {feat} coefficient sign correct ({c:+.4f})")
            else:
                logger.warning(f"  ⚠ {feat} has unexpected sign ({c:+.4f}, expected {expected_sign})")

    return calibrated, coef_df


def main():
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
    model_a, coefs_a = train_track(df_train, df_calib, TRACK_A_FEATURES, "trackA")

    # Train Track B
    model_b, coefs_b = train_track(df_train, df_calib, TRACK_B_FEATURES, "trackB")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
