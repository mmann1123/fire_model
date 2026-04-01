"""Model registry for fire probability models.

Provides a uniform sklearn-compatible interface for multiple model types:
- logistic_regression: StandardScaler + LogisticRegression
- elasticnet_logreg: FeatureTransformer + StandardScaler + SGDClassifier(elasticnet)
- random_forest: FeatureTransformer + RandomForestClassifier
- lightgbm: FeatureTransformer + LGBMClassifier (GPU)
- tabnet: FeatureTransformer + StandardScaler + TabNetWrapper (GPU)
- ecoregion_logreg: EcoregionClassifier (per-ecoregion LogReg)

All models are returned as sklearn Pipelines so that `model.predict_proba(X)[:, 1]`
works identically in 03_evaluate.py and forecast.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FeatureTransformer — serialized inside pipeline pickle
# ---------------------------------------------------------------------------

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn transformer that computes interaction features.

    Interactions are specified as (col_a, col_b, name) tuples.
    Bare names (e.g. 'cwd_cum6_anom') are resolved to track-specific
    suffixed names ('cwd_cum6_anom_a' or '_b') based on feature_names.
    """

    def __init__(self, feature_names, interactions=None):
        self.feature_names = list(feature_names)
        self.interactions = interactions or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.interactions:
            return X
        idx = {n: i for i, n in enumerate(self.feature_names)}
        extras = []
        for col_a, col_b, name in self.interactions:
            a = self._resolve(col_a, idx)
            b = self._resolve(col_b, idx)
            extras.append((X[:, a] * X[:, b]).reshape(-1, 1))
        return np.hstack([X] + extras)

    def _resolve(self, col, idx):
        """Resolve bare name to track-specific suffix if needed."""
        if col in idx:
            return idx[col]
        # Try _a and _b suffixes
        for suffix in ("_a", "_b"):
            if col + suffix in idx:
                return idx[col + suffix]
        raise KeyError(
            f"Feature '{col}' not found in feature list "
            f"(also tried '{col}_a', '{col}_b')"
        )

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names)
        for _, _, name in self.interactions:
            names.append(name)
        return names


# ---------------------------------------------------------------------------
# TabNetWrapper — sklearn-compatible wrapper
# ---------------------------------------------------------------------------

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible wrapper around pytorch_tabnet.TabNetClassifier.

    Handles config translation, GPU selection, class imbalance via sample
    weights, and feature_importances_ exposure.
    """

    def __init__(self, n_d=32, n_a=32, n_steps=5, gamma=1.5,
                 lambda_sparse=1e-4, lr=2e-3, batch_size=4096,
                 max_epochs=200, patience=20, dropout=0.1,
                 momentum=0.02, device_name="cuda", seed=42,
                 eval_set=None):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.dropout = dropout
        self.momentum = momentum
        self.device_name = device_name
        self.seed = seed
        self.eval_set = eval_set  # (X_val, y_val) for early stopping
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, eval_set=None):
        from pytorch_tabnet.tab_model import TabNetClassifier

        self.model_ = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a,
            n_steps=self.n_steps, gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_params={"lr": self.lr},
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=None,
            mask_type="sparsemax",
            seed=self.seed,
            device_name=self.device_name,
            verbose=1,
        )

        # Compute sample weights for class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        weights = np.where(y == 1, n_neg / max(n_pos, 1), 1.0).astype(np.float32)

        # Use provided eval_set or self.eval_set
        es = eval_set or self.eval_set
        fit_kwargs = {
            "X_train": X.astype(np.float32),
            "y_train": y.astype(np.int64),
            "weights": weights,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "drop_last": False,
        }
        if es is not None:
            X_val, y_val = es
            fit_kwargs["eval_set"] = [(X_val.astype(np.float32), y_val.astype(np.int64))]
            fit_kwargs["eval_name"] = ["val"]
            fit_kwargs["eval_metric"] = ["auc"]

        self.model_.fit(**fit_kwargs)
        self.feature_importances_ = self.model_.feature_importances_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X.astype(np.float32))

    def predict(self, X):
        return self.model_.predict(X.astype(np.float32))


# ---------------------------------------------------------------------------
# EcoregionClassifier — per-ecoregion LogReg
# ---------------------------------------------------------------------------

class CalibratedWithCoords(BaseEstimator, ClassifierMixin):
    """CalibratedClassifierCV wrapper that forwards pixel_indices kwarg."""

    def __init__(self, base_estimator, method="isotonic", cv="prefit"):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.calibrator_ = CalibratedClassifierCV(
            self.base_estimator, method=self.method, cv=self.cv
        )
        self.calibrator_.fit(X, y)
        return self

    def predict_proba(self, X, **kwargs):
        # Forward kwargs to base estimator if it supports them
        # CalibratedClassifierCV doesn't forward kwargs, so for ecoregion
        # models we need to handle this specially
        if hasattr(self.base_estimator, "needs_pixel_indices") and "pixel_indices" in kwargs:
            # Store pixel_indices on the base estimator temporarily
            self.base_estimator._pixel_indices = kwargs["pixel_indices"]
        return self.calibrator_.predict_proba(X)

    def predict(self, X, **kwargs):
        proba = self.predict_proba(X, **kwargs)
        return np.argmax(proba, axis=1)

    @property
    def needs_pixel_indices(self):
        return hasattr(self.base_estimator, "needs_pixel_indices")


class EcoregionClassifier(BaseEstimator, ClassifierMixin):
    """Fits separate LogisticRegression models per L3 ecoregion.

    Tests the hypothesis that spatial heterogeneity in fire drivers is a
    larger limitation than nonlinearity.

    During predict_proba, if pixel_indices kwarg is provided, samples are
    routed to their ecoregion's model. Otherwise, the global model is used.
    """

    needs_pixel_indices = True

    def __init__(self, ecoregion_tif, C=1.0, class_weight="balanced",
                 max_iter=2000, min_pos_per_eco=500, seed=42):
        self.ecoregion_tif = ecoregion_tif
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.min_pos_per_eco = min_pos_per_eco
        self.seed = seed
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, rows=None, cols=None):
        """Fit per-ecoregion models.

        Parameters
        ----------
        X : array (n_samples, n_features)
        y : array (n_samples,)
        rows, cols : arrays of pixel row/col indices for ecoregion assignment
        """
        import rasterio
        from rasterio.warp import Resampling, reproject

        # Load and store ecoregion raster
        with rasterio.open(self.ecoregion_tif) as src:
            eco_raw = src.read(1)
            src_transform = src.transform
            src_crs = src.crs

        # Reproject to BCM grid if needed
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        from rasterio.transform import Affine
        H = cfg["grid"]["height"]
        W = cfg["grid"]["width"]
        tx = cfg["grid"]["transform"]
        dst_transform = Affine(tx[0], tx[1], tx[2], tx[3], tx[4], tx[5])

        self.eco_raster_ = np.full((H, W), 0, dtype=np.int32)
        reproject(
            source=eco_raw,
            destination=self.eco_raster_,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:3310",
            resampling=Resampling.nearest,
        )

        self.n_features_in_ = X.shape[1]

        # Fit global model as fallback
        self.global_scaler_ = StandardScaler()
        self.global_model_ = LogisticRegression(
            C=self.C, class_weight=self.class_weight,
            max_iter=self.max_iter, solver="lbfgs",
            random_state=self.seed, n_jobs=-1,
        )
        X_scaled = self.global_scaler_.fit_transform(X)
        self.global_model_.fit(X_scaled, y)
        logger.info(f"  Global model: {len(X)} samples, {y.sum()} positive")

        # Fit per-ecoregion models
        self.eco_models_ = {}
        self.eco_scalers_ = {}
        self.eco_ids_ = []

        if rows is not None and cols is not None:
            eco_labels = self.eco_raster_[rows, cols]
        else:
            logger.warning("No row/col provided for ecoregion assignment, using global model only")
            return self

        unique_ecos = np.unique(eco_labels)
        for eco_id in unique_ecos:
            if eco_id == 0:  # nodata
                continue
            mask = eco_labels == eco_id
            y_eco = y[mask]
            n_pos = y_eco.sum()
            if n_pos < self.min_pos_per_eco:
                logger.info(f"  Ecoregion {eco_id}: {n_pos} positives < {self.min_pos_per_eco}, using global fallback")
                continue

            X_eco = X[mask]
            scaler = StandardScaler()
            X_eco_scaled = scaler.fit_transform(X_eco)
            model = LogisticRegression(
                C=self.C, class_weight=self.class_weight,
                max_iter=self.max_iter, solver="lbfgs",
                random_state=self.seed, n_jobs=-1,
            )
            model.fit(X_eco_scaled, y_eco)
            self.eco_models_[eco_id] = model
            self.eco_scalers_[eco_id] = scaler
            self.eco_ids_.append(eco_id)
            logger.info(f"  Ecoregion {eco_id}: {len(X_eco)} samples, {n_pos} positive")

        logger.info(f"  Fitted {len(self.eco_models_)} ecoregion models + 1 global fallback")
        return self

    def predict_proba(self, X, **kwargs):
        """Predict probabilities, routing to ecoregion models if possible."""
        pixel_indices = kwargs.get("pixel_indices", None)

        # Check if pixel_indices was stashed by CalibratedWithCoords
        if pixel_indices is None and hasattr(self, "_pixel_indices"):
            pixel_indices = self._pixel_indices
            del self._pixel_indices

        proba = np.zeros((X.shape[0], 2), dtype=np.float64)

        if pixel_indices is not None and len(self.eco_models_) > 0:
            rows, cols = pixel_indices
            eco_labels = self.eco_raster_[rows, cols]

            # Default: global model for all
            X_global = self.global_scaler_.transform(X)
            proba[:] = self.global_model_.predict_proba(X_global)

            # Override with ecoregion-specific predictions
            for eco_id, model in self.eco_models_.items():
                mask = eco_labels == eco_id
                if mask.sum() == 0:
                    continue
                X_eco = self.eco_scalers_[eco_id].transform(X[mask])
                proba[mask] = model.predict_proba(X_eco)
        else:
            # No pixel indices — use global model
            X_global = self.global_scaler_.transform(X)
            proba[:] = self.global_model_.predict_proba(X_global)

        return proba

    def predict(self, X, **kwargs):
        proba = self.predict_proba(X, **kwargs)
        return np.argmax(proba, axis=1)


# ---------------------------------------------------------------------------
# build_model — factory function
# ---------------------------------------------------------------------------

def build_model(cfg_model, feature_names, seed=42):
    """Build an sklearn Pipeline for the specified model type.

    Parameters
    ----------
    cfg_model : dict
        Model config section from config.yaml.
    feature_names : list[str]
        Feature column names (for FeatureTransformer).
    seed : int
        Random seed.

    Returns
    -------
    pipeline : sklearn Pipeline or EcoregionClassifier
        Model with .fit() and .predict_proba() interface.
    """
    model_type = cfg_model.get("type", "logistic_regression")
    interactions = cfg_model.get("interactions", [])
    # Convert interaction lists to tuples
    interactions = [(a, b, name) for a, b, name in interactions] if interactions else []

    transformer = FeatureTransformer(feature_names, interactions)

    if model_type == "logistic_regression":
        steps = [
            ("transformer", transformer),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=cfg_model.get("C", 1.0),
                class_weight=cfg_model.get("class_weight", "balanced"),
                max_iter=cfg_model.get("max_iter", 2000),
                solver=cfg_model.get("solver", "lbfgs"),
                random_state=seed,
                n_jobs=-1,
            )),
        ]
        return Pipeline(steps)

    elif model_type == "elasticnet_logreg":
        steps = [
            ("transformer", transformer),
            ("scaler", StandardScaler()),
            ("lr", SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=cfg_model.get("alpha", 0.001),
                l1_ratio=cfg_model.get("l1_ratio", 0.5),
                class_weight=cfg_model.get("class_weight", "balanced"),
                max_iter=cfg_model.get("max_iter", 2000),
                random_state=seed,
                n_jobs=-1,
            )),
        ]
        return Pipeline(steps)

    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        steps = [
            ("transformer", transformer),
            ("rf", RandomForestClassifier(
                n_estimators=cfg_model.get("n_estimators", 500),
                max_depth=cfg_model.get("max_depth", 15),
                min_samples_leaf=cfg_model.get("min_samples_leaf", 20),
                max_features=cfg_model.get("max_features", "sqrt"),
                class_weight=cfg_model.get("class_weight", "balanced"),
                random_state=seed,
                n_jobs=-1,
            )),
        ]
        return Pipeline(steps)

    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        steps = [
            ("transformer", transformer),
            ("lgbm", LGBMClassifier(
                num_leaves=cfg_model.get("num_leaves", 63),
                learning_rate=cfg_model.get("learning_rate", 0.05),
                n_estimators=cfg_model.get("n_estimators", 1000),
                min_child_samples=cfg_model.get("min_child_samples", 20),
                reg_alpha=cfg_model.get("reg_alpha", 0.1),
                reg_lambda=cfg_model.get("reg_lambda", 0.1),
                subsample=cfg_model.get("subsample", 0.8),
                colsample_bytree=cfg_model.get("colsample_bytree", 0.8),
                class_weight=cfg_model.get("class_weight", "balanced"),
                device=cfg_model.get("device", "gpu"),
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )),
        ]
        return Pipeline(steps)

    elif model_type == "tabnet":
        steps = [
            ("transformer", transformer),
            ("scaler", StandardScaler()),
            ("tabnet", TabNetWrapper(
                n_d=cfg_model.get("n_d", 32),
                n_a=cfg_model.get("n_a", 32),
                n_steps=cfg_model.get("n_steps", 5),
                gamma=cfg_model.get("gamma", 1.5),
                lambda_sparse=cfg_model.get("lambda_sparse", 1e-4),
                lr=cfg_model.get("lr", 2e-3),
                batch_size=cfg_model.get("batch_size", 4096),
                max_epochs=cfg_model.get("max_epochs", 200),
                patience=cfg_model.get("patience", 20),
                dropout=cfg_model.get("dropout", 0.1),
                momentum=cfg_model.get("momentum", 0.02),
                device_name=cfg_model.get("device_name", "cuda"),
                seed=seed,
            )),
        ]
        return Pipeline(steps)

    elif model_type == "ecoregion_logreg":
        return EcoregionClassifier(
            ecoregion_tif=cfg_model.get("ecoregion_tif",
                                         "/home/mmann1123/extra_space/Regions/ca_eco_l3.tif"),
            C=cfg_model.get("C", 1.0),
            class_weight=cfg_model.get("class_weight", "balanced"),
            max_iter=cfg_model.get("max_iter", 2000),
            min_pos_per_eco=cfg_model.get("min_pos_per_eco", 500),
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def extract_feature_importance(pipeline, feature_names):
    """Extract feature importance from a fitted pipeline.

    Parameters
    ----------
    pipeline : sklearn Pipeline or EcoregionClassifier
        Fitted model.
    feature_names : list[str]
        Original feature names (before interaction expansion).

    Returns
    -------
    df : DataFrame with columns (feature, importance, importance_type, coefficient)
    """
    # Determine expanded feature names (including interactions)
    if hasattr(pipeline, "named_steps") and "transformer" in pipeline.named_steps:
        transformer = pipeline.named_steps["transformer"]
        expanded_names = transformer.get_feature_names_out()
    elif isinstance(pipeline, EcoregionClassifier):
        expanded_names = list(feature_names)
    else:
        expanded_names = list(feature_names)

    # Extract importances based on model type
    if isinstance(pipeline, EcoregionClassifier):
        # Mean |coefficient| across ecoregion models
        all_coefs = []
        if pipeline.eco_models_:
            for eco_id, model in pipeline.eco_models_.items():
                all_coefs.append(np.abs(model.coef_[0]))
            importances = np.mean(all_coefs, axis=0)
        else:
            importances = np.abs(pipeline.global_model_.coef_[0])
        importance_type = "mean_abs_coefficient"
        coefficients = pipeline.global_model_.coef_[0]

    elif hasattr(pipeline, "named_steps"):
        # Pipeline — find the classifier step
        clf_name, clf = list(pipeline.named_steps.items())[-1]

        if hasattr(clf, "coef_"):
            # Linear model (LogReg, SGD)
            importances = np.abs(clf.coef_[0])
            importance_type = "abs_coefficient"
            coefficients = clf.coef_[0]
        elif hasattr(clf, "feature_importances_"):
            # Tree-based or TabNet
            importances = clf.feature_importances_
            importance_type = "feature_importance"
            coefficients = importances  # No directional coef for trees
        else:
            logger.warning(f"Cannot extract importance from {type(clf)}")
            importances = np.zeros(len(expanded_names))
            importance_type = "unknown"
            coefficients = np.zeros(len(expanded_names))
    else:
        logger.warning(f"Cannot extract importance from {type(pipeline)}")
        importances = np.zeros(len(expanded_names))
        importance_type = "unknown"
        coefficients = np.zeros(len(expanded_names))

    df = pd.DataFrame({
        "feature": expanded_names[:len(importances)],
        "importance": importances,
        "importance_type": importance_type,
        "coefficient": coefficients,
    })

    # Add odds_ratio for linear models
    if importance_type in ("abs_coefficient", "mean_abs_coefficient"):
        df["odds_ratio"] = np.exp(df["coefficient"])

    df = df.sort_values("importance", ascending=False, key=abs)
    return df
