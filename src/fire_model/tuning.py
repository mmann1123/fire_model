"""Optuna hyperparameter tuning for fire probability models.

Usage:
    from src.fire_model.tuning import tune_model
    best_params = tune_model("lightgbm", X_train, y_train, df_train, feature_names, cfg_model)
"""

import logging

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Model types that use forward-chaining temporal CV
USES_TEMPORAL_CV = {"lightgbm", "random_forest", "tabnet"}

# Forward-chaining temporal folds over training period (1984-2016)
# Each fold trains on all prior years, validates on a held-out window.
TEMPORAL_FOLDS = [
    (range(1984, 1990), range(1990, 1992)),
    (range(1984, 1996), range(1996, 1998)),
    (range(1984, 2002), range(2002, 2004)),
    (range(1984, 2008), range(2008, 2010)),
    (range(1984, 2014), range(2014, 2017)),
]


def build_temporal_cv_splits(df, folds=TEMPORAL_FOLDS):
    """Forward-chaining temporal CV splits.

    Each fold holds out 2-3 complete water years.
    Preserves temporal ordering — no future data leaks into training.

    Returns list of (train_idx, val_idx) tuples.
    """
    years = df["year"].values
    splits = []

    for train_years, val_years in folds:
        train_mask = np.isin(years, list(train_years))
        val_mask = np.isin(years, list(val_years))

        if val_mask.sum() == 0:
            continue

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        splits.append((train_idx, val_idx))

    return splits


def tune_model(model_type, X_train, y_train, df_train, feature_names, cfg_model,
               n_trials=100, cv_folds=3, tune_subsample=0.2, seed=42):
    """Run Optuna hyperparameter search.

    Parameters
    ----------
    model_type : str
        One of: logistic_regression, elasticnet_logreg, random_forest,
        lightgbm, tabnet, ecoregion_logreg.
    X_train, y_train : arrays
        Training data (tuning uses a stratified subsample for tree/neural models).
    df_train : DataFrame
        Training DataFrame with 'fire', 'year', 'row', 'col' columns.
        Used for temporal CV splits on tree models.
    feature_names : list[str]
        Feature column names.
    cfg_model : dict
        Model config section (for defaults and interactions).
    n_trials : int
        Number of Optuna trials.
    cv_folds : int
        Number of CV folds for evaluation (used by StratifiedKFold only).
    tune_subsample : float
        Fraction of training data to use for tuning (tree/neural models only).
    seed : int
        Random seed.

    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    use_temporal = model_type in USES_TEMPORAL_CV

    # Subsample for expensive models (only for non-temporal CV;
    # temporal CV needs full years to preserve temporal structure)
    use_subsample = (not use_temporal
                     and model_type in ("random_forest", "lightgbm", "tabnet")
                     and tune_subsample < 1.0)
    if use_subsample:
        rng = np.random.RandomState(seed)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        n_pos = int(len(pos_idx) * tune_subsample)
        n_neg = int(len(neg_idx) * tune_subsample)
        sub_idx = np.concatenate([
            rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
        ])
        X_sub = X_train[sub_idx]
        y_sub = y_train[sub_idx]
        df_sub = df_train.iloc[sub_idx].reset_index(drop=True)
        logger.info(f"Tuning with {len(X_sub)} samples ({tune_subsample:.0%} subsample)")
    elif use_temporal and tune_subsample < 1.0:
        # For temporal CV, subsample within each year to preserve temporal structure
        rng = np.random.RandomState(seed)
        keep = []
        for yr in df_train["year"].unique():
            yr_idx = np.where(df_train["year"].values == yr)[0]
            pos_mask = y_train[yr_idx] == 1
            pos_in_yr = yr_idx[pos_mask]
            neg_in_yr = yr_idx[~pos_mask]
            n_pos = max(1, int(len(pos_in_yr) * tune_subsample))
            n_neg = int(len(neg_in_yr) * tune_subsample)
            keep.append(pos_in_yr if len(pos_in_yr) <= n_pos
                        else rng.choice(pos_in_yr, n_pos, replace=False))
            keep.append(rng.choice(neg_in_yr, min(n_neg, len(neg_in_yr)), replace=False))
        sub_idx = np.sort(np.concatenate(keep))
        X_sub = X_train[sub_idx]
        y_sub = y_train[sub_idx]
        df_sub = df_train.iloc[sub_idx].reset_index(drop=True)
        logger.info(f"Tuning with {len(X_sub)} samples "
                    f"({tune_subsample:.0%} stratified subsample per year)")
    else:
        X_sub = X_train
        y_sub = y_train
        df_sub = df_train.reset_index(drop=True)
        logger.info(f"Tuning with full {len(X_sub)} samples")

    # Build CV splits
    if use_temporal:
        splits = build_temporal_cv_splits(df_sub)
        logger.info(f"Using forward-chaining temporal CV ({len(splits)} folds)")
        for fold_i, (tr_idx, val_idx) in enumerate(splits):
            n_pos_val = y_sub[val_idx].sum()
            val_years = sorted(df_sub.iloc[val_idx]["year"].unique())
            logger.info(f"  Fold {fold_i+1}: val years {min(val_years)}-{max(val_years)}, "
                        f"{len(val_idx):,} val rows, {int(n_pos_val):,} positive")
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X_sub, y_sub))
        logger.info(f"Using StratifiedKFold ({cv_folds} folds)")

    def objective(trial):
        from src.fire_model.models import build_model

        params = _suggest_params(trial, model_type, cfg_model)
        trial_cfg = {**cfg_model, **params, "type": model_type}

        fold_aucs = []
        weights = []
        for train_idx, val_idx in splits:
            try:
                model = build_model(trial_cfg, feature_names, seed=seed)
                model.fit(X_sub[train_idx], y_sub[train_idx])
                prob = model.predict_proba(X_sub[val_idx])[:, 1]
                auc = roc_auc_score(y_sub[val_idx], prob)
                n_pos = y_sub[val_idx].sum()
                fold_aucs.append(auc * n_pos)
                weights.append(n_pos)
            except Exception as e:
                logger.warning(f"Trial {trial.number} fold failed: {e}")
                return 0.0

        # Weighted mean AUC (weight by fire count per fold)
        return sum(fold_aucs) / sum(weights) if weights else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner() if model_type == "tabnet" else optuna.pruners.NopPruner(),
    )

    logger.info(f"Starting Optuna: {n_trials} trials, model={model_type}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Best trial: AUC={study.best_value:.4f}")
    logger.info(f"Best params: {best}")

    return best


def _suggest_params(trial, model_type, cfg_model):
    """Suggest hyperparameters for an Optuna trial."""

    if model_type == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 100, log=True),
        }

    elif model_type == "elasticnet_logreg":
        return {
            "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        }

    elif model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_features": trial.suggest_categorical("max_features",
                                                       ["sqrt", "log2", 0.3, 0.5, 0.8]),
        }

    elif model_type == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        }

    elif model_type == "tabnet":
        n_d = trial.suggest_int("n_d", 16, 64, step=8)
        return {
            "n_d": n_d,
            "n_a": n_d,  # keep equal to n_d
            "n_steps": trial.suggest_int("n_steps", 3, 7),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-3, log=True),
            "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2048, 4096, 8192]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        }

    elif model_type == "ecoregion_logreg":
        return {
            "C": trial.suggest_float("C", 1e-4, 100, log=True),
            "min_pos_per_eco": trial.suggest_int("min_pos_per_eco", 200, 1000, step=100),
        }

    else:
        raise ValueError(f"No tuning search space defined for: {model_type}")
