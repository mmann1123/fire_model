"""Modular sampling strategies for fire panel construction.

Each strategy selects positive and negative pixel-months from a fire raster.
Positives are always a 100% census of burned pixel-months. Strategies differ
only in how negatives are selected.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sample_panel(fire_raster, valid_mask, config):
    """Sample positive and negative pixel-months for fire panel.

    Args:
        fire_raster: (T, H, W) uint8 binary fire raster
        valid_mask: (H, W) bool mask of valid pixels
        config: cfg["sampling"] dict from config.yaml

    Returns:
        (all_t, all_r, all_c, all_fire) — index arrays + labels
    """
    strategy = config.get("strategy", "grid_thin")
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    # Positives: 100% census of burned pixel-months
    pos_t, pos_r, pos_c = np.where((fire_raster == 1) & valid_mask[np.newaxis, :, :])
    n_pos = len(pos_t)
    logger.info(f"Positive samples: {n_pos}")

    # Dispatch to negative sampling strategy
    dispatch = {
        "grid_thin": _negatives_grid_thin,
        "matched_ratio": _negatives_matched_ratio,
        "temporal_thin": _negatives_temporal_thin,
        "random_subsample": _negatives_random_subsample,
    }
    if strategy not in dispatch:
        raise ValueError(
            f"Unknown sampling strategy '{strategy}'. "
            f"Choose from: {list(dispatch.keys())}"
        )

    neg_t, neg_r, neg_c = dispatch[strategy](fire_raster, valid_mask, config, rng)
    n_neg = len(neg_t)
    logger.info(f"Negative samples: {n_neg} (strategy={strategy})")

    # Combine
    all_t = np.concatenate([pos_t, neg_t])
    all_r = np.concatenate([pos_r, neg_r])
    all_c = np.concatenate([pos_c, neg_c])
    all_fire = np.concatenate([
        np.ones(n_pos, dtype=np.int8),
        np.zeros(n_neg, dtype=np.int8),
    ])

    return all_t, all_r, all_c, all_fire


def _negatives_grid_thin(fire_raster, valid_mask, config, rng):
    """Spatially thinned grid, all non-fire months. Original behavior."""
    spacing = config.get("grid_thin", {}).get("neg_grid_spacing", config.get("neg_grid_spacing", 5))
    H, W = valid_mask.shape

    neg_rows = np.arange(0, H, spacing)
    neg_cols = np.arange(0, W, spacing)
    neg_rr, neg_cc = np.meshgrid(neg_rows, neg_cols, indexing="ij")
    neg_rr, neg_cc = neg_rr.ravel(), neg_cc.ravel()

    # Filter to valid pixels
    neg_valid = valid_mask[neg_rr, neg_cc]
    neg_rr = neg_rr[neg_valid]
    neg_cc = neg_cc[neg_valid]
    logger.info(f"Negative grid: {len(neg_rr)} pixels (every {spacing}th)")

    # All non-fire months for negative pixels
    neg_fire_subset = fire_raster[:, neg_rr, neg_cc]  # (T, n_neg_pixels)
    neg_t_idx, neg_p_idx = np.where(neg_fire_subset == 0)
    neg_r_out = neg_rr[neg_p_idx]
    neg_c_out = neg_cc[neg_p_idx]

    return neg_t_idx, neg_r_out, neg_c_out


def _negatives_matched_ratio(fire_raster, valid_mask, config, rng):
    """Random subsample of negatives to achieve target_ratio negatives per positive."""
    params = config.get("matched_ratio", {})
    target_ratio = params.get("target_ratio", 10)

    n_pos = int(((fire_raster == 1) & valid_mask[np.newaxis]).sum())
    n_target = n_pos * target_ratio

    # Use grid_thin to get the full candidate pool, then subsample
    # This avoids materializing all ~122M candidates at once
    all_neg_t, all_neg_r, all_neg_c = _negatives_grid_thin(
        fire_raster, valid_mask,
        {"grid_thin": {"neg_grid_spacing": 1}, "neg_grid_spacing": 1},
        rng,
    )

    n_available = len(all_neg_t)
    if n_target >= n_available:
        logger.warning(
            f"Requested {n_target} negatives but only {n_available} available. "
            f"Using all available."
        )
        return all_neg_t, all_neg_r, all_neg_c

    # Chunked approach: generate candidate pool from valid pixels in chunks
    # to avoid memory issues with very large grids
    idx = rng.choice(n_available, size=n_target, replace=False)
    idx.sort()  # preserve temporal ordering
    return all_neg_t[idx], all_neg_r[idx], all_neg_c[idx]


def _negatives_temporal_thin(fire_raster, valid_mask, config, rng):
    """Grid-thin spatially + keep every Nth month for negatives."""
    params = config.get("temporal_thin", {})
    spacing = params.get("neg_grid_spacing", 5)
    month_step = params.get("month_step", 3)
    H, W = valid_mask.shape

    neg_rows = np.arange(0, H, spacing)
    neg_cols = np.arange(0, W, spacing)
    neg_rr, neg_cc = np.meshgrid(neg_rows, neg_cols, indexing="ij")
    neg_rr, neg_cc = neg_rr.ravel(), neg_cc.ravel()

    neg_valid = valid_mask[neg_rr, neg_cc]
    neg_rr = neg_rr[neg_valid]
    neg_cc = neg_cc[neg_valid]
    logger.info(f"Temporal-thin grid: {len(neg_rr)} pixels (every {spacing}th), month_step={month_step}")

    # Only keep months at step intervals
    T = fire_raster.shape[0]
    keep_months = np.arange(0, T, month_step)
    neg_fire_subset = fire_raster[keep_months][:, neg_rr, neg_cc]  # (len(keep_months), n_pix)
    local_t_idx, neg_p_idx = np.where(neg_fire_subset == 0)
    neg_t_out = keep_months[local_t_idx]
    neg_r_out = neg_rr[neg_p_idx]
    neg_c_out = neg_cc[neg_p_idx]

    return neg_t_out, neg_r_out, neg_c_out


def _negatives_random_subsample(fire_raster, valid_mask, config, rng):
    """Probabilistic subsampling: keep each valid negative pixel-month with probability `fraction`."""
    params = config.get("random_subsample", {})
    fraction = params.get("fraction", 0.01)

    T = fire_raster.shape[0]
    valid_rows, valid_cols = np.where(valid_mask)
    n_valid = len(valid_rows)

    logger.info(f"Random subsample: fraction={fraction}, valid pixels={n_valid}, T={T}")

    # Process in chunks of pixels to avoid materializing all T*n_valid candidates
    chunk_size = 5000
    neg_t_parts = []
    neg_r_parts = []
    neg_c_parts = []

    for start in range(0, n_valid, chunk_size):
        end = min(start + chunk_size, n_valid)
        rows_chunk = valid_rows[start:end]
        cols_chunk = valid_cols[start:end]
        n_pix = end - start

        # Get fire status for this chunk: (T, n_pix)
        fire_chunk = fire_raster[:, rows_chunk, cols_chunk]

        # Find non-fire pixel-months
        t_idx, p_idx = np.where(fire_chunk == 0)

        # Probabilistic keep
        keep = rng.random(len(t_idx)) < fraction
        t_keep = t_idx[keep]
        p_keep = p_idx[keep]

        neg_t_parts.append(t_keep)
        neg_r_parts.append(rows_chunk[p_keep])
        neg_c_parts.append(cols_chunk[p_keep])

    return (
        np.concatenate(neg_t_parts),
        np.concatenate(neg_r_parts),
        np.concatenate(neg_c_parts),
    )
