"""Create a frozen snapshot of the current fire model outputs.

Copies model artifacts, metrics, spatial maps, and config into
snapshots/{run-id}/ for reproducibility and experiment tracking.

Usage:
    conda run -n deep_field python scripts/create_snapshot.py --run-id v1-baseline \
        --notes "Logistic regression baseline, all features, 300-acre min"
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# Load config
with open(PROJECT_ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

OUTPUT_DIR = Path((PROJECT_ROOT / cfg["paths"]["output_dir"]).resolve())
SNAPSHOT_DIR = Path((PROJECT_ROOT / cfg["paths"]["snapshot_dir"]).resolve())


def main():
    parser = argparse.ArgumentParser(description="Create experiment snapshot")
    parser.add_argument("--run-id", required=True, help="Snapshot identifier (e.g., v1-baseline)")
    parser.add_argument("--notes", default="", help="Description of this experiment")
    parser.add_argument("--force", action="store_true", help="Overwrite existing snapshot")
    args = parser.parse_args()

    snap_dir = SNAPSHOT_DIR / args.run_id
    if snap_dir.exists():
        if args.force:
            logger.warning(f"Overwriting existing snapshot: {snap_dir}")
            shutil.rmtree(snap_dir)
        else:
            logger.error(f"Snapshot already exists: {snap_dir}. Use --force to overwrite.")
            sys.exit(1)

    snap_dir.mkdir(parents=True)
    logger.info(f"Creating snapshot: {snap_dir}")

    # 1. Copy config
    shutil.copy2(PROJECT_ROOT / "config.yaml", snap_dir / "config.yaml")

    # 2. Copy models
    for track in ["trackA", "trackB"]:
        model_src = OUTPUT_DIR / "model" / track
        if model_src.exists():
            model_dst = snap_dir / "model" / track
            model_dst.mkdir(parents=True)
            for f in model_src.iterdir():
                shutil.copy2(f, model_dst / f.name)

    # 3. Copy evaluation outputs
    eval_src = OUTPUT_DIR / "evaluation"
    if eval_src.exists():
        shutil.copytree(eval_src, snap_dir / "evaluation")

    # 4. Copy comparison outputs
    comp_src = OUTPUT_DIR / "comparison"
    if comp_src.exists():
        shutil.copytree(comp_src, snap_dir / "comparison")

    # 5. Copy spatial maps
    maps_src = OUTPUT_DIR / "spatial_maps"
    if maps_src.exists():
        shutil.copytree(maps_src, snap_dir / "spatial_maps")

    # 6. Copy manifest
    manifest_src = OUTPUT_DIR / "manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, snap_dir / "manifest.json")

    # 7. Create/update manifest with snapshot metadata
    manifest = {}
    if manifest_src.exists():
        with open(manifest_src) as f:
            manifest = json.load(f)

    # Add snapshot metadata
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(PROJECT_ROOT)
        ).strip()[:12]
    except Exception:
        pass

    manifest["snapshot"] = {
        "run_id": args.run_id,
        "notes": args.notes,
        "git_hash": git_hash,
    }

    with open(snap_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    n_files = sum(1 for _ in snap_dir.rglob("*") if _.is_file())
    total_mb = sum(f.stat().st_size for f in snap_dir.rglob("*") if f.is_file()) / 1e6
    logger.info(f"Snapshot created: {n_files} files, {total_mb:.1f} MB")
    logger.info(f"  Run ID: {args.run_id}")
    logger.info(f"  Notes: {args.notes}")
    if "trackA_overall_auc" in manifest:
        logger.info(f"  AUC-A: {manifest['trackA_overall_auc']:.4f}")
        logger.info(f"  AUC-B: {manifest['trackB_overall_auc']:.4f}")
        logger.info(f"  Delta: {manifest['auc_delta']:+.4f}")


if __name__ == "__main__":
    main()
