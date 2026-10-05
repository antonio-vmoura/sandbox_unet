"""Export the Phase 1 (Baseline) validation masks of the U-Net for the side-by-side figure.

Predicts the 100 official validation images with the Phase 1 ``best.pt`` through the Phase 5 path
(:func:`inference.evaluate_ids`: 256 × 256 probability map, bilinear upsampling to the dataset resolution,
threshold 0.5) and writes::

    <project>/phase1_val_masks/unet/masks/<ISIC_ID>.png   # 0/255, dataset resolution
    <project>/phase1_val_masks/unet/per_image.csv         # id + every pixel_scores key
    <project>/phase1_val_masks/unet/meta.json

Read by ``analysis/results_aggregator.py`` (``fig_phase1_side_by_side``). Runs on CPU by default (seconds).
Validation data only; the test set is untouched.

Usage:
    python unet/export_phase1_val_masks.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from common import DEFAULT_CACHE_DIR, DEFAULT_PIPELINE_ROOT, SEED, PipelinePaths, seed_everything, sha256_file
from data import CacheData
from inference import evaluate_ids, resolve_data_root
from segmentation_metrics import aggregate_scores
from training import load_model


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT)
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Phase 0 cache directory.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    seed_everything(SEED)
    torch.set_num_threads(8)
    cache = CacheData(args.cache)
    ids = cache.ids("val")
    assert len(ids) == 100, f"expected the 100 official validation images, found {len(ids)}"
    device = torch.device(args.device)
    weights = PipelinePaths(Path(args.project)).phase1_best_pt("unet")
    out = Path(args.project) / "phase1_val_masks" / "unet"
    rows = evaluate_ids(load_model(weights, device), cache, ids, resolve_data_root(cache), device,
                        mask_dir=out / "masks")
    with (out / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", *rows[0].keys()])
        w.writeheader()
        w.writerows({"id": Path(r["image"]).stem, **r} for r in rows)
    agg = aggregate_scores(rows, seed=SEED)
    (out / "meta.json").write_text(json.dumps({
        "arch": "unet", "model": "unet", "label": "U-Net", "split": "val (official, n=100)",
        "source": "Phase 1 best.pt", "weights": str(weights), "weights_sha256": sha256_file(weights),
        "rule": "probability map upsampled bilinearly, threshold 0.5", "device": args.device,
        "jsi_mean": agg["jsi"]["mean"], "dsc_mean": agg["dsc"]["mean"]}, indent=2))
    print(f"unet: JSI={agg['jsi']['mean']:.4f} DSC={agg['dsc']['mean']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
