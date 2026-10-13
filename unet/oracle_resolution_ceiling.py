"""Resolution ceiling of the 256 × 256 U-Net — an oracle through exactly the U-Net prediction path.

For every image of a split, the **reference 256 × 256 mask of the Phase 0 cache** (the network's training target:
official mask area-resized to 256 × 256 and thresholded at 0.5) is used as if it were a perfect probability map and
sent through the U-Net inference path (:func:`inference.predict_full_res`: bilinear upsampling to the dataset
resolution with ``align_corners=False``, threshold ``PROB_THRESHOLD`` = 0.5). The result is scored against the
**official mask at the dataset resolution** with the shared :func:`segmentation_metrics.pixel_scores` — the same ground
truth and metrics as Phase 5. The scores are the best a perfect 256 × 256 U-Net could reach: the accuracy lost to the
input resolution alone. A one-pixel shift or a flip anywhere in the path would also show up here as a large drop.

Outputs (CPU only, seconds)::

    <project>/oracle_ceiling/<split>_per_image.csv
    <project>/oracle_ceiling/<split>_summary.json      # per-image means with the seeded bootstrap CI, medians

Usage:
    python unet/oracle_resolution_ceiling.py --split test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import DEFAULT_CACHE_DIR, DEFAULT_PIPELINE_ROOT, SEED, atomic_write_json, seed_everything, utc_now_iso
from data import CacheData
from inference import PROB_THRESHOLD, resolve_data_root
from segmentation_metrics import SCORE_KEYS, aggregate_scores, ground_truth_mask, pixel_scores


def oracle_mask(mask256: np.ndarray, h: int, w: int) -> np.ndarray:
    """The 256 × 256 reference mask through the U-Net upsampling of ``inference.predict_full_res``."""
    prob = torch.from_numpy((mask256 > 0).astype(np.float32))[None, None]
    up = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    return (up[0, 0] > PROB_THRESHOLD).numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Phase 0 cache directory.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root (outputs in oracle_ceiling/).")
    args = p.parse_args()
    seed_everything(SEED)
    cache = CacheData(args.cache)
    data_root = resolve_data_root(cache)
    ids = cache.ids(args.split)
    _, masks = cache.arrays(ids)
    rows = []
    for i, iid in enumerate(ids):
        rec = cache.record(iid)
        h, w = int(rec["orig_h"]), int(rec["orig_w"])
        gt = ground_truth_mask(data_root / rec["image"], h, w)
        rows.append({"id": iid, "height": h, "width": w, **pixel_scores(gt, oracle_mask(masks[i], h, w))})
        if (i + 1) % 250 == 0 or i + 1 == len(ids):
            print(f"  {i + 1}/{len(ids)}", flush=True)
    out = Path(args.project) / "oracle_ceiling"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.split}_per_image.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    agg = aggregate_scores(rows, seed=SEED)
    summary = {"split": args.split, "n_images": agg["n_images"], "n_empty_pred": agg["n_empty_pred"],
               "pooled_dsc": agg["pooled_dsc"], "pooled_jsi": agg["pooled_jsi"],
               "metrics": {k: agg[k] for k in SCORE_KEYS},
               "path": "cache mask 256x256 -> F.interpolate(bilinear, align_corners=False) -> > 0.5 vs official mask",
               "created_at": utc_now_iso()}
    atomic_write_json(out / f"{args.split}_summary.json", summary)
    for k in ("jsi", "dsc", "biou", "nsd", "hd95"):
        m = agg[k]
        print(f"  {k:5s} mean {m['mean']:.4f} [{m['ci95_low']:.4f}, {m['ci95_high']:.4f}]  median {m['median']:.4f}  "
              f"min {min(r[k] for r in rows):.4f}  max {max(r[k] for r in rows):.4f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
