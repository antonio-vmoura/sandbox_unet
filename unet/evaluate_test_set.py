"""Phase 5a — Accuracy of the Baseline and Optimised U-Net on the held-out TEST set.

Mirror of YOLO26's ``yolo26_seg/evaluate_test_set.py``: same metrics, same
ground truth, same resolution and the same output schema, so the two
architectures can be compared directly. For every ``variant ∈ {baseline,
optimized}`` × ``model`` × ``precision ∈ {fp32, fp16}`` the final ``best.pt``
is evaluated on the ``test`` split — the only phase that ever touches it:

* **Pixel metrics** (:mod:`inference`) — per image, batch = 1: the U-Net
  probability map (256×256) is upsampled bilinearly to the image's original
  resolution (dataset resolution, as YOLO26's evaluation) and thresholded at 0.5; it is
  scored against the ground truth rasterised from the **same YOLO labels**
  with :func:`segmentation_metrics.pixel_scores`: DSC, JSI, ISIC thresholded
  JSI, sensitivity, specificity, accuracy. Empty predictions score 0 (never
  skipped). Aggregates: per-image mean, sample std, median, IQR, seeded
  bootstrap 95 % CI, pooled DSC/JSI (:func:`segmentation_metrics.aggregate_scores`).
* **Instance metrics** — YOLO26's box/mask mAP, P, R and F1 are Ultralytics
  instance-level metrics that do not exist for a semantic-segmentation
  network; the keys are kept (value NaN) so the JSON schema is identical.

FP32 is the primary result (training was FP32); FP16 quantifies the accuracy
cost of half-precision deployment (CUDA only).

Outputs (per variant/model/precision), as YOLO26::

    <project>/phase5_test/accuracy/<variant>_<model>_<precision>.json
    <project>/phase5_test/per_image/<variant>_<model>_<precision>.csv
    <project>/phase5_test/masks/<variant>_<model>/<image stem>.png   # FP32 only

Each JSON records the SHA-256 of the weights and of the test-ID list; up-to-date
results are skipped.

Usage:
    python evaluate_test_set.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    IMGSZ,
    SEED,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    parse_device,
    read_json,
    seed_everything,
    sha256_file,
    torch_device,
    utc_now_iso,
)
from data import CacheData, ids_fingerprint
from inference import EVAL_VERSION, PROB_THRESHOLD, evaluate_ids, resolve_data_root
from segmentation_metrics import aggregate_scores
from train_cv_unet import YOLO_INSTANCE_KEYS
from training import RUN_STATE_FILE, load_model

VARIANTS: tuple[str, ...] = ("baseline", "optimized")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 5a."""
    p = argparse.ArgumentParser(description="Phase 5a — U-Net test-set accuracy (pixel metrics, YOLO26 schema).")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Phase 0 cache directory.")
    p.add_argument("--data-root", default=None, help="YOLO dataset root (default: from the cache).")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0) or 'cpu'.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--no-save-masks", action="store_true", help="Do not save predicted masks.")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if results are up to date.")
    return p.parse_args()


def _require_trained(paths: PipelinePaths, variant: str, model: str) -> Path:
    """Weights of a completed training run, or raise."""
    weights = paths.best_pt(variant, model)
    state = read_json(weights.parent.parent / RUN_STATE_FILE)
    if state is None or state.get("status") != "complete" or not weights.exists():
        raise RuntimeError(f"{variant}/{model}: training run not complete ({weights.parent.parent})")
    return weights


def evaluate_one(variant: str, model_name: str, precision: str, args, device: torch.device,
                 paths: PipelinePaths, cache: CacheData, test_ids: list[str]) -> dict[str, Any]:
    """Evaluate one ``(variant, model, precision)`` on the test set (or skip if current)."""
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("FP16 evaluation requires a CUDA device")
    weights = _require_trained(paths, variant, model_name)
    out_json = paths.phase5_accuracy_json(variant, model_name, precision)
    settings = {
        "weights_sha256": sha256_file(weights), "test_list_sha256": ids_fingerprint(test_ids),
        "conf": PROB_THRESHOLD, "imgsz": IMGSZ, "precision": precision,
        "eval_version": EVAL_VERSION, "cache": cache.fingerprint(),
    }
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        return {"tag": out_json.stem, "skipped": True, "payload": previous}

    tag = paths.phase5_tag(variant, model_name, precision)
    print(f"\n=== {tag}  ({len(test_ids)} test images, weights={weights})")
    t0 = time.perf_counter()
    model = load_model(weights, device)
    if precision == "fp16":
        model = model.half()
    save_masks = precision == "fp32" and not args.no_save_masks
    rows = evaluate_ids(model, cache, test_ids, resolve_data_root(cache, args.data_root), device,
                        half=precision == "fp16",
                        mask_dir=paths.phase5_mask_dir(variant, model_name) if save_masks else None)
    pixel = aggregate_scores(rows, seed=SEED)
    print(f"  pixel   : DSC={pixel['dsc']['mean']:.4f} (95% CI {pixel['dsc']['ci95_low']:.4f}-"
          f"{pixel['dsc']['ci95_high']:.4f})  JSI={pixel['jsi']['mean']:.4f}  "
          f"JSI_thr={pixel['jsi_thr']['mean']:.4f}  empty_pred={pixel['n_empty_pred']}")

    per_image_csv = paths.phase5_per_image_csv(variant, model_name, precision)
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_image_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mean_ms = sum(r["infer_ms"] for r in rows) / len(rows)
    payload = {
        "variant": variant, "model": model_name, "precision": precision, "split": "test",
        "weights": str(weights), "data": str(cache.meta.get("source_data_yaml")), "n_images": len(test_ids),
        "settings": settings, "settings_hash": config_hash(settings),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "instance_metrics": {k: math.nan for k in YOLO_INSTANCE_KEYS},
        "instance_conf": None,
        "pixel_metrics": pixel,
        "pixel_conf": PROB_THRESHOLD,
        "ultralytics_speed_ms": {"preprocess": math.nan, "inference": mean_ms, "postprocess": math.nan},
        "per_image_csv": str(per_image_csv),
        "mask_dir": str(paths.phase5_mask_dir(variant, model_name)) if save_masks else None,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "created_at": utc_now_iso(),
    }
    atomic_write_json(out_json, payload)
    return {"tag": tag, "skipped": False, "payload": payload}


def main() -> int:
    """Evaluate every requested combination.

    Returns:
        ``0`` on success, ``1`` if any combination failed.
    """
    args = parse_args()
    seed_everything()
    device = torch_device(parse_device(args.device))
    paths = PipelinePaths(Path(args.project))
    cache = CacheData(args.cache)
    test_ids = cache.ids("test")
    print(f"Phase 5a — test set: {len(test_ids)} images | variants={args.variants} "
          f"models={args.models} precisions={args.precisions} device={device}")
    failures = 0
    for variant in args.variants:
        for m in args.models:
            for precision in args.precisions:
                try:
                    r = evaluate_one(variant, m, precision, args, device, paths, cache, test_ids)
                    if r["skipped"]:
                        print(f"  [skip] {r['tag']} up to date")
                except Exception:
                    failures += 1
                    print(f"  [fail] {variant}/{m}/{precision}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
