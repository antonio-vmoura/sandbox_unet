"""Phase 2 (post-step) — DSC/JSI of every CV fold on its held-out fold, at full resolution.

Each fold's ``best.pt`` is scored on that fold's held-out images with the
same pixel metrics, ground truth and resolution as YOLO26's Phase 2 pixel step
(:mod:`inference`: 640 × 640, labels rasterised from the YOLO polygons), so
the CV DSC/JSI of both architectures are directly comparable — they are even
computed on identical folds. The test set is never used here.

Outputs (per model), in the YOLO26 format::

    <project>/phase2_cv_<protocol>/unet/
    ├── pixel_metrics_per_fold.csv    # per-fold means of DSC, JSI, ...
    └── pixel_metrics_summary.json    # mean ± sample std (ddof=1) across folds

Up-to-date results (same fold weights, splits and method version) are skipped.

Usage:
    python evaluate_cv_pixels.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import traceback
from pathlib import Path

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
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
from data import CacheData
from inference import EVAL_VERSION, PROB_THRESHOLD, evaluate_ids, resolve_data_root
from segmentation_metrics import SCORE_KEYS, aggregate_scores
from train_cv_unet import build_kfold_splits, build_splits_manifest
from training import RUN_STATE_FILE, load_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Full-resolution DSC/JSI of each CV fold on its held-out fold.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--protocol", choices=["baseline", "optimized"], default="baseline")
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Phase 0 cache directory.")
    p.add_argument("--data-root", default=None, help="YOLO dataset root (default: from the cache).")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0) or 'cpu'.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if up to date.")
    return p.parse_args()


def evaluate_model(model_name: str, args, device, cache: CacheData, paths: PipelinePaths) -> None:
    """Score every fold of one model and write the per-fold CSV + summary JSON."""
    cv_root = paths.cv_model_dir(model_name, args.protocol)
    manifest = read_json(cv_root / "splits_manifest.json")
    if manifest is None:
        raise RuntimeError(f"{cv_root}: no splits_manifest.json — run Phase 2 training first")
    pool = cache.ids("train") + cache.ids("val")
    splits = build_kfold_splits(pool, manifest["k"], manifest["seed"])
    if build_splits_manifest(pool, splits, manifest["k"], manifest["seed"], len(cache.ids("test"))) != manifest:
        raise RuntimeError(f"{cv_root}: the cache no longer reproduces the recorded folds")

    folds = []
    for k, (_, val_ids) in enumerate(splits):
        run = cv_root / "runs" / f"fold_{k}"
        state = read_json(run / RUN_STATE_FILE)
        if state is None or state.get("status") != "complete":
            raise RuntimeError(f"{run}: fold training not complete")
        folds.append((k, run / "weights" / "best.pt", val_ids))

    settings = {"weights_sha256": [sha256_file(w) for _, w, _ in folds], "threshold": PROB_THRESHOLD,
                "manifest": manifest, "eval_version": EVAL_VERSION, "cache": cache.fingerprint()}
    out_json = cv_root / "pixel_metrics_summary.json"
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        print(f"  [skip] {model_name}: pixel metrics up to date")
        return

    data_root = resolve_data_root(cache, args.data_root)
    per_fold = []
    for k, weights, val_ids in folds:
        print(f"  [{model_name}] fold {k}: {len(val_ids)} held-out images")
        rows = evaluate_ids(load_model(weights, device), cache, val_ids, data_root, device)
        agg = aggregate_scores(rows, seed=SEED)
        per_fold.append({"fold": k, "n_images": agg["n_images"], "n_empty_pred": agg["n_empty_pred"],
                         "pooled_dsc": agg["pooled_dsc"], "pooled_jsi": agg["pooled_jsi"],
                         **{key: agg[key].get("mean", float("nan")) for key in SCORE_KEYS}})
        print(f"    DSC={per_fold[-1]['dsc']:.4f}  JSI={per_fold[-1]['jsi']:.4f}")

    with (cv_root / "pixel_metrics_per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()))
        w.writeheader()
        w.writerows(per_fold)
    summary = {}
    for key in (*SCORE_KEYS, "pooled_dsc", "pooled_jsi"):
        vals = [r[key] for r in per_fold]
        summary[key] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    atomic_write_json(out_json, {
        "model": model_name, "protocol": args.protocol, "split": "cv_heldout_fold",
        "resolution": "original (YOLO export, 640x640)", "n_folds": len(per_fold), "std_ddof": 1,
        "conf": PROB_THRESHOLD, "per_fold": per_fold, "summary": summary,
        "settings_hash": config_hash(settings), "created_at": utc_now_iso(),
    })
    print(f"  [{model_name}] DSC={summary['dsc']['mean']:.4f}±{summary['dsc']['std']:.4f}  "
          f"JSI={summary['jsi']['mean']:.4f}±{summary['jsi']['std']:.4f}")


def main() -> int:
    """Run the CV pixel evaluation for every requested model.

    Returns:
        ``0`` on success, ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = torch_device(parse_device(args.device))
    paths = PipelinePaths(Path(args.project))
    cache = CacheData(args.cache)
    print(f"Phase 2 pixel metrics (protocol={args.protocol}) for {args.models} on {device}")
    failures = 0
    for m in args.models:
        try:
            evaluate_model(m, args, device, cache, paths)
        except Exception:
            failures += 1
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
