"""Phase 2 — Deterministic K-Fold cross-validation of the U-Net (PyTorch).

For every requested model, ``k`` U-Nets are trained on a deterministic K-Fold
partition of the **train + val pool** of the Phase 0 cache and each is
selected on its held-out fold. Per-fold validation metrics (per-image mean
DSC, JSI, ISIC thresholded JSI, sensitivity, specificity, accuracy — at the
256×256 network resolution) are written to CSV and aggregated as mean ±
**sample** standard deviation (ddof = 1). The DSC/JSI of each fold at the
dataset resolution evaluation resolution (same ground truth as YOLO26) are computed by
:mod:`evaluate_cv_pixels`.

Protocols (``--protocol``): ``baseline`` (default, **Phase 2**) = the Phase 1
protocol; ``optimized`` (optional) = base setup + Phase 3 HPs.

Design notes:

* **Same folds as YOLO26.** The pool is the train IDs followed by the val IDs
  in cache order — identical to the image order YOLO26's cross-validation
  uses — and :func:`build_kfold_splits` is YOLO26's algorithm
  (``numpy.random.RandomState(seed)`` shuffle + contiguous folds, bit-exact
  with ``sklearn.model_selection.KFold(shuffle=True, random_state=seed)``).
  Both pipelines therefore train and validate on identical folds.
* **Test-set isolation.** The ``test`` split is never in the pool; the script
  additionally aborts if any test ID appears in it.
* **Split manifest.** ``splits_manifest.json`` fingerprints the pool and every
  fold (by ISIC ID); a re-run whose partition differs is refused.
* **Fault tolerance.** Each fold is a bit-exactly resumable run
  (:mod:`training`); completed folds are skipped.
* **Schema.** Per-fold rows and summaries carry the YOLO26 column names; the
  Ultralytics-only instance metrics (box/mask mAP, P, R, F1) do not exist for
  a semantic-segmentation network and are written as NaN.

Outputs::

    <project>/phase2_cv_<protocol>/unet/
    ├── splits_manifest.json
    ├── runs/fold_<k>/{weights/, results.csv, run_state.json}
    ├── metrics_per_fold.csv
    └── metrics_summary.json

Usage:
    python train_cv_unet.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    SEED,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    PipelinePaths,
    atomic_write_json,
    baseline_protocol,
    optimized_protocol,
    parse_device,
    read_json,
    seed_everything,
)
from data import CacheData
from training import (
    backup_dir,
    load_tuned_hp,
    print_phase_summary,
    require_complete_hpo,
    train_or_resume,
)

K_FOLDS_DEFAULT: int = 5

#: YOLO26 instance-metric columns (Ultralytics-only → NaN for the U-Net).
YOLO_INSTANCE_KEYS: tuple[str, ...] = (
    "precision_b", "recall_b", "map50_b", "map5095_b",
    "precision_m", "recall_m", "map50_m", "map5095_m", "f1_b", "f1_m",
)

#: U-Net validation metrics reported per fold (from results.csv of the best epoch).
VAL_KEYS: tuple[str, ...] = (
    "val_dsc", "val_jsi", "val_jsi_thr", "val_sensitivity", "val_specificity",
    "val_accuracy", "val_pooled_dsc", "val_pooled_jsi", "val_loss",
)


def build_kfold_splits(ids: list[str], k: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """K deterministic folds over ``ids`` — YOLO26's algorithm, applied to IDs.

    Indices are shuffled with ``numpy.random.RandomState(seed)`` and split into
    ``k`` contiguous blocks; the first ``n % k`` blocks get one extra element.

    Raises:
        ValueError: If ``k < 2`` or there are fewer than ``k`` IDs.
    """
    if k < 2:
        raise ValueError(f"k_folds must be >= 2 (got {k}).")
    n = len(ids)
    if n < k:
        raise ValueError(f"Pool of {n} images is smaller than k={k}.")
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    splits, start = [], 0
    for size in fold_sizes:
        stop = start + size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append(([ids[i] for i in train_idx], [ids[i] for i in val_idx]))
        start = stop
    return splits


def _sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build_splits_manifest(pool: list[str], splits, k: int, seed: int, n_test: int) -> dict[str, Any]:
    """Describe the pool and every fold so later runs can verify the partition."""
    return {
        "k": k, "seed": seed, "pool_size": len(pool), "pool_sha256": _sha(pool),
        "test_size_excluded": n_test,
        "folds": [{"fold": i, "n_train": len(tr), "n_val": len(va), "val_sha256": _sha(va)}
                  for i, (tr, va) in enumerate(splits)],
    }


def check_or_write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest on first use; refuse to continue if the splits changed."""
    existing = read_json(path)
    if existing is None:
        atomic_write_json(path, manifest)
    elif existing != manifest:
        raise RuntimeError(f"K-Fold splits differ from {path} (data, k or seed changed). "
                           f"Use --force to start this model's CV from scratch.")


def fold_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """Per-fold row in the YOLO26 schema (+ U-Net validation metrics)."""
    row: dict[str, Any] = {k: math.nan for k in YOLO_INSTANCE_KEYS}
    row.update({k: metrics[k] for k in VAL_KEYS})
    row.update(fitness=metrics["val_jsi"], best_epoch=metrics["best_epoch"],
               epochs_trained=metrics["epochs_trained"], best_epoch_source="best.pt")
    return row


def aggregate_fold_metrics(per_fold: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """``{key: {mean, std}}`` over folds with the **sample** std (ddof = 1); NaN-preserving."""
    summary = {}
    for k in per_fold[0]:
        vals = [m[k] for m in per_fold]
        if not all(isinstance(v, (int, float)) for v in vals):
            continue
        if any(math.isnan(v) for v in vals):
            summary[k] = {"mean": math.nan, "std": math.nan}
        else:
            summary[k] = {"mean": statistics.mean(vals),
                          "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    return summary


def save_metrics_artifacts(model: str, protocol: str, per_fold, summary, out_dir: Path,
                           hp_source: str | None) -> tuple[Path, Path]:
    """Write ``metrics_per_fold.csv`` and ``metrics_summary.json`` (YOLO26 format)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out_dir / "metrics_per_fold.csv", out_dir / "metrics_summary.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", *per_fold[0].keys()])
        w.writeheader()
        for i, m in enumerate(per_fold):
            w.writerow({"fold": i, **m})
    atomic_write_json(json_path, {"model": model, "protocol": protocol, "hp_source": hp_source,
                                  "n_folds": len(per_fold), "std_ddof": 1,
                                  "per_fold": per_fold, "summary": summary})
    return csv_path, json_path


def cross_validate_one_model(model: str, args, device, cache: CacheData, paths: PipelinePaths) -> dict:
    """Run (or resume) the K-Fold CV of one model."""
    cv_root = paths.cv_model_dir(model, args.protocol)
    if args.force:
        backup_dir(cv_root)
    pool = cache.ids("train") + cache.ids("val")
    test_ids = set(cache.ids("test"))
    leaked = [i for i in pool if i in test_ids]
    if leaked:
        raise RuntimeError(f"{len(leaked)} test IDs in the CV pool, e.g. {leaked[:5]}")
    splits = build_kfold_splits(pool, args.k_folds, args.seed)
    check_or_write_manifest(cv_root / "splits_manifest.json",
                            build_splits_manifest(pool, splits, args.k_folds, args.seed, len(test_ids)))
    if args.protocol == "baseline":
        protocol, hp_source = baseline_protocol(device, args.epochs, args.patience), None
    else:
        require_complete_hpo(paths.phase3_state(model))
        hp_source = str(paths.phase3_best_yaml(model))
        protocol = optimized_protocol(device, load_tuned_hp(paths.phase3_best_yaml(model)), args.epochs, args.patience)

    print(f"  protocol = {args.protocol}   hp_source = {hp_source or 'base setup + default HPs'}")
    print(f"  pool     = {len(pool)} images (k={args.k_folds}, seed={args.seed}); test excluded = {len(test_ids)}")
    t0 = time.perf_counter()
    per_fold, any_resumed, all_skipped = [], False, True
    for k, (tr, va) in enumerate(splits):
        print(f"\n  [fold {k}/{args.k_folds - 1}] train={len(tr)} val={len(va)}")
        stats = train_or_resume(phase=f"phase2_cv_{args.protocol}", model_name=model, protocol=protocol,
                                cache=cache, train_ids=tr, val_ids=va,
                                project=cv_root / "runs", name=f"fold_{k}")
        any_resumed |= stats["resumed"]
        all_skipped &= stats["skipped"]
        per_fold.append(fold_row(stats["metrics"]))
        m = stats["metrics"]
        print(f"  fold {k}: val JSI={m['val_jsi']:.4f} DSC={m['val_dsc']:.4f} (best epoch {m['best_epoch']})")

    summary = aggregate_fold_metrics(per_fold)
    csv_path, json_path = save_metrics_artifacts(model, args.protocol, per_fold, summary, cv_root, hp_source)
    print(f"\n  [{model}] artefacts: {csv_path} | {json_path}")
    return {"model": model, "skipped": all_skipped, "resumed": any_resumed, "reason": None,
            "elapsed_min": (time.perf_counter() - t0) / 60,
            "metrics": {k: v["mean"] for k, v in summary.items()} | {"best_epoch": "-"}}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 2."""
    p = argparse.ArgumentParser(description="Phase 2 — deterministic K-Fold cross-validation of the U-Net.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--protocol", choices=["baseline", "optimized"], default="baseline")
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Phase 0 cache directory.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0) or 'cpu'.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--k-folds", type=int, default=K_FOLDS_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED, help="Seed of the K-Fold shuffle (default: 0).")
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Epochs per fold (must match Phases 1/4).")
    p.add_argument("--patience", type=int, default=TRAIN_PATIENCE)
    p.add_argument("--force", action="store_true", help="Re-run a model's CV from scratch (→ *.bak-<UTC>).")
    return p.parse_args()


def main() -> int:
    """Run Phase 2.

    Returns:
        ``0`` on success, ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything(args.seed)
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))
    cache = CacheData(args.cache)
    print(f"Phase 2 (CV, protocol={args.protocol}) for {args.models}; output {paths.cv_dir(args.protocol)}")
    summary, t0 = [], time.perf_counter()
    for m in args.models:
        print("\n" + "=" * 80 + f"\n=== PHASE 2 CV ({args.protocol}): {m}\n" + "=" * 80)
        try:
            summary.append(cross_validate_one_model(m, args, device, cache, paths))
        except Exception as e:
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "failed": True, "reason": str(e)})
    print_phase_summary(f"PHASE 2 CV ({args.protocol}, fold means)", summary, (time.perf_counter() - t0) / 60)
    return 1 if any(s.get("failed") for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
