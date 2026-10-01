"""Phase 1 — Baseline training of the U-Net on ISIC 2018 Task 1 (PyTorch).

Trains the U-Net with the fixed **base setup** shared by every phase and the
**default hyperparameters** — the original Keras baseline (Adam lr 1e-3, no
weight decay, dropout 0.1, the original ImageDataGenerator augmentation) —
via :func:`common.baseline_protocol`. Baseline and Optimised (Phase 4) share
the identical base setup and architecture and differ only in the tuned
hyperparameters.

Data: the Phase 0 cache (same images and splits as YOLO26): ``train`` for
fitting, ``val`` for model selection / early stopping. The ``test`` split is
never touched.

The run is fault-tolerant and bit-exactly resumable (see :mod:`training`); a
completed run is skipped unless ``--force`` is passed.

Outputs:
    ``<project>/phase1_baseline/unet_baseline/{weights/{best,last}.pt, results.csv, run_state.json}``

Usage:
    python train_baseline_models.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    PipelinePaths,
    baseline_protocol,
    parse_device,
    seed_everything,
)
from data import CacheData
from training import print_phase_summary, train_or_resume

PHASE: str = "phase1_baseline"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 1."""
    p = argparse.ArgumentParser(description="Phase 1 — Baseline training (base setup + default HPs) of the U-Net.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Phase 0 cache directory.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0) or 'cpu'.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS,
                   help=f"Training budget (default: {TRAIN_EPOCHS}). Must match Phases 2 and 4.")
    p.add_argument("--patience", type=int, default=TRAIN_PATIENCE,
                   help=f"Early-stopping patience on val JSI (default: {TRAIN_PATIENCE}).")
    p.add_argument("--force", action="store_true", help="Retrain from scratch (old run → *.bak-<UTC>).")
    return p.parse_args()


def main() -> int:
    """Run Phase 1 for the requested models.

    Returns:
        ``0`` on success (including skipped models), ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))
    cache = CacheData(args.cache)
    protocol = baseline_protocol(device, args.epochs, args.patience)
    train_ids, val_ids = cache.ids("train"), cache.ids("val")

    print(f"Phase 1 (Baseline) for models: {args.models}")
    print(f"  device = {device}   cache = {args.cache}   output = {paths.phase1_dir}")
    print(f"  data   = {len(train_ids)} train / {len(val_ids)} val images (test untouched)")
    print(f"  budget = {args.epochs} epochs, patience {args.patience} (val JSI), FP32, seed 0")
    print("  setup  = U-Net 16 filters, ReLU, BN, AdamW (eps 1e-7), constant LR, BCE+Dice, batch 16")
    print("  HPs    = Keras-baseline defaults (lr0 1e-3, wd 0, beta1 0.9, dropout 0.1, original augmentation)")

    summary: list[dict] = []
    t0 = time.perf_counter()
    for m in args.models:
        print("\n" + "=" * 80 + f"\n=== PHASE 1 (BASELINE): {m}\n" + "=" * 80)
        try:
            summary.append(train_or_resume(
                phase=PHASE, model_name=m, protocol=protocol, cache=cache,
                train_ids=train_ids, val_ids=val_ids,
                project=paths.phase1_dir, name=paths.phase1_run_name(m), force=args.force,
            ))
        except Exception as e:
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "failed": True, "reason": str(e)})

    print_phase_summary("PHASE 1 (BASELINE)", summary, (time.perf_counter() - t0) / 60)
    return 1 if any(s.get("failed") for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
