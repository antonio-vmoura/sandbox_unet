"""Resumable, deterministic U-Net training shared by Phases 1, 2, 3 and 4.

Contract (mirrors ``sandbox_yolo26/yolo26_seg/training.py``):

* ``<run_dir>/run_state.json`` records the lifecycle (atomic writes):
  ``status`` (``running``/``complete``), ``phase``, ``model``, ``protocol`` /
  ``protocol_hash``, ``data`` (cache fingerprint + ID-list fingerprints),
  ``events`` (start / resume / complete), ``metrics`` (validation metrics of
  the best epoch), ``epochs_trained``, ``best_epoch``, versions.
* ``<run_dir>/weights/last.pt`` is written atomically after every epoch with
  the **complete** training state: model, optimiser, epoch, best score,
  early-stopping counter and the torch (CPU + CUDA), NumPy and Python RNG
  states. Data order and augmentation are pure functions of (seed, epoch,
  sample) (see :mod:`data`). A resumed run is therefore **bit-identical** to
  an uninterrupted one (on a given device and software stack).
* ``<run_dir>/weights/best.pt`` holds the weights of the best epoch, the epoch
  number and its validation metrics, plus the model configuration, so it can
  be loaded stand-alone by Phase 5.
* ``<run_dir>/results.csv`` has one row per epoch. ``last.pt`` also stores the
  complete per-epoch history, and on resume ``results.csv`` is **rewritten
  from the checkpoint**, so the file always matches the checkpointed training
  exactly (rows of an epoch that was never checkpointed, duplicates or any
  other damage cannot survive a resume).
* Completed runs are skipped; a changed protocol or data selection is refused
  (``--force`` moves the old run to ``*.bak-<UTC>``); a lock prevents two
  processes from training the same run.

Model selection: the epoch with the highest validation **per-image mean JSI**
(``val_jsi``, the ISIC 2018 Task 1 ranking metric) is kept; an epoch must be
*strictly* better to replace it (ties keep the earliest epoch, as Keras'
``ModelCheckpoint``/``EarlyStopping``). Training stops after ``patience``
epochs without improvement or after ``epochs`` epochs. ``best.pt`` and the
reported metrics are selected by the same rule, so they always agree.

Loss: BCE + soft Dice, as the original Keras ``bce_dice_loss``: pixel-mean
binary cross-entropy (computed from logits — the numerically stable form of
Keras' BCE on sigmoid outputs, without Keras' 1e-7 probability clipping) plus
``1 − Dice`` with the Dice coefficient computed over the whole batch
(smooth = 1e-6).

Validation metrics are computed per image with exactly the definitions of
:mod:`segmentation_metrics` (threshold 0.5, empty-mask conventions), on the
256×256 validation masks.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    atomic_write_json,
    config_hash,
    exclusive_lock,
    read_json,
    torch_device,
    utc_now_iso,
    utc_stamp,
)
from data import AUG_KEYS, CacheData, SegDataset, ids_fingerprint, make_loader
from model import build_model, count_parameters
from segmentation_metrics import ISIC_JSI_THRESHOLD

#: File that records the lifecycle of one training run.
RUN_STATE_FILE: str = "run_state.json"

#: Columns of ``results.csv`` (one row per epoch).
RESULT_COLUMNS: tuple[str, ...] = (
    "epoch", "time_s", "lr", "train_loss", "val_loss",
    "val_dsc", "val_jsi", "val_jsi_thr", "val_sensitivity", "val_specificity", "val_accuracy",
    "val_pooled_dsc", "val_pooled_jsi", "val_n_empty_pred", "improved",
)

#: Protocol keys that do not affect the result (allowed to differ on resume).
_HASH_EXCLUDED: frozenset[str] = frozenset({"device", "workers"})

#: Dice smoothing of the original Keras loss.
DICE_SMOOTH: float = 1e-6


# ----------------------------------------------------------------------------
# Loss and metrics
# ----------------------------------------------------------------------------
def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE (pixel mean, from logits) + (1 − batch-global soft Dice)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = (2 * inter + DICE_SMOOTH) / (p.sum() + target.sum() + DICE_SMOOTH)
    return bce + (1 - dice)


def batch_pixel_scores(logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-image confusion counts and scores (same definitions as :mod:`segmentation_metrics`).

    The prediction is ``sigmoid(logit) > 0.5`` ⇔ ``logit > 0``. Returns tensors
    of shape ``(B,)``; undefined sensitivity/specificity are NaN.
    """
    pred = (logits > 0).flatten(1)
    gt = target.flatten(1) > 0.5
    tp = (pred & gt).sum(1).double()
    fp = (pred & ~gt).sum(1).double()
    fn = (~pred & gt).sum(1).double()
    tn = pred.shape[1] - tp - fp - fn
    both_empty = (tp + fn == 0) & (tp + fp == 0)
    denom = tp + fp + fn
    dsc = torch.where(both_empty, torch.ones_like(tp), 2 * tp / (2 * tp + fp + fn).clamp_min(1))
    jsi = torch.where(both_empty, torch.ones_like(tp), tp / denom.clamp_min(1))
    nan = torch.full_like(tp, math.nan)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dsc": dsc, "jsi": jsi,
        "jsi_thr": torch.where(jsi >= ISIC_JSI_THRESHOLD, jsi, torch.zeros_like(jsi)),
        "sensitivity": torch.where(tp + fn > 0, tp / (tp + fn).clamp_min(1), nan),
        "specificity": torch.where(tn + fp > 0, tn / (tn + fp).clamp_min(1), nan),
        "accuracy": (tp + tn) / pred.shape[1],
        "empty_pred": tp + fp == 0,
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    """Validation loss and per-image mean pixel metrics."""
    model.eval()
    loss_sum, n = 0.0, 0
    acc: dict[str, list[torch.Tensor]] = {}
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += bce_dice_loss(logits, y).item() * len(x)
        n += len(x)
        for k, v in batch_pixel_scores(logits, y).items():
            acc.setdefault(k, []).append(v.cpu())
    cat = {k: torch.cat(v) for k, v in acc.items()}
    tp, fp, fn = (cat[k].sum().item() for k in ("tp", "fp", "fn"))
    out = {"val_loss": loss_sum / max(n, 1)}
    for k in ("dsc", "jsi", "jsi_thr", "accuracy"):
        out[f"val_{k}"] = cat[k].mean().item()
    for k in ("sensitivity", "specificity"):
        out[f"val_{k}"] = float(np.nanmean(cat[k].numpy())) if (~torch.isnan(cat[k])).any() else math.nan
    out["val_pooled_dsc"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan
    out["val_pooled_jsi"] = tp / (tp + fp + fn) if (tp + fp + fn) else math.nan
    out["val_n_empty_pred"] = int(cat["empty_pred"].sum().item())
    return out


# ----------------------------------------------------------------------------
# Checkpoint helpers
# ----------------------------------------------------------------------------
def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _rewrite_results(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically rewrite ``results.csv`` from the checkpointed history."""
    tmp = csv_path.with_name(f".{csv_path.name}.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)


def _append_result(csv_path: Path, row: dict[str, Any]) -> None:
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def read_results(csv_path: Path) -> list[dict[str, Any]]:
    """Parsed ``results.csv`` rows (numbers as float)."""
    with Path(csv_path).open() as f:
        return [{k: (float(v) if k != "improved" else v == "True") for k, v in r.items()} for r in csv.DictReader(f)]


def best_metrics(run_dir: Path) -> dict[str, Any]:
    """Validation metrics of the epoch stored in ``best.pt`` (read from ``results.csv``)."""
    best = torch.load(Path(run_dir) / "weights" / "best.pt", map_location="cpu", weights_only=False)
    rows = {int(r["epoch"]): r for r in read_results(Path(run_dir) / "results.csv")}
    row = rows[int(best["epoch"])]
    return {**{k: row[k] for k in RESULT_COLUMNS if k.startswith("val_")},
            "best_epoch": int(best["epoch"]), "epochs_trained": len(rows)}


def backup_dir(path: Path) -> Path | None:
    """Move ``path`` to ``<path>.bak-<UTC>``; return the backup path."""
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak-{utc_stamp()}")
    path.rename(backup)
    print(f"  [force] moved previous run to {backup}")
    return backup


def load_model(best_pt: Path, device: torch.device) -> torch.nn.Module:
    """Rebuild the U-Net from a ``best.pt`` (Phase 5 / notebooks)."""
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


# ----------------------------------------------------------------------------
# Resumable training
# ----------------------------------------------------------------------------
def protocol_fingerprint(protocol: dict[str, Any]) -> dict[str, Any]:
    """Protocol subset that must match to skip/resume a run."""
    return {k: v for k, v in protocol.items() if k not in _HASH_EXCLUDED}


def train_or_resume(
    *,
    phase: str,
    model_name: str,
    protocol: dict[str, Any],
    cache: CacheData,
    train_ids: list[str],
    val_ids: list[str],
    project: Path,
    name: str,
    force: bool = False,
) -> dict[str, Any]:
    """Train ``project/name`` to completion, resuming or skipping as needed.

    Args:
        phase: Tag stored in ``run_state.json`` (e.g. ``"phase1_baseline"``).
        model_name: Model name (``"unet"``).
        protocol: Full protocol (base setup + hyperparameters + budget + device).
        cache: Phase 0 cache.
        train_ids / val_ids: ISIC IDs of the training and validation samples.
        project: Parent directory of the run.
        name: Run directory name.
        force: Move an existing run aside and train from scratch.

    Returns:
        Summary dict with ``model``, ``skipped``, ``resumed``, ``reason``,
        ``elapsed_min`` and ``metrics``.

    Raises:
        RuntimeError: If an existing run used another protocol or data
            selection, or another process is training the same run.
    """
    if set(train_ids) & set(val_ids):
        raise RuntimeError(f"{name}: {len(set(train_ids) & set(val_ids))} IDs in both train and val")
    with exclusive_lock(Path(project), f".{name}.lock"):
        return _train_locked(phase, model_name, protocol, cache, train_ids, val_ids,
                             Path(project), name, force)


def _train_locked(phase, model_name, protocol, cache, train_ids, val_ids, project, name, force):
    run_dir = project / name
    wdir = run_dir / "weights"
    state_path, csv_path = run_dir / RUN_STATE_FILE, run_dir / "results.csv"
    last_pt, best_pt = wdir / "last.pt", wdir / "best.pt"
    data_spec = {"cache": cache.fingerprint(), "train": ids_fingerprint(train_ids),
                 "val": ids_fingerprint(val_ids), "n_train": len(train_ids), "n_val": len(val_ids)}
    phash = config_hash({"protocol": protocol_fingerprint(protocol), "data": data_spec})

    if force:
        backup_dir(run_dir)
    state = read_json(state_path)
    if state is not None and state.get("protocol_hash") != phash:
        raise RuntimeError(f"{run_dir} was trained with a different protocol/data "
                           f"({state.get('protocol_hash')} != {phash}). Use --force to retrain.")
    if state is not None and state.get("status") == "complete":
        return {"model": model_name, "skipped": True, "resumed": False, "elapsed_min": 0.0,
                "reason": f"complete ({state_path})", "metrics": best_metrics(run_dir)}
    if state is None and run_dir.exists():
        backup_dir(run_dir)  # foreign or pre-state folder: start clean
    wdir.mkdir(parents=True, exist_ok=True)
    if state is None:
        state = {"status": "running", "phase": phase, "model": model_name, "run_dir": str(run_dir),
                 "protocol": protocol_fingerprint(protocol), "protocol_hash": phash, "data": data_spec,
                 "torch_version": torch.__version__, "events": []}

    def log(event: str, **info: Any) -> None:
        state["events"].append({"at": utc_now_iso(), "event": event, **info})
        atomic_write_json(state_path, state)

    device = torch_device(protocol["device"])
    model = build_model(protocol).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=protocol["lr0"],
                            betas=(protocol["beta1"], protocol["beta2"]),
                            eps=protocol["adam_eps"], weight_decay=protocol["weight_decay"])
    aug = {k: protocol[k] for k in AUG_KEYS}
    tr_img, tr_msk = cache.arrays(train_ids)
    va_img, va_msk = cache.arrays(val_ids)
    train_ds = SegDataset(tr_img, tr_msk, aug=aug, seed=protocol["seed"])
    val_ds = SegDataset(va_img, va_msk, aug=None)
    pin = device.type == "cuda"
    train_loader, train_sampler = make_loader(train_ds, protocol["batch"], True, protocol["seed"],
                                              protocol["workers"], drop_last=True, pin_memory=pin)
    val_loader, _ = make_loader(val_ds, protocol["batch"], False, protocol["seed"],
                                protocol["workers"], drop_last=False, pin_memory=pin)
    if len(train_loader) == 0:
        raise RuntimeError(f"{len(train_ids)} training images < batch {protocol['batch']}")

    start, best_value, best_epoch, bad_epochs, resumed = 0, -math.inf, 0, 0, False
    if last_pt.exists():
        ck = torch.load(last_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        _set_rng_state(ck["rng"])
        start, best_value, best_epoch, bad_epochs = ck["epoch"], ck["best_value"], ck["best_epoch"], ck["bad_epochs"]
        history = ck["history"]
        _rewrite_results(csv_path, history)
        if not ck.get("final"):
            resumed = True
            log("resume", from_epoch=start)
            print(f"  [resume] continuing {run_dir.name} from epoch {start + 1}")
        final = ck.get("final", False)
    else:
        history = []
        _rewrite_results(csv_path, history)
        log("start", n_params=count_parameters(model), device=str(device))
        final = False

    t0 = time.perf_counter()
    for epoch in range(start + 1, protocol["epochs"] + 1):
        if final:
            break
        te = time.perf_counter()
        model.train()
        train_ds.set_epoch(epoch)
        train_sampler.set_epoch(epoch)
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = bce_dice_loss(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(x)
            n += len(x)
        val = evaluate(model, val_loader, device)
        improved = val[protocol["monitor"]] > best_value
        if improved:
            best_value, best_epoch, bad_epochs = val[protocol["monitor"]], epoch, 0
        else:
            bad_epochs += 1
        final = bad_epochs >= protocol["patience"] or epoch == protocol["epochs"]
        row = {"epoch": epoch, "time_s": round(time.perf_counter() - te, 3),
               "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / n, **val, "improved": improved}
        history.append(row)
        _append_result(csv_path, row)
        if improved:
            _atomic_torch_save({"epoch": epoch, "model": model.state_dict(), "metrics": val,
                                "model_config": {k: protocol[k] for k in ("n_filters", "dropout", "batchnorm", "activation")},
                                "protocol_hash": phash}, best_pt)
        _atomic_torch_save({"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                            "best_value": best_value, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                            "rng": _rng_state(), "final": final, "history": history,
                            "protocol_hash": phash}, last_pt)
        print(f"  epoch {epoch:>3}/{protocol['epochs']}  loss {loss_sum / n:.4f}  "
              f"val_jsi {val['val_jsi']:.4f}  val_dsc {val['val_dsc']:.4f}"
              f"{'  *' if improved else ''}", flush=True)

    metrics = best_metrics(run_dir)
    state.update(status="complete", metrics=metrics, epochs_trained=metrics["epochs_trained"],
                 best_epoch=metrics["best_epoch"], best_pt=str(best_pt),
                 n_params=count_parameters(model))
    log("complete", elapsed_min=round((time.perf_counter() - t0) / 60, 2))
    return {"model": model_name, "skipped": False, "resumed": resumed, "reason": None,
            "elapsed_min": (time.perf_counter() - t0) / 60, "metrics": metrics}


# ----------------------------------------------------------------------------
# Helpers shared by the phase scripts (mirror of the YOLO26 helpers)
# ----------------------------------------------------------------------------
def load_tuned_hp(path: Path) -> dict[str, Any]:
    """Load the ``best_hyperparameters.yaml`` written by Phase 3.

    Raises:
        ValueError: If the YAML is empty (signals a failed tune).
    """
    import yaml

    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(f"Empty YAML at {path}. Did Phase 3 complete?")
    return data


def require_complete_hpo(state_path: Path) -> None:
    """Raise unless Phase 3's ``hpo_state.json`` reports a complete search."""
    state = read_json(state_path)
    if state is None:
        raise RuntimeError(f"HPO checkpoint not found: {state_path}. Run Phase 3 first.")
    if state.get("status") != "complete":
        raise RuntimeError(
            f"HPO is not complete ({state.get('completed_trials')}/{state.get('target_trials')} "
            f"trials, status={state.get('status')!r}). Re-run Phase 3 to resume it.",
        )


def print_phase_summary(title: str, summary: list[dict], total_min: float) -> None:
    """Per-model summary table shared by the training phases."""
    print("\n" + "=" * 80)
    print(f"=== {title} — SUMMARY")
    print("=" * 80)
    for s in summary:
        m = s.get("metrics") or {}
        score = (f"  val JSI={m['val_jsi']:.4f}  DSC={m['val_dsc']:.4f}  best epoch {m.get('best_epoch')}"
                 if "val_jsi" in m else "")
        if s.get("failed"):
            status = f"FAILED ({s.get('reason')})"
        elif s.get("skipped"):
            status = "skipped (complete)"
        else:
            status = f"ok{' (resumed)' if s.get('resumed') else ''} in {s['elapsed_min']:.1f} min"
        print(f"  {s['model']:<8} : {status}{score}")
    print(f"\nTotal time: {total_min:.1f} min ({total_min / 60:.2f} h)")
