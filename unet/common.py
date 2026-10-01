"""Shared configuration and helpers for the 5-phase U-Net pipeline (PyTorch).

Mirror of ``sandbox_yolo26/yolo26_seg/common.py``: every phase script imports
its constants, training protocols and output paths from here, so the
experimental protocol is defined **once**.

Protocols
---------
* :func:`baseline_protocol` — Phase 1 (baseline) and Phase 2 (baseline CV):
  the fixed **base setup** (:data:`BASE_SETUP`) + the **default
  hyperparameters** (:data:`DEFAULT_HPS`, equivalent to the original Keras
  baseline: Adam lr 1e-3, no weight decay, dropout 0.1, the original
  ``ImageDataGenerator`` augmentation).
* :func:`optimized_protocol` — Phase 4: the same base setup + tuned HPs.
* :func:`hpo_trial_protocol` — Phase 3 trials: the same base setup with the
  short per-trial budget.

Baseline = base setup + default HPs; Optimised = base setup + tuned HPs. Only
the **learning dynamics and augmentation** (:data:`TUNABLE_KEYS`) can differ;
the architecture (width, activation, normalisation), optimiser type, loss,
input size, batch, budget and precision are part of the base setup and can
never be overridden (:data:`PROTECTED_KEYS`), so Baseline and Optimised have
identical architecture and computational cost.

Output layout (see :class:`PipelinePaths`)::

    <root>/                              # e.g. /workspace/logs/pipeline_final_v1
    ├── phase1_baseline/unet_baseline/
    ├── phase2_cv_baseline/unet/{splits/, runs/fold_<k>/, metrics_*}
    ├── phase3_hpo/tune_unet/{trials.csv, best_hyperparameters.yaml,
    │                         hpo_state.json, optuna_study.db}
    ├── phase4_optimized/unet_optimized/
    ├── phase5_test/{accuracy/, per_image/, masks/, efficiency/}
    ├── summary/
    └── pipeline_runs/<UTC-timestamp>/
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Union

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
#: Model names handled by the pipeline (one U-Net width; kept as a list so the
#: scripts, summaries and notebooks share the YOLO "--models" interface).
DEFAULT_ORDER: list[str] = ["unet"]

#: YOLO-format dataset — the single source of truth shared with YOLO26
#: (identical images, splits and polygon labels).
DEFAULT_YOLO_DATA_YAML: str = "/workspace/datasets/isic_2018_task1_yolo26/data.yaml"

#: Phase 0 output: 256×256 cached arrays + ID manifests.
DEFAULT_CACHE_DIR: str = "/workspace/datasets/isic_2018_task1_unet256"

#: Default pipeline root; isolates this study from older runs under ``logs/``.
DEFAULT_PIPELINE_ROOT: str = "/workspace/logs/pipeline_final_v1"

#: Global seed for every RNG in the pipeline.
SEED: int = 0

#: Training budget shared by Phases 1, 2 and 4 (identical to YOLO26). Patience =
#: epochs, i.e. no early stopping: with patience 25 on the 100-image validation
#: split, the Baseline and Optimised runs both stopped at epoch 81 while the CV
#: folds (530 validation images) kept improving to epochs 96-118.
TRAIN_EPOCHS: int = 120
TRAIN_PATIENCE: int = 120

#: Phase 3 per-trial budget (patience = epochs: every trial runs its 30 epochs;
#: 10 of 30 trials stopped early with patience 10).
HPO_EPOCHS: int = 30
HPO_PATIENCE: int = 30

#: Network input size (images and masks are cached at this size in Phase 0).
IMGSZ: int = 256

#: Fixed base setup shared by EVERY phase. Never searched, never overridden.
BASE_SETUP: dict[str, Any] = {
    # Architecture (original Keras baseline: get_unet_baseline)
    "n_filters": 16,
    "activation": "relu",
    "batchnorm": True,
    # Optimisation
    "optimizer": "AdamW",        # decoupled weight decay == Keras Adam(weight_decay=...)
    "adam_eps": 1e-7,            # Keras default epsilon
    "beta2": 0.999,
    "lr_schedule": "constant",   # the original Keras baseline used a constant LR
    "loss": "bce_dice",          # BCE + soft Dice (original Keras bce_dice_loss)
    # Data / budget / numerics
    "imgsz": IMGSZ,
    "batch": 16,
    "workers": 8,
    "amp": False,
    "seed": SEED,
    "deterministic": True,
    "monitor": "val_jsi",        # best-epoch and early-stopping criterion (ISIC ranking metric)
}

#: Default hyperparameters = the original Keras baseline, expressed with the
#: names of the search space. Augmentation reproduces ImageDataGenerator(
#: rotation_range=15, width/height_shift_range=0.1, zoom_range=0.1,
#: horizontal_flip=True, vertical_flip=True, fill_mode="reflect").
DEFAULT_HPS: dict[str, Any] = {
    "lr0": 1e-3,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "dropout": 0.1,
    "degrees": 15.0,     # rotation uniform in [-degrees, +degrees]
    "translate": 0.1,    # shift uniform in [-translate, +translate] × size (x and y independently)
    "scale": 0.1,        # zoom factors uniform in [1 - scale, 1 + scale] (x and y independently)
    "fliplr": 0.5,       # horizontal-flip probability
    "flipud": 0.5,       # vertical-flip probability
}

#: The only keys an HPO search space / tuned YAML may contain.
TUNABLE_KEYS: frozenset[str] = frozenset(DEFAULT_HPS)

#: Keys that define the base setup, budget and reproducibility contract.
PROTECTED_KEYS: frozenset[str] = frozenset({*BASE_SETUP, "epochs", "patience", "device"})

#: Type alias for the device argument.
DeviceArg = Union[int, str]


# ----------------------------------------------------------------------------
# Device & reproducibility
# ----------------------------------------------------------------------------
def parse_device(arg: str) -> DeviceArg:
    """Parse ``--device``: a single GPU id (``"0"``) or ``"cpu"``.

    Raises:
        ValueError: For a multi-GPU list — the U-Net is trained on one GPU.
    """
    if "," in arg:
        raise ValueError("the U-Net pipeline uses a single device (e.g. --device 0)")
    return "cpu" if arg == "cpu" else int(arg)


def torch_device(device: DeviceArg):
    """Return the ``torch.device`` for a parsed device argument."""
    import torch

    return torch.device("cpu" if device == "cpu" else f"cuda:{device}")


def seed_everything(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed every RNG of the process and request deterministic kernels.

    Must be called at the very start of ``main()`` — before any CUDA context is
    created — because ``CUBLAS_WORKSPACE_CONFIG`` is only honoured at CUDA
    initialisation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------------
# Training protocols
# ----------------------------------------------------------------------------
def _check_tunable(hps: dict[str, Any], what: str) -> None:
    """Reject protected or unknown keys in a hyperparameter mapping."""
    clash = PROTECTED_KEYS.intersection(hps)
    if clash:
        raise ValueError(f"{what} must not override base-setup keys: {sorted(clash)}")
    unknown = set(hps) - TUNABLE_KEYS
    if unknown:
        raise ValueError(f"{what} contains unknown keys: {sorted(unknown)} (allowed: {sorted(TUNABLE_KEYS)})")


def baseline_protocol(
    device: DeviceArg,
    epochs: int = TRAIN_EPOCHS,
    patience: int = TRAIN_PATIENCE,
) -> dict[str, Any]:
    """Return the Phase 1 / Phase 2 protocol: base setup + default HPs.

    Args:
        device: Parsed device argument.
        epochs: Training budget (override only for smoke tests, identically
            for Phases 1, 2 and 4).
        patience: Early-stopping patience on ``val_jsi``.
    """
    return {**BASE_SETUP, **DEFAULT_HPS, "epochs": epochs, "patience": patience, "device": device}


def optimized_protocol(
    device: DeviceArg,
    tuned_hp: dict[str, Any] | None = None,
    epochs: int = TRAIN_EPOCHS,
    patience: int = TRAIN_PATIENCE,
) -> dict[str, Any]:
    """Return the Phase 4 protocol: base setup + tuned HPs (defaults for the rest).

    Raises:
        ValueError: If ``tuned_hp`` contains a protected or unknown key.
    """
    tuned_hp = dict(tuned_hp or {})
    _check_tunable(tuned_hp, "Tuned hyperparameters")
    return {**baseline_protocol(device, epochs, patience), **tuned_hp}


def hpo_trial_protocol(
    device: DeviceArg,
    trial_hp: dict[str, Any],
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    """Return the protocol of one Phase 3 trial (short budget, same base setup)."""
    return optimized_protocol(device, trial_hp, epochs, patience)


# ----------------------------------------------------------------------------
# Output layout
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelinePaths:
    """Canonical output layout of one pipeline run rooted at ``root``."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # ---- Phase 1 — baseline ------------------------------------------------
    @property
    def phase1_dir(self) -> Path:
        return self.root / "phase1_baseline"

    @staticmethod
    def phase1_run_name(model: str) -> str:
        return f"{model}_baseline"

    def phase1_best_pt(self, model: str) -> Path:
        return self.phase1_dir / self.phase1_run_name(model) / "weights" / "best.pt"

    # ---- Phase 2 — cross-validation ----------------------------------------
    def cv_dir(self, protocol: str = "baseline") -> Path:
        """CV root for a protocol; Phase 2 is ``baseline``, ``optimized`` is optional."""
        return self.root / f"phase2_cv_{protocol}"

    def cv_model_dir(self, model: str, protocol: str = "baseline") -> Path:
        return self.cv_dir(protocol) / model

    # ---- Phase 3 — HPO -----------------------------------------------------
    @property
    def phase3_dir(self) -> Path:
        return self.root / "phase3_hpo"

    @staticmethod
    def phase3_tune_name(model: str) -> str:
        return f"tune_{model}"

    def phase3_tune_dir(self, model: str) -> Path:
        return self.phase3_dir / self.phase3_tune_name(model)

    def phase3_best_yaml(self, model: str) -> Path:
        return self.phase3_tune_dir(model) / "best_hyperparameters.yaml"

    def phase3_state(self, model: str) -> Path:
        return self.phase3_tune_dir(model) / "hpo_state.json"

    # ---- Phase 4 — optimised fine-tune -------------------------------------
    @property
    def phase4_dir(self) -> Path:
        return self.root / "phase4_optimized"

    @staticmethod
    def phase4_run_name(model: str) -> str:
        return f"{model}_optimized"

    def phase4_best_pt(self, model: str) -> Path:
        return self.phase4_dir / self.phase4_run_name(model) / "weights" / "best.pt"

    # ---- Phase 5 & summaries -----------------------------------------------
    def best_pt(self, variant: str, model: str) -> Path:
        """Weights evaluated in Phase 5: ``baseline`` (Phase 1) or ``optimized`` (Phase 4)."""
        if variant == "baseline":
            return self.phase1_best_pt(model)
        if variant == "optimized":
            return self.phase4_best_pt(model)
        raise ValueError(f"unknown variant {variant!r}")

    @property
    def phase5_dir(self) -> Path:
        return self.root / "phase5_test"

    @staticmethod
    def phase5_tag(variant: str, model: str, precision: str) -> str:
        return f"{variant}_{model}_{precision}"

    def phase5_accuracy_json(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "accuracy" / f"{self.phase5_tag(variant, model, precision)}.json"

    def phase5_per_image_csv(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "per_image" / f"{self.phase5_tag(variant, model, precision)}.csv"

    def phase5_mask_dir(self, variant: str, model: str) -> Path:
        """Predicted test masks (FP32 only) for the visualisation notebook."""
        return self.phase5_dir / "masks" / f"{variant}_{model}"

    def phase5_efficiency_json(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "efficiency" / f"{self.phase5_tag(variant, model, precision)}.json"

    @property
    def summary_dir(self) -> Path:
        return self.root / "summary"

    @property
    def pipeline_runs_dir(self) -> Path:
        return self.root / "pipeline_runs"


# ----------------------------------------------------------------------------
# Small I/O helpers (identical to the YOLO26 pipeline)
# ----------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_stamp() -> str:
    """Compact UTC timestamp for backup names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so ``path`` is never half-written (tmp + fsync + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON at ``path``, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def config_hash(payload: Any) -> str:
    """Return a short, stable SHA-256 fingerprint of a JSON-serialisable config."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@contextmanager
def exclusive_lock(directory: Path, name: str = ".lock") -> Iterator[None]:
    """Hold a non-blocking exclusive POSIX lock (``lockf``) on ``directory/name``.

    ``lockf`` (not ``flock``): a POSIX record lock belongs to the *process* and
    is not inherited by ``fork()``-ed children. With ``flock`` the lock belongs
    to the open file and survives in forked DataLoader workers, so after a
    ``kill -9`` the orphaned workers kept the run locked and an immediate
    restart was refused. The lock is released as soon as the process dies.

    Raises:
        RuntimeError: If another process already holds the lock.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w") as fh:
        try:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError(f"another process is already working in {directory}") from e
        try:
            yield
        finally:
            fcntl.lockf(fh, fcntl.LOCK_UN)
