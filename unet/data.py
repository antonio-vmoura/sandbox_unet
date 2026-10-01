"""Data access and augmentation for the U-Net pipeline (reads the Phase 0 cache).

Samples are addressed by their **ISIC identifier**, never by array position,
so every split, fold or subset is an explicit, fingerprintable list of IDs.

Augmentation
------------
:class:`SegDataset` re-implements the original Keras
``ImageDataGenerator(rotation_range, width/height_shift_range, zoom_range,
horizontal_flip, vertical_flip, fill_mode="reflect")`` jointly on image and
mask, with the hyperparameter names of the search space:

* rotation angle ~ U(−``degrees``, +``degrees``);
* shifts ~ U(−``translate``, +``translate``) × image size, x and y independent;
* zoom factors ~ U(1 − ``scale``, 1 + ``scale``), x and y independent
  (Keras semantics: a factor > 1 zooms out);
* the affine transform is centred on the image; borders are filled by
  reflection (scipy/Keras ``"reflect"`` ≡ ``cv2.BORDER_REFLECT``);
* horizontal / vertical flips with probabilities ``fliplr`` / ``flipud``.

Images are interpolated bilinearly; **masks use nearest-neighbour
interpolation and therefore stay binary** — the original Keras pipeline
interpolated masks bilinearly, producing soft labels.

Determinism
-----------
The random parameters of sample *i* in epoch *e* are drawn from
``numpy.random.default_rng([seed, e, i])`` and the epoch's sample order from
``default_rng([seed, e, SHUFFLE_STREAM])``. Augmentation and order are thus
pure functions of (seed, epoch, sample), independent of the number of
data-loader workers and of interruptions — a resumed run sees exactly the same
data as an uninterrupted one.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from common import read_json

#: Extra entropy word separating the shuffle stream from augmentation streams.
SHUFFLE_STREAM: int = 7_919_000

#: Augmentation keys (subset of the tunable hyperparameters).
AUG_KEYS: tuple[str, ...] = ("degrees", "translate", "scale", "fliplr", "flipud")


# ----------------------------------------------------------------------------
# Cache access
# ----------------------------------------------------------------------------
class CacheData:
    """Read-only view of the Phase 0 cache, addressed by ISIC ID."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)
        self.meta = read_json(self.dir / "meta.json")
        if self.meta is None:
            raise FileNotFoundError(f"{self.dir}/meta.json missing — run prepare_dataset.py (Phase 0)")
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.manifest: dict[str, list[dict[str, Any]]] = {}
        self.where: dict[str, tuple[str, int]] = {}
        for split in self.meta["splits"]:
            with (self.dir / split / "manifest.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.manifest[split] = rows
            for r in rows:
                self.where[r["id"]] = (split, int(r["index"]))

    def ids(self, split: str) -> list[str]:
        """IDs of a split in cache (= YOLO sorted-path) order."""
        return [r["id"] for r in self.manifest[split]]

    def _split_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split not in self._arrays:
            self._arrays[split] = (
                np.load(self.dir / split / "images.npy", mmap_mode="r"),
                np.load(self.dir / split / "masks.npy", mmap_mode="r"),
            )
        return self._arrays[split]

    def arrays(self, ids: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(images N×S×S×3 uint8, masks N×S×S uint8)`` for ``ids`` (in order)."""
        ids = list(ids)
        imgs = np.empty((len(ids), *self._split_arrays(self.where[ids[0]][0])[0].shape[1:]), np.uint8)
        msks = np.empty((len(ids), *imgs.shape[1:3]), np.uint8)
        for i, iid in enumerate(ids):
            split, row = self.where[iid]
            a_img, a_msk = self._split_arrays(split)
            imgs[i], msks[i] = a_img[row], a_msk[row]
        return imgs, msks

    def record(self, iid: str) -> dict[str, Any]:
        """Manifest row of one ID (original image / label paths, sizes)."""
        split, row = self.where[iid]
        return self.manifest[split][row]

    def fingerprint(self) -> str:
        """Digest of the cache content (array and manifest SHA-256 of every split)."""
        parts = {s: {k: m[k] for k in ("images_sha256", "masks_sha256", "manifest_sha256")}
                 for s, m in self.meta["splits"].items()}
        return hashlib.sha256(repr(sorted(parts.items())).encode()).hexdigest()[:16]


def ids_fingerprint(ids: Iterable[str]) -> str:
    """Order-sensitive digest of an ID list."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def random_affine(rng: np.random.Generator, size: int, hp: dict[str, float]) -> tuple[np.ndarray, bool, bool]:
    """Draw one augmentation: (2×3 output→input affine matrix, flip_lr, flip_ud)."""
    theta = np.deg2rad(rng.uniform(-hp["degrees"], hp["degrees"]))
    tx, ty = rng.uniform(-hp["translate"], hp["translate"], size=2) * size
    zx, zy = rng.uniform(1 - hp["scale"], 1 + hp["scale"], size=2)
    flip_lr = rng.random() < hp["fliplr"]
    flip_ud = rng.random() < hp["flipud"]
    c = (size - 1) / 2.0
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    zoom = np.diag([zx, zy, 1.0])
    to_c, from_c = np.array([[1, 0, c], [0, 1, c], [0, 0, 1]]), np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
    m = to_c @ rot @ shift @ zoom @ from_c          # output pixel → input pixel (Keras composition)
    return m[:2], flip_lr, flip_ud


def augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
            hp: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Apply one joint random augmentation to an image (H×W×3) and its mask (H×W)."""
    size = img.shape[0]
    m, flip_lr, flip_ud = random_affine(rng, size, hp)
    flags = cv2.WARP_INVERSE_MAP
    img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR | flags, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, m, (size, size), flags=cv2.INTER_NEAREST | flags, borderMode=cv2.BORDER_REFLECT)
    if flip_lr:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if flip_ud:
        img, mask = img[::-1], mask[::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class SegDataset(Dataset):
    """Image/mask pairs as float tensors: image ``3×S×S`` in [0, 1], mask ``1×S×S`` in {0, 1}."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, aug: dict[str, float] | None = None,
                 seed: int = 0) -> None:
        self.images, self.masks, self.aug, self.seed = images, masks, aug, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the augmentation stream of an epoch (call before iterating)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.images[i], self.masks[i]
        if self.aug is not None:
            img, mask = augment(img, mask, np.random.default_rng([self.seed, self.epoch, i]), self.aug)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
        y = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        return x, y


class EpochSampler(Sampler[int]):
    """Deterministic per-epoch permutation (or identity order when ``shuffle=False``)."""

    def __init__(self, n: int, shuffle: bool, seed: int) -> None:
        self.n, self.shuffle, self.seed, self.epoch = n, shuffle, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(range(self.n))
        return iter(np.random.default_rng([self.seed, self.epoch, SHUFFLE_STREAM]).permutation(self.n).tolist())

    def __len__(self) -> int:
        return self.n


def make_loader(dataset: SegDataset, batch: int, shuffle: bool, seed: int, workers: int,
                drop_last: bool, pin_memory: bool) -> tuple[DataLoader, EpochSampler]:
    """DataLoader + its sampler; call ``set_epoch`` on both dataset and sampler each epoch.

    The loader gets its own ``torch.Generator``: otherwise PyTorch draws the
    worker base seed from the *global* RNG at every epoch when ``workers > 0``,
    which would shift the global stream used by dropout and make results
    depend on the worker count. (Workers themselves draw no randomness:
    augmentation uses the per-sample generators above.)
    """
    sampler = EpochSampler(len(dataset), shuffle, seed)
    loader = DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=workers,
                        drop_last=drop_last, pin_memory=pin_memory, persistent_workers=False,
                        generator=torch.Generator().manual_seed(seed))
    return loader, sampler
