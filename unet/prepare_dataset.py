"""Phase 0 — Build the U-Net data cache from the YOLO26 dataset (single source of truth).

``--task 1`` (default): ISIC 2018 Task 1, one binary lesion mask per image (``masks.npy`` of shape N×S×S).
``--task 2``: ISIC 2018 Task 2, the five lesion attributes (``segmentation_metrics.ISIC2018_ATTRIBUTES``) as a
**multi-label** target: ``masks.npy`` has shape N×S×S×5, channel *k* = attribute *k* (binary, independent —
attributes may overlap, and an image may have none); input = the Task 2 YOLO26 dataset
(``isic2018_task2_official``, built by YOLO26's ``prepare_dataset.py --task 2``), whose per-attribute official
masks (``masks/<attribute>/<id>.png``) are read with :func:`segmentation_metrics.ground_truth_class_mask`.
Everything else (image processing, ID checks, official split sizes, fingerprints) is identical for both tasks.

The U-Net is trained and evaluated on **exactly the same images, splits and
annotations** as YOLO26-seg: this script reads the YOLO-format dataset
(``data.yaml``: ``train`` / ``val`` / ``test`` image folders + polygon labels)
and writes, per split::

    <out>/<split>/images.npy    (N, S, S, 3) uint8, RGB
    <out>/<split>/masks.npy     (N, S, S)    uint8, strictly binary {0, 1}
    <out>/<split>/manifest.csv  index, id, image, label, orig_h, orig_w, gt_px, gt_px_net, empty_gt
    <out>/meta.json             provenance: source fingerprints, array SHA-256, parameters

with ``S = 256`` (the U-Net input size).

Processing of each image (deterministic):

* **Ground truth** — the official ISIC mask of the image, as written by YOLO26's
  Phase 0 (``masks/<id>.png``, read with :func:`segmentation_metrics.ground_truth_mask`,
  the very function every pipeline's Phase 5 scores predictions against; datasets
  without mask images fall back to the rasterised YOLO polygons).
* **Image** — resized to ``S × S`` with area interpolation (``cv2.INTER_AREA``).
* **Mask** — the full-resolution binary mask is area-resized to ``S × S``
  (each output pixel = fraction of lesion pixels it covers) and thresholded at
  0.5, which keeps the mask **strictly binary** and places the boundary at the
  sub-pixel majority (a plain nearest-neighbour resize would alias it).

Guarantees and checks:

* Image order inside a split is ``sorted(Path.rglob("*"))``, the same order the
  YOLO26 cross-validation uses to build its pool, so both pipelines can share
  identical K-Fold partitions.
* ISIC identifiers must be unique across splits (no image in two splits) —
  otherwise the script aborts.
* The split sizes must be **exactly** the official ISIC 2018 Task 1 ones
  (2,594 / 100 / 1,000; :data:`EXPECTED_COUNTS`) — asserted on input and output.
* Every mask is verified to contain only {0, 1}.
* The cache is rebuilt only when the source dataset (SHA-256 of every image
  and label file) or the parameters change; ``--force`` rebuilds it anyway.
  The new cache is written to a temporary folder and swapped in atomically;
  a replaced cache is kept as ``<out>.bak-<UTC>``.

Usage:
    python prepare_dataset.py \\
        --yolo-data /workspace/yolo26_dataset/data.yaml \\
        --out /workspace/datasets/isic_2018_task1_unet256
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_YOLO_DATA_YAML,
    IMGSZ,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now_iso,
    utc_stamp,
)
from segmentation_metrics import (
    ISIC2018_ATTRIBUTES,
    class_mask_path_for,
    ground_truth_class_mask,
    ground_truth_mask,
    label_path_for,
    mask_path_for,
)

#: Version of the preprocessing method (part of the cache fingerprint).
PREP_VERSION: int = 1

#: Splits of ``data.yaml`` (key → cache folder name).
SPLITS: dict[str, str] = {"train": "train", "val": "val", "test": "test"}

#: Official ISIC 2018 Task 1 split sizes (3,694 images) — enforced with assertions.
EXPECTED_COUNTS: dict[str, int] = {"train": 2594, "val": 100, "test": 1000}

#: Image extensions recognised (same as YOLO26's cross-validation).
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

#: Majority threshold applied to the area-resized mask.
MASK_THRESHOLD: float = 0.5


def isic_id(image: Path) -> str:
    """ISIC identifier of a (possibly Roboflow-renamed) image file.

    ``ISIC_0012169_jpg.rf.<hash>.jpg`` → ``ISIC_0012169``.
    """
    stem = image.name.split(".rf.")[0]
    for ext in ("_jpg", "_jpeg", "_png"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return Path(stem).stem


def resolve_split(data_yaml: Path, key: str) -> tuple[Path, list[Path]]:
    """Return ``(dataset_root, sorted image paths)`` of one ``data.yaml`` split."""
    data = yaml.safe_load(data_yaml.read_text()) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    if not root.is_dir() or not any(root.iterdir()):
        # e.g. a /workspace/... container path used on the host, or the empty mount-point
        # folder an older nested ``docker run -v`` left behind
        root = data_yaml.parent
    value = data.get(key)
    if value is None:
        raise ValueError(f"{data_yaml}: no {key!r} split")
    dirs = [Path(v) if Path(v).is_absolute() else (root / v).resolve()
            for v in (value if isinstance(value, list) else [value])]
    images = [p for d in dirs for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError(f"{data_yaml}: split {key!r} has no images ({dirs})")
    return root, images


def source_fingerprint(images: list[Path], root: Path, task: int = 1) -> str:
    """SHA-256 over (relative path, image, label and mask SHA-256) of a split."""
    h = hashlib.sha256()
    for img in images:
        lab = label_path_for(img)
        msks = [mask_path_for(img)] if task == 1 else [class_mask_path_for(img, a) for a in ISIC2018_ATTRIBUTES]
        h.update(str(img.relative_to(root)).encode())
        h.update(sha256_file(img).encode())
        h.update((sha256_file(lab) if lab.exists() else "no-label").encode())
        for msk in msks:
            h.update((sha256_file(msk) if msk.exists() else "no-mask").encode())
    return h.hexdigest()


def _resize_mask(gt: np.ndarray, size: int) -> np.ndarray:
    """Area-resize a binary mask to ``size × size`` and re-binarise at :data:`MASK_THRESHOLD`."""
    frac = cv2.resize(gt.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return (frac >= MASK_THRESHOLD).astype(np.uint8)


def process_image(img_path: Path, size: int, task: int = 1) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(image S×S×3 uint8 RGB, mask, manifest fields)``.

    Task 1: mask S×S uint8 {0,1}. Task 2: mask S×S×5 uint8 {0,1}, one independent channel per attribute.
    """
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read {img_path}")
    h, w = bgr.shape[:2]
    img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
    if task == 2:
        chans, info = [], {"orig_h": h, "orig_w": w}
        for k, name in enumerate(ISIC2018_ATTRIBUTES):
            gt = ground_truth_class_mask(img_path, name, k, h, w)
            chans.append(_resize_mask(gt, size))
            info[f"{name}_px"] = int(gt.sum())
            info[f"{name}_px_net"] = int(chans[-1].sum())
        info["n_attributes"] = sum(info[f"{n}_px"] > 0 for n in ISIC2018_ATTRIBUTES)
        info["empty_gt"] = info["n_attributes"] == 0
        return img, np.stack(chans, axis=-1), info
    gt = ground_truth_mask(img_path, h, w)
    frac = cv2.resize(gt.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    mask = (frac >= MASK_THRESHOLD).astype(np.uint8)
    return img, mask, {"orig_h": h, "orig_w": w, "gt_px": int(gt.sum()),
                       "gt_px_net": int(mask.sum()), "empty_gt": bool(gt.sum() == 0)}


def build_split(images: list[Path], root: Path, out_dir: Path, size: int, task: int = 1) -> dict[str, Any]:
    """Process one split into ``out_dir``; return its metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(images)
    arr_img = np.zeros((n, size, size, 3), dtype=np.uint8)
    mask_shape = (n, size, size) if task == 1 else (n, size, size, len(ISIC2018_ATTRIBUTES))
    arr_msk = np.zeros(mask_shape, dtype=np.uint8)
    rows = []
    for i, img_path in enumerate(images):
        arr_img[i], arr_msk[i], info = process_image(img_path, size, task)
        rows.append({"index": i, "id": isic_id(img_path),
                     "image": str(img_path.relative_to(root)),
                     "label": str(label_path_for(img_path).relative_to(root)), **info})
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"    {i + 1}/{n}", flush=True)
    values = np.unique(arr_msk)
    if not set(values.tolist()) <= {0, 1}:
        raise RuntimeError(f"non-binary mask values {values} in {out_dir}")
    np.save(out_dir / "images.npy", arr_img)
    np.save(out_dir / "masks.npy", arr_msk)
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "n_images": n,
        "n_empty_gt": sum(r["empty_gt"] for r in rows),
        "images_sha256": sha256_file(out_dir / "images.npy"),
        "masks_sha256": sha256_file(out_dir / "masks.npy"),
        "manifest_sha256": sha256_file(out_dir / "manifest.csv"),
        "mask_foreground_fraction": float(arr_msk.mean()),
    }
    if task == 2:
        meta["mask_channels"] = list(ISIC2018_ATTRIBUTES)
        meta["images_per_attribute"] = {a: int(arr_msk[..., k].reshape(n, -1).any(axis=1).sum())
                                        for k, a in enumerate(ISIC2018_ATTRIBUTES)}
        meta["n_images_with_overlapping_attributes"] = int(((arr_msk.sum(axis=-1) >= 2).reshape(n, -1)).any(axis=1).sum())
    return meta


#: Task 2 defaults: the Task 2 YOLO26 dataset (mount it at /workspace/yolo26_dataset_task2) and its cache.
TASK2_YOLO_DATA_YAML: str = "/workspace/yolo26_dataset_task2/data.yaml"
TASK2_CACHE_DIR: str = "/workspace/datasets/isic_2018_task2_unet256"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Phase 0 — build the 256×256 U-Net cache from the YOLO dataset.")
    p.add_argument("--task", type=int, choices=(1, 2), default=1,
                   help="ISIC 2018 task: 1 = lesion segmentation (default), 2 = five lesion attributes (5-channel masks).")
    p.add_argument("--yolo-data", default=None,
                   help=f"YOLO data.yaml (default: Task 1 {DEFAULT_YOLO_DATA_YAML}; Task 2 {TASK2_YOLO_DATA_YAML}).")
    p.add_argument("--out", default=None,
                   help=f"Cache directory (default: Task 1 {DEFAULT_CACHE_DIR}; Task 2 {TASK2_CACHE_DIR}).")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help=f"Cache size (default: {IMGSZ}).")
    p.add_argument("--force", action="store_true", help="Rebuild even if the cache is up to date.")
    return p.parse_args()


def main() -> int:
    """Build (or validate) the cache.

    Returns:
        ``0`` on success / up to date, ``2`` on invalid input (missing split,
        duplicate IDs across splits).
    """
    args = parse_args()
    args.yolo_data = args.yolo_data or (DEFAULT_YOLO_DATA_YAML if args.task == 1 else TASK2_YOLO_DATA_YAML)
    args.out = args.out or (DEFAULT_CACHE_DIR if args.task == 1 else TASK2_CACHE_DIR)
    data_yaml = Path(args.yolo_data).resolve()
    out = Path(args.out)
    if args.task == 2:
        names = (yaml.safe_load(data_yaml.read_text()) or {}).get("names")
        names = list(names.values()) if isinstance(names, dict) else names
        assert names == list(ISIC2018_ATTRIBUTES), (
            f"{data_yaml}: classes {names} are not the five ISIC 2018 Task 2 attributes "
            f"{list(ISIC2018_ATTRIBUTES)} — build it with YOLO26's prepare_dataset.py --task 2")
    try:
        resolved = {name: resolve_split(data_yaml, key) for key, name in SPLITS.items()}
    except (OSError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    # ---- Official split sizes ----------------------------------------------
    for name, (_, images) in resolved.items():
        assert len(images) == EXPECTED_COUNTS[name], (
            f"{name}: {len(images)} images in {data_yaml}, expected exactly {EXPECTED_COUNTS[name]} "
            f"(official ISIC 2018 Task 1) — rebuild the YOLO26 dataset from the raw release")

    # ---- No image may appear in two splits --------------------------------
    owner: dict[str, str] = {}
    for name, (_, images) in resolved.items():
        for img in images:
            iid = isic_id(img)
            if iid in owner:
                print(f"[error] {iid} appears in both {owner[iid]!r} and {name!r}", file=sys.stderr)
                return 2
            owner[iid] = name

    print(f"Phase 0 — U-Net cache from {data_yaml}")
    for name, (_, images) in resolved.items():
        print(f"  {name:<5}: {len(images)} images")
    print("  fingerprinting source files ...")
    params = {"imgsz": args.imgsz, "prep_version": PREP_VERSION, "mask_threshold": MASK_THRESHOLD,
              "image_interp": "INTER_AREA", "mask_interp": "INTER_AREA+threshold"}
    if args.task == 2:   # Task 1 keeps its historical key, so existing caches stay valid
        params.update(task=2, mask_channels=list(ISIC2018_ATTRIBUTES))
    sources = {name: source_fingerprint(images, root, args.task) for name, (root, images) in resolved.items()}

    meta = read_json(out / "meta.json")
    if meta and meta.get("sources") == sources and meta.get("params") == params and not args.force:
        print(f"  [skip] cache up to date: {out}")
        return 0

    tmp = out.with_name(f"{out.name}.tmp-{utc_stamp()}")
    splits_meta = {}
    for name, (root, images) in resolved.items():
        print(f"  building {name} ...")
        splits_meta[name] = build_split(images, root, tmp / name, args.imgsz, args.task)
        assert splits_meta[name]["n_images"] == EXPECTED_COUNTS[name], f"{name}: cached {splits_meta[name]['n_images']} images"
    atomic_write_json(tmp / "meta.json", {
        "created_at": utc_now_iso(),
        "source_data_yaml": str(data_yaml),
        "source_root": str(next(iter(resolved.values()))[0]),
        "sources": sources,
        "params": params,
        "splits": splits_meta,
    })
    if out.exists():
        backup = out.with_name(f"{out.name}.bak-{utc_stamp()}")
        out.rename(backup)
        print(f"  previous cache kept as {backup}")
    tmp.rename(out)

    print(f"\nCache written to {out}")
    for name, m in splits_meta.items():
        print(f"  {name:<5}: {m['n_images']} images, {m['n_empty_gt']} empty masks, "
              f"foreground {m['mask_foreground_fraction']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
