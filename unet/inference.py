"""Full-resolution U-Net inference and pixel scoring (Phase 2 pixel step, Phase 5).

To make U-Net and YOLO26 scores directly comparable, predictions are scored
**exactly like YOLO26's**: against the same official ground-truth masks
(:func:`segmentation_metrics.ground_truth_mask`) at each image's resolution in
the shared dataset (the working resolution of Phase 0), with the same
per-image metric definitions (:func:`segmentation_metrics.pixel_scores`).

Prediction pipeline (per image):

1. Network input: the image resized to 256 × 256 with area interpolation and
   scaled to [0, 1] — byte-identical to the Phase 0 cache, which is therefore
   used directly.
2. Sigmoid probability map (256 × 256) → **bilinear upsampling** to the
   original resolution (``align_corners=False``) → threshold 0.5.
3. Per-image confusion counts and scores against the full-resolution ground
   truth.

Output rows use YOLO26's per-image CSV columns: ``image``, ``height``,
``width``, ``n_pred`` (number of connected components of the predicted mask —
the analogue of YOLO's instance count), ``max_conf`` (maximum pixel
probability), ``infer_ms`` and every key of :func:`pixel_scores`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data import CacheData
from segmentation_metrics import ground_truth_mask, pixel_scores

#: Version of the evaluation method (part of the result cache keys).
EVAL_VERSION: int = 2   # 2: + boundary metrics (BIoU, NSD)

#: Probability threshold of the binary prediction.
PROB_THRESHOLD: float = 0.5

#: Container locations of the YOLO dataset (fallbacks for the data root): the
#: mount used by the README / wait_gpu_unet.sh, then the former nested mount.
DEFAULT_YOLO_ROOTS: tuple[str, ...] = ("/workspace/yolo26_dataset", "/workspace/datasets/isic2018_task1_official",
                                       "/workspace/datasets/isic_2018_task1_yolo26")


def resolve_data_root(cache: CacheData, override: str | None = None) -> Path:
    """Root of the YOLO dataset holding the original images and labels.

    Order: explicit override → the root recorded by Phase 0 → the Docker mounts.
    Empty folders are skipped (an older nested ``docker run -v`` left an empty
    mount-point folder at the former location).
    """
    for cand in (override, cache.meta.get("source_root"), *DEFAULT_YOLO_ROOTS):
        if cand and Path(cand).is_dir() and any(Path(cand).iterdir()):
            return Path(cand)
    raise FileNotFoundError("YOLO dataset root not found; pass --data-root")


def to_input(images: np.ndarray, device: torch.device, half: bool) -> torch.Tensor:
    """``N×S×S×3`` uint8 → ``N×3×S×S`` float in [0, 1] on ``device``."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).to(device)
    return x.half().div_(255.0) if half else x.float().div_(255.0)


@torch.no_grad()
def predict_full_res(model: torch.nn.Module, x: torch.Tensor, h: int, w: int) -> tuple[np.ndarray, float]:
    """Binary mask at ``h × w`` and the maximum probability for one input ``1×3×S×S``."""
    prob = torch.sigmoid(model(x).float())
    up = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    return (up[0, 0] > PROB_THRESHOLD).cpu().numpy(), float(prob.max())


def evaluate_ids(
    model: torch.nn.Module,
    cache: CacheData,
    ids: list[str],
    data_root: Path,
    device: torch.device,
    *,
    half: bool = False,
    mask_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Predict every ID (batch = 1) and score it at full resolution.

    Args:
        model: U-Net in eval mode (FP16 weights when ``half``).
        cache: Phase 0 cache (network inputs + manifest).
        ids: ISIC IDs to evaluate.
        data_root: YOLO dataset root (original labels).
        device: Inference device.
        half: FP16 inference.
        mask_dir: If given, each predicted mask is saved as ``<image stem>.png``
            (0/255) for the visualisation notebook.

    Returns:
        One row per image in the YOLO26 per-image schema.
    """
    if mask_dir is not None:
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
    images, _ = cache.arrays(ids)
    model.eval()
    rows = []
    for i, iid in enumerate(ids):
        rec = cache.record(iid)
        h, w = int(rec["orig_h"]), int(rec["orig_w"])
        t0 = time.perf_counter()
        pred, max_prob = predict_full_res(model, to_input(images[i:i + 1], device, half), h, w)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        infer_ms = (time.perf_counter() - t0) * 1000
        gt = ground_truth_mask(data_root / rec["image"], h, w)
        image_path = data_root / rec["image"]
        if mask_dir is not None:
            cv2.imwrite(str(Path(mask_dir) / f"{image_path.stem}.png"), pred.astype(np.uint8) * 255)
        n_pred = cv2.connectedComponents(pred.astype(np.uint8))[0] - 1
        rows.append({"image": str(image_path), "height": h, "width": w, "n_pred": int(n_pred),
                     "max_conf": max_prob, "infer_ms": infer_ms, **pixel_scores(gt, pred)})
        if (i + 1) % 100 == 0 or i + 1 == len(ids):
            print(f"    {i + 1}/{len(ids)} images", flush=True)
    return rows
