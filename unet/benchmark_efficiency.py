"""Phase 5b — Hardware-efficiency benchmark (batch=1) of the Baseline and Optimised U-Net.

Same measurement method and output schema as YOLO26's
``yolo26_seg/benchmark_efficiency.py`` (this file is derived from it; only the
model-specific parts differ). For every ``variant`` × ``model`` ×
``precision ∈ {fp32, fp16}`` it measures, on a **single GPU** with
**batch = 1**:

* **Latency** (ms) — mean, std, median, P90, P95, P99, min, max — of two
  scopes:

  - ``forward``: the **fused** U-Net (BatchNorm folded into the convolutions,
    :func:`model.fuse` — the inference form, as Ultralytics' ``fuse()`` for
    YOLO26) on a fixed 1×3×256×256 input (the U-Net's native input size),
    timed with ``torch.cuda.Event`` pairs and a ``synchronize`` after every
    iteration. Primary "model latency" figure.
  - ``end_to_end``: the deployed prediction pipeline on a real test image as
    stored in the dataset (640×640): BGR→RGB, area-resize to 256, scaling to
    [0, 1], host→device copy, forward, sigmoid, bilinear upsampling back to
    640×640, threshold 0.5, device→host copy of the mask. Timed with
    ``time.perf_counter`` around a synchronised call (it includes CPU work).

* **FPS** = 1000 / mean latency (``fps``) and 1000 / median (``fps_median``).
* **Memory** — steady-state peak VRAM allocated / reserved by PyTorch during
  the timed iterations (peak counters reset **after** warm-up, so cuDNN
  autotuning workspaces are excluded; the warm-up peak is kept separately as
  ``vram_peak_warmup_mb``), the VRAM of the weights alone, and host RAM (RSS
  after loading / benchmarking and peak RSS). The CUDA context is excluded.
* **Model size** — size of ``best.pt`` on disk (FP32 state dict + metadata),
  theoretical FP32/FP16 weight sizes (fused params × 4 / × 2 bytes),
  parameter count (unfused and fused), GFLOPs (2 × MACs, thop — the YOLO26
  convention) at the native 256×256 input (``gflops``) and, for a
  resolution-matched comparison with YOLO26, at 640×640 (``gflops_640``).

Isolation & fairness (as YOLO26): every configuration runs in a **fresh
subprocess**; ``cudnn.benchmark=True`` (recorded); warm-up iterations are
discarded; GPU utilisation of other processes is recorded before/after each
run and a busy GPU is flagged ``contended``.

Outputs::

    <project>/phase5_test/efficiency/<variant>_<model>_<precision>.json

(including the raw per-iteration latencies). Results whose weights and
settings are unchanged are skipped.

Usage:
    python benchmark_efficiency.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    IMGSZ,
    SEED,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    read_json,
    sha256_file,
    utc_now_iso,
)

VARIANTS: tuple[str, ...] = ("baseline", "optimized")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")

#: Version of the measurement method. Part of the cache key: bump it whenever
#: what or how this script measures changes, so stale results are recomputed.
BENCHMARK_VERSION: int = 2

#: GPU utilisation (%) above which a run is flagged as contended.
CONTENTION_UTIL_PCT: int = 5

MB: float = 1024 ** 2


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def latency_stats(ms: list[float]) -> dict[str, float]:
    """Summarise per-iteration latencies (ms) and derive FPS."""
    a = np.asarray(ms, dtype=float)
    mean, median = float(a.mean()), float(np.median(a))
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": median,
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "fps": 1000.0 / mean,
        "fps_median": 1000.0 / median,
    }


# ----------------------------------------------------------------------------
# Worker (runs in its own process)
# ----------------------------------------------------------------------------
def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / MB


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark one weights file at one precision; return the measurement dict."""
    import copy

    import cv2
    import thop
    import torch
    import torch.nn.functional as F

    from model import count_parameters, fuse
    from training import load_model

    torch.manual_seed(SEED)
    cuda = args.device != "cpu"
    dev = torch.device(f"cuda:{args.device}" if cuda else "cpu")
    half = args.precision == "fp16"
    if half and not cuda:
        raise RuntimeError("FP16 benchmarking requires a CUDA device")
    dtype = torch.float16 if half else torch.float32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    rss_start = _rss_mb()
    if cuda:
        torch.cuda.set_device(dev)
        torch.zeros(1, device=dev)  # create the CUDA context before measuring
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    rss_ctx = _rss_mb()
    alloc0 = torch.cuda.memory_allocated(dev) if cuda else 0

    # ---- Model statistics (CPU, FP32) --------------------------------------
    net = load_model(Path(args.weights), torch.device("cpu"))
    params = count_parameters(net)
    net = fuse(net)
    params_fused = count_parameters(net)

    def _gflops(size: int) -> float:  # 2 × MACs (thop), as Ultralytics' get_flops
        macs, _ = thop.profile(copy.deepcopy(net), inputs=(torch.zeros(1, 3, size, size),), verbose=False)
        return 2 * macs / 1e9

    gflops, gflops_640 = _gflops(IMGSZ), _gflops(640)

    # ---- Forward-pass benchmark ---------------------------------------------
    net = net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if half:
        net = net.half()
    weights_vram = (torch.cuda.memory_allocated(dev) - alloc0) / MB if cuda else None
    rss_model = _rss_mb()

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.rand(1, 3, IMGSZ, IMGSZ, generator=gen).to(dev, dtype)
    fwd_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            net(x)
        if cuda:
            torch.cuda.synchronize(dev)
            warmup_peak = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB
            torch.cuda.reset_peak_memory_stats(dev)  # exclude cuDNN autotune workspaces
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            if cuda:
                start.record()
                net(x)
                end.record()
                torch.cuda.synchronize(dev)
                fwd_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                net(x)
                fwd_ms.append((time.perf_counter() - t0) * 1000)
    fwd_peak_alloc = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB if cuda else None
    fwd_peak_reserved = torch.cuda.max_memory_reserved(dev) / MB if cuda else None

    # ---- End-to-end prediction pipeline benchmark ---------------------------
    image = cv2.imread(args.sample_image)
    if image is None:
        raise RuntimeError(f"cannot read sample image {args.sample_image}")
    h0, w0 = image.shape[:2]

    @torch.inference_mode()
    def predict(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(small).permute(2, 0, 1)[None].to(dev).to(dtype).div_(255.0)
        prob = torch.sigmoid(net(t).float())
        up = F.interpolate(prob, size=(h0, w0), mode="bilinear", align_corners=False)
        return (up[0, 0] > 0.5).cpu().numpy()

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(args.e2e_warmup):
        predict(image)
    if cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    e2e_ms: list[float] = []
    for _ in range(args.e2e_iters):
        if cuda:
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        predict(image)
        if cuda:
            torch.cuda.synchronize(dev)
        e2e_ms.append((time.perf_counter() - t0) * 1000)
    e2e_peak_alloc = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB if cuda else None
    e2e_peak_reserved = torch.cuda.max_memory_reserved(dev) / MB if cuda else None

    size_disk = os.path.getsize(args.weights) / MB
    return {
        "forward": {**latency_stats(fwd_ms), "timer": "torch.cuda.Event" if cuda else "perf_counter",
                    "input_shape": [1, 3, IMGSZ, IMGSZ], "raw_ms": fwd_ms},
        "end_to_end": {**latency_stats(e2e_ms), "timer": "perf_counter+synchronize",
                       "sample_image": args.sample_image, "raw_ms": e2e_ms},
        "memory": {
            "vram_weights_mb": weights_vram,
            "vram_peak_allocated_mb": fwd_peak_alloc,
            "vram_peak_reserved_mb": fwd_peak_reserved,
            "vram_peak_allocated_e2e_mb": e2e_peak_alloc,
            "vram_peak_reserved_e2e_mb": e2e_peak_reserved,
            "vram_peak_warmup_mb": warmup_peak if cuda else None,
            "ram_rss_start_mb": rss_start,
            "ram_rss_after_cuda_init_mb": rss_ctx,
            "ram_rss_model_loaded_mb": rss_model,
            "ram_rss_end_mb": _rss_mb(),
            "ram_peak_rss_mb": _peak_rss_mb(),
            "ram_model_delta_mb": rss_model - rss_ctx,
            "note": "VRAM from the PyTorch allocator (CUDA context excluded), steady state "
                    "after warm-up; vram_peak_warmup_mb includes cuDNN autotune workspaces; "
                    "*_e2e_* covers the deployed pipeline (resize, forward, upsample, threshold).",
        },
        "model": {
            "params": int(params),
            "params_fused": int(params_fused),
            "gflops": gflops,
            "gflops_640": gflops_640,
            "gflops_input": f"{IMGSZ}x{IMGSZ} (native); gflops_640 at 640x640",
            "size_mb_disk": size_disk,
            "size_disk_dtype": "fp32 (PyTorch state_dict + metadata)",
            "size_mb_fp32_theoretical": params_fused * 4 / MB,
            "size_mb_fp16_theoretical": params_fused * 2 / MB,
        },
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": args.cudnn_benchmark,
            "gpu": torch.cuda.get_device_name(dev) if cuda else None,
            "cpu": platform.processor() or platform.machine(),
        },
    }


# ----------------------------------------------------------------------------
# Parent orchestration
# ----------------------------------------------------------------------------
def gpu_snapshot(device: str) -> dict[str, Any]:
    """Utilisation / memory of the benchmark GPU right before a run (via nvidia-smi)."""
    if device == "cpu":
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(device), "--format=csv,noheader,nounits",
             "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,"
             "clocks.sm,clocks.max.sm,temperature.gpu"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip().split(", ")
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    keys = ["name", "driver", "util_pct", "mem_used_mb", "mem_total_mb", "sm_clock_mhz",
            "sm_clock_max_mhz", "temp_c"]
    snap: dict[str, Any] = dict(zip(keys, out))
    for k in keys[2:]:
        try:
            snap[k] = float(snap[k])
        except (KeyError, ValueError):
            pass
    snap["contended"] = isinstance(snap.get("util_pct"), float) and snap["util_pct"] > CONTENTION_UTIL_PCT
    return snap


def benchmark_one(
    variant: str, model_size: str, precision: str, args: argparse.Namespace,
    paths: PipelinePaths, sample_image: str,
) -> dict[str, Any]:
    """Run (or skip) one configuration in a fresh worker process and save its JSON."""
    weights = paths.best_pt(variant, model_size)
    if not weights.exists():
        raise RuntimeError(f"weights not found: {weights} (run Phase 1/4 first)")
    out_json = paths.phase5_efficiency_json(variant, model_size, precision)
    settings = {
        "weights_sha256": sha256_file(weights), "precision": precision, "imgsz": IMGSZ,
        "warmup": args.warmup, "iters": args.iters, "e2e_warmup": args.e2e_warmup,
        "e2e_iters": args.e2e_iters, "cudnn_benchmark": args.cudnn_benchmark,
        "device": args.device, "sample_image": sample_image,
        "benchmark_version": BENCHMARK_VERSION,
    }
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        return {"tag": out_json.stem, "skipped": True, "payload": previous}

    before = gpu_snapshot(args.device)
    if before.get("contended"):
        print(f"  [warn] GPU {args.device} is busy ({before['util_pct']:.0f}% util, "
              f"{before['mem_used_mb']:.0f} MB used) — latencies will be flagged as contended")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_json = Path(tmp) / "result.json"
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--weights", str(weights), "--precision", precision, "--device", args.device,
            "--sample-image", sample_image, "--out", str(tmp_json),
            "--warmup", str(args.warmup), "--iters", str(args.iters),
            "--e2e-warmup", str(args.e2e_warmup), "--e2e-iters", str(args.e2e_iters),
        ]
        if not args.cudnn_benchmark:
            cmd.append("--no-cudnn-benchmark")
        subprocess.run(cmd, check=True)
        result = json.loads(tmp_json.read_text())
    after = gpu_snapshot(args.device)

    payload = {
        "variant": variant, "model": model_size, "precision": precision, "batch": 1,
        "weights": str(weights), **result,
        "gpu_before": before, "gpu_after": after,
        "contended": bool(before.get("contended") or after.get("contended")),
        "settings": settings, "settings_hash": config_hash(settings), "created_at": utc_now_iso(),
    }
    atomic_write_json(out_json, payload)
    return {"tag": out_json.stem, "skipped": False, "payload": payload}


def _first_test_image(cache_dir: str, data_root: str | None) -> str:
    """Deterministic sample image for the end-to-end benchmark (first test image, 640×640)."""
    from data import CacheData
    from inference import resolve_data_root

    cache = CacheData(cache_dir)
    first = cache.record(cache.ids("test")[0])
    return str(resolve_data_root(cache, data_root) / first["image"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (parent and ``--worker`` modes)."""
    p = argparse.ArgumentParser(description="Phase 5b — batch=1 efficiency benchmark (FP32 / FP16).")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--cache", default=DEFAULT_CACHE_DIR, help="Phase 0 cache (first test image = e2e sample).")
    p.add_argument("--data-root", default=None, help="YOLO dataset root (default: from the cache).")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0); 'cpu' for smoke tests.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--warmup", type=int, default=50, help="Discarded forward iterations (default: 50).")
    p.add_argument("--iters", type=int, default=500, help="Timed forward iterations (default: 500).")
    p.add_argument("--e2e-warmup", type=int, default=20, help="Discarded predict() calls (default: 20).")
    p.add_argument("--e2e-iters", type=int, default=200, help="Timed predict() calls (default: 200).")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false",
                   help="Disable cuDNN autotuning (default: enabled, as in deployment).")
    p.add_argument("--force", action="store_true", help="Re-run even if results are up to date.")
    # Worker mode (internal).
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--weights", help=argparse.SUPPRESS)
    p.add_argument("--precision", choices=PRECISIONS, help=argparse.SUPPRESS)
    p.add_argument("--sample-image", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    """Benchmark every requested configuration (or run one worker).

    Returns:
        ``0`` on success, ``1`` if any configuration failed, ``2`` on bad usage.
    """
    args = parse_args()
    if args.worker:
        Path(args.out).write_text(json.dumps(run_worker(args)))
        return 0
    if "," in args.device:
        print("[error] Phase 5b must run on a single device (e.g. --device 0).", file=sys.stderr)
        return 2

    paths = PipelinePaths(Path(args.project))
    sample_image = _first_test_image(args.cache, args.data_root)
    print(f"Phase 5b — efficiency (batch=1) on device {args.device}")
    print(f"  variants={args.variants} models={args.models} precisions={args.precisions}")
    print(f"  forward: {args.warmup} warm-up + {args.iters} timed | "
          f"predict(): {args.e2e_warmup} warm-up + {args.e2e_iters} timed")

    failures = 0
    for variant in args.variants:
        for m in args.models:
            for precision in args.precisions:
                try:
                    r = benchmark_one(variant, m, precision, args, paths, sample_image)
                    pl = r["payload"]
                    status = "skip (up to date)" if r["skipped"] else "ok"
                    print(
                        f"  [{status}] {r['tag']:<26} fwd median={pl['forward']['median_ms']:.2f} ms "
                        f"P95={pl['forward']['p95_ms']:.2f} ms  FPS={pl['forward']['fps']:.1f} | "
                        f"e2e median={pl['end_to_end']['median_ms']:.2f} ms | "
                        f"VRAM peak={pl['memory']['vram_peak_allocated_mb'] or 0:.0f} MB"
                        f"{'  [CONTENDED]' if pl['contended'] else ''}",
                    )
                except Exception:
                    failures += 1
                    print(f"  [fail] {variant}/{m}/{precision}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
