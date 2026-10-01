# U-Net — Skin Lesion Segmentation (ISIC 2018 Task 1), PyTorch

This repository contains the U-Net arm of the study that compares segmentation architectures (YOLO26-seg, U-Net,
SAM) on ISIC 2018 Task 1. It runs the **same 5-phase protocol, metrics, evaluation resolution, efficiency
profiling and output schema as `sandbox_yolo26`**, so the two pipelines can be compared directly and analysed
with the same notebooks.

---

## What makes the comparison with YOLO26 fair

| Aspect | How it is guaranteed |
|---|---|
| **Same data** | Phase 0 builds the U-Net cache **from the YOLO26 dataset itself** (same images, same train/val/test split — 2,594/100/1,000, the official split — same polygon labels). Every sample is addressed by its ISIC ID. |
| **Same CV folds** | The CV pool is in YOLO26's order and uses YOLO26's K-Fold algorithm (NumPy `RandomState(0)`, no scikit-learn) → **identical folds** (verified). |
| **Same metrics, ground truth and resolution** | `unet/segmentation_metrics.py` is a **byte-identical copy** of YOLO26's. Predictions (256×256) are upsampled to the original dataset resolution and scored against the ground truth rasterised from the same YOLO labels. |
| **Same profiling** | `benchmark_efficiency.py` is derived from YOLO26's: same `torch.cuda.Event` timing, statistics, steady-state VRAM, contention checks and JSON schema; same PyTorch version. |
| **Same software stack** | The Docker image pins `torch==2.5.1` / `torchvision==0.20.1` (cu121) like YOLO26, plus `optuna==5.0.0`. |

**Resolution ceiling.** A perfect 256×256 prediction, processed by the U-Net inference pipeline (bilinear
upsampling to dataset resolution, threshold 0.5), scores **DSC 0.9967** on the test set (min 0.965) — the maximum
achievable at the U-Net's input resolution (measured with an oracle model on the 994 test images of the earlier
Roboflow export; **[re-measure on the official 1,000-image test set]**).

---

## The 5-phase protocol

| Phase | What | Data | Script(s) |
|---|---|---|---|
| **0 — Data cache** | 256×256 arrays from the YOLO26 dataset; strictly binary masks; ID manifests; SHA-256 provenance | train / val / test | `prepare_dataset.py` |
| **1 — Baseline** | Base setup + **Keras-baseline default** hyperparameters | train / val | `train_baseline_models.py` |
| **2 — Baseline CV** | 5-fold CV with the Phase 1 configuration; DSC/JSI per fold at dataset resolution | train ∪ val pool (**test excluded and verified**) | `train_cv_unet.py`, `consolidate_cv_results_unet.py`, `evaluate_cv_pixels.py` |
| **3 — HPO** | Optuna TPE, **seeded per proposal**, fault-tolerant and resumable | train / val | `tune_unet.py`, `check_hpo_validity.py` |
| **4 — Optimised** | Same base setup + Phase 3 hyperparameters | train / val | `train_optimized_unet.py` |
| **5 — Test set** | Baseline **and** Optimised: DSC, JSI, ISIC thresholded JSI, sensitivity, specificity (FP32 + FP16); batch-1 efficiency (FP32 + FP16); final report | **test** (only here) | `evaluate_test_set.py`, `benchmark_efficiency.py`, `build_final_report.py` |

Everything is orchestrated by **`run_pipeline_unet.sh`**; the protocol is defined once in **`unet/common.py`**.

### Base setup vs. tuned hyperparameters

**Baseline = base setup + default hyperparameters; Optimised = base setup + tuned hyperparameters.** The
base setup is identical in every phase and can never be overridden by a tuned file or a search space:

| Base setup (fixed) | Value |
|---|---|
| Architecture | U-Net of the original Keras baseline: 16 base filters, ReLU, BatchNorm, 4 levels + bottleneck (2,161,649 parameters) |
| Optimiser / schedule | AdamW (decoupled weight decay ≡ Keras `Adam(weight_decay)`), eps 1e-7, β2 0.999, constant LR |
| Loss | BCE + soft Dice (original Keras `bce_dice_loss`) |
| Input / batch | 256×256, batch 16 |
| Budget | 120 epochs, no early stopping (patience 120; HPO trials: 30 epochs, patience 30); `best.pt` = best validation JSI |
| Numerics | FP32 (`amp=False`), seed 0, deterministic algorithms |

| Tuned (Phase 3) | Default (Baseline) | Search range |
|---|---|---|
| `lr0` | 1e-3 | [1e-4, 1e-2] log |
| `weight_decay` | 0 | [1e-6, 1e-3] log *(the default 0 lies outside the log range; the first trial is clipped to 1e-6)* |
| `beta1` | 0.9 | [0.80, 0.95] |
| `dropout` | 0.1 | [0.10, 0.40] |
| `degrees` (rotation) | 15 | [0, 45] |
| `translate` (shift) | 0.1 | [0, 0.20] |
| `scale` (zoom) | 0.1 | [0, 0.30] |
| `fliplr` / `flipud` (probabilities) | 0.5 / 0.5 | [0, 0.50] |

The augmentation re-implements the original Keras `ImageDataGenerator` jointly on image and mask, with one
deliberate fix: **masks are warped with nearest-neighbour interpolation and stay binary** (the Keras pipeline
interpolated them bilinearly into soft labels).

### Faithful port of the Keras model

`unet/model.py` reproduces the Keras U-Net layer by layer (Keras BatchNorm momentum/epsilon, `he_normal` and
`glorot_uniform` initialisation, Keras "same" alignment of the transposed convolutions). With weights copied
from the original Keras model the PyTorch output is **identical (max difference 0.0)** and the parameter
count matches exactly.

### Reproducibility and fault tolerance

* **Bit-exact resume.** `last.pt` stores the model, optimiser, epoch, early-stopping state and all RNG states;
  data order and augmentation are pure functions of (seed, epoch, sample). A training run killed at any point
  and resumed is **bit-identical** to an uninterrupted one (verified on CPU and GPU, also with 0 vs. 8 workers).
  `results.csv` is rebuilt from the checkpoint on resume.
* **Model selection.** The epoch with the highest validation per-image mean JSI (the ISIC ranking metric) is
  `best.pt`; reported validation metrics are always those of that epoch.
* **HPO.** `hpo_state.json` (same schema as YOLO26) is checkpointed before every trial; interrupted trials
  are resumed **including their training**; failed trials are retried with identical parameters (≤ 2 times);
  GPU failures exit with code 75 and are retried by the orchestrator; the proposal of each trial is recorded
  and re-verified; changes of search space, base setup, seed, data or library versions are refused.
* **Locks** (POSIX `lockf`) prevent two processes from writing the same run; orphaned DataLoader workers of a
  killed process cannot hold them.
* `--force` never deletes: previous outputs are moved to `*.bak-<UTC>`.
* **Statistics:** CV mean ± sample SD (ddof = 1); test per-image mean with seeded bootstrap 95 % CI; HPO gain
  as paired per-image difference with bootstrap CI and Wilcoxon signed-rank test.

---

## Running the pipeline

### Build the image

```bash
docker build -t unet_ft .
```

### Full run (all phases)

```bash
GPU=1                                  # host GPU index
PIPELINE_NAME="pipeline_final_v1"

mkdir -p "logs/${PIPELINE_NAME}"     # the terminal log goes inside the pipeline folder
docker run --gpus "\"device=${GPU}\"" -it --rm --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/cache -e TORCH_HOME=/workspace/cache/torch \
    -e GPU_DEVICE=0 -e PIPELINE_NAME="${PIPELINE_NAME}" \
    -e YOLO_DATA_YAML=/workspace/yolo26_dataset/data.yaml \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/../sandbox_yolo26/datasets/isic2018_task1_official:/workspace/yolo26_dataset:ro" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/unet:/workspace/unet" \
    -v "$(pwd)/run_pipeline_unet.sh:/workspace/run_pipeline_unet.sh:ro" \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    unet_ft \
    bash /workspace/run_pipeline_unet.sh \
    2>&1 | tee "logs/${PIPELINE_NAME}/terminal_$(date -u +%Y%m%dT%H%M%SZ).log"
```

* The YOLO26 dataset is mounted **read-only** (it is the source of truth) at `/workspace/yolo26_dataset`, a
  sibling of the `datasets` mount — never inside it: a mount nested in a bind mount makes Docker create an
  empty, root-owned mount-point folder on the host (`datasets/isic_2018_task1_yolo26/`; delete it with
  `sudo rmdir` if an older command created it). The Phase 0 cache is written to `datasets/isic_2018_task1_unet256/`.
* Inside the container the selected GPU is index `0` (hence `GPU_DEVICE=0`).
* If the run is interrupted for any reason, **run the same command again** — it resumes.

### Common variations (arguments after `bash /workspace/run_pipeline_unet.sh`)

```bash
--phases "3 4 5"                # a subset of phases
--dry-run                       # print the commands only
--epochs 3 --patience 2         # SMOKE TEST ONLY (applied to Phases 1, 2 and 4 together)
--bench-device 0                # GPU used for Phase 5 (default: --device)
--pipeline-name pipeline_final_v2   # a fresh, isolated study
--force                         # start the selected phases over (old outputs → *.bak-<UTC>)
```

Environment overrides (defaults): `CV_K_FOLDS=5`, `CV_SEED=0`, `HPO_ITERATIONS=30`, `HPO_EPOCHS_PER_TRIAL=30`,
`HPO_PATIENCE=30`, `HPO_MAX_RETRIES=5`, `HPO_RETRY_WAIT=600`, `EVAL_PRECISIONS="fp32 fp16"`, `YOLO_DATA_YAML`,
`CACHE_DIR`, `LOGS_ROOT`, `PROJECT`.

Exit codes: `0` success · `75` the HPO gave up after repeated GPU failures (fix the driver and re-run to
resume) · otherwise the exit code of the failing step (log in `pipeline_runs/<UTC>/`).

### Waiting for an idle GPU

```bash
GPU_DEVICE=1 ./wait_gpu_unet.sh     # polls nvidia-smi, then launches the docker command above
```

---

## Phase 5 — what exactly is measured

**Accuracy (`evaluate_test_set.py`)** — test split only, batch 1, FP32 (primary) and FP16. The 256×256
probability map is upsampled bilinearly to dataset resolution and thresholded at 0.5; per image: DSC, JSI, ISIC
thresholded JSI (`JSI < 0.65 → 0`), sensitivity, specificity, accuracy (empty prediction → 0, never skipped).
Same aggregates and JSON/CSV schema as YOLO26; YOLO26's Ultralytics-only instance metrics (box/mask mAP, P, R,
F1) are present as `NaN`.

**Efficiency (`benchmark_efficiency.py`)** — batch 1, one GPU, each configuration in a fresh process:

* forward latency of the **fused** U-Net (BatchNorm folded, as Ultralytics' `fuse()` for YOLO26) on
  1×3×256×256 with `torch.cuda.Event` (50 warm-up + 500 timed); end-to-end latency of the deployed pipeline
  (640 image → resize → forward → upsample → threshold) with `perf_counter` (20 + 200);
* mean, SD, median, P90/P95/P99, FPS = 1000 / mean;
* steady-state peak VRAM after warm-up (cuDNN autotune workspaces excluded and reported separately), weight
  VRAM, host RAM;
* parameters (2,161,649; fused 2,158,705), GFLOPs (2 × MACs, thop) at the native 256×256 input (6.40) and,
  for a resolution-matched comparison with YOLO26, at 640×640 (`gflops_640`, 40.0);
* contended runs (another process using the GPU) are flagged — re-run them on an idle GPU.

Note for the comparison: the U-Net is benchmarked at its native 256×256 input, YOLO26 at 640×640; report
both `gflops` and `gflops_640`.

---

## Output layout

```
logs/pipeline_final_v1/
├── phase1_baseline/unet_baseline/          weights/{best,last}.pt, results.csv, run_state.json
├── phase2_cv_baseline/unet/                splits_manifest.json, runs/fold_<k>/, metrics_per_fold.csv,
│                                           metrics_summary.json, pixel_metrics_per_fold.csv, pixel_metrics_summary.json
├── phase3_hpo/tune_unet/                   tune_results.csv, best_hyperparameters.yaml, hpo_state.json,
│                                           optuna_study.db, trials/trial_<i>/
├── phase4_optimized/unet_optimized/        weights/, results.csv, run_state.json, tuned_hyperparameters.yaml
├── phase5_test/{accuracy, per_image, masks, efficiency}/
├── summary/                                phase1_val, phase2_cv_baseline, phase2_cv_pixel, phase4_val,
│                                           test_accuracy, efficiency, hpo_gain, final_results (.csv/.json)
├── figures/  tables/                       written by the notebooks
└── pipeline_runs/<UTC>/                    pipeline.log + one log per step
```

The `summary/` files have **exactly the same columns as YOLO26's**.

## Project structure

```
sandbox_unet/
├── run_pipeline_unet.sh   wait_gpu_unet.sh   Dockerfile
├── unet/
│   ├── common.py              # base setup, default HPs, protocols, paths, locks
│   ├── prepare_dataset.py     # Phase 0
│   ├── model.py               # PyTorch port of the Keras U-Net (+ Conv-BN fusion)
│   ├── data.py                # cache access, Keras-equivalent augmentation, deterministic loaders
│   ├── training.py            # bit-exact resumable training
│   ├── inference.py           # full-resolution prediction + scoring
│   ├── segmentation_metrics.py  # byte-identical to YOLO26's
│   ├── train_baseline_models.py  train_cv_unet.py  consolidate_cv_results_unet.py  evaluate_cv_pixels.py
│   ├── tune_unet.py  check_hpo_validity.py  train_optimized_unet.py  collect_phase_metrics_unet.py
│   ├── evaluate_test_set.py  benchmark_efficiency.py  build_final_report.py
│   └── legacy/                # previous TensorFlow/Keras scripts (not used)
├── notebooks/
│   ├── 01_Segmentation_Visualizer.ipynb
│   └── 02_Metrics_and_Efficiency_Analysis.ipynb
├── utils/                     # earlier notebooks
└── datasets/  logs/           # not versioned
```

## Analysis notebooks

Same notebooks as YOLO26, adapted to the U-Net (they read only the pipeline outputs; no GPU needed):
`01_Segmentation_Visualizer` (ground truth green/solid vs. prediction red/dashed, Baseline vs. Optimised) and
`02_Metrics_and_Efficiency_Analysis` (DSC/JSI across phases, paired HPO gain, accuracy vs. size, latency vs.
FPS, latency distribution, memory, accuracy–latency trade-off, LaTeX tables). The YOLO26 dataset is located
automatically (`datasets/` or `../sandbox_yolo26/datasets/`; the command below mounts the parent folder so the
sibling repository is visible).

```bash
docker run --rm -it -p 8888:8888 --user "$(id -u):$(id -g)" -e HOME=/workspace/cache \
    -v "$(pwd)/..:/projects" -w /projects/sandbox_unet \
    unet_ft jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/projects/sandbox_unet
```

---

## Running on a remote server

```bash
screen -S unet_ft        # start; run the docker command above
# Ctrl + A, then D       # detach
screen -r unet_ft        # reattach
```

Copy the results:

```bash
rsync -avz --progress -e "ssh -p 13508" \
    antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/sandbox_unet/logs/pipeline_final_v1 \
    /home/avmoura_linux/Documents/unb/SANDBOX_UNET/logs/
```

Hardware monitoring: `nvidia-smi`, `nvtop`.
