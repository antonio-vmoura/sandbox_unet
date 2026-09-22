# U-Net Baseline Training

This repository provides scripts and configuration files for training the **U-Net** model on a custom dataset, focusing on skin lesion segmentation. 

The goal is to establish a rigorous, reproducible baseline on medical images (ISIC 2018 Task 1) to evaluate and compare segmentation performance against state-of-the-art architectures like YOLO and SAM.

---

## Overview

The training pipeline includes:

* Loading and preparing the dataset (Pre-processed NumPy arrays `.npy`)
* Training a vanilla U-Net baseline for semantic segmentation
* Automatic saving of checkpoints, history, and metrics
* Fully reproducible execution via Docker with GPU support

---

## Requirements

* Docker with NVIDIA GPU support
* NVIDIA Container Toolkit installed
* Dataset (in NumPy format: `TRAINING_IMAGES.npy`, `TRAINING_MASKS.npy`, etc.) available at:

```
./datasets/<dataset_name>
```

---

## Expected Project Structure

```
sandbox_unet/
│
├── logs/               # Training outputs
├── datasets/           # Dataset (Numpy arrays)
├── unet_seg/           # Model source code (train_baseline_unet.py)
└── utils/              # Useful scripts
```

---

## Environment Setup

### Build the Docker Image

Build the environment containing CUDA, TensorFlow/PyTorch, and all necessary dependencies:

```bash
docker build -t unet_ft .
```

---

## Training Execution

### Option A: Run training using ALL available GPUs

```bash
docker run --gpus all --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/unet_seg:/workspace/unet_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  unet_ft \
  python /workspace/unet_seg/train_baseline_unet.py \
    --data_dir /workspace/datasets/isic_2018_task1_unet \
    --project /workspace/logs/pipeline_unet_v1 2>&1 | tee logs/unet_baseline.log
```

### Option B: Run training using a SINGLE GPU

```bash
docker run --gpus '"device=0"' --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/unet_seg:/workspace/unet_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  unet_ft \
  python /workspace/unet_seg/train_baseline_unet.py \
    --data_dir /workspace/datasets/isic_2018_task1_unet \
    --project /workspace/logs/pipeline_unet_v1 2>&1 | tee logs/unet_baseline_gpu0.log
```

### Option C: Automated Training (Wait for Free GPUs)

```bash
chmod +x wait_gpu_unet.sh
```

```bash
nohup ./wait_gpu_unet.sh > nohup_unet_wait.log 2>&1 &
```

---

## Running on a Remote Server

### Run training in the background

Create a screen session:

```bash
screen -S unet_ft
```

Run the Docker orchestrator command normally (e.g., `bash run_pipeline_unet.sh`). Detach while keeping the process running:

```text
Ctrl + A, then D
```

Reattach later:

```bash
screen -r unet_ft
```

---

### Copy results from the server

```bash
rsync -avz --progress -e "ssh -p 13508 -v" antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/SANDBOX_UNET/logs/pipeline_unet_v1 /home/avmoura_linux/Documents/unb/SANDBOX_UNET
```

---

### Environment Setup (Local/Host)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy pandas jupyterlab
```

---

### Hardware Monitoring

```bash
nvidia-smi
nvtop
```

---

## End-to-End Pipeline (`run_pipeline_unet.sh`)

The master orchestrator `run_pipeline_unet.sh` sets up the training environment and executes the required phases for the ISIC 2018 Task 1 study using the U-Net architecture.

Currently, the orchestrator handles:

1. **Phase 1 — Baseline:** Standard Keras compilation, `epochs=120`, `patience=20`, `deterministic=True`, `seed=0`, `amp=False` (FP32 precision for strict reproducibility). Implemented by `unet_seg/train_baseline_unet.py`.

### Recommended Logs Layout

Every artefact of a pipeline run is isolated under `logs/<PIPELINE_NAME>/` (default `logs/pipeline_unet_v1/`) so it does **not** mix with other runs. 

```
logs/
├── pipeline_unet_v1/                                # ← PIPELINE_NAME
│   ├── phase1_baseline/
│   │   └── unet_baseline/{best_model.h5, results.csv}
│   └── pipeline_runs/<UTC-timestamp>/{pipeline_unet.log}
│
├── pipeline_unet_v2/                                # ← a future re-run
│   └── ...
```

### Idempotency

The Python script already detects existing artefacts:

* `train_baseline_unet.py` skips the training if an existing `best_model.h5` is found. 
Pass `--force` to the script if you need to re-execute the phase and overwrite existing weights.

### Running the pipeline inside Docker

Run the orchestrator with explicit GPU allocation (e.g., `GPU_DEVICE_IDS="0"`):

```bash
GPU_DEVICE_IDS="0"
PIPELINE_NAME="pipeline_unet_v1"

docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -e GPU_DEVICE_IDS="${GPU_DEVICE_IDS}" \
    -e PIPELINE_NAME="${PIPELINE_NAME}" \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/unet_seg:/workspace/unet_seg" \
    -v "$(pwd)/run_pipeline_unet.sh:/workspace/run_pipeline_unet.sh:ro" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    unet_ft \
    bash /workspace/run_pipeline_unet.sh
```