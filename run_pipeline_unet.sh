#!/usr/bin/env bash
# =============================================================================
# run_pipeline_unet.sh — Orquestrador para o fine-tuning do Baseline da U-Net
# no dataset ISIC 2018 Task 1 (Arrays Numpy).
# =============================================================================
set -euo pipefail

# ---------- Caminhos no HOST (Servidor Físico) -------------------------------
HOST_LOGS_DIR="$(pwd)/logs"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_unet_v1}"
HOST_PROJECT_DIR="${HOST_LOGS_DIR}/${PIPELINE_NAME}"

UNET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/unet"

# Criação das pastas de log diretamente na máquina local
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG_DIR="${HOST_PROJECT_DIR}/pipeline_runs/${RUN_TS}"
mkdir -p "${PIPELINE_LOG_DIR}"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/pipeline_unet.log"

# ---------- Caminhos no CONTAINER (Mapeamento interno do Docker) -------------
DATA_DIR_CONTAINER="/workspace/datasets/isic_2018_task1_numpy"
PROJECT_CONTAINER="/workspace/logs/${PIPELINE_NAME}"

# Se estiver a usar 1 GPU para a U-Net, defina 0 ou 1.
GPU_DEVICE_IDS="${GPU_DEVICE_IDS:-0}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${PIPELINE_LOG}"; }

log "=============================================================="
log "U-Net ISIC 2018 Task 1 — End-to-End Pipeline (Fase 1)"
log "=============================================================="
log "  Host Logs Dir  = ${HOST_LOGS_DIR}"
log "  Container Data = ${DATA_DIR_CONTAINER}"
log "  device         = ${GPU_DEVICE_IDS}"
log "  pipeline_log   = ${PIPELINE_LOG}"
log "--------------------------------------------------------------"

# Execução do Docker
docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e TF_FORCE_GPU_ALLOW_GROWTH=true \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "${UNET_DIR}:/workspace/unet" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    unet_ft \
    python /workspace/unet/train_baseline_models.py \
        --data_dir "${DATA_DIR_CONTAINER}" \
        --project "${PROJECT_CONTAINER}" \
        --epochs 120 \
        --patience 20 \
        --batch 16 \
        --seed 0 \
    2>&1 | tee -a "${PIPELINE_LOG}"

log "=============================================================="
log "Fase 1 da U-Net finalizada. Logs: ${PIPELINE_LOG}"
log "=============================================================="