#!/usr/bin/env bash
# =============================================================================
# run_pipeline_unet.sh — Orquestrador para o fine-tuning do Baseline da U-Net
# no dataset ISIC 2018 Task 1 (Arrays Numpy).
# =============================================================================
set -euo pipefail

# ---------- Variáveis e Diretórios -------------------------------------------
# 1. Aqui alteramos para a pasta "isic_2018_task1_numpy"
DATA_DIR="${DATA_DIR:-/workspace/datasets/isic_2018_task1_numpy}"

LOGS_ROOT="${LOGS_ROOT:-/workspace/logs}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_unet_v1}"
PROJECT="${LOGS_ROOT}/${PIPELINE_NAME}"

# Se estiver usando 1 GPU para a U-Net, defina 0 ou 1.
GPU_DEVICE_IDS="${GPU_DEVICE_IDS:-0}"

# 2. Aqui alteramos a variável para apontar para a pasta "unet" do seu host
UNET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/unet"

# Diretório de logs do orquestrador
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG_DIR="${PROJECT}/pipeline_runs/${RUN_TS}"
mkdir -p "${PIPELINE_LOG_DIR}"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/pipeline_unet.log"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${PIPELINE_LOG}"; }

log "=============================================================="
log "U-Net ISIC 2018 Task 1 — End-to-End Pipeline (Fase 1)"
log "=============================================================="
log "  data_dir       = ${DATA_DIR}"
log "  project        = ${PROJECT}"
log "  device         = ${GPU_DEVICE_IDS}"
log "  pipeline_log   = ${PIPELINE_LOG}"
log "--------------------------------------------------------------"

# Execução do Docker com os mesmos mapeamentos do YOLO26
# OBS: Retiramos a flag '-it' (interativa) para evitar travamentos ao rodar via nohup/SSH fechado.
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
        --data_dir "${DATA_DIR}" \
        --project "${PROJECT}" \
        --epochs 120 \
        --patience 20 \
        --batch 16 \
        --seed 0 \
    2>&1 | tee -a "${PIPELINE_LOG}"

log "=============================================================="
log "Fase 1 da U-Net finalizada. Logs: ${PIPELINE_LOG}"
log "=============================================================="