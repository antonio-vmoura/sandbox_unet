#!/usr/bin/env bash
# =============================================================================
# run_pipeline_unet.sh — Orquestrador para o fine-tuning da U-Net 
# no dataset ISIC 2018 Task 1 (Arrays Numpy).
# Inclui Fases 1 (Baseline), 2 (HPO), 3 (Optimized) e 4 (5-Fold CV).
# =============================================================================
set -euo pipefail

# ---------- Caminhos no HOST (Servidor Físico) -------------------------------
HOST_LOGS_DIR="$(pwd)/logs"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_unet_v2}"
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

# Configuração da GPU
GPU_DEVICE_IDS="${GPU_DEVICE_IDS:-1}"

# Parâmetros HPO (Fase 2)
HPO_ITERATIONS="${HPO_ITERATIONS:-30}"
HPO_EPOCHS="${HPO_EPOCHS:-30}"
HPO_PATIENCE="${HPO_PATIENCE:-10}"
HPO_PROJECT_CONTAINER="${PROJECT_CONTAINER}/hpo"

# Parâmetros CV (Fase 4)
CV_K_FOLDS="${CV_K_FOLDS:-5}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${PIPELINE_LOG}"; }

log "=============================================================="
log "U-Net ISIC 2018 Task 1 — End-to-End Pipeline (Fases 1 a 4)"
log "=============================================================="
log "  Host Logs Dir  = ${HOST_LOGS_DIR}"
log "  Container Data = ${DATA_DIR_CONTAINER}"
log "  device         = ${GPU_DEVICE_IDS}"
log "  pipeline_log   = ${PIPELINE_LOG}"
log "--------------------------------------------------------------"

# =============================================================================
# FASE 1: Baseline
# =============================================================================
log "### Fase 1 — Iniciando Baseline..."
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

# Recolha de Métricas da Fase 1
log "### Fase 1 — Coletando métricas do baseline..."
docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "${UNET_DIR}:/workspace/unet" \
    unet_ft \
    python /workspace/unet/collect_phase_metrics_unet.py \
        --phase baseline \
        --project "${PROJECT_CONTAINER}" \
    2>&1 | tee -a "${PIPELINE_LOG}"

# =============================================================================
# FASE 2: Hyperparameter Optimization (Optuna)
# =============================================================================
log "### Fase 2 — Iniciando HPO (Optuna)..."
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
    python /workspace/unet/tune_unet.py \
        --data_dir "${DATA_DIR_CONTAINER}" \
        --project "${HPO_PROJECT_CONTAINER}" \
        --iterations ${HPO_ITERATIONS} \
        --epochs ${HPO_EPOCHS} \
        --patience ${HPO_PATIENCE} \
        --batch 16 \
        --seed 0 \
    2>&1 | tee -a "${PIPELINE_LOG}"

# =============================================================================
# FASE 3: Treino Otimizado (Single-split)
# =============================================================================
log "### Fase 3 — Iniciando Treino Otimizado (usa best_hyperparameters.yaml)..."
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
    python /workspace/unet/train_optimized_unet.py \
        --data_dir "${DATA_DIR_CONTAINER}" \
        --project "${PROJECT_CONTAINER}" \
        --hpo_dir "${HPO_PROJECT_CONTAINER}/hpo_v3/tune_isic_2018_task_1_unet" \
        --epochs 120 \
        --patience 25 \
        --batch 16 \
        --seed 0 \
    2>&1 | tee -a "${PIPELINE_LOG}"

# Recolha de Métricas da Fase 3
log "### Fase 3 — Coletando métricas otimizadas para pipeline_summary/..."
docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "${UNET_DIR}:/workspace/unet" \
    unet_ft \
    python /workspace/unet/collect_phase_metrics_unet.py \
        --phase optimized \
        --project "${PROJECT_CONTAINER}" \
    2>&1 | tee -a "${PIPELINE_LOG}"

# =============================================================================
# FASE 4: 5-Fold Cross-Validation
# =============================================================================
log "### Fase 4 — Iniciando ${CV_K_FOLDS}-Fold CV..."
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
    python /workspace/unet/train_cv_unet.py \
        --data_dir "${DATA_DIR_CONTAINER}" \
        --project "${PROJECT_CONTAINER}" \
        --hpo_dir "${HPO_PROJECT_CONTAINER}/hpo_v3/tune_isic_2018_task_1_unet" \
        --k_folds ${CV_K_FOLDS} \
        --epochs 120 \
        --patience 25 \
        --batch 16 \
        --seed 0 \
    2>&1 | tee -a "${PIPELINE_LOG}"

# Consolidação da Fase 4
log "### Fase 4 — Consolidando resultados do CV..."
docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "${UNET_DIR}:/workspace/unet" \
    unet_ft \
    python /workspace/unet/consolidate_cv_results_unet.py \
        --project "${PROJECT_CONTAINER}" \
        --k_folds ${CV_K_FOLDS} \
    2>&1 | tee -a "${PIPELINE_LOG}"

log "=============================================================="
log "Pipeline U-Net (Fases 1 a 4) finalizado."
log "=============================================================="