#!/bin/bash
# =============================================================================
# wait_gpu_unet.sh — Aguarda a GPU ficar ociosa e então lança o pipeline U-Net
# (run_pipeline_unet.sh) dentro do container ``unet_ft``.
#
# 1. Consulta ``nvidia-smi`` uma vez por minuto na GPU ${GPU_DEVICE}.
# 2. A GPU é considerada ociosa quando memory.used < 1000 MiB E
#    utilization.gpu < 10%.
# 3. Após ``REQUIRED_IDLE_MINUTES`` verificações ociosas consecutivas, executa
#    o bloco ``docker run`` abaixo.
#
# O pipeline é retomável: relançar o mesmo comando (SEM --force) continua um
# estudo interrompido em vez de recomeçá-lo. Argumentos extras são repassados
# ao run_pipeline_unet.sh (ex.: ./wait_gpu_unet.sh --phases "1 2 3 4 5" --force).
# O log do terminal fica em logs/${PIPELINE_NAME}/terminal_<UTC>.log.
# =============================================================================

GPU_DEVICE="${GPU_DEVICE:-0}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
CHECK_INTERVAL=60
REQUIRED_IDLE_MINUTES=3
IDLE_COUNT=0

echo "Aguardando a GPU ${GPU_DEVICE} ficar ociosa por ${REQUIRED_IDLE_MINUTES} minuto(s)..."

while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_DEVICE}")
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU_DEVICE}")
    if [ "$MEM" -lt 1000 ] && [ "$UTIL" -lt 10 ]; then
        ((IDLE_COUNT++))
        echo "$(date) | GPU${GPU_DEVICE}: ${MEM}MiB ${UTIL}% -> ociosa há $IDLE_COUNT minuto(s)."
        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_MINUTES" ]; then
            echo "GPU liberada — iniciando o pipeline U-Net."
            break
        fi
    else
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "$(date) | Atividade detectada — zerando contador de ociosidade."
        else
            echo "$(date) | GPU${GPU_DEVICE}: ${MEM}MiB ${UTIL}% -> ocupada."
        fi
        IDLE_COUNT=0
    fi
    sleep $CHECK_INTERVAL
done

# O dataset YOLO26 (fonte única de verdade) é montado somente-leitura em
# /workspace/yolo26_dataset — ao lado de datasets/, nunca dentro (um mount
# aninhado faz o Docker criar no host uma pasta vazia de root); o cache
# 256×256 da Fase 0 é escrito em datasets/isic_2018_task1_unet256.
mkdir -p "logs/${PIPELINE_NAME}"
docker run --gpus "\"device=${GPU_DEVICE}\"" --rm --ipc=host \
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
  bash /workspace/run_pipeline_unet.sh "$@" \
  2>&1 | tee "logs/${PIPELINE_NAME}/terminal_$(date -u +%Y%m%dT%H%M%SZ).log"
