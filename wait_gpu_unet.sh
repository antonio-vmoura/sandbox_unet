#!/bin/bash
# =============================================================================
# wait_gpu_unet.sh — Aguarda a GPU ficar ociosa e então lança o pipeline U-Net
# (run_pipeline_unet.sh) dentro do container ``unet_ft``.
#
# 1. Consulta ``nvidia-smi`` a cada ``POLL_INTERVAL`` segundos (padrão 5) na
#    GPU ${GPU_DEVICE}.
# 2. A GPU está "livre" quando não roda nenhum processo de computação E
#    memory.used < ``MAX_MEM_MIB`` (1000) E utilization.gpu < ``MAX_UTIL``
#    (10%). Uma falha do ``nvidia-smi`` conta como ocupada.
# 3. O pipeline inicia IMEDIATAMENTE na primeira verificação livre
#    (``CONFIRM_CHECKS=1``). Use ``CONFIRM_CHECKS=N`` para exigir N
#    verificações livres consecutivas (ex.: ignorar o intervalo curto entre
#    dois jobs de outro usuário). Em seguida executa o bloco ``docker run``.
#
# O pipeline é retomável: relançar o mesmo comando (SEM --force) continua um
# estudo interrompido em vez de recomeçá-lo. Argumentos extras são repassados
# ao run_pipeline_unet.sh (ex.: ./wait_gpu_unet.sh --phases "1 2 3 4 5" --force).
# O log do terminal fica em logs/${PIPELINE_NAME}/terminal_<UTC>.log.
# =============================================================================

GPU_DEVICE="${GPU_DEVICE:-0}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
CONFIRM_CHECKS="${CONFIRM_CHECKS:-1}"
MAX_MEM_MIB="${MAX_MEM_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-10}"

gpu_free() {
    local q mem util uuid apps
    STATUS=""
    q=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,uuid --format=csv,noheader,nounits -i "${GPU_DEVICE}" 2>/dev/null) || return 1
    IFS=', ' read -r mem util uuid <<<"${q}"
    [[ "${mem}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || return 1
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "${uuid}")
    STATUS="${mem}MiB ${util}% ${apps} proc"
    [ "${apps}" -eq 0 ] && [ "${mem}" -lt "${MAX_MEM_MIB}" ] && [ "${util}" -lt "${MAX_UTIL}" ]
}

echo "Aguardando a GPU ${GPU_DEVICE} ficar livre (consulta a cada ${POLL_INTERVAL}s, ${CONFIRM_CHECKS} verificação(ões))..."
FREE_COUNT=0
LAST_STATE=""
while true; do
    if gpu_free; then
        ((FREE_COUNT++))
        STATE="livre"
    else
        FREE_COUNT=0
        STATE="ocupada"
    fi
    # Registra só as mudanças de estado (a consulta a cada 5 s inundaria o terminal).
    if [ "${STATE}" != "${LAST_STATE}" ]; then
        echo "$(date) | GPU${GPU_DEVICE}: ${STATUS:-nvidia-smi falhou} -> ${STATE}."
        LAST_STATE="${STATE}"
    fi
    [ "${FREE_COUNT}" -ge "${CONFIRM_CHECKS}" ] && break
    sleep "${POLL_INTERVAL}"
done
echo "GPU liberada — iniciando o pipeline U-Net."

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
