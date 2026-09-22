#!/bin/bash
# =============================================================================
# wait_gpu_unet.sh — Aguarda a GPU 0 ficar ociosa para lançar o pipeline U-Net.
# =============================================================================

echo "Aguardando a GPU 0 ficar ociosa por alguns minutos..."

CHECK_INTERVAL=60
REQUIRED_IDLE_MINUTES=3
IDLE_COUNT=0

while true; do
    # Verifica memória (MiB) e utilização (%) da GPU 0
    GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    GPU0_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)

    # Condição de ociosidade: memória < 1000 MiB E utilização < 10%
    if [ "$GPU0_MEM" -lt 1000 ] && [ "$GPU0_UTIL" -lt 10 ]; then
        ((IDLE_COUNT++))
        echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% -> ociosa há $IDLE_COUNT minuto(s)."

        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_MINUTES" ]; then
            echo "GPU liberada! Iniciando o treinamento da U-Net..."
            break
        fi
    else
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "$(date) | Atividade detectada — zerando contador de ociosidade."
        else
            echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% -> ocupada."
        fi
        IDLE_COUNT=0
    fi
    sleep $CHECK_INTERVAL
done

# Chama o orquestrador que acabamos de criar
bash run_pipeline_unet.sh