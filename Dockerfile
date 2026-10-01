# U-Net (PyTorch) — mesma base e mesmas versões fixadas do pipeline YOLO26
# (sandbox_yolo26/Dockerfile), para que latência, memória, parâmetros e GFLOPs
# sejam medidos com a mesma pilha de software em todas as arquiteturas.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependências do sistema e Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

# Versões fixadas (reprodutibilidade):
# - torch/torchvision/torchaudio: idênticas ao YOLO26 (kernels e determinismo);
# - optuna: versão usada no HPO da U-Net (o checkpoint da Fase 3 recusa retomar
#   uma busca sob outra versão);
# - ultralytics-thop: mesma contagem de GFLOPs do YOLO26;
# - pandas/scipy/matplotlib/opencv: avaliação, relatório e notebooks.
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install \
        numpy==2.3.5 \
        optuna==5.0.0 \
        ultralytics-thop==2.0.18 \
        opencv-python-headless==4.13.0.92 \
        scipy==1.17.1 \
        pandas==3.0.1 \
        matplotlib==3.10.8 \
        psutil==7.2.2 \
        pyyaml \
        jupyterlab

COPY . /workspace

CMD ["/bin/bash"]
