# Imagem oficial da NVIDIA com CUDA 12.1 e Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema e Python 3.11
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

# Instala o PIP para Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Faz o link simbólico para que o comando "python" chame o "python3.11"
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Define o diretório de trabalho
WORKDIR /workspace

# Instala o TensorFlow (que já suporta CUDA nativamente no pip) e as 
# dependências contidas no seu notebook da U-Net, além do jupyterlab
RUN pip install --upgrade pip && \
    pip install tensorflow numpy matplotlib scikit-image scikit-learn tqdm Pillow jupyterlab

# Copia o código para dentro do container
COPY . /workspace

# Comando padrão
CMD ["/bin/bash"]