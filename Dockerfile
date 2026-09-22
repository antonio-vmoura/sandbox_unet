# Usa a imagem oficial do TensorFlow com suporte total a GPU
FROM tensorflow/tensorflow:2.15.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Restringimos o NumPy para a versão 1.x para manter compatibilidade com o TF 2.15
RUN pip install --upgrade pip && \
    pip install "numpy<2.0.0" matplotlib scikit-image scikit-learn tqdm Pillow jupyterlab optuna pyyaml

COPY . /workspace

CMD ["/bin/bash"]