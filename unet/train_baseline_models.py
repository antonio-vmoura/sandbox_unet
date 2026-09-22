"""Phase 1 — Baseline training of U-Net on ISIC 2018 Task 1.

Este script executa o treinamento do Baseline da U-Net utilizando 
parâmetros fora da caixa (sem customização de otimizadores ou augmentations agressivas). 
Mantemos as constantes alinhadas ao estudo original do YOLO: 
epochs=120, patience=20, deterministic=True e seed=0 para total reprodutibilidade.
O AMP (Automatic Mixed Precision) fica desativado.

Os dados lidos são os arrays .npy pré-processados.

Outputs gerados na estrutura:
    <project>/phase1_baseline/unet_baseline/{best_model.h5, results.csv, ...}
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf
import csv
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
# Importando as camadas para a construção da U-Net Baseline
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout
from tensorflow.keras import Model

# ----------------------------------------------------------------------------
# Module-level configuration
# ----------------------------------------------------------------------------
VERSION = "phase1_baseline"
MODEL_NAME = "unet_baseline"

# ----------------------------------------------------------------------------
# Reprodutibilidade e Constantes
# ----------------------------------------------------------------------------
def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Configuração determinística para operações no TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# ----------------------------------------------------------------------------
# Construção do Modelo U-Net Baseline
# ----------------------------------------------------------------------------
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # primeira camada de convolução
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # segunda camada de convolução
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet_baseline(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    # Contraction path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # Expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

# ----------------------------------------------------------------------------
# CLI Parsing
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 — Baseline training of U-Net.")
    p.add_argument("--data_dir", default="/workspace/datasets/isic_2018_task1_unet",
                   help="Diretório onde estão localizados os arquivos .npy")
    p.add_argument("--project", default="/workspace/logs/pipeline_unet_v1",
                   help="Diretório raiz para os logs")
    p.add_argument("--epochs", type=int, default=120, help="Épocas por modelo (default: 120).")
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience (default: 20).")
    p.add_argument("--imgsz", type=int, default=256, help="Input image size (default: 256).")
    p.add_argument("--batch", type=int, default=16, help="Tamanho do batch.")
    p.add_argument("--seed", type=int, default=0, help="Semente determinística (default: 0).")
    p.add_argument("--force", action="store_true", help="Re-treinar mesmo se os pesos existirem.")
    return p.parse_args()

def main():
    args = parse_args()
    
    # 1. Isolamento e Determismo
    set_seeds(args.seed)
    # Desativa o AMP para garantir consistência
    tf.keras.mixed_precision.set_global_policy('float32')
    
    run_root = Path(args.project) / VERSION / MODEL_NAME
    run_root.mkdir(parents=True, exist_ok=True)
    
    best_pt = run_root / "weights" / "best_model.h5"
    best_pt.parent.mkdir(exist_ok=True)
    csv_path = run_root / "results.csv"

    if best_pt.exists() and not args.force:
        print(f"[SKIP] O modelo {best_pt} já existe. Use --force para re-treinar.")
        return 0

    print(f"\n=== Iniciando PHASE 1 (BASELINE U-NET) ===")
    print(f"Epochs: {args.epochs}, Patience: {args.patience}, Seed: {args.seed}")
    
    # 2. Carregamento dos Dados .npy
    try:
        x_train = np.load(os.path.join(args.data_dir, "TRAINING_IMAGES.npy"))
        y_train = np.load(os.path.join(args.data_dir, "TRAINING_MASKS.npy"))
        x_val = np.load(os.path.join(args.data_dir, "VALIDATION_IMAGES.npy"))
        y_val = np.load(os.path.join(args.data_dir, "VALIDATION_MASKS.npy"))
        
        # Ajuste de dimensões e normalização básica (caso necessário)
        # Assegurar shape (N, H, W, C)
        if len(x_train.shape) == 3: x_train = np.expand_dims(x_train, axis=-1)
        if len(y_train.shape) == 3: y_train = np.expand_dims(y_train, axis=-1)
        if len(x_val.shape) == 3: x_val = np.expand_dims(x_val, axis=-1)
        if len(y_val.shape) == 3: y_val = np.expand_dims(y_val, axis=-1)
        
    except Exception as e:
        print(f"[ERRO] Falha ao carregar os arrays .npy. Verifique o caminho {args.data_dir}.")
        print(e)
        return 1

    # 3. Construção do Modelo
    input_img = Input((args.imgsz, args.imgsz, 3))
    model = get_unet_baseline(input_img, n_filters=16, dropout=0.1, batchnorm=True)
    
    # Optimizer padrão do Keras sem calibração profunda
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy", MeanIoU(num_classes=2)])

    # 4. Callbacks para o Baseline (Idempotência e Arquivos Canônicos)
    callbacks = [
        EarlyStopping(patience=args.patience, verbose=1, restore_best_weights=True),
        ModelCheckpoint(str(best_pt), verbose=1, save_best_only=True, monitor='val_loss'),
        CSVLogger(str(csv_path), separator=',', append=False)
    ]
    
    # 5. Treinamento
    t0 = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    elapsed = (time.perf_counter() - t0) / 60
    
    print(f"\n[SUCESSO] Treinamento Baseline U-Net concluído em {elapsed:.1f} minutos.")
    print(f"Artefatos salvos em: {run_root}")
    return 0

if __name__ == "__main__":
    main()