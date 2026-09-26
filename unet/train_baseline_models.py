"""Phase 1 — Baseline training of U-Net on ISIC 2018 Task 1.

Este script executa o treinamento do Baseline da U-Net utilizando 
parâmetros fora da caixa (16 filtros, dropout 0.1, Adam standard).
Mantemos as constantes alinhadas ao estudo: epochs=120, patience=20, seed=0.

Implementa os geradores de Data Augmentation estática e a separação correta
entre métricas (com threshold) e Losses (sem threshold, diferenciáveis)
para garantir a justiça na comparação do baseline contra a YOLO.
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

VERSION = "phase1_baseline"
MODEL_NAME = "unet_baseline"

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# --- MÉTRICAS (Usam Threshold para Avaliação Exata) ---
def metric_iou(y_true, y_pred, smooth=1e-6):
    y_pred_th = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.cast(y_true, tf.float32)
    intersection = K.sum(y_true_f * y_pred_th)
    union = K.sum(y_true_f) + K.sum(y_pred_th) - intersection
    return (intersection + smooth) / (union + smooth)

def metric_dice(y_true, y_pred, smooth=1e-6):
    y_pred_th = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.cast(y_true, tf.float32)
    intersection = K.sum(y_true_f * y_pred_th)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_th) + smooth)

# --- LOSSES (Contínuas e Diferenciáveis) ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - dice_coeff

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# --- Construtor U-Net Baseline ---
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
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

def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 — Baseline training of U-Net.")
    p.add_argument("--data_dir", default="/workspace/datasets/isic_2018_task1_numpy")
    p.add_argument("--project", default="/workspace/logs/pipeline_unet_v1")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seeds(args.seed)
    tf.keras.mixed_precision.set_global_policy('float32')
    
    run_root = Path(args.project) / VERSION / MODEL_NAME
    run_root.mkdir(parents=True, exist_ok=True)
    best_pt = run_root / "weights" / "best_model.h5"
    best_pt.parent.mkdir(exist_ok=True)
    csv_path = run_root / "results.csv"

    if best_pt.exists() and not args.force:
        print(f"[SKIP] O modelo {best_pt} já existe. Use --force para re-treinar.")
        return 0

    print(f"\n=== Iniciando PHASE 1 (BASELINE U-NET COM AUGMENTATION) ===")
    
    data_path = Path(args.data_dir)
    try:
        x_train = np.load(data_path / "ISIC2018_Task1-2_Training_Input" / "ISIC2018_Task1-2_Training_Input.npy")
        y_train = np.load(data_path / "ISIC2018_Task1_Training_GroundTruth" / "ISIC2018_Task1_Training_GroundTruth.npy")
        x_val = np.load(data_path / "ISIC2018_Task1-2_Validation_Input" / "ISIC2018_Task1-2_Validation_Input.npy")
        y_val = np.load(data_path / "ISIC2018_Task1_Validation_GroundTruth" / "ISIC2018_Task1_Validation_GroundTruth.npy")
        
        if len(x_train.shape) == 3: x_train = np.expand_dims(x_train, axis=-1)
        if len(y_train.shape) == 3: y_train = np.expand_dims(y_train, axis=-1)
        if len(x_val.shape) == 3: x_val = np.expand_dims(x_val, axis=-1)
        if len(y_val.shape) == 3: y_val = np.expand_dims(y_val, axis=-1)
        
    except Exception as e:
        print(f"[ERRO] Falha ao carregar os arrays .npy.")
        print(e)
        return 1

    # --- Data Augmentation FIXA (Baseline Estável) ---
    data_gen_args = dict(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed_gen = 42
    image_generator = image_datagen.flow(x_train, batch_size=args.batch, seed=seed_gen)
    mask_generator = mask_datagen.flow(y_train, batch_size=args.batch, seed=seed_gen)
    train_generator = zip(image_generator, mask_generator)
    steps_per_epoch = len(x_train) // args.batch

    input_img = Input((args.imgsz, args.imgsz, 3))
    model = get_unet_baseline(input_img, n_filters=16, dropout=0.1, batchnorm=True)
    
    # Compilamos usando a BCE_Dice contínua e mantemos as métricas restritas para tracking
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_dice_loss, metrics=["accuracy", metric_iou, metric_dice])

    callbacks = [
        EarlyStopping(patience=args.patience, verbose=1, restore_best_weights=True, monitor='val_metric_iou', mode='max'),
        ModelCheckpoint(str(best_pt), verbose=1, save_best_only=True, monitor='val_metric_iou', mode='max'),
        CSVLogger(str(csv_path), separator=',', append=False)
    ]
    
    t0 = time.perf_counter()
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
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