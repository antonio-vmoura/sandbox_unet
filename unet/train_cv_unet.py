"""Phase 4 — 5-Fold Cross-Validation of U-Net on ISIC 2018 Task 1.

Este script junta os arrays .npy (Treino + Validação) num único pool.
Aplica um split K-Fold determinístico (semelhante ao scikit-learn, mas apenas com NumPy)
e treina a U-Net 5 vezes utilizando os hiperparâmetros da Fase 2.

Outputs gerados:
    <project>/cv/cv_v1/unet_cv_isic_2018/fold_{0..4}/{best_model.h5, results.csv}
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout
from tensorflow.keras import Model
import tensorflow.keras.backend as K

VERSION = "cv_v1"
MODEL_NAME = "unet"

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# --- Métricas Customizadas ---
def custom_iou(y_true, y_pred, smooth=1e-6):
    y_pred_th = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.cast(y_true, tf.float32)
    intersection = K.sum(y_true_f * y_pred_th)
    union = K.sum(y_true_f) + K.sum(y_pred_th) - intersection
    return (intersection + smooth) / (union + smooth)

def custom_dice(y_true, y_pred, smooth=1e-6):
    y_pred_th = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.cast(y_true, tf.float32)
    intersection = K.sum(y_true_f * y_pred_th)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_th) + smooth)

# --- Construtor U-Net ---
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet_cv(input_img, n_filters, dropout, batchnorm):
    c1 = conv2d_block(input_img, n_filters=n_filters*1, batchnorm=batchnorm)
    p1 = Dropout(dropout)(MaxPooling2D((2, 2))(c1))
    c2 = conv2d_block(p1, n_filters=n_filters*2, batchnorm=batchnorm)
    p2 = Dropout(dropout)(MaxPooling2D((2, 2))(c2))
    c3 = conv2d_block(p2, n_filters=n_filters*4, batchnorm=batchnorm)
    p3 = Dropout(dropout)(MaxPooling2D((2, 2))(c3))
    c4 = conv2d_block(p3, n_filters=n_filters*8, batchnorm=batchnorm)
    p4 = Dropout(dropout)(MaxPooling2D(pool_size=(2, 2))(c4))
    c5 = conv2d_block(p4, n_filters=n_filters*16, batchnorm=batchnorm)
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    c6 = conv2d_block(Dropout(dropout)(concatenate([u6, c4])), n_filters=n_filters*8, batchnorm=batchnorm)
    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    c7 = conv2d_block(Dropout(dropout)(concatenate([u7, c3])), n_filters=n_filters*4, batchnorm=batchnorm)
    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    c8 = conv2d_block(Dropout(dropout)(concatenate([u8, c2])), n_filters=n_filters*2, batchnorm=batchnorm)
    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    c9 = conv2d_block(Dropout(dropout)(concatenate([u9, c1], axis=3)), n_filters=n_filters*1, batchnorm=batchnorm)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return Model(inputs=[input_img], outputs=[outputs])

# --- K-Fold com NumPy ---
def build_kfold_splits(n_samples, k, seed):
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[: n_samples % k] += 1
    
    splits = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_idx, val_idx))
        start = stop
    return splits

# --- CLI e Main ---
def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 — CV training of U-Net.")
    p.add_argument("--data_dir", default="/workspace/datasets/isic_2018_task1_numpy")
    p.add_argument("--project", default="/workspace/logs/pipeline_unet_v1")
    p.add_argument("--hpo_dir", default="/workspace/logs/pipeline_unet_v1/hpo/hpo_v3/tune_isic_2018_task_1_unet")
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def load_tuned_hp(path: Path):
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(f"YAML vazio em {path}. A Fase 2 pode ter falhado.")
    return data

def main():
    args = parse_args()
    set_seeds(args.seed)
    tf.keras.mixed_precision.set_global_policy('float32')
    
    cv_root = Path(args.project) / "cv" / VERSION / f"{MODEL_NAME}_cv_isic_2018"
    
    hp_yaml = Path(args.hpo_dir) / "best_hyperparameters.yaml"
    if not hp_yaml.exists():
        print(f"[ERRO] YAML não encontrado em {hp_yaml}.")
        return 1
    tuned_hp = load_tuned_hp(hp_yaml)

    print(f"\n=== Iniciando PHASE 4 ({args.k_folds}-FOLD CV U-NET) ===")
    
    # Carregamento e União dos Dados
    data_path = Path(args.data_dir)
    try:
        x_train_orig = np.load(data_path / "ISIC2018_Task1-2_Training_Input" / "ISIC2018_Task1-2_Training_Input.npy")
        y_train_orig = np.load(data_path / "ISIC2018_Task1_Training_GroundTruth" / "ISIC2018_Task1_Training_GroundTruth.npy")
        x_val_orig = np.load(data_path / "ISIC2018_Task1-2_Validation_Input" / "ISIC2018_Task1-2_Validation_Input.npy")
        y_val_orig = np.load(data_path / "ISIC2018_Task1_Validation_GroundTruth" / "ISIC2018_Task1_Validation_GroundTruth.npy")
        
        X_pool = np.concatenate((x_train_orig, x_val_orig), axis=0)
        Y_pool = np.concatenate((y_train_orig, y_val_orig), axis=0)
        
        if len(X_pool.shape) == 3: X_pool = np.expand_dims(X_pool, axis=-1)
        if len(Y_pool.shape) == 3: Y_pool = np.expand_dims(Y_pool, axis=-1)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dados: {e}")
        return 1

    splits = build_kfold_splits(len(X_pool), args.k_folds, args.seed)
    
    t_total = time.perf_counter()
    
    for k, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Treinando Fold {k+1}/{args.k_folds} ---")
        fold_dir = cv_root / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        best_pt = fold_dir / "best_model.h5"
        csv_path = fold_dir / "results.csv"
        
        if best_pt.exists() and not args.force:
            print(f"[SKIP] Fold {k} já treinado.")
            continue
            
        K.clear_session()
        
        x_train, y_train = X_pool[train_idx], Y_pool[train_idx]
        x_val, y_val = X_pool[val_idx], Y_pool[val_idx]

        input_img = Input((args.imgsz, args.imgsz, 3))
        model = get_unet_cv(
            input_img, 
            n_filters=tuned_hp.get("n_filters", 16), 
            dropout=tuned_hp.get("dropout", 0.1), 
            batchnorm=tuned_hp.get("batchnorm", True)
        )
        
        lr = tuned_hp.get("lr0", 1e-3)
        if tuned_hp.get("optimizer", "Adam") == "Adam":
            opt = Adam(learning_rate=lr)
        else:
            opt = SGD(learning_rate=lr, momentum=0.9)
            
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy", custom_iou, custom_dice])

        callbacks = [
            EarlyStopping(patience=args.patience, verbose=1, restore_best_weights=True, monitor='val_custom_iou', mode='max'),
            ModelCheckpoint(str(best_pt), verbose=1, save_best_only=True, monitor='val_custom_iou', mode='max'),
            CSVLogger(str(csv_path), separator=',', append=False)
        ]
        
        model.fit(
            x_train, y_train,
            batch_size=args.batch,
            epochs=args.epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
    elapsed = (time.perf_counter() - t_total) / 60
    print(f"\n[SUCESSO] {args.k_folds}-Fold CV concluído em {elapsed:.1f} minutos.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())