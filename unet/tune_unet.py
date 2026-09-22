"""Phase 2 — Hyperparameter Optimization (HPO) for U-Net using Optuna.

Otimização de hiperparâmetros blindada com Garbage Collection agressivo 
para evitar deadlocks do TensorFlow e backup em SQLite (retomada automática).
"""

import os
import argparse
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import optuna
import gc

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout
from tensorflow.keras import Model
import tensorflow.keras.backend as K

VERSION = "hpo_v3"
MODEL_NAME = "unet"
TUNE_PREFIX = "tune_isic_2018_task_1_"

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters, dropout, batchnorm):
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

def create_objective(args, x_train, y_train, x_val, y_val):
    def objective(trial):
        K.clear_session()
        gc.collect() # Limpeza forçada antes de inicializar pesos
        
        lr = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        batchnorm = trial.suggest_categorical("batchnorm", [True, False])
        n_filters = trial.suggest_categorical("n_filters", [8, 16, 32])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD_Momentum"])
        
        input_img = Input((args.imgsz, args.imgsz, 3))
        model = get_unet(input_img, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm)
        
        if optimizer_name == "Adam": opt = Adam(learning_rate=lr)
        else: opt = SGD(learning_rate=lr, momentum=0.9)
            
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[custom_iou, custom_dice])
        callbacks = [EarlyStopping(patience=args.patience, restore_best_weights=True, monitor='val_custom_iou', mode='max')]
        
        history = model.fit(
            x_train, y_train,
            batch_size=args.batch,
            epochs=args.epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=0 
        )
        
        best_iou = max(history.history.get("val_custom_iou", [0.0]))
        
        # Destruição explícita do modelo na memória de vídeo
        del model
        del history
        K.clear_session()
        gc.collect()
        
        return best_iou
    return objective

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/workspace/datasets/isic_2018_task1_numpy")
    p.add_argument("--project", default="/workspace/logs/pipeline_unet_v1/hpo")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seeds(args.seed)
    tf.keras.mixed_precision.set_global_policy('float32')
    
    out_dir = Path(args.project) / VERSION / f"{TUNE_PREFIX}{MODEL_NAME}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_yaml = out_dir / "best_hyperparameters.yaml"
    db_path = out_dir / "optuna_study.db"

    if best_yaml.exists() and not args.force:
        print(f"[SKIP] O YAML de hiperparâmetros já existe em {best_yaml}.")
        return 0

    print(f"\n=== Iniciando PHASE 2 (HPO U-NET) ===")
    
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
        print(f"[ERRO] Falha ao carregar dados: {e}")
        return 1

    t0 = time.perf_counter()
    
    # Integração do SQLite para persistência e retomada
    study = optuna.create_study(
        study_name="unet_hpo",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    # Executa apenas as iterações restantes
    remaining_trials = args.iterations - len(study.trials)
    if remaining_trials > 0:
        objective = create_objective(args, x_train, y_train, x_val, y_val)
        study.optimize(objective, n_trials=remaining_trials)

    elapsed = (time.perf_counter() - t0) / 60
    
    print("\n[SUCESSO] Otimização concluída em {:.1f} minutos.".format(elapsed))
    for k, v in study.best_params.items(): print(f"  {k}: {v}")

    with best_yaml.open("w") as f:
        yaml.safe_dump(study.best_params, f, default_flow_style=False)
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())