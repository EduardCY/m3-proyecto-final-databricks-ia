"""
Orquestador de entrenamiento con tolerancia a fallos, checkpoints y MLflow Tracking.

Implementa guardado periódico en Google Drive, doble registro (MLflow + CSVLogger),
reanudación automática desde checkpoints y registro integral de artefactos clínicos.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.architecture import (
    build_model,
    compile_model,
    enable_mixed_precision_if_available,
    set_seeds,
)
from src.config import (
    BATCH_SIZE,
    CLASSES,
    EARLY_STOPPING_PATIENCE,
    ENABLE_DRIVE_SYNC,
    FINAL_MODEL_PATH,
    GDRIVE_CHECKPOINTS_DIR,
    INPUT_EXAMPLE_INFO_PATH,
    IS_COLAB,
    MIN_LEARNING_RATE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_CHECKPOINTS_DIR,
    NUM_CLASSES,
    REDUCE_LR_PATIENCE,
    SEED,
)
from src.metrics import (
    compute_comprehensive_metrics,
    plot_confusion_matrix,
    plot_multiclass_roc_curve,
    plot_training_history,
)

logger = logging.getLogger(__name__)


class DriveSyncCallback(tf.keras.callbacks.Callback):
    """Callback de resiliencia: replica checkpoints en Google Drive tras cada época."""

    def __init__(self, local_checkpoint_path: Path, drive_checkpoint_dir: Path):
        super().__init__()
        self.local_path = local_checkpoint_path
        self.drive_dir = drive_checkpoint_dir

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if not ENABLE_DRIVE_SYNC or not self.drive_dir.exists():
            return
        try:
            if self.local_path.exists():
                drive_dest = self.drive_dir / self.local_path.name
                shutil.copy2(self.local_path, drive_dest)
                logger.info(f"[Drive Sync] Checkpoint respaldado en Drive: {drive_dest.name}")
        except Exception as e:
            logger.warning(f"No se pudo sincronizar checkpoint con Drive: {e}")


def train_and_log_run(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    run_name: str,
    backbone_name: str = "resnet50",
    epochs: int = 15,
    learning_rate: float = 1e-3,
    dropout: float = 0.3,
    base_trainable: bool = False,
    fine_tune_at: Optional[int] = None,
    optimizer_name: str = "adam",
    use_cosine_decay: bool = False,
    resume_from_checkpoint: Optional[Path] = None,
    weights: str = "imagenet",
) -> tuple[tf.keras.Model, Any, dict[str, float]]:
    """Ejecuta un experimento completo registrando parámetros, métricas y artefactos.

    Incluye múltiples capas de resiliencia:
    1. Checkpoints locales y sincronizados a Google Drive.
    2. Respaldo de métricas en CSVLogger independiente de SQLite.
    3. Reanudación desde checkpoints previos si hubo desconexión.
    """
    set_seeds(SEED)
    mixed_precision_active = enable_mixed_precision_if_available()

    # Configuración de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 1. Construcción o reanudación del modelo
    model = build_model(
        backbone_name=backbone_name,
        dropout=dropout,
        num_classes=NUM_CLASSES,
        base_trainable=base_trainable,
        fine_tune_at=fine_tune_at,
        weights=weights,
    )

    compile_model(
        model=model,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        use_cosine_decay=use_cosine_decay,
    )

    if resume_from_checkpoint and resume_from_checkpoint.exists():
        logger.info(f"[Resiliencia] Reanudando pesos desde: {resume_from_checkpoint}")
        model.load_weights(resume_from_checkpoint)

    # 2. Configuración de Callbacks y Puntos de Guardado
    MODEL_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = MODEL_CHECKPOINTS_DIR / f"{run_name}_best.keras"
    csv_backup_path = MODEL_CHECKPOINTS_DIR / f"{run_name}_history.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        CSVLogger(filename=str(csv_backup_path)),
    ]

    # Añadir sincronización a Google Drive si está activo
    if ENABLE_DRIVE_SYNC and GDRIVE_CHECKPOINTS_DIR.exists():
        callbacks.append(DriveSyncCallback(best_checkpoint_path, GDRIVE_CHECKPOINTS_DIR))

    with mlflow.start_run(run_name=run_name) as run:
        # A. Registrar parámetros del experimento
        mlflow.log_params(
            {
                "run_name": run_name,
                "backbone": backbone_name,
                "epochs_max": epochs,
                "batch_size": BATCH_SIZE,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "optimizer": optimizer_name,
                "use_cosine_decay": use_cosine_decay,
                "base_trainable": base_trainable,
                "fine_tune_at": str(fine_tune_at),
                "mixed_precision": mixed_precision_active,
                "is_colab": IS_COLAB,
                "seed": SEED,
            }
        )

        # B. Ciclo de Entrenamiento
        logger.info(f"Iniciando entrenamiento para '{run_name}'...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # C. Registrar métricas de entrenamiento por época en MLflow
        for epoch_idx in range(len(history.history["loss"])):
            epoch_metrics = {
                "train_loss": history.history["loss"][epoch_idx],
                "train_accuracy": history.history["accuracy"][epoch_idx],
                "val_loss": history.history["val_loss"][epoch_idx],
                "val_accuracy": history.history["val_accuracy"][epoch_idx],
            }
            mlflow.log_metrics(epoch_metrics, step=epoch_idx + 1)

        # D. Evaluación sobre el Test Set
        logger.info("Evaluando sobre el conjunto de test...")
        y_true_list = []
        y_prob_list = []

        for images_batch, labels_batch in test_dataset:
            probs = model.predict(images_batch, verbose=0)
            y_prob_list.append(probs)
            y_true_list.append(np.argmax(labels_batch.numpy(), axis=1))

        y_prob = np.concatenate(y_prob_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)

        metrics_dict, report_str = compute_comprehensive_metrics(y_true, y_prob, classes=CLASSES)
        mlflow.log_metrics(metrics_dict)

        # E. Generación y Registro de Artefactos Visuales
        artifacts_dir = MODEL_CHECKPOINTS_DIR / f"{run_name}_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        cm_path = artifacts_dir / "confusion_matrix.png"
        roc_path = artifacts_dir / "roc_curves.png"
        hist_path = artifacts_dir / "training_history.png"
        report_path = artifacts_dir / "classification_report.txt"

        plot_confusion_matrix(y_true, np.argmax(y_prob, axis=1), cm_path, classes=CLASSES)
        plot_multiclass_roc_curve(np.eye(len(CLASSES))[y_true], y_prob, roc_path, classes=CLASSES)
        plot_training_history(history, hist_path)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_str)

        # Firma de entrada/salida como JSON
        input_info = {
            "input_shape": [None, 224, 224, 3],
            "input_dtype": "float32",
            "output_shape": [None, NUM_CLASSES],
            "output_classes": CLASSES,
        }
        with open(INPUT_EXAMPLE_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(input_info, f, indent=2)

        # Registrar artefactos en MLflow
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(hist_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(INPUT_EXAMPLE_INFO_PATH))

        # Registrar el modelo serializado en MLflow
        try:
            mlflow.keras.log_model(model, artifact_path="model")
        except Exception as e:
            logger.warning(f"Aviso al serializar modelo en MLflow: {e}")

        logger.info(f"Run '{run_name}' finalizado exitosamente. Macro-F1={metrics_dict['test_macro_f1']:.4f}")

    return model, history, metrics_dict
