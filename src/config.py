"""
Configuración central y resiliente del sistema NeuroScan AI.

Centraliza rutas dinámicas (Local, Colab, Google Drive), parámetros de clases,
hiperparámetros por defecto, tolerancia a fallos y configuración de MLflow.
"""

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Detección dinámica del entorno de ejecución
# --------------------------------------------------------------------------
IS_COLAB: bool = "google.colab" in sys.modules

# --------------------------------------------------------------------------
# Gestión de Rutas y Directorios
# --------------------------------------------------------------------------
# Raíz del proyecto en local o en Colab (/content/... o carpeta de workspace)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Rutas de Google Drive (habilitadas si se ejecuta en Google Colab con Drive montado)
GDRIVE_MOUNT_POINT: Path = Path("/content/drive/MyDrive/m3_project")
GDRIVE_BACKUP_DIR: Path = GDRIVE_MOUNT_POINT / "backups"
GDRIVE_CHECKPOINTS_DIR: Path = GDRIVE_MOUNT_POINT / "checkpoints"

# Rutas de datos
DEFAULT_RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_DIR: Path = Path(os.environ.get("NEUROSCAN_DATA_DIR", str(DEFAULT_RAW_DATA_DIR)))

DATA_CHECKPOINTS_DIR: Path = PROJECT_ROOT / "data" / "checkpoints"
MANIFEST_PATH: Path = PROJECT_ROOT / "data" / "manifest.csv"
SPLIT_MANIFEST_PATH: Path = PROJECT_ROOT / "data" / "manifest_split.csv"

# Rutas de artefactos de modelos y checkpoints periódicos
MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
FINAL_MODEL_PATH: Path = MODELS_DIR / "neuro_mri_model.keras"
INPUT_EXAMPLE_INFO_PATH: Path = MODELS_DIR / "input_example_info.json"

# Asegurar la existencia de directorios críticos
for directory in [DATA_CHECKPOINTS_DIR, MODEL_CHECKPOINTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Configuración de MLflow Tracking y Registry
# --------------------------------------------------------------------------
if IS_COLAB and GDRIVE_MOUNT_POINT.exists():
    DEFAULT_MLFLOW_DB = f"sqlite:///{GDRIVE_MOUNT_POINT / 'mlflow.db'}"
else:
    DEFAULT_MLFLOW_DB = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

MLFLOW_TRACKING_URI: str = os.environ.get("NEUROSCAN_MLFLOW_URI", DEFAULT_MLFLOW_DB)
MLFLOW_EXPERIMENT_NAME: str = "NeuroScan-MRI-BrainTumor"
MLFLOW_MODEL_NAME: str = "neuro_mri_classifier"

# --------------------------------------------------------------------------
# Constantes del Problema (Brain Tumor MRI)
# --------------------------------------------------------------------------
# 4 clases canónicas del dataset de referencia
CLASSES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX: dict[str, int] = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: cls_name for i, cls_name in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)

# Especificaciones de imagen
IMG_SIZE: tuple[int, int] = (224, 224)  # Compatible con ResNet50 y EfficientNet
IMG_CHANNELS: int = 3
BATCH_SIZE: int = 32

# Proporciones de partición sin fuga
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED: int = 42

# --------------------------------------------------------------------------
# Parámetros de Resiliencia y Failsafes
# --------------------------------------------------------------------------
EARLY_STOPPING_PATIENCE: int = 5
REDUCE_LR_PATIENCE: int = 2
MIN_LEARNING_RATE: float = 1e-7
CHECKPOINT_SAVE_FREQ: str = "epoch"
ENABLE_DRIVE_SYNC: bool = IS_COLAB

# Disclaimer médico normativo para inferencia y reportes
MEDICAL_DISCLAIMER: str = (
    "AVISO CLÍNICO OBLIGATORIO: NeuroScan AI es un sistema académico de asistencia "
    "diagnóstica basado en visión computacional profunda. NO sustituye el criterio "
    "médico, biopsia histopatológica ni la interpretación radiológica oficial."
)
