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

# Candidatos para montaje de Google Drive (prioriza m3_project en Colab y carpeta sincronizada)
_GDRIVE_CANDIDATES = [
    Path("/content/drive/MyDrive/m3_project"),
    Path("/content/drive/MyDrive/01_Development/Cursos/DevSenior/m3-proyecto-final-databricks-ia"),
]


def get_gdrive_mount_point() -> Path:
    """Resuelve dinámicamente la ruta de Google Drive en cualquier instante de ejecución.

    Why: En Google Colab, `src.config` puede importarse antes de ejecutar `drive.mount('/content/drive')`.
    Evaluar dinámicamente los candidatos evita quedarse atado a una ruta inexistente evaluada en import-time.
    """
    env_override = os.environ.get("NEUROSCAN_GDRIVE_DIR")
    if env_override:
        return Path(env_override)
    for candidate in _GDRIVE_CANDIDATES:
        if candidate.exists():
            return candidate
    return _GDRIVE_CANDIDATES[0]


GDRIVE_MOUNT_POINT: Path = get_gdrive_mount_point()
GDRIVE_BACKUP_DIR: Path = GDRIVE_MOUNT_POINT / "backups"
GDRIVE_CHECKPOINTS_DIR: Path = GDRIVE_MOUNT_POINT / "checkpoints"
GDRIVE_DATA_DIR: Path = GDRIVE_MOUNT_POINT / "data"

# Rutas de datos: en Colab se prioriza disco efímero NVMe local para evitar FUSE
DEFAULT_RAW_DATA_DIR: Path = (
    Path("/content/data/raw") if IS_COLAB else PROJECT_ROOT / "data" / "raw"
)
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
# Why: Se utiliza el disco local (NVMe en Colab) como backend activo de SQLite para evitar
# bloqueos de concurrencia ('database is locked') sobre el sistema FUSE de Google Drive.
# La sincronización hacia/desde Google Drive se realiza atómicamente vía scripts/sync_drive.py.
LOCAL_MLFLOW_DB_PATH: Path = PROJECT_ROOT / "mlflow.db"
DEFAULT_MLFLOW_DB: str = f"sqlite:///{LOCAL_MLFLOW_DB_PATH.as_posix()}"


def resolve_active_mlflow_uri() -> str:
    """Resuelve el URI de MLflow garantizando la restauración desde Google Drive si la DB local está vacía.

    Why: Cuando se abre el Notebook 04 en una nueva sesión de Colab, `PROJECT_ROOT / 'mlflow.db'`
    no existe aún o fue inicializada vacía. Esta función verifica si existe una copia poblada
    en Google Drive y la restaura antes de conectar el cliente MLflow.
    """
    env_uri = os.environ.get("NEUROSCAN_MLFLOW_URI")
    if env_uri:
        return env_uri

    gdrive_db = get_gdrive_mount_point() / "mlflow.db"
    if gdrive_db.exists() and gdrive_db.stat().st_size > 0:
        local_size = LOCAL_MLFLOW_DB_PATH.stat().st_size if LOCAL_MLFLOW_DB_PATH.exists() else 0
        if local_size < gdrive_db.stat().st_size:
            import shutil

            try:
                shutil.copy2(gdrive_db, LOCAL_MLFLOW_DB_PATH)
            except Exception:
                return f"sqlite:///{gdrive_db.as_posix()}"

    return f"sqlite:///{LOCAL_MLFLOW_DB_PATH.as_posix()}"


MLFLOW_TRACKING_URI: str = resolve_active_mlflow_uri()
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
