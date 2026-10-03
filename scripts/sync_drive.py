"""
Script de sincronización de artefactos y bases de datos con Google Drive.

Permite respaldar y recuperar checkpoints, base de datos SQLite de MLflow
y el modelo final .keras entre Google Colab Pro y la máquina local.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FINAL_MODEL_PATH,
    GDRIVE_BACKUP_DIR,
    GDRIVE_CHECKPOINTS_DIR,
    GDRIVE_MOUNT_POINT,
    IS_COLAB,
    MODEL_CHECKPOINTS_DIR,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def mount_google_drive() -> bool:
    """Monta Google Drive si el script se ejecuta dentro de Google Colab."""
    if not IS_COLAB:
        logger.info("Entorno local detectado. No se requiere 'google.colab.drive.mount'.")
        return False

    try:
        from google.colab import drive
        drive.mount("/content/drive")
        logger.info("Google Drive montado exitosamente en /content/drive.")
        return True
    except Exception as e:
        logger.warning(f"No se pudo montar Google Drive automáticamente: {e}")
        return False


def backup_to_gdrive(source_dir: Path = PROJECT_ROOT, drive_root: Path = GDRIVE_MOUNT_POINT) -> None:
    """Copia la base de datos de MLflow, checkpoints y modelos hacia Google Drive."""
    drive_root.mkdir(parents=True, exist_ok=True)
    GDRIVE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    GDRIVE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Respaldar base de datos de MLflow
    local_mlflow_db = source_dir / "mlflow.db"
    if local_mlflow_db.exists():
        dest_db = drive_root / "mlflow.db"
        shutil.copy2(local_mlflow_db, dest_db)
        logger.info(f"[Backup] MLflow DB respaldada en Drive: {dest_db}")

    # 2. Respaldar modelo final si existe
    if FINAL_MODEL_PATH.exists():
        dest_model = drive_root / FINAL_MODEL_PATH.name
        shutil.copy2(FINAL_MODEL_PATH, dest_model)
        logger.info(f"[Backup] Modelo final respaldado en Drive: {dest_model}")

    # 3. Respaldar checkpoints
    if MODEL_CHECKPOINTS_DIR.exists():
        for ckpt in MODEL_CHECKPOINTS_DIR.glob("*.keras"):
            dest_ckpt = GDRIVE_CHECKPOINTS_DIR / ckpt.name
            shutil.copy2(ckpt, dest_ckpt)
            logger.info(f"[Backup] Checkpoint respaldado: {dest_ckpt.name}")

    logger.info("Respaldo completo a Google Drive finalizado con éxito.")


def restore_from_gdrive(drive_root: Path = GDRIVE_MOUNT_POINT, dest_dir: Path = PROJECT_ROOT) -> None:
    """Restaura los artefactos desde Google Drive hacia el entorno local de trabajo."""
    if not drive_root.exists():
        logger.warning(f"La ruta de Google Drive no existe: {drive_root}")
        return

    # Restaurar base de datos de MLflow
    drive_mlflow_db = drive_root / "mlflow.db"
    if drive_mlflow_db.exists():
        shutil.copy2(drive_mlflow_db, dest_dir / "mlflow.db")
        logger.info(f"[Restore] Base de datos restaurada en: {dest_dir / 'mlflow.db'}")

    # Restaurar modelo
    drive_model = drive_root / "neuro_mri_model.keras"
    if drive_model.exists():
        dest_model_path = dest_dir / "models" / "neuro_mri_model.keras"
        dest_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_model, dest_model_path)
        logger.info(f"[Restore] Modelo restaurado en: {dest_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Sincronizador de artefactos con Google Drive.")
    parser.add_argument("--backup", action="store_true", help="Copia artefactos locales hacia Google Drive.")
    parser.add_argument("--restore", action="store_true", help="Restaura artefactos desde Google Drive al proyecto.")
    args = parser.parse_args()

    if IS_COLAB:
        mount_google_drive()

    if args.backup:
        backup_to_gdrive()
    elif args.restore:
        restore_from_gdrive()
    else:
        logger.info("Especifica --backup o --restore.")


if __name__ == "__main__":
    main()
