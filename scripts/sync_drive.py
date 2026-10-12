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

    # 4. Respaldar manifiesto estratificado
    split_manifest = source_dir / "data" / "manifest_split.csv"
    if split_manifest.exists():
        dest_manifest = drive_root / "data" / "manifest_split.csv"
        dest_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_manifest, dest_manifest)
        logger.info(f"[Backup] Manifiesto estratificado respaldado en: {dest_manifest}")

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

    # Restaurar manifiesto
    drive_manifest = drive_root / "data" / "manifest_split.csv"
    if drive_manifest.exists():
        dest_manifest_path = dest_dir / "data" / "manifest_split.csv"
        dest_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_manifest, dest_manifest_path)
        logger.info(f"[Restore] Manifiesto restaurado en: {dest_manifest_path}")


def sync_dataset_to_ephemeral(
    drive_root: Path = GDRIVE_MOUNT_POINT,
    target_data_dir: Path = Path("/content/data/raw"),
) -> bool:
    """Copia el dataset comprimido desde Google Drive a disco efímero NVMe y lo descomprime.

    Why: El sistema de archivos FUSE de Google Colab (/content/drive) impone alta latencia
    por llamada I/O y agota cuotas de API al leer miles de imágenes pequeñas sueltas.
    Transferir un único archivo comprimido secuencial y descomprimirlo localmente en NVMe
    desbloquea el ancho de banda máximo para que la GPU nunca sufra cuellos de botella.
    """
    import zipfile

    # Buscar zip en ubicaciones estándar de Google Drive
    candidate_zips = [
        drive_root / "data" / "brain_tumor_mri_clean.zip",
        drive_root / "brain_tumor_mri_clean.zip",
        drive_root / "brain-tumor-mri-dataset.zip",
    ]

    source_zip = None
    for cand in candidate_zips:
        if cand.exists():
            source_zip = cand
            break

    if not source_zip:
        logger.warning(
            f"No se encontró un zip del dataset en Google Drive ({drive_root}). "
            "Se requerirá descarga remota a través de download_dataset.py."
        )
        return False

    target_data_dir.mkdir(parents=True, exist_ok=True)
    temp_zip_dest = target_data_dir.parent / "dataset_ephemeral.zip"

    logger.info(f"Copiando secuencialmente {source_zip.name} a disco efímero ({temp_zip_dest})...")
    shutil.copy2(source_zip, temp_zip_dest)

    logger.info(f"Descomprimiendo en disco efímero local: {target_data_dir}...")
    with zipfile.ZipFile(temp_zip_dest, "r") as zf:
        zf.extractall(target_data_dir)

    temp_zip_dest.unlink(missing_ok=True)
    logger.info("✅ Dataset desplegado en almacenamiento efímero local con CERO latencia FUSE.")
    return True


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
