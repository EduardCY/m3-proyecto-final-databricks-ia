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
    INPUT_EXAMPLE_INFO_PATH,
    IS_COLAB,
    MODEL_CHECKPOINTS_DIR,
    PROJECT_ROOT,
    get_gdrive_mount_point,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def mount_google_drive(force_remount: bool = False) -> bool:
    """Monta Google Drive si el script se ejecuta dentro de Google Colab."""
    if not IS_COLAB:
        logger.info("Entorno local detectado. No se requiere 'google.colab.drive.mount'.")
        return False

    # Si ya está montado y accesible MyDrive, no volver a montar
    if Path("/content/drive/MyDrive").exists() or Path("/content/drive/Shareddrives").exists():
        logger.info("Google Drive ya se encuentra montado en /content/drive/MyDrive.")
        return True

    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=force_remount)
        logger.info("Google Drive montado exitosamente en /content/drive.")
        return True
    except Exception as e:
        logger.warning(
            f"No se pudo montar Google Drive automáticamente: {e}. "
            "Si ejecutas en Google Colab, pulsa el botón 'Montar Drive' en el panel lateral de archivos para autorizar el acceso."
        )
        return False


def backup_to_gdrive(source_dir: Path = PROJECT_ROOT, drive_root: Path | None = None) -> None:
    """Copia la base de datos de MLflow, mlruns, checkpoints y modelos hacia Google Drive."""
    effective_drive_root = drive_root if drive_root is not None else get_gdrive_mount_point()
    is_drive_mounted = Path("/content/drive/MyDrive").exists() or Path("/content/drive/Shareddrives").exists()
    if IS_COLAB and not is_drive_mounted:
        logger.warning(
            "Google Drive no está montado en /content/drive/MyDrive; se omite el respaldo en nube "
            f"(los artefactos se preservan localmente en {source_dir}). "
            "Para activar el respaldo en Drive, autoriza la ventana modal de Google Drive en Colab."
        )
        return

    effective_drive_root.mkdir(parents=True, exist_ok=True)
    drive_ckpt_dir = effective_drive_root / "checkpoints"
    drive_backup_dir = effective_drive_root / "backups"
    drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
    drive_backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Respaldar base de datos de MLflow (solo si contiene tablas inicializadas)
    local_mlflow_db = source_dir / "mlflow.db"
    if local_mlflow_db.exists() and local_mlflow_db.stat().st_size > 0:
        dest_db = effective_drive_root / "mlflow.db"
        if local_mlflow_db.resolve() != dest_db.resolve():
            shutil.copy2(local_mlflow_db, dest_db)
            logger.info(f"[Backup] MLflow DB respaldada en Drive: {dest_db}")

    # 2. Respaldar directorio mlruns (artefactos, gráficas y modelos registrados en MLflow)
    local_mlruns = source_dir / "mlruns"
    if local_mlruns.exists():
        dest_mlruns = effective_drive_root / "mlruns"
        if local_mlruns.resolve() != dest_mlruns.resolve():
            shutil.copytree(local_mlruns, dest_mlruns, dirs_exist_ok=True)
            logger.info(f"[Backup] Directorio mlruns respaldado en Drive: {dest_mlruns}")

    # 3. Respaldar modelo final y firma de entrada si existen
    if FINAL_MODEL_PATH.exists():
        dest_model = effective_drive_root / FINAL_MODEL_PATH.name
        if FINAL_MODEL_PATH.resolve() != dest_model.resolve():
            shutil.copy2(FINAL_MODEL_PATH, dest_model)
            logger.info(f"[Backup] Modelo final respaldado en Drive: {dest_model}")

    if INPUT_EXAMPLE_INFO_PATH.exists():
        dest_info = effective_drive_root / INPUT_EXAMPLE_INFO_PATH.name
        if INPUT_EXAMPLE_INFO_PATH.resolve() != dest_info.resolve():
            shutil.copy2(INPUT_EXAMPLE_INFO_PATH, dest_info)

    # 4. Respaldar checkpoints (.keras, historiales .csv, metadatos y gráficas por corrida)
    if MODEL_CHECKPOINTS_DIR.exists():
        for item in MODEL_CHECKPOINTS_DIR.iterdir():
            dest_item = drive_ckpt_dir / item.name
            if item.resolve() == dest_item.resolve():
                continue
            if item.is_file() and item.suffix in {".keras", ".csv", ".json"}:
                shutil.copy2(item, dest_item)
                logger.info(f"[Backup] Checkpoint respaldado: {dest_item.name}")
            elif item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)

    # 5. Respaldar manifiesto estratificado
    split_manifest = source_dir / "data" / "manifest_split.csv"
    if split_manifest.exists():
        dest_manifest = effective_drive_root / "data" / "manifest_split.csv"
        dest_manifest.parent.mkdir(parents=True, exist_ok=True)
        if split_manifest.resolve() != dest_manifest.resolve():
            shutil.copy2(split_manifest, dest_manifest)
            logger.info(f"[Backup] Manifiesto estratificado respaldado en: {dest_manifest}")

    logger.info("Respaldo completo a Google Drive finalizado con éxito.")


def restore_from_gdrive(drive_root: Path | None = None, dest_dir: Path = PROJECT_ROOT) -> None:
    """Restaura los artefactos desde Google Drive hacia el entorno local de trabajo."""
    effective_drive_root = drive_root if drive_root is not None else get_gdrive_mount_point()
    if not effective_drive_root.exists():
        logger.warning(f"La ruta de Google Drive no existe: {effective_drive_root}")
        return

    # 1. Restaurar base de datos de MLflow
    drive_mlflow_db = effective_drive_root / "mlflow.db"
    local_mlflow_db = dest_dir / "mlflow.db"
    if drive_mlflow_db.exists() and drive_mlflow_db.stat().st_size > 0:
        if drive_mlflow_db.resolve() != local_mlflow_db.resolve():
            shutil.copy2(drive_mlflow_db, local_mlflow_db)
            logger.info(f"[Restore] Base de datos restaurada en: {local_mlflow_db}")

    # 2. Restaurar directorio mlruns
    drive_mlruns = effective_drive_root / "mlruns"
    local_mlruns = dest_dir / "mlruns"
    if drive_mlruns.exists() and drive_mlruns.resolve() != local_mlruns.resolve():
        shutil.copytree(drive_mlruns, local_mlruns, dirs_exist_ok=True)
        logger.info(f"[Restore] Directorio mlruns restaurado en: {local_mlruns}")

    # 3. Restaurar modelo final y firma JSON
    drive_model = effective_drive_root / "neuro_mri_model.keras"
    if drive_model.exists():
        dest_model_path = dest_dir / "models" / "neuro_mri_model.keras"
        dest_model_path.parent.mkdir(parents=True, exist_ok=True)
        if drive_model.resolve() != dest_model_path.resolve():
            shutil.copy2(drive_model, dest_model_path)
            logger.info(f"[Restore] Modelo restaurado en: {dest_model_path}")

    drive_info = effective_drive_root / "input_example_info.json"
    if drive_info.exists():
        dest_info_path = dest_dir / "models" / "input_example_info.json"
        dest_info_path.parent.mkdir(parents=True, exist_ok=True)
        if drive_info.resolve() != dest_info_path.resolve():
            shutil.copy2(drive_info, dest_info_path)

    # 4. Restaurar checkpoints por corrida
    drive_ckpt_dir = effective_drive_root / "checkpoints"
    local_ckpt_dir = dest_dir / "models" / "checkpoints"
    if drive_ckpt_dir.exists() and drive_ckpt_dir.resolve() != local_ckpt_dir.resolve():
        local_ckpt_dir.mkdir(parents=True, exist_ok=True)
        for item in drive_ckpt_dir.iterdir():
            dest_item = local_ckpt_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest_item)
            elif item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
        logger.info(f"[Restore] Checkpoints restaurados en: {local_ckpt_dir}")

    # 5. Restaurar manifiesto
    drive_manifest = effective_drive_root / "data" / "manifest_split.csv"
    if drive_manifest.exists():
        dest_manifest_path = dest_dir / "data" / "manifest_split.csv"
        dest_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if drive_manifest.resolve() != dest_manifest_path.resolve():
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
    logger.info("[OK] Dataset desplegado en almacenamiento efímero local con CERO latencia FUSE.")
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
