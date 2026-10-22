"""
Script resiliente para la descarga, verificación y descompresión del dataset Brain Tumor MRI.

Soporta descarga mediante Kaggle API, HuggingFace Hub, enlaces directos y
generación de datos sintéticos (smoke testing) para pruebas unitarias sin conexión.
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Añadir la raíz del proyecto al sys.path para importar la configuración
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLASSES,
    DATA_DIR,
    GDRIVE_DATA_DIR,
    GDRIVE_MOUNT_POINT,
    IMG_SIZE,
    IS_COLAB,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Enlaces de descarga directa del dataset oficial (164.67 MB)
KAGGLE_DATASET_ID = "masoudnickparvar/brain-tumor-mri-dataset"
KAGGLE_DIRECT_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset"
)
BACKUP_DATASET_URL = KAGGLE_DIRECT_URL


def download_stream(url: str, output_path: Path, timeout: int = 60) -> bool:
    """Descarga un archivo grande vía streaming HTTP con User-Agent estándar para evitar bloqueos 403/404."""
    import urllib.request

    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )
    logger.info(f"Conectando a {url}...")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB por bloque
            with open(output_path, "wb") as out_file:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) < block_size:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"Progreso descarga: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
        logger.info(f"[OK] Descarga completada exitosamente: {output_path.name} ({downloaded / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        logger.warning(f"Error descargando desde {url}: {e}")
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return False


def generate_smoke_test_dataset(output_dir: Path, samples_per_class: int = 15) -> None:
    """Genera un dataset sintético ligero para validar pipelines y pruebas automatizadas offline.

    Args:
        output_dir: Directorio destino donde se crearán las subcarpetas de clase.
        samples_per_class: Cantidad de imágenes de prueba a crear por cada clase.
    """
    logger.info(f"Generando dataset mock de prueba en: {output_dir}")
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        logger.error("Se requiere pillow y numpy para generar datos de prueba.")
        return

    for split in ["Training", "Testing"]:
        for cls_name in CLASSES:
            folder = output_dir / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(samples_per_class if split == "Training" else max(3, samples_per_class // 3)):
                # Generar imagen sintética simulando un corte de resonancia magnética
                arr = np.random.randint(20, 230, size=(*IMG_SIZE, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(folder / f"mock_{cls_name}_{split}_{i:03d}.jpg")

    logger.info("Dataset mock generado exitosamente para validación de pruebas.")


def download_from_kaggle(dataset_name: str, target_dir: Path) -> bool:
    """Intenta descargar el dataset utilizando la API oficial de Kaggle."""
    try:
        import kaggle
        logger.info(f"Descargando {dataset_name} vía Kaggle API...")
        kaggle.api.dataset_download_files(dataset_name, path=str(target_dir), unzip=True)
        logger.info("Descarga y descompresión con Kaggle API completada.")
        return True
    except Exception as e:
        logger.warning(f"No se pudo descargar vía Kaggle API: {e}")
        return False


def verify_dataset_structure(
    data_dir: Path, min_samples_per_class: int = 10, require_real: bool = False, verbose: bool = True
) -> bool:
    """Verifica si el dataset en el directorio tiene imágenes válidas para las clases requeridas."""
    total_valid = 0
    for cls_name in CLASSES:
        all_imgs = (
            list(data_dir.rglob(f"*{cls_name}*/*.jpg"))
            + list(data_dir.rglob(f"*{cls_name}*/*.jpeg"))
            + list(data_dir.rglob(f"*{cls_name}*/*.png"))
        )
        if require_real:
            images = [p for p in all_imgs if not p.name.startswith("mock_")]
        else:
            images = all_imgs

        if len(images) >= min_samples_per_class:
            if verbose:
                logger.info(f"Clase detectada '{cls_name}': {len(images)} imágenes.")
            total_valid += len(images)
        else:
            if verbose:
                logger.warning(f"No se detectaron suficientes imágenes para '{cls_name}' en {data_dir} ({len(images)} halladas)")
            return False

    if require_real and total_valid < 1000:
        if verbose:
            logger.warning(f"Total imágenes reales ({total_valid}) menor al umbral esperado (1000).")
        return False
    return total_valid > 0


def download_via_kagglehub(target_dir: Path) -> bool:
    """Descarga el dataset público oficial usando kagglehub (disponible en Colab sin credenciales)."""
    try:
        import kagglehub

        logger.info(f"Descargando {KAGGLE_DATASET_ID} vía kagglehub (alta velocidad sin credenciales)...")
        cache_path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
        logger.info(f"Dataset descargado en caché de kagglehub: {cache_path}")
        src_path = Path(cache_path)
        for item in src_path.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return verify_dataset_structure(target_dir, require_real=True)
    except Exception as e:
        logger.warning(f"kagglehub no disponible o falló: {e}")
        return False


def ensure_dataset_ready(data_dir: Path = DATA_DIR, force_download: bool = False) -> bool:
    """Garantiza la disponibilidad inmediata del dataset en el directorio destino.

    Estrategia de resolución en capas:
    1. Reutilización si ya existen imágenes reales (>= 1,000 cortes).
    2. Copia local desde Google Drive si existe el zip respaldado.
    3. Descarga ultrarrápida vía `kagglehub` (estándar en Google Colab sin credenciales).
    4. Descarga directa en streaming HTTP desde el endpoint público de Kaggle (~164.67 MB).
    5. Fallback a datos mock solo en caso de desconexión absoluta.
    """
    target_path = Path(data_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    if not force_download and verify_dataset_structure(target_path, require_real=True, verbose=False):
        logger.info(f"[OK] Dataset real disponible y validado en: {target_path}")
        return True

    # 1. Comprobar si existe un archivo zip en Google Drive montado
    candidate_drive_zips = [
        GDRIVE_DATA_DIR / "brain_tumor_mri_clean.zip",
        GDRIVE_MOUNT_POINT / "data" / "brain_tumor_mri_clean.zip",
        GDRIVE_MOUNT_POINT / "brain_tumor_mri_clean.zip",
    ]
    for drive_zip in candidate_drive_zips:
        if drive_zip.exists():
            logger.info(f"Detectado archivo zip en Google Drive: {drive_zip}. Copiando a almacenamiento local...")
            local_zip = target_path.parent / "dataset_from_drive.zip"
            shutil.copy2(drive_zip, local_zip)
            with zipfile.ZipFile(local_zip, "r") as zf:
                zf.extractall(target_path)
            local_zip.unlink(missing_ok=True)
            if verify_dataset_structure(target_path, require_real=True):
                logger.info("[OK] Dataset descomprimido en local desde Google Drive con exito.")
                return True

    # 2. Intentar descarga vía kagglehub (óptimo en Google Colab)
    if download_via_kagglehub(target_path):
        logger.info("[OK] Dataset descargado y descomprimido exitosamente vía kagglehub.")
        return True

    # 3. Descarga directa streaming del ZIP oficial de Kaggle (~164.67 MB)
    logger.info("Iniciando descarga streaming directa del dataset Brain Tumor MRI (~164.67 MB)...")
    zip_tmp = target_path.parent / "dataset_download_temp.zip"
    temp_extract = target_path.parent / "temp_extracted"

    download_ok = download_stream(KAGGLE_DIRECT_URL, zip_tmp)
    if download_ok and zip_tmp.exists():
        try:
            with zipfile.ZipFile(zip_tmp, "r") as zip_ref:
                zip_ref.extractall(temp_extract)

            # Localizar subcarpeta extraída o raíz
            extracted_subfolders = list(temp_extract.glob("brain-tumor-mri-dataset*"))
            source_root = extracted_subfolders[0] if extracted_subfolders else temp_extract

            # Limpiar datos mock previos en destino si existían
            for mock_file in target_path.rglob("mock_*.jpg"):
                mock_file.unlink(missing_ok=True)

            for item in source_root.iterdir():
                dest = target_path / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(target_path))

            shutil.rmtree(temp_extract, ignore_errors=True)

            # Guardar una copia comprimida en Google Drive para acelerar ejecuciones futuras
            if GDRIVE_MOUNT_POINT.exists() or Path("/content/drive/MyDrive").exists():
                try:
                    GDRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    dest_drive_zip = GDRIVE_DATA_DIR / "brain_tumor_mri_clean.zip"
                    if not dest_drive_zip.exists() and zip_tmp.exists():
                        shutil.copy2(zip_tmp, dest_drive_zip)
                        logger.info(f"Copia comprimida guardada en Google Drive: {dest_drive_zip}")
                except Exception as e_drive:
                    logger.warning(f"No se pudo guardar copia en Google Drive: {e_drive}")

            zip_tmp.unlink(missing_ok=True)
            if verify_dataset_structure(target_path, require_real=True):
                logger.info("[OK] Dataset descargado y descomprimido exitosamente en almacenamiento local.")
                return True
        except Exception as e_extract:
            logger.warning(f"Error al extraer archivo zip: {e_extract}")
            shutil.rmtree(temp_extract, ignore_errors=True)
            zip_tmp.unlink(missing_ok=True)

    # 4. Fallback final de contingencia
    logger.warning("[AVISO] No se pudo obtener el dataset real. Generando dataset de contingencia sintético...")
    generate_smoke_test_dataset(target_path, samples_per_class=30)
    return False


def main():
    parser = argparse.ArgumentParser(description="Descarga y verificación del dataset NeuroScan AI.")
    parser.add_argument("--mock", action="store_true", help="Genera datos sintéticos para pruebas locales rápidas.")
    parser.add_argument("--dataset", type=str, default="masoudnickparvar/brain-tumor-mri-dataset", help="Identificador Kaggle.")
    parser.add_argument("--target-dir", type=str, default=str(DATA_DIR), help="Directorio destino.")
    parser.add_argument("--force", action="store_true", help="Fuerza la descarga remota del dataset.")
    args = parser.parse_args()

    target_path = Path(args.target_dir)

    if args.mock:
        generate_smoke_test_dataset(target_path)
        return

    ensure_dataset_ready(target_path, force_download=args.force)


if __name__ == "__main__":
    main()

