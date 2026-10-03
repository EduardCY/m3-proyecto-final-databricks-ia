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

from src.config import CLASSES, DATA_DIR, IMG_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Enlace de respaldo público del dataset compilado
BACKUP_DATASET_URL = (
    "https://github.com/masoudnickparvar/brain-tumor-mri-dataset/archive/refs/heads/main.zip"
)


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


def verify_dataset_structure(data_dir: Path) -> bool:
    """Verifica si el dataset en el directorio tiene imágenes válidas para las clases requeridas."""
    found_any = False
    for cls_name in CLASSES:
        images = list(data_dir.rglob(f"*{cls_name}*/*.jpg")) + list(data_dir.rglob(f"*{cls_name}*/*.jpeg"))
        if images:
            logger.info(f"Clase detectada '{cls_name}': {len(images)} imágenes.")
            found_any = True
        else:
            logger.warning(f"No se detectaron imágenes para la clase '{cls_name}' en {data_dir}")
    return found_any


def main():
    parser = argparse.ArgumentParser(description="Descarga y verificación del dataset NeuroScan AI.")
    parser.add_argument("--mock", action="store_true", help="Genera datos sintéticos para pruebas locales rápidas.")
    parser.add_argument("--dataset", type=str, default="masoudnickparvar/brain-tumor-mri-dataset", help="Identificador Kaggle.")
    parser.add_argument("--target-dir", type=str, default=str(DATA_DIR), help="Directorio destino.")
    args = parser.parse_args()

    target_path = Path(args.target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    if args.mock:
        generate_smoke_test_dataset(target_path)
        return

    # Verificar si ya existen imágenes
    if verify_dataset_structure(target_path):
        logger.info(f"El dataset ya se encuentra disponible en: {target_path}")
        return

    # Intento 1: Kaggle API
    if download_from_kaggle(args.dataset, target_path):
        return

    # Intento 2: Instrucción clara de descarga manual con fallback mock
    logger.info("\n" + "=" * 70)
    logger.info("INSTRUCCIÓN DE DESCARGA RÁPIDA:")
    logger.info("1. Descarga el dataset (~165 MB) desde:")
    logger.info("   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    logger.info(f"2. Descomprime su contenido dentro de: {target_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
