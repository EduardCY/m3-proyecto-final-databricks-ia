"""
Script para descargar modelos entrenados, datasets y bases de datos desde Google Drive.

Permite a los evaluadores y profesores descargar los artefactos pesados con un solo comando,
manteniendo el repositorio de GitHub ultra-ligero y libre de binarios innecesarios.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FINAL_MODEL_PATH, MODELS_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Enlaces públicos de Google Drive (Reemplazar con los IDs públicos definitivos del estudiante)
# Formato Google Drive directo: https://drive.google.com/uc?export=download&id=FILE_ID
GDRIVE_PUBLIC_RESOURCES = {
    "model": {
        "description": "Modelo final entrenado (ResNet50 Fine-Tuning - 95.8 MB)",
        "target_path": FINAL_MODEL_PATH,
        "default_url": "https://drive.google.com/uc?export=download&id=TU_DRIVE_MODEL_ID",
    },
    "mlflow_db": {
        "description": "Base de datos SQLite de experimentos de MLflow (mlflow.db)",
        "target_path": PROJECT_ROOT / "mlflow.db",
        "default_url": "https://drive.google.com/uc?export=download&id=TU_DRIVE_MLFLOW_ID",
    },
    "dataset": {
        "description": "Dataset limpio de resonancias magnéticas (~165 MB zip)",
        "target_path": PROJECT_ROOT / "data" / "brain_tumor_mri_clean.zip",
        "default_url": "https://drive.google.com/uc?export=download&id=TU_DRIVE_DATASET_ID",
    },
}


def download_file_from_drive(file_id_or_url: str, output_path: Path) -> bool:
    """Descarga un archivo público de Google Drive usando urllib o gdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extraer ID si se pasa la URL completa
    if "drive.google.com" in file_id_or_url and "id=" in file_id_or_url:
        file_id = file_id_or_url.split("id=")[-1].split("&")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    else:
        download_url = file_id_or_url

    logger.info(f"Descargando recurso hacia: {output_path}...")
    try:
        # Intentar con gdown si está disponible
        import gdown
        gdown.download(download_url, str(output_path), quiet=False)
        logger.info(f"✅ Descarga exitosa con gdown: {output_path.name}")
        return True
    except ImportError:
        pass

    try:
        urlretrieve(download_url, str(output_path))
        logger.info(f"✅ Descarga completada: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error al descargar desde Google Drive: {e}")
        logger.info("Instrucción manual: Puedes descargar el archivo manualmente y colocarlo en la ruta indicada.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Gestor de descarga de artefactos desde Google Drive.")
    parser.add_argument("--model", action="store_true", help="Descarga el modelo campeón entrenado neuro_mri_model.keras.")
    parser.add_argument("--mlflow", action="store_true", help="Descarga la base de datos de experimentos mlflow.db.")
    parser.add_argument("--dataset", action="store_true", help="Descarga el dataset completo comprimido.")
    parser.add_argument("--all", action="store_true", help="Descarga todos los recursos pesados.")
    parser.add_argument("--url", type=str, default=None, help="URL o ID de Google Drive personalizada.")
    args = parser.parse_args()

    if not (args.model or args.mlflow or args.dataset or args.all):
        parser.print_help()
        print("\nEjemplo de uso:")
        print("  python scripts/download_drive_assets.py --model")
        print("  python scripts/download_drive_assets.py --all")
        return

    if args.model or args.all:
        res = GDRIVE_PUBLIC_RESOURCES["model"]
        url = args.url if args.url else res["default_url"]
        download_file_from_drive(url, res["target_path"])

    if args.mlflow or args.all:
        res = GDRIVE_PUBLIC_RESOURCES["mlflow_db"]
        url = args.url if args.url else res["default_url"]
        download_file_from_drive(url, res["target_path"])

    if args.dataset or args.all:
        res = GDRIVE_PUBLIC_RESOURCES["dataset"]
        url = args.url if args.url else res["default_url"]
        download_file_from_drive(url, res["target_path"])


if __name__ == "__main__":
    main()
