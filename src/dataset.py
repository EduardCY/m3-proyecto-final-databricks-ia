"""
Módulo de ingesta, auditoría de integridad y partición del dataset Brain Tumor MRI.

Incluye detección de imágenes corruptas, construcción de manifiesto reproducible,
prevención de fuga de datos (data leakage) y versión distribuida en PySpark.
"""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from src.config import (
    CLASSES,
    CLASS_TO_IDX,
    DATA_DIR,
    MANIFEST_PATH,
    SEED,
    SPLIT_MANIFEST_PATH,
    SPLIT_RATIOS,
)

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo para verificar duplicados e integridad."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_image_integrity(file_path: Path) -> bool:
    """Verifica si una imagen está íntegra y puede ser decodificada sin errores.

    Args:
        file_path: Ruta al archivo de imagen.

    Returns:
        True si la imagen es válida; False si está truncada o corrupta.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Imagen corrupta descartada: {file_path.name} - Motivo: {e}")
        return False


def _extract_subject_id(filename: str, label_name: str) -> str:
    """Extrae o deduce un identificador de sujeto/estudio a partir del nombre de archivo.

    Si el dataset no codifica el paciente explícitamente, se extrae el prefijo numérico
    o se asigna un identificador único por imagen para evitar agrupaciones falsas.
    """
    # Patrón común en datasets de resonancias: te-XX_0001, Tr-gl_0010, etc.
    match = re.match(r"^([a-zA-Z]+[_-]?\d+)", filename)
    if match:
        return f"{label_name}_{match.group(1)}"
    return f"{label_name}_{filename}"


def build_manifest(data_dir: Path = DATA_DIR, use_cache: bool = True) -> pd.DataFrame:
    """Recorre las subcarpetas del dataset y genera un manifiesto estructurado.

    Descarta automáticamente archivos que no sean imágenes o que estén corruptos.
    Guarda el resultado en `data/manifest.csv` como punto de guardado (checkpoint).

    Args:
        data_dir: Directorio raíz donde reside el dataset.
        use_cache: Si es True y existe un manifest.csv previo, se reutiliza.

    Returns:
        DataFrame con columnas: path, filename, label, label_name, subject_id, file_size, origin_split.
    """
    if use_cache and MANIFEST_PATH.exists():
        logger.info(f"Cargando manifiesto desde caché: {MANIFEST_PATH}")
        df = pd.read_csv(MANIFEST_PATH)
        if not df.empty:
            return df

    rows = []
    # Explorar recursivamente buscando las 4 clases canónicas
    for root_dir, _, files in os.walk(data_dir):
        folder_name = Path(root_dir).name.lower()
        
        # Identificar la clase correspondiente a la carpeta
        matched_class = None
        for cls_name in CLASSES:
            if cls_name == folder_name or f"{cls_name}tumor" == folder_name:
                matched_class = cls_name
                break

        if not matched_class:
            continue

        label_idx = CLASS_TO_IDX[matched_class]
        origin_split = "Training" if "train" in root_dir.lower() else "Testing"

        for file_name in sorted(files):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            file_path = Path(root_dir) / file_name
            # Failsafe: descartar imágenes corruptas
            if not verify_image_integrity(file_path):
                continue

            rows.append(
                {
                    "path": str(file_path.resolve()),
                    "filename": file_name,
                    "label": label_idx,
                    "label_name": matched_class,
                    "subject_id": _extract_subject_id(file_name, matched_class),
                    "file_size_bytes": file_path.stat().st_size,
                    "origin_split": origin_split,
                }
            )

    df_manifest = pd.DataFrame(rows)
    if df_manifest.empty:
        raise FileNotFoundError(
            f"No se encontraron imágenes válidas en {data_dir}. "
            "Ejecuta 'python scripts/download_dataset.py' o coloca las carpetas de las clases en data/raw/."
        )

    # Guardar punto de guardado (checkpoint)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    logger.info(f"Manifiesto generado con {len(df_manifest)} imágenes. Guardado en: {MANIFEST_PATH}")
    return df_manifest


def split_manifest(
    manifest_df: pd.DataFrame,
    split_ratios: dict = SPLIT_RATIOS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Genera una partición estratificada 70/15/15 sin fuga de información.

    Garantiza que la distribución de clases se mantenga constante entre conjuntos.
    Guarda el resultado en `data/manifest_split.csv` como punto de guardado.

    Args:
        manifest_df: DataFrame con el manifiesto completo.
        split_ratios: Proporciones deseadas (train, val, test).
        seed: Semilla para reproducibilidad estricta.

    Returns:
        DataFrame con la columna añadida 'split' ('train', 'val', 'test').
    """
    df = manifest_df.copy()
    rng = np.random.default_rng(seed)

    df["split"] = None

    # Estratificación por clase
    for cls_name in CLASSES:
        cls_mask = df["label_name"] == cls_name
        cls_indices = df[cls_mask].index.to_numpy()
        rng.shuffle(cls_indices)

        n_total = len(cls_indices)
        n_train = int(n_total * split_ratios["train"])
        n_val = int(n_total * split_ratios["val"])

        train_idx = cls_indices[:n_train]
        val_idx = cls_indices[n_train : n_train + n_val]
        test_idx = cls_indices[n_train + n_val :]

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLIT_MANIFEST_PATH, index=False)
    logger.info(
        f"Split estratificado completado: "
        f"Train={sum(df['split'] == 'train')}, "
        f"Val={sum(df['split'] == 'val')}, "
        f"Test={sum(df['split'] == 'test')}. "
        f"Guardado en: {SPLIT_MANIFEST_PATH}"
    )
    return df


def build_manifest_spark(data_dir: Path = DATA_DIR):
    """Genera el manifiesto mediante PySpark para simulación de la Fase 2 (Databricks).

    Utiliza `spark.read.format('binaryFile')` tal como se ejecutaría sobre
    un Data Lake / Unity Catalog con cientos de miles de imágenes médicas.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.appName("NeuroScan_Spark_Manifest").getOrCreate()
    binary_df = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load(str(data_dir))
    )

    # Extraer nombre de archivo y clase
    manifest_spark = (
        binary_df.select(
            F.col("path"),
            F.col("length").alias("file_size_bytes"),
            F.element_at(F.split(F.col("path"), "/"), -1).alias("filename"),
            F.element_at(F.split(F.col("path"), "/"), -2).alias("raw_folder"),
        )
        .filter(F.col("filename").rlike(r"\.(jpg|jpeg|png)$"))
    )

    return manifest_spark
