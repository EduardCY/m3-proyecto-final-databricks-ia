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


def build_manifest(
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
    output_path: Optional[Path] = None,
    save_to_disk: bool = True,
) -> pd.DataFrame:
    """Recorre las subcarpetas del dataset y genera un manifiesto estructurado.

    Descarta automáticamente archivos que no sean imágenes o que estén corruptos.
    Guarda el resultado en `output_path` (por defecto `data/manifest.csv`) como punto de guardado.

    Args:
        data_dir: Directorio raíz donde reside el dataset.
        use_cache: Si es True y existe un manifest.csv previo, se reutiliza.
        output_path: Ruta personalizada para guardar o leer la caché del manifiesto.
        save_to_disk: Si es True, persiste el DataFrame en formato CSV.

    Returns:
        DataFrame con columnas: path, filename, label, label_name, subject_id, file_size, origin_split.
    """
    target_manifest_path = Path(output_path) if output_path else MANIFEST_PATH

    if use_cache and target_manifest_path.exists():
        logger.info(f"Cargando manifiesto desde caché: {target_manifest_path}")
        df = pd.read_csv(target_manifest_path)
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
    if save_to_disk:
        target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df_manifest.to_csv(target_manifest_path, index=False)
        logger.info(f"Manifiesto generado con {len(df_manifest)} imágenes. Guardado en: {target_manifest_path}")
    return df_manifest


def split_manifest(
    manifest_df: pd.DataFrame,
    split_ratios: Optional[dict] = None,
    ratios: Optional[dict] = None,
    seed: int = SEED,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Genera una partición estratificada 70/15/15 sin fuga de información.

    Garantiza que la distribución de clases se mantenga constante entre conjuntos.
    Guarda el resultado en `output_path` (por defecto `data/manifest_split.csv`).

    Args:
        manifest_df: DataFrame con el manifiesto completo.
        split_ratios: Proporciones deseadas (train, val, test).
        ratios: Alias de compatibilidad para split_ratios.
        seed: Semilla para reproducibilidad estricta.
        output_path: Ruta de guardado para el archivo CSV del split.

    Returns:
        DataFrame con la columna añadida 'split' ('train', 'val', 'test').
    """
    effective_ratios = split_ratios or ratios or SPLIT_RATIOS
    target_output_path = Path(output_path) if output_path else SPLIT_MANIFEST_PATH

    df = manifest_df.copy()
    rng = np.random.default_rng(seed)

    df["split"] = None

    # Estratificación por clase
    for cls_name in CLASSES:
        cls_mask = df["label_name"] == cls_name
        cls_indices = df[cls_mask].index.to_numpy()
        rng.shuffle(cls_indices)

        n_total = len(cls_indices)
        n_train = int(n_total * effective_ratios["train"])
        n_val = int(n_total * effective_ratios["val"])

        train_idx = cls_indices[:n_train]
        val_idx = cls_indices[n_train : n_train + n_val]
        test_idx = cls_indices[n_train + n_val :]

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    target_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_output_path, index=False)
    logger.info(
        f"Split estratificado completado: "
        f"Train={sum(df['split'] == 'train')}, "
        f"Val={sum(df['split'] == 'val')}, "
        f"Test={sum(df['split'] == 'test')}. "
        f"Guardado en: {target_output_path}"
    )
    return df


# Alias de compatibilidad para notebooks y scripts
create_stratified_split = split_manifest



def build_manifest_spark(data_dir: Path = DATA_DIR):
    """Genera el manifiesto mediante PySpark para simulación de la Fase 2 (Databricks).

    Utiliza `spark.read.format('binaryFile')` tal como se ejecutaría sobre
    un Data Lake / Unity Catalog con cientos de miles de imágenes médicas.
    El esquema resultante replica el del manifiesto pandas (`label_name`, `label`,
    `origin_split`) para que ambas rutas sean intercambiables.

    Args:
        data_dir: Directorio raíz con las carpetas de clases.

    Returns:
        DataFrame de Spark con columnas path, filename, raw_folder, label_name,
        label, origin_split y file_size_bytes; o None si Spark/Java no está disponible.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError:
        logger.warning("PySpark no está instalado; se omite el manifiesto distribuido.")
        return None

    try:
        spark = SparkSession.builder.appName("NeuroScan_Spark_Manifest").getOrCreate()
        binary_df = (
            spark.read.format("binaryFile")
            .option("recursiveFileLookup", "true")
            # Excluye el contenido binario del plan: solo se necesitan metadatos.
            .load(str(data_dir))
            .drop("content")
        )
    except Exception as e:  # Java ausente, ruta inexistente, etc.
        logger.warning(f"Spark no pudo inicializarse o leer {data_dir}: {e}")
        return None

    path_parts = F.split(F.col("path"), "/")
    raw_folder = F.lower(F.element_at(path_parts, -2))

    # Mapeo explícito carpeta -> clase canónica. No se usa regexp_replace("tumor$")
    # porque convertiría 'notumor' en 'no'.
    label_name_expr = None
    for cls_name in CLASSES:
        cond = (raw_folder == cls_name) | (raw_folder == f"{cls_name}tumor")
        label_name_expr = (
            F.when(cond, F.lit(cls_name))
            if label_name_expr is None
            else label_name_expr.when(cond, F.lit(cls_name))
        )

    label_idx_expr = None
    for cls_name, idx in CLASS_TO_IDX.items():
        cond = F.col("label_name") == cls_name
        label_idx_expr = (
            F.when(cond, F.lit(idx))
            if label_idx_expr is None
            else label_idx_expr.when(cond, F.lit(idx))
        )

    manifest_spark = (
        binary_df.select(
            F.col("path"),
            F.col("length").alias("file_size_bytes"),
            F.element_at(path_parts, -1).alias("filename"),
            raw_folder.alias("raw_folder"),
        )
        .filter(F.lower(F.col("filename")).rlike(r"\.(jpg|jpeg|png)$"))
        .withColumn("label_name", label_name_expr)
        .filter(F.col("label_name").isNotNull())
        .withColumn("label", label_idx_expr)
        .withColumn(
            "origin_split",
            F.when(F.lower(F.col("path")).contains("train"), F.lit("Training")).otherwise(
                F.lit("Testing")
            ),
        )
    )

    return manifest_spark


__all__ = [
    "compute_file_hash",
    "verify_image_integrity",
    "build_manifest",
    "split_manifest",
    "create_stratified_split",
    "build_manifest_spark",
]

