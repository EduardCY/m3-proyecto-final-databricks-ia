"""
Pipeline de preprocesamiento de alto rendimiento con TensorFlow (tf.data).

Implementa decodificación segura de 3 canales, normalización específica de pesos
de ImageNet, Data Augmentation clínicamente seguro y optimización con prefetch.
"""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from src.config import BATCH_SIZE, IMG_CHANNELS, IMG_SIZE, NUM_CLASSES, PROJECT_ROOT, SEED


def _resolve_image_path(raw_path: str, project_root: Path) -> str:
    """Resuelve la ruta de la imagen asegurando compatibilidad multiplataforma (Windows/Linux/Colab)."""
    p = Path(raw_path)
    if p.is_file():
        return str(p.resolve())

    # Fallback 1: Buscar relativo a data/raw con la subcarpeta de clase
    candidate = project_root / "data" / "raw" / p.parent.name / p.name
    if candidate.is_file():
        return str(candidate.resolve())

    # Fallback 2: Buscar en data/raw recursivamente por nombre de archivo
    matches = list((project_root / "data" / "raw").rglob(p.name))
    if matches:
        return str(matches[0].resolve())

    return str(p)


def augment_image_clinically_safe(image: tf.Tensor) -> tf.Tensor:
    """Aplica Data Augmentation seguro para resonancias magnéticas cerebrales.

    Aviso clínico: NO se aplica flip vertical porque alteraría la orientación
    cráneo-caudal (superior/inferior) del encéfalo, lo cual induciría artefactos
    anatómicos irreales. Solo se permite flip horizontal (simetría hemisférica)
    y ligeras variaciones de contraste/brillo.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
    return image


def load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor,
    is_training: bool = False,
    model_type: str = "resnet50",
) -> tuple[tf.Tensor, tf.Tensor]:
    """Carga y procesa un tensor de imagen a partir de su ruta.

    Garantiza 3 canales de color (RGB) independientemente de si la resonancia
    original está almacenada en escala de grises.
    """
    # Leer archivo crudo
    raw_img = tf.io.read_file(path)
    # Decodificar siempre a 3 canales para compatibilidad con la base convolucional
    image = tf.io.decode_image(raw_img, channels=IMG_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)

    # Data Augmentation exclusivo para el conjunto de entrenamiento
    if is_training:
        image = augment_image_clinically_safe(image)

    # Normalización estricta según el backbone
    if model_type == "resnet50":
        image = resnet_preprocess(image)
    elif model_type.startswith("efficientnet"):
        image = efficientnet_preprocess(image)
    else:
        image = tf.cast(image, tf.float32) / 255.0

    # Codificación One-Hot para clasificación multiclase (4 clases)
    one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, one_hot_label


def create_tf_dataset(
    df: "pd.DataFrame",
    split_name: str,
    batch_size: int = BATCH_SIZE,
    model_type: str = "resnet50",
    shuffle_buffer: int = 1024,
) -> tf.data.Dataset:
    """Construye un pipeline tf.data optimizado con prefetch y paralelismo.

    Args:
        df: DataFrame que contiene al menos las columnas 'path', 'label' y 'split'.
        split_name: 'train', 'val' o 'test'.
        batch_size: Tamaño del lote.
        model_type: 'resnet50' o 'efficientnet'.
        shuffle_buffer: Tamaño del búfer de mezcla.

    Returns:
        tf.data.Dataset listo para alimentar el entrenamiento o evaluación.
    """
    split_df = df[df["split"] == split_name]
    if split_df.empty:
        raise ValueError(f"No hay registros en el DataFrame para el split '{split_name}'.")

    paths = [_resolve_image_path(str(p), PROJECT_ROOT) for p in split_df["path"].values]
    labels = split_df["label"].values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    is_training = (split_name == "train")

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(paths), shuffle_buffer), seed=SEED)

    dataset = dataset.map(
        lambda p, l: load_and_preprocess_image(p, l, is_training=is_training, model_type=model_type),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
