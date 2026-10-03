"""
Pruebas automatizadas de la Etapa 2: Pipeline de Preprocesamiento y Verificación de Tensores.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import tensorflow as tf

from src.config import CLASSES, IMG_CHANNELS, IMG_SIZE, NUM_CLASSES
from src.preprocessing import create_tf_dataset, load_and_preprocess_image


@pytest.fixture
def sample_image_and_df(tmp_path: Path):
    """Crea una imagen temporal y un DataFrame mínimo para probar el pipeline."""
    img_path = tmp_path / "sample_scan.jpg"
    arr = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    df = pd.DataFrame(
        [
            {"path": str(img_path), "label": 0, "label_name": CLASSES[0], "split": "train"},
            {"path": str(img_path), "label": 1, "label_name": CLASSES[1], "split": "train"},
            {"path": str(img_path), "label": 2, "label_name": CLASSES[2], "split": "val"},
            {"path": str(img_path), "label": 3, "label_name": CLASSES[3], "split": "test"},
        ]
    )
    return img_path, df


def test_load_and_preprocess_image_shape(sample_image_and_df):
    """Verifica que la salida de preprocesamiento tenga dimensiones (224, 224, 3) y (4,)."""
    img_path, _ = sample_image_and_df
    image_tensor, label_tensor = load_and_preprocess_image(
        path=tf.constant(str(img_path)),
        label=tf.constant(2),
        is_training=False,
        model_type="resnet50",
    )

    assert image_tensor.shape == (*IMG_SIZE, IMG_CHANNELS)
    assert label_tensor.shape == (NUM_CLASSES,)
    # Comprobar que es un vector one-hot válido
    assert tf.reduce_sum(label_tensor).numpy() == 1.0
    assert label_tensor.numpy()[2] == 1.0


def test_create_tf_dataset_batch_integrity(sample_image_and_df):
    """Verifica que el dataset tf.data por lotes no contenga NaNs ni infinitos."""
    _, df = sample_image_and_df
    dataset = create_tf_dataset(df, split_name="train", batch_size=2, model_type="resnet50")

    for images, labels in dataset.take(1):
        assert images.shape == (2, *IMG_SIZE, IMG_CHANNELS)
        assert labels.shape == (2, NUM_CLASSES)
        assert not np.isnan(images.numpy()).any(), "Se detectaron valores NaN en los tensores de imagen"
        assert not np.isinf(images.numpy()).any(), "Se detectaron valores Inf en los tensores de imagen"
