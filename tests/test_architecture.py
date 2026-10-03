"""
Pruebas automatizadas de la Etapa 3: Arquitectura, Capas Congeladas y Forward Pass.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.architecture import build_model, compile_model
from src.config import IMG_CHANNELS, IMG_SIZE, NUM_CLASSES


def test_resnet50_build_and_forward_pass():
    """Verifica la construcción y forward pass de ResNet50 con pesos aleatorios."""
    model = build_model(
        backbone_name="resnet50",
        dropout=0.3,
        num_classes=NUM_CLASSES,
        base_trainable=False,
        weights=None,
    )
    compile_model(model, learning_rate=1e-3)

    assert model.output_shape == (None, NUM_CLASSES)

    # Simular un lote de 2 imágenes sintéticas
    dummy_input = np.random.randn(2, *IMG_SIZE, IMG_CHANNELS).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)

    assert predictions.shape == (2, NUM_CLASSES)
    # Comprobar que las probabilidades de softmax suman 1.0 (margen numérico 1e-4)
    row_sums = np.sum(predictions, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-4)


def test_efficientnet_build_and_forward_pass():
    """Verifica la construcción y forward pass de EfficientNetB2 para comparación multimodelo."""
    model = build_model(
        backbone_name="efficientnetb2",
        dropout=0.2,
        num_classes=NUM_CLASSES,
        base_trainable=False,
        weights=None,
    )
    compile_model(model, learning_rate=1e-3)

    assert model.output_shape == (None, NUM_CLASSES)

    dummy_input = np.random.randn(1, *IMG_SIZE, IMG_CHANNELS).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    assert predictions.shape == (1, NUM_CLASSES)


def test_frozen_base_layer_trainability():
    """Verifica que la base convolucional quede congelada cuando base_trainable=False."""
    model = build_model(
        backbone_name="resnet50",
        base_trainable=False,
        weights=None,
    )

    base_layer = model.get_layer("resnet50")
    assert base_layer.trainable is False

    # El cabezal denso sí debe ser entrenable
    pred_layer = model.get_layer("prediction")
    assert pred_layer.trainable is True


def test_fine_tuning_layer_unfreezing():
    """Verifica que al descongelar con fine_tune_at solo se activen las capas superiores."""
    model = build_model(
        backbone_name="resnet50",
        base_trainable=True,
        fine_tune_at=140,
        weights=None,
    )

    base_layer = model.get_layer("resnet50")
    assert base_layer.trainable is True

    # Comprobar que las capas previas a 140 están congeladas
    assert base_layer.layers[10].trainable is False
    assert base_layer.layers[139].trainable is False

    # Comprobar que a partir de la capa 140 están descongeladas
    assert base_layer.layers[140].trainable is True
    assert base_layer.layers[-1].trainable is True
