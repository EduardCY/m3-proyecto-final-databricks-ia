"""
Arquitecturas de Redes Convolucionales Profundas con Transfer Learning.

Soporta ResNet50 y EfficientNetB2 preentrenadas en ImageNet, con cabezal clasificador
multiclase (Softmax), descongelamiento selectivo por capas y soporte para Mixed Precision.
"""

import random
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB2, ResNet50

from src.config import IMG_CHANNELS, IMG_SIZE, NUM_CLASSES, SEED


def set_seeds(seed: int = SEED) -> None:
    """Fija las semillas de aleatoriedad para garantizar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def enable_mixed_precision_if_available() -> bool:
    """Habilita la política mixed_float16 si hay GPU disponible (ej. en Colab Pro).

    Acelera el entrenamiento hasta 2x en GPUs NVIDIA con Tensor Cores (A100, V100, T4).
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        return True
    return False


def build_model(
    backbone_name: str = "resnet50",
    dropout: float = 0.3,
    num_classes: int = NUM_CLASSES,
    base_trainable: bool = False,
    fine_tune_at: Optional[int] = None,
    weights: str = "imagenet",
) -> tf.keras.Model:
    """Construye un modelo de Transfer Learning con cabezal clasificador multiclase.

    Args:
        backbone_name: 'resnet50' o 'efficientnetb2'.
        dropout: Tasa de regularización Dropout antes de la capa final.
        num_classes: 4 para Brain Tumor MRI.
        base_trainable: Si es False, toda la base convolucional queda congelada.
        fine_tune_at: Índice de capa a partir del cual se descongela la base convolucional.
        weights: 'imagenet' para transfer learning; None para pruebas sin pesos preentrenados.

    Returns:
        tf.keras.Model compilable.
    """
    input_shape = (*IMG_SIZE, IMG_CHANNELS)

    if backbone_name == "resnet50":
        base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    elif backbone_name == "efficientnetb2":
        base_model = EfficientNetB2(weights=weights, include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Backbone no soportado: '{backbone_name}'. Usa 'resnet50' o 'efficientnetb2'.")

    # Control de descongelamiento de capas
    if not base_trainable:
        base_model.trainable = False
    else:
        base_model.trainable = True
        if fine_tune_at is not None:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_tensor")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout, name="dropout")(x)

    # Nota: dtype='float32' explícito en la salida para estabilidad con Mixed Precision
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="prediction",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"neuroscan_{backbone_name}")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer_name: str = "adam",
    use_cosine_decay: bool = False,
    decay_steps: int = 1000,
) -> None:
    """Compila el modelo con optimizador adaptable, métricas multiclase y Crossentropy."""
    if use_cosine_decay:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=0.01,
        )
        opt_lr = lr_schedule
    else:
        opt_lr = learning_rate

    if optimizer_name.lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(learning_rate=opt_lr, weight_decay=1e-4)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=opt_lr)

    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
