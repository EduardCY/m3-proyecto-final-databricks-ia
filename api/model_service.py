"""
Servicio singleton para carga y ejecución de inferencia del modelo NeuroScan AI.
"""

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from src.architecture import build_model, compile_model
from src.config import (
    CLASSES,
    FINAL_MODEL_PATH,
    IMG_CHANNELS,
    IMG_SIZE,
    MODEL_CHECKPOINTS_DIR,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


class NeuroScanModelService:
    """Singleton para gestionar la carga en memoria y la inferencia del modelo."""

    _instance: Optional["NeuroScanModelService"] = None

    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.model_version: str = "1.0.0"
        self._load_model()

    @classmethod
    def get_instance(cls) -> "NeuroScanModelService":
        if cls._instance is None:
            cls._instance = NeuroScanModelService()
        return cls._instance

    def _load_model(self) -> None:
        """Carga el modelo final o el mejor checkpoint disponible."""
        # 1. Intentar cargar el modelo final exportado
        if FINAL_MODEL_PATH.exists():
            try:
                self.model = tf.keras.models.load_model(str(FINAL_MODEL_PATH))
                logger.info(f"Modelo cargado exitosamente desde: {FINAL_MODEL_PATH}")
                return
            except Exception as e:
                logger.error(f"Error al cargar modelo desde {FINAL_MODEL_PATH}: {e}")

        # 2. Intentar cargar el mejor checkpoint disponible
        checkpoints = sorted(MODEL_CHECKPOINTS_DIR.glob("*_best.keras"))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            try:
                self.model = tf.keras.models.load_model(str(latest_checkpoint))
                logger.info(f"Modelo cargado desde el mejor checkpoint: {latest_checkpoint.name}")
                return
            except Exception as e:
                logger.error(f"Error al cargar checkpoint {latest_checkpoint}: {e}")

        # 3. Failsafe: Modelo de fallback para permitir inicialización y testing inicial
        logger.warning(
            "No se encontró un modelo entrenado en disco. Inicializando arquitectura "
            "de respaldo para permitir pruebas de endpoints y verificación de API."
        )
        self.model = build_model(backbone_name="resnet50", weights=None)
        compile_model(self.model, learning_rate=1e-3)

    def is_loaded(self) -> bool:
        """Comprueba si el modelo está instanciado y listo para inferir."""
        return self.model is not None

    def predict_image_bytes(self, image_bytes: bytes) -> dict:
        """Procesa una imagen en memoria y devuelve la predicción con probabilidades.

        Args:
            image_bytes: Bytes del archivo de imagen subido.

        Returns:
            Diccionario con clase predicha, confianza y probabilidades por clase.
        """
        # Cargar y validar imagen con Pillow
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # Asegurar 3 canales RGB (las resonancias suelen ser escala de grises)
            pil_img = pil_img.convert("RGB")
            pil_img = pil_img.resize(IMG_SIZE)
            img_array = np.array(pil_img, dtype=np.float32)

        # Preprocesamiento idéntico al entrenamiento (ImageNet mean centering)
        img_preprocessed = resnet_preprocess(img_array)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # Inferencia
        predictions = self.model.predict(img_batch, verbose=0)[0]

        # Mapear probabilidades a cada patología
        probabilities = {
            CLASSES[i]: float(predictions[i])
            for i in range(NUM_CLASSES)
        }

        best_idx = int(np.argmax(predictions))
        best_class = CLASSES[best_idx]
        best_confidence = float(predictions[best_idx])

        return {
            "predicted_class": best_class,
            "confidence": round(best_confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "model_version": self.model_version,
        }
