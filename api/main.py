"""
Servidor FastAPI para el sistema de inferencia clínica NeuroScan AI.

Expone endpoints para monitoreo (/health), clasificación de imágenes (/predict)
y generación de reportes explicativos asistidos por Google AI Pro (/predict_explained).
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from api.model_service import NeuroScanModelService
from api.schemas import (
    ExplainedPredictionResponse,
    HealthResponse,
    PredictionResponse,
)
from src.config import CLASSES, MEDICAL_DISCLAIMER

logger = logging.getLogger("neuroscan_api")
logging.basicConfig(level=logging.INFO)

# Formatos de imagen permitidos
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB límite de seguridad (OWASP)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa el modelo de deep learning una sola vez al arrancar el servidor."""
    logger.info("Inicializando servicio de inferencia NeuroScan AI...")
    NeuroScanModelService.get_instance()
    logger.info("Servicio de inferencia listo para recibir peticiones.")
    yield


app = FastAPI(
    title="NeuroScan AI - Brain Tumor MRI Classification API",
    description=(
        "API REST para clasificación automatizada de resonancias magnéticas cerebrales "
        "(Glioma, Meningioma, Pituitary, Sin Tumor) y reporte clínico asistido por IA."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar el estado del servicio y del modelo",
)
async def health_check():
    """Endpoint de monitoreo y health check."""
    service = NeuroScanModelService.get_instance()
    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded(),
        model_name="neuroscan_resnet50",
        classes=CLASSES,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Clasificar una imagen de resonancia magnética",
)
async def predict_mri(file: UploadFile = File(...)):
    """Recibe un archivo de imagen (JPEG/PNG) y devuelve la predicción con probabilidades."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no permitido: '{file.content_type}'. Solo se admiten imágenes JPEG o PNG.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"El archivo supera el tamaño máximo permitido de 10 MB.",
        )

    service = NeuroScanModelService.get_instance()
    try:
        result = service.predict_image_bytes(file_bytes)
        return PredictionResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version=result["model_version"],
            disclaimer=MEDICAL_DISCLAIMER,
        )
    except Exception as e:
        logger.error(f"Error procesando la imagen: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No se pudo decodificar el archivo como una imagen médica válida.",
        )


@app.post(
    "/predict_explained",
    response_model=ExplainedPredictionResponse,
    summary="Clasificar resonancia y generar reporte asistido con Google AI Pro",
)
async def predict_with_explanation(file: UploadFile = File(...)):
    """Clasifica la resonancia y genera una justificación radiológica con Google Gemini."""
    base_prediction = await predict_mri(file)
    predicted_class = base_prediction.predicted_class
    confidence = base_prediction.confidence

    # Failsafe: Guía radiológica offline si la API de Gemini no está configurada o falla
    clinical_rationale = ""
    suggested_protocols = []
    provider = "Offline Clinical Fallback"

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            from google import genai

            client = genai.Client(api_key=gemini_api_key)
            prompt = (
                f"Actúa como un radiólogo consultor de neuroimagen. Se ha analizado una resonancia "
                f"magnética cerebral y el modelo de visión convolucional predijo '{predicted_class}' "
                f"con una probabilidad de {confidence * 100:.1f}%. "
                f"Genera un resumen clínico preliminar breve (máximo 3 oraciones) describiendo las características "
                f"típicas de esta entidad y menciona 2 secuencias recomendadas de confirmación."
            )
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
            )
            clinical_rationale = response.text.strip()
            provider = "Google Gemini 1.5 Flash (Google AI Pro)"
        except Exception as e:
            logger.warning(f"[Failsafe AI] Falló llamada a Gemini API, usando plantilla local: {e}")

    # Fallback determinístico clínico si Gemini no está activo
    if not clinical_rationale:
        fallbacks = {
            "glioma": (
                "Hallazgo compatible con lesión infiltrativa intraaxial en parénquima cerebral. "
                "Suele presentar heterogeneidad con áreas de edema vasogénico perilesional.",
                ["Secuencia T1 post-gadolinio", "Secuencia FLAIR y Espectroscopía por RM"],
            ),
            "meningioma": (
                "Masa extraaxial circunscrita con probable base de implantación dural. "
                "Típicamente muestra realce intenso y homogéneo con posible cola dural.",
                ["Secuencia T1 axial y coronal con contraste", "Secuencia T2/SWI para calcificaciones"],
            ),
            "pituitary": (
                "Lesión localizada en la región selar/paraselar con posible efecto de masa "
                "sobre el quiasma óptico. Requiere evaluación endocrinológica completa.",
                ["RM de Silla Turca con cortes finos T1 dinámico", "Evaluación de campos visuales"],
            ),
            "notumor": (
                "Estructuras anatómicas encefálicas y sistema ventricular sin evidencias de masa, "
                "efecto de volumen ni desvío de la línea media.",
                ["Control de rutina según criterio clínico", "Secuencia T2 de screening"],
            ),
        }
        clinical_rationale, suggested_protocols = fallbacks.get(
            predicted_class,
            ("Estudio procesado. Requiere correlación clínica inmediata con el especialista.", ["Revisión por especialista"])
        )

    return ExplainedPredictionResponse(
        predicted_class=base_prediction.predicted_class,
        confidence=base_prediction.confidence,
        probabilities=base_prediction.probabilities,
        model_version=base_prediction.model_version,
        disclaimer=base_prediction.disclaimer,
        clinical_rationale=clinical_rationale,
        suggested_mri_protocols=suggested_protocols if suggested_protocols else ["T1 con contraste", "T2 FLAIR"],
        ai_assistant_provider=provider,
    )
