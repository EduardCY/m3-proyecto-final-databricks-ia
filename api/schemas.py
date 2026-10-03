"""
Contratos Pydantic de entrada y salida para el servicio REST de NeuroScan AI.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Esquema de respuesta para el endpoint de monitoreo /health."""
    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_name: str = Field(..., json_schema_extra={"example": "neuroscan_resnet50"})
    classes: List[str] = Field(..., json_schema_extra={"example": ["glioma", "meningioma", "notumor", "pituitary"]})


class PredictionResponse(BaseModel):
    """Esquema de respuesta para el endpoint de inferencia /predict."""
    predicted_class: str = Field(..., json_schema_extra={"example": "meningioma"})
    confidence: float = Field(..., json_schema_extra={"example": 0.964})
    probabilities: Dict[str, float] = Field(..., json_schema_extra={"example": {
        "glioma": 0.015,
        "meningioma": 0.964,
        "notumor": 0.003,
        "pituitary": 0.018,
    }})
    model_version: str = Field(default="1.0.0", json_schema_extra={"example": "1.0.0"})
    disclaimer: str = Field(...)


class ExplainedPredictionResponse(PredictionResponse):
    """Esquema extendido con reporte clínico asistido por Google AI Pro (Gemini)."""
    clinical_rationale: str = Field(..., json_schema_extra={"example": "La masa extraaxial presenta realce homogéneo..."})
    suggested_mri_protocols: List[str] = Field(..., json_schema_extra={"example": ["T1 con contraste Gadolinio", "T2 / FLAIR axial"]})
    ai_assistant_provider: str = Field(default="Google Gemini / AI Pro", json_schema_extra={"example": "Google Gemini / AI Pro"})
