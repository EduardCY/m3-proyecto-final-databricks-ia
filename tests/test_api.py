"""
Pruebas automatizadas de la Etapa 4: Servicio REST API (FastAPI) y Failsafes de Inferencia.
"""

import io
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.config import CLASSES


@pytest.fixture
def client():
    """Cliente de pruebas para FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def mock_image_bytes():
    """Genera bytes de una imagen JPEG sintética."""
    arr = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_api_health_endpoint(client):
    """Verifica que el endpoint /health devuelva 200 OK y el modelo esté instanciado."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert set(data["classes"]) == set(CLASSES)


def test_api_predict_valid_image(client, mock_image_bytes):
    """Verifica que /predict procese una imagen válida y retorne probabilidades completas."""
    files = {"file": ("test_mri.jpg", mock_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert set(data["probabilities"].keys()) == set(CLASSES)
    assert len(data["disclaimer"]) > 20


def test_api_predict_invalid_mime_type(client):
    """Verifica el rechazo seguro con HTTP 400 para archivos que no son imágenes."""
    files = {"file": ("document.txt", b"Texto plano simulado", "text/plain")}
    response = client.post("/predict", files=files)

    assert response.status_code == 400
    assert "Formato no permitido" in response.json()["detail"]


def test_api_predict_explained_with_fallback(client, mock_image_bytes):
    """Verifica que /predict_explained responda con reporte clínico (usando fallback seguro)."""
    files = {"file": ("test_mri.jpg", mock_image_bytes, "image/jpeg")}
    response = client.post("/predict_explained", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in CLASSES
    assert len(data["clinical_rationale"]) > 10
    assert len(data["suggested_mri_protocols"]) >= 1
    assert "ai_assistant_provider" in data
