"""
Tests d'integration avec le VRAI modele ONNX.
Ils sont ignores automatiquement si le modele est absent (cas de la CI) :
il faut d'abord faire 'dvc pull'.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.inference import MODEL_PATH

pytestmark = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="Modele ONNX absent (lancer 'dvc pull')",
)

SAMPLES_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "processed" / "test" / "images"
)


@pytest.fixture(scope="module")
def real_client():
    with TestClient(main_module.app) as test_client:
        yield test_client


def _sample_image_bytes() -> bytes:
    images = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
    if images:
        return images[0].read_bytes()
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), (128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_real_model_is_loaded(real_client):
    assert real_client.get("/health").json()["model_loaded"] is True


def test_real_model_metadata(real_client):
    data = real_client.get("/model/metadata").json()
    assert data["num_classes"] == 9


def test_real_model_predict(real_client):
    files = {"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    response = real_client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["num_detections"] == len(data["detections"])
    for detection in data["detections"]:
        assert 0.0 <= detection["confidence"] <= 1.0


def test_real_model_visualize(real_client):
    files = {"file": ("sample.jpg", _sample_image_bytes(), "image/jpeg")}
    response = real_client.post("/predict/visualize", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
