"""
test_api.py
============
Tests unitaires Pytest pour l'API SafeVision AI.

Usage :
    cd backend
    pytest tests/test_api.py -v
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app



@pytest.fixture(scope="module")
def client():
    """
    Demarre l'application AVEC ses evenements de demarrage (lifespan), ce qui
    charge le modele une seule fois pour tous les tests du module.
    Sans le bloc 'with', FastAPI ne lance pas le lifespan et le modele reste
    non charge (d'ou les erreurs 503).
    """
    with TestClient(app) as c:
        yield c


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "images"


def _get_sample_image_path() -> Path:
    """Retourne le chemin de la premiere image de test disponible."""
    images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    assert images, f"Aucune image de test trouvee dans {TEST_IMAGES_DIR}"
    return images[0]


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_structure(client):
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "gpu_available" in data
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["gpu_available"], bool)


def test_health_model_is_loaded(client):
    """Verifie que le modele est bien charge (precondition pour les autres tests)."""
    response = client.get("/health")
    data = response.json()
    assert data["model_loaded"] is True, "Le modele doit etre charge pour que les tests suivants aient un sens."


# ------------------------------------------------------------------
# GET /model/metadata
# ------------------------------------------------------------------

def test_model_metadata_returns_200(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200


def test_model_metadata_structure(client):
    response = client.get("/model/metadata")
    data = response.json()
    assert data["name"] == "safevision-yolo"
    assert data["num_classes"] == 9
    assert len(data["classes"]) == 9
    assert "Person" in data["classes"]
    assert "Hardhat" in data["classes"]
    assert "mAP50" in data["metrics"]


# ------------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------------

def test_predict_with_valid_image_returns_200(client):
    image_path = _get_sample_image_path()
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    assert response.status_code == 200


def test_predict_response_structure(client):
    image_path = _get_sample_image_path()
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    data = response.json()
    assert "detections" in data
    assert "num_detections" in data
    assert "image_width" in data
    assert "image_height" in data
    assert "inference_time_ms" in data
    assert isinstance(data["detections"], list)
    assert data["num_detections"] == len(data["detections"])


def test_predict_detection_fields(client):
    """Si au moins une detection est trouvee, verifie la structure de chaque champ."""
    image_path = _get_sample_image_path()
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    data = response.json()
    if data["num_detections"] > 0:
        detection = data["detections"][0]
        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert 0.0 <= detection["confidence"] <= 1.0
        assert "bbox" in detection
        assert set(detection["bbox"].keys()) == {"x1", "y1", "x2", "y2"}


def test_predict_with_invalid_file_returns_400(client):
    """Un fichier texte (pas une image) doit etre rejete avec un code 400."""
    fake_file = io.BytesIO(b"ceci n'est pas une image")
    response = client.post(
        "/predict",
        files={"file": ("not_an_image.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400


def test_predict_with_empty_file_returns_400(client):
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", empty_file, "image/jpeg")},
    )
    assert response.status_code == 400


# ------------------------------------------------------------------
# POST /predict/visualize
# ------------------------------------------------------------------

def test_predict_visualize_returns_200(client):
    image_path = _get_sample_image_path()
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict/visualize",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    assert response.status_code == 200


def test_predict_visualize_returns_valid_jpeg(client):
    image_path = _get_sample_image_path()
    with open(image_path, "rb") as f:
        response = client.post(
            "/predict/visualize",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    assert response.headers["content-type"] == "image/jpeg"
    # Verifie que les octets retournes forment bien une image JPEG valide
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "JPEG"


def test_predict_visualize_with_invalid_file_returns_400(client):
    fake_file = io.BytesIO(b"pas une image")
    response = client.post(
        "/predict/visualize",
        files={"file": ("fake.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400


# ------------------------------------------------------------------
# Racine
# ------------------------------------------------------------------

def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
