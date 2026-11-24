"""
Tests de l'API SafeVision AI avec un modele simule (voir conftest.py).
Ils verifient les endpoints, les formats de reponse et les codes d'erreur,
sans dependre du modele ni des donnees : ils tournent en integration continue.
"""

import io

from PIL import Image


def post_image(client, path, content, name="test.jpg", mime="image/jpeg"):
    return client.post(path, files={"file": (name, io.BytesIO(content), mime)})


# ------------------------------ GET / ------------------------------


def test_root_returns_200(client):
    assert client.get("/").status_code == 200


# ---------------------------- GET /health ----------------------------


def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_response_structure(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert isinstance(data["gpu_available"], bool)
    assert data["model_version"] == "1"


# ------------------------- GET /model/metadata -------------------------


def test_model_metadata_structure(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "safevision-yolo"
    assert data["num_classes"] == 9
    assert len(data["classes"]) == 9
    assert "Hardhat" in data["classes"]
    assert "mAP50" in data["metrics"]


# ---------------------------- POST /predict ----------------------------


def test_predict_returns_valid_structure(client, jpeg_bytes):
    response = post_image(client, "/predict", jpeg_bytes)
    assert response.status_code == 200
    data = response.json()
    assert data["num_detections"] == len(data["detections"])
    assert data["image_width"] == 64
    assert data["image_height"] == 48
    assert data["inference_time_ms"] >= 0


def test_predict_detection_fields(client, jpeg_bytes):
    detection = post_image(client, "/predict", jpeg_bytes).json()["detections"][0]
    assert detection["class_name"] == "Hardhat"
    assert 0.0 <= detection["confidence"] <= 1.0
    assert set(detection["bbox"]) == {"x1", "y1", "x2", "y2"}


def test_predict_rejects_non_image_with_400(client):
    response = post_image(client, "/predict", b"ceci n'est pas une image", "x.txt")
    assert response.status_code == 400


def test_predict_rejects_empty_file_with_400(client):
    assert post_image(client, "/predict", b"").status_code == 400


# ------------------------ POST /predict/visualize ------------------------


def test_visualize_returns_valid_jpeg(client, jpeg_bytes):
    response = post_image(client, "/predict/visualize", jpeg_bytes)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(response.content)).format == "JPEG"


def test_visualize_rejects_non_image_with_400(client):
    response = post_image(client, "/predict/visualize", b"pas une image", "x.txt")
    assert response.status_code == 400


# ------------------------- Modele indisponible -------------------------


def test_health_is_degraded_without_model(client_without_model):
    data = client_without_model.get("/health").json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


def test_predict_returns_503_without_model(client_without_model, jpeg_bytes):
    assert post_image(client_without_model, "/predict", jpeg_bytes).status_code == 503


def test_visualize_returns_503_without_model(client_without_model, jpeg_bytes):
    response = post_image(client_without_model, "/predict/visualize", jpeg_bytes)
    assert response.status_code == 503


def test_metadata_returns_503_without_model(client_without_model):
    assert client_without_model.get("/model/metadata").status_code == 503
