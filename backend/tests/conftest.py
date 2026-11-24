"""
Fixtures partagees par les tests.

Un faux moteur d'inference (FakeEngine) remplace le vrai modele : les tests
tournent donc sans modele ni jeu de donnees, par exemple dans GitHub Actions.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module

CLASSES = [
    "Person",
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
    "Mask",
    "NO-Mask",
    "Gloves",
    "Safety Cone",
]


def make_jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 120, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeEngine:
    """Imite InferenceEngine sans charger de modele."""

    is_loaded = True

    def predict(self, source, **kwargs):
        width, height = source.size
        return {
            "detections": [
                {
                    "class_id": 1,
                    "class_name": "Hardhat",
                    "confidence": 0.93,
                    "bbox": {"x1": 5.0, "y1": 5.0, "x2": 30.0, "y2": 30.0},
                }
            ],
            "num_detections": 1,
            "image_width": width,
            "image_height": height,
            "inference_time_ms": 12.5,
            "raw_result": None,
        }

    def get_annotated_image_bytes(self, result):
        return make_jpeg_bytes()

    def get_metadata(self):
        return {
            "name": "safevision-yolo",
            "architecture": "YOLOv8s",
            "version": "1",
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "metrics": {
                "mAP50": 0.626,
                "mAP50_95": 0.360,
                "precision": 0.866,
                "recall": 0.572,
            },
            "input_size": 640,
            "format": "onnx",
        }


@pytest.fixture
def client(monkeypatch):
    """Client de test avec un modele simule, charge au demarrage (lifespan)."""
    monkeypatch.setattr(main_module, "InferenceEngine", FakeEngine)
    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(monkeypatch):
    """Client de test dont le chargement du modele echoue."""

    def fail():
        raise RuntimeError("modele absent")

    monkeypatch.setattr(main_module, "InferenceEngine", fail)
    with TestClient(main_module.app) as test_client:
        yield test_client


@pytest.fixture
def jpeg_bytes():
    return make_jpeg_bytes()
