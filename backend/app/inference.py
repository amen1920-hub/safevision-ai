"""
inference.py
=============
Classe wrapper qui charge le modele ONNX (via le wrapper Ultralytics, qui
sait aussi bien charger un .pt qu'un .onnx) et expose une methode simple
pour detecter les objets sur une image, prete a etre utilisee par l'API.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Le chemin du modele peut etre surcharge par la variable d'environnement
# MODEL_PATH (utilise dans Docker, ou le modele est monte dans /models).
MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "safevision_yolov8s_v1.onnx"),
))

CLASS_NAMES = [
    "Person", "Hardhat", "NO-Hardhat", "Safety Vest",
    "NO-Safety Vest", "Mask", "NO-Mask", "Gloves", "Safety Cone",
]

MODEL_METRICS = {
    "mAP50": 0.626,
    "mAP50_95": 0.360,
    "precision": 0.866,
    "recall": 0.572,
}

DEVICE = "cpu"  # onnxruntime-gpu non disponible sur cette machine (voir rapport)
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45


class InferenceEngine:
    """
    Wrapper autour du modele ONNX SafeVision AI.
    Charge le modele une seule fois (au demarrage de l'API) et reutilise
    cette instance pour chaque requete, au lieu de recharger a chaque fois.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self._load()

    def _load(self):
        try:
            self.model = YOLO(str(self.model_path))
            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError(f"Impossible de charger le modele depuis {self.model_path} : {e}")

    def predict(
        self,
        source: Any,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Execute une detection sur l'image donnee. 'source' peut etre un
        chemin de fichier (str), une image PIL, ou un tableau numpy.
        Retourne un dict pret a etre serialise dans PredictResponse.
        """
        if not self.is_loaded:
            raise RuntimeError("Le modele n'est pas charge.")

        start = time.perf_counter()
        results = self.model.predict(
            source,
            device=DEVICE,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        inference_time_ms = (time.perf_counter() - start) * 1000

        result = results[0]  # une seule image en entree -> un seul resultat
        height, width = result.orig_shape

        detections: List[Dict[str, Any]] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                "confidence": confidence,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_width": width,
            "image_height": height,
            "inference_time_ms": round(inference_time_ms, 2),
            "raw_result": result,  # garde l'objet brut pour /predict/visualize
        }

    def get_annotated_image_bytes(self, result) -> bytes:
        """
        Dessine les bounding boxes + labels sur l'image a partir du resultat
        brut d'ultralytics, et retourne les octets JPEG prets a renvoyer
        dans la reponse HTTP.
        """
        annotated_bgr = result.plot()  # numpy array BGR avec les boxes dessinees
        success, buffer = cv2.imencode(".jpg", annotated_bgr)
        if not success:
            raise RuntimeError("Echec de l'encodage JPEG de l'image annotee.")
        return buffer.tobytes()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "safevision-yolo",
            "architecture": "YOLOv8s",
            "version": "1",
            "classes": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
            "metrics": MODEL_METRICS,
            "input_size": 640,
            "format": "onnx",
        }
