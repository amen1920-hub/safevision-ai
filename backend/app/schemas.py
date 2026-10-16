"""
schemas.py
==========
Schemas Pydantic pour la validation stricte des payloads entrants et
sortants de l'API FastAPI, conformement au cahier des charges.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., description="Etat global du serveur : 'ok' ou 'degraded'")
    model_loaded: bool = Field(..., description="Le modele est-il charge en memoire ?")
    gpu_available: bool = Field(..., description="Un GPU est-il disponible sur la machine ?")
    model_version: Optional[str] = Field(None, description="Version du modele actuellement charge")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_loaded": True,
                "gpu_available": False,
                "model_version": "safevision-yolo-v1",
            }
        }


# ------------------------------------------------------------------
# /model/metadata
# ------------------------------------------------------------------

class ModelMetadataResponse(BaseModel):
    name: str = Field(..., description="Nom du modele")
    architecture: str = Field(..., description="Architecture du modele (ex: YOLOv8s)")
    version: str = Field(..., description="Version du modele")
    classes: List[str] = Field(..., description="Liste des classes detectables")
    num_classes: int = Field(..., description="Nombre de classes")
    metrics: dict = Field(..., description="Metriques de validation du modele")
    input_size: int = Field(..., description="Taille d'entree attendue (pixels, carre)")
    format: str = Field(..., description="Format du fichier modele (ex: onnx)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "safevision-yolo",
                "architecture": "YOLOv8s",
                "version": "1",
                "classes": ["Person", "Hardhat", "NO-Hardhat", "Safety Vest",
                             "NO-Safety Vest", "Mask", "NO-Mask", "Gloves", "Safety Cone"],
                "num_classes": 9,
                "metrics": {"mAP50": 0.626, "mAP50_95": 0.360, "precision": 0.866, "recall": 0.572},
                "input_size": 640,
                "format": "onnx",
            }
        }


# ------------------------------------------------------------------
# /predict
# ------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Coordonnee X du coin superieur gauche (pixels)")
    y1: float = Field(..., description="Coordonnee Y du coin superieur gauche (pixels)")
    x2: float = Field(..., description="Coordonnee X du coin inferieur droit (pixels)")
    y2: float = Field(..., description="Coordonnee Y du coin inferieur droit (pixels)")


class Detection(BaseModel):
    class_id: int = Field(..., description="Identifiant numerique de la classe")
    class_name: str = Field(..., description="Nom de la classe detectee")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score de confiance (0 a 1)")
    bbox: BoundingBox = Field(..., description="Boite englobante de l'objet detecte")


class PredictResponse(BaseModel):
    detections: List[Detection] = Field(..., description="Liste des objets detectes")
    num_detections: int = Field(..., description="Nombre total d'objets detectes")
    image_width: int = Field(..., description="Largeur de l'image traitee (pixels)")
    image_height: int = Field(..., description="Hauteur de l'image traitee (pixels)")
    inference_time_ms: float = Field(..., description="Temps d'inference cote serveur (ms)")

    class Config:
        json_schema_extra = {
            "example": {
                "detections": [
                    {
                        "class_id": 1,
                        "class_name": "Hardhat",
                        "confidence": 0.94,
                        "bbox": {"x1": 120.5, "y1": 45.0, "x2": 210.3, "y2": 140.7},
                    }
                ],
                "num_detections": 1,
                "image_width": 640,
                "image_height": 480,
                "inference_time_ms": 185.3,
            }
        }


# ------------------------------------------------------------------
# Erreurs
# ------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Description de l'erreur")
