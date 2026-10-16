"""
main.py
========
API FastAPI pour SafeVision AI - detection d'EPI sur chantier.

Endpoints :
    GET  /health            - etat du serveur, GPU, modele
    GET  /model/metadata    - caracteristiques du modele
    POST /predict           - detection -> JSON structure
    POST /predict/visualize - detection -> image annotee (JPEG)

Documentation interactive automatique : /docs (Swagger UI)
"""

import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from app.inference import InferenceEngine
from app.schemas import (
    HealthResponse,
    ModelMetadataResponse,
    PredictResponse,
)

# ------------------------------------------------------------------
# Chargement du modele au demarrage (une seule fois, pas a chaque requete)
# ------------------------------------------------------------------

engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        engine = InferenceEngine()
    except Exception as e:
        print(f"ATTENTION : echec du chargement du modele au demarrage : {e}")
        engine = None
    yield
    # (rien a nettoyer a l'arret)


app = FastAPI(
    title="SafeVision AI API",
    description="API de detection d'equipements de protection individuelle (EPI) sur chantier.",
    version="1.0.0",
    lifespan=lifespan,
)

# Autorise le frontend (React, autre origine) a appeler cette API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # a restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------------

def _validate_and_open_image(file_bytes: bytes) -> Image.Image:
    """Verifie que les octets recus forment bien une image valide."""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # verifie l'integrite sans charger entierement
        # verify() invalide l'objet pour un usage ulterieur -> on le rouvre
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return image
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image valide.")


def _ensure_model_loaded():
    if engine is None or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Le modele n'est pas disponible pour le moment.")


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    model_loaded = engine is not None and engine.is_loaded
    gpu_available = torch.cuda.is_available()
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        gpu_available=gpu_available,
        model_version=engine.get_metadata()["version"] if model_loaded else None,
    )


# ------------------------------------------------------------------
# GET /model/metadata
# ------------------------------------------------------------------

@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Monitoring"])
async def model_metadata():
    _ensure_model_loaded()
    return ModelMetadataResponse(**engine.get_metadata())


# ------------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse, tags=["Detection"])
async def predict(file: UploadFile = File(...)):
    _ensure_model_loaded()

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    image = _validate_and_open_image(file_bytes)

    try:
        result = engine.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'inference : {e}")

    return PredictResponse(
        detections=result["detections"],
        num_detections=result["num_detections"],
        image_width=result["image_width"],
        image_height=result["image_height"],
        inference_time_ms=result["inference_time_ms"],
    )


# ------------------------------------------------------------------
# POST /predict/visualize
# ------------------------------------------------------------------

@app.post("/predict/visualize", tags=["Detection"])
async def predict_visualize(file: UploadFile = File(...)):
    _ensure_model_loaded()

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    image = _validate_and_open_image(file_bytes)

    try:
        result = engine.predict(image)
        annotated_bytes = engine.get_annotated_image_bytes(result["raw_result"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'inference : {e}")

    return Response(content=annotated_bytes, media_type="image/jpeg")


# ------------------------------------------------------------------
# Racine (pratique pour verifier rapidement que l'API tourne)
# ------------------------------------------------------------------

@app.get("/", tags=["Monitoring"])
async def root():
    return {"message": "SafeVision AI API - voir /docs pour la documentation interactive"}
