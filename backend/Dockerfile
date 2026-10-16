# syntax=docker/dockerfile:1

# ---------- Etape 1 : construction des dependances ----------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# PyTorch en version CPU : beaucoup plus legere que la version CUDA
RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Etape 2 : image finale (sans outils de build) ----------
FROM python:3.11-slim-bookworm AS runtime

# Bibliotheques systeme necessaires a OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/safevision_yolov8s_v1.onnx

# Utilisateur non-root (bonne pratique de securite)
RUN useradd --create-home appuser

WORKDIR /app
COPY app ./app
USER appuser

EXPOSE 8000

# Le modele ONNX est monte au lancement (-v ...:/models), il n'est pas dans l'image
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
