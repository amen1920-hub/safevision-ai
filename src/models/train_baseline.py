"""
train_baseline.py
==================
Entraine un modele BASELINE (YOLOv8n, peu d'epochs, config par defaut)
sur le dataset SafeVision AI. Sert de point de reference pour comparer
avec le modele avance (fine-tuning plus pousse) entraine plus tard.

Le logging (parametres, metriques, artefacts) est fait automatiquement
dans MLflow par l'integration native d'Ultralytics.

Usage :
    python src/models/train_baseline.py
"""

import os
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------
# 1. Configuration MLflow
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Le serveur MLflow doit deja tourner (mlflow ui) dans un autre terminal
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "safevision-ai"

# ------------------------------------------------------------------
# 2. Configuration + lancement de l'entrainement BASELINE
#    (protege par if __name__ == "__main__" -> obligatoire sous Windows
#    a cause du multiprocessing utilise par le DataLoader de PyTorch)
# ------------------------------------------------------------------

def main():
    DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"

    # Modele de depart : YOLOv8n pre-entraine sur COCO (le plus petit/rapide)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=30,              # peu d'epochs : c'est volontairement une baseline simple
        imgsz=640,
        batch=8,                # adapte a une carte 4 Go de VRAM (GTX 1650 Ti)
        patience=10,            # arret anticipe si pas d'amelioration pendant 10 epochs
        project="runs_safevision",
        name="baseline_yolov8n_v2",
        exist_ok=True,
        seed=42,
        workers=4,               # corrige : le vrai fix etait le if __name__, pas workers=0
        verbose=True,
    )

    print("=" * 60)
    print("ENTRAINEMENT BASELINE TERMINE")
    print("=" * 60)
    print(f"Resultats sauvegardes dans : runs_safevision/baseline_yolov8n/")
    print(f"Consultez MLflow sur http://127.0.0.1:5000 pour voir les metriques")


if __name__ == "__main__":
    main()
