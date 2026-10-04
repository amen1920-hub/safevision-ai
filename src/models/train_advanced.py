"""
train_advanced.py
==================
Entraine le modele AVANCE (YOLOv8s, plus d'epochs, augmentation renforcee)
sur le dataset SafeVision AI, a comparer avec le baseline (YOLOv8n).

Ameliorations par rapport au baseline :
- Modele plus grand (YOLOv8s au lieu de YOLOv8n) -> plus de capacite
- Plus d'epochs (80 au lieu de 30) -> plus de temps pour apprendre
- Augmentation renforcee (mixup, copy_paste) -> aide les classes rares
  (Gloves, Safety Cone) qui avaient tres peu d'instances dans le baseline
- workers=4 au lieu de 0 -> entrainement beaucoup plus rapide

Le logging (parametres, metriques, artefacts) est fait automatiquement
dans MLflow par l'integration native d'Ultralytics.

Usage :
    python src/models/train_advanced.py
"""

import os
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------
# 1. Configuration MLflow
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "safevision-ai"

# ------------------------------------------------------------------
# 2. Configuration + lancement de l'entrainement AVANCE
#    (protege par if __name__ == "__main__" -> obligatoire sous Windows)
# ------------------------------------------------------------------

def main():
    DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"

    # Modele de depart : YOLOv8s pre-entraine sur COCO (plus grand que le nano)
    model = YOLO("yolov8s.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=80,
        imgsz=640,
        batch=4,                 # YOLOv8s est plus lourd -> batch reduit pour tenir dans 4 Go VRAM
        patience=20,              # un peu plus de patience vu le plus grand nombre d'epochs
        project="runs_safevision",
        name="advanced_yolov8s",
        exist_ok=True,
        seed=42,
        workers=4,                # OK maintenant que le script est protege par if __name__
        # --- Augmentation renforcee (aide les classes rares) ---
        mosaic=1.0,                # deja actif par defaut, on le rend explicite
        mixup=0.15,                # melange deux images -> plus de diversite
        copy_paste=0.15,           # colle des objets d'une image sur une autre -> plus d'exemples
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,               # legere rotation aleatoire
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        verbose=True,
    )

    print("=" * 60)
    print("ENTRAINEMENT AVANCE TERMINE")
    print("=" * 60)
    print(f"Resultats sauvegardes dans : runs_safevision/advanced_yolov8s/")
    print(f"Consultez MLflow sur http://127.0.0.1:5000 pour comparer avec le baseline")


if __name__ == "__main__":
    main()
