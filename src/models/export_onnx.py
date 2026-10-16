"""
export_onnx.py
===============
Exporte le modele final (safevision_yolov8s_v1.pt) au format ONNX,
standardise pour une inference optimisee via ONNX Runtime.

Usage :
    python src/models/export_onnx.py
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "safevision_yolov8s_v1.pt"


def main():
    print(f"Chargement du modele : {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("Export vers ONNX en cours...")
    onnx_path = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,   # taille d'entree fixe -> plus rapide et plus simple pour l'API
        simplify=True,   # simplifie le graphe ONNX (optimisation)
        opset=12,
    )

    print("=" * 60)
    print("EXPORT ONNX TERMINE")
    print("=" * 60)
    print(f"Fichier genere : {onnx_path}")


if __name__ == "__main__":
    main()
