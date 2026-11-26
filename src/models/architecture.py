"""
architecture.py
================
Definition des deux modeles compares dans le projet et de leurs hyperparametres.

    - baseline : YOLOv8n (le plus petit), 30 epochs, reglages par defaut
    - advanced : YOLOv8s (plus de capacite), 80 epochs, augmentation renforcee
                 (mixup, copy_paste) pour aider les classes rares

Les valeurs ci-dessous sont celles utilisees pour les runs MLflow
"baseline_yolov8n_v2" et "advanced_yolov8s" (voir train_baseline.py et
train_advanced.py). Elles sont rassemblees ici pour que train.py n'ait qu'un seul
endroit ou les lire.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"

MODEL_CONFIGS = {
    "baseline": {
        "weights": "yolov8n.pt",  # pre-entraine sur COCO
        "run_name": "baseline_yolov8n_v2",
        "train_args": {
            "epochs": 30,  # peu d'epochs : baseline volontairement simple
            "imgsz": 640,
            "batch": 8,  # adapte a une carte de 4 Go de VRAM
            "patience": 10,  # arret anticipe si pas d'amelioration
            "seed": 42,
            "workers": 4,
        },
    },
    "advanced": {
        "weights": "yolov8s.pt",  # pre-entraine sur COCO
        "run_name": "advanced_yolov8s",
        "train_args": {
            "epochs": 80,
            "imgsz": 640,
            "batch": 4,  # modele plus lourd -> batch reduit pour tenir en 4 Go
            "patience": 20,
            "seed": 42,
            "workers": 4,
            # Augmentation renforcee (aide les classes rares : Gloves, Safety Cone)
            "mosaic": 1.0,
            "mixup": 0.15,  # melange deux images
            "copy_paste": 0.15,  # colle des objets d'une image sur une autre
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 5.0,  # legere rotation
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
        },
    },
}


def get_config(name: str) -> dict:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Modele inconnu '{name}'. Choix : {list(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[name]


def get_train_args(name: str, run_name: str | None = None) -> dict:
    """Arguments complets passes a model.train() (donnees, dossier, nom du run...)."""
    config = get_config(name)
    return {
        "data": str(DATA_YAML),
        "project": "runs_safevision",
        "name": run_name or config["run_name"],
        "exist_ok": True,
        "verbose": True,
        **config["train_args"],
    }


def build_model(name: str):
    """Cree le modele YOLO de depart (poids pre-entraines COCO) pour 'baseline' ou 'advanced'."""
    from ultralytics import YOLO  # import tardif : ultralytics est lourd

    return YOLO(get_config(name)["weights"])
