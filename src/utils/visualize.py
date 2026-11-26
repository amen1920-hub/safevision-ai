"""
visualize.py
=============
Compare visuellement deux modeles sur des images du jeu de test : pour chaque
image, le modele de reference (YOLOv8n) a gauche et le modele avance (YOLOv8s)
a droite, avec les boites et les scores de confiance.

Les images produites sont enregistrees dans reports/comparison/ et peuvent
illustrer le README ou servir de support pendant la demonstration.

Usage :
    python src/utils/visualize.py
    python src/utils/visualize.py --images 6 --conf 0.25
    python src/utils/visualize.py --baseline chemin/vers/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "images"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "comparison"

# Emplacements possibles des poids, dans l'ordre de preference
BASELINE_CANDIDATES = [
    PROJECT_ROOT / "models" / "safevision_yolov8n_baseline.pt",
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs_safevision"
    / "baseline_yolov8n_v2"
    / "weights"
    / "best.pt",
    PROJECT_ROOT / "runs_safevision" / "baseline_yolov8n_v2" / "weights" / "best.pt",
]
ADVANCED_CANDIDATES = [
    PROJECT_ROOT / "models" / "safevision_yolov8s_v1.pt",
    PROJECT_ROOT / "models" / "safevision_yolov8s_v1.onnx",
]

HEADER_HEIGHT = 40


def first_existing(candidates, label):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Poids du modele {label} introuvables. Indique-les avec --{label} <chemin>."
    )


def pick_images(n: int):
    """Choisit n images reparties regulierement dans le jeu de test."""
    images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"Aucune image dans {TEST_IMAGES_DIR} (lancer dvc pull)"
        )
    step = max(1, len(images) // n)
    return images[::step][:n]


def annotate(model, image_path: Path, conf: float) -> np.ndarray:
    """Retourne l'image avec les boites dessinees (format BGR)."""
    result = model.predict(str(image_path), conf=conf, device="cpu", verbose=False)[0]
    return result.plot()


def with_header(image: np.ndarray, text: str) -> np.ndarray:
    """Ajoute un bandeau sombre avec le nom du modele au-dessus de l'image."""
    bar = np.full((HEADER_HEIGHT, image.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([bar, image])


def side_by_side(
    left: np.ndarray, right: np.ndarray, left_label: str, right_label: str
):
    """Assemble deux images de meme hauteur cote a cote, avec un bandeau chacune."""
    height = min(left.shape[0], right.shape[0])

    def fit(img):
        scale = height / img.shape[0]
        return cv2.resize(img, (round(img.shape[1] * scale), height))

    separator = np.full((height + HEADER_HEIGHT, 6, 3), 255, dtype=np.uint8)
    return np.hstack(
        [
            with_header(fit(left), left_label),
            separator,
            with_header(fit(right), right_label),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Comparaison visuelle baseline / avance"
    )
    parser.add_argument("--images", type=int, default=4, help="nombre d'images")
    parser.add_argument("--conf", type=float, default=0.25, help="seuil de confiance")
    parser.add_argument(
        "--baseline", type=Path, default=None, help="poids du baseline (.pt)"
    )
    parser.add_argument(
        "--advanced", type=Path, default=None, help="poids du modele avance"
    )
    args = parser.parse_args()

    from ultralytics import YOLO  # import tardif : ultralytics est lourd

    baseline_path = args.baseline or first_existing(BASELINE_CANDIDATES, "baseline")
    advanced_path = args.advanced or first_existing(ADVANCED_CANDIDATES, "advanced")
    print(f"Baseline : {baseline_path}")
    print(f"Avance   : {advanced_path}")
    baseline = YOLO(str(baseline_path))
    advanced = YOLO(str(advanced_path), task="detect")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in pick_images(args.images):
        merged = side_by_side(
            annotate(baseline, image_path, args.conf),
            annotate(advanced, image_path, args.conf),
            "Baseline YOLOv8n",
            "Avance YOLOv8s",
        )
        out = OUTPUT_DIR / f"comparison_{image_path.stem[:40]}.png"
        cv2.imwrite(str(out), merged)
        print(f"  enregistre : {out}")
    print(f"\nTermine : {args.images} comparaisons dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
