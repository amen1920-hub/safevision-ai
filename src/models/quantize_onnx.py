"""
quantize_onnx.py
=================
Optimise le modele ONNX par quantification et compare trois versions :

    - FP32 : modele de reference (celui servi par l'API)
    - FP16 : poids en demi-precision (fichier ~2x plus petit)
    - INT8 : quantification statique (poids et activations en entiers 8 bits),
             calibree sur quelques images d'entrainement

Pour chaque version, le script mesure : taille du fichier, latence moyenne
sur CPU et precision (mAP) sur le jeu de TEST. Le tableau est affiche, enregistre
dans reports/quantization/ et logge dans MLflow.

Prerequis :
    pip install onnx onnxruntime onnxconverter-common

Usage :
    python src/models/quantize_onnx.py
    python src/models/quantize_onnx.py --skip-int8   (FP16 seulement)
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FP32_PATH = MODELS_DIR / "safevision_yolov8s_v1.onnx"
FP16_PATH = MODELS_DIR / "safevision_yolov8s_v1_fp16.onnx"
INT8_PATH = MODELS_DIR / "safevision_yolov8s_v1_int8.onnx"
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"
CALIB_DIR = PROJECT_ROOT / "data" / "processed" / "train" / "images"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "images"
REPORT_DIR = PROJECT_ROOT / "reports" / "quantization"

IMGSZ = 640
N_WARMUP = 5
N_RUNS = 30


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# ----------------------------------------------------------------------
# FP16 : conversion des poids en demi-precision
# ----------------------------------------------------------------------
def _loads_in_onnxruntime(path: Path) -> bool:
    """Verifie que onnxruntime sait charger le fichier (le graphe est coherent)."""
    try:
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        print(f"    modele invalide : {str(e)[:160]}")
        return False


def make_fp16(src: Path, dst: Path) -> bool:
    """Convertit en FP16. Essaie plusieurs reglages : sur YOLO, le noeud Resize
    produit parfois un graphe invalide avec les reglages par defaut."""
    from onnxconverter_common import float16

    blocked = list(float16.DEFAULT_OP_BLOCK_LIST) + ["Resize"]
    attempts = [
        ("sans inference de formes", {"disable_shape_infer": True}),
        ("Resize garde en FP32", {"op_block_list": blocked}),
        (
            "les deux reglages",
            {"disable_shape_infer": True, "op_block_list": blocked},
        ),
    ]
    for label, options in attempts:
        print(f"  FP16 : essai ({label})")
        try:
            model = onnx.load(str(src))
            # Efface les types intermediaires de l'export : ils entrent en conflit
            # avec ceux que la conversion recalcule.
            model.graph.ClearField("value_info")
            # keep_io_types=True : entree et sortie restent en float32, donc
            # l'API et Ultralytics n'ont rien a changer.
            converted = float16.convert_float_to_float16(
                model, keep_io_types=True, **options
            )
            onnx.save(converted, str(dst))
        except Exception as e:
            print(f"    conversion impossible : {str(e)[:160]}")
            continue
        if _loads_in_onnxruntime(dst):
            return True
    dst.unlink(missing_ok=True)
    return False


# ----------------------------------------------------------------------
# INT8 : quantification statique avec calibration
# ----------------------------------------------------------------------
def preprocess(image_path: Path) -> np.ndarray:
    """Meme pretraitement que YOLO : redimensionnement + bandes grises, RGB, [0,1]."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    scale = IMGSZ / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    canvas[top : top + nh, left : left + nw] = img
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[None]


class ImageCalibrationReader(CalibrationDataReader):
    """Fournit quelques images reelles pour mesurer l'etendue des activations."""

    def __init__(self, model_path: Path, image_paths):
        session_input = onnx.load(str(model_path)).graph.input[0].name
        self.input_name = session_input
        self.iterator = iter(image_paths)

    def get_next(self):
        path = next(self.iterator, None)
        if path is None:
            return None
        return {self.input_name: preprocess(path)}


def ensure_opset(src: Path, min_opset: int = 13) -> Path:
    """La quantification par canal exige l'opset 13 ou plus (le modele est exporte en 12)."""
    model = onnx.load(str(src))
    current = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    if current >= min_opset:
        return src
    print(f"  Conversion de l'opset {current} vers {min_opset}...")
    converted = version_converter.convert_version(model, min_opset)
    tmp = src.with_name(src.stem + "_opset13.onnx")
    onnx.save(converted, str(tmp))
    return tmp


def detection_head_nodes(model_path: Path):
    """Noms des noeuds de la derniere couche (la tete de detection de YOLO).

    Quantifier cette tete detruit les predictions : elle melange des coordonnees
    (jusqu'a 640) et des scores (entre 0 et 1) sur une meme echelle 8 bits, et les
    scores sont alors ecrases a zero. On la laisse donc en precision normale.
    """
    model = onnx.load(str(model_path))
    layers = {}
    for node in model.graph.node:
        match = re.match(r"^/model\.(\d+)/", node.name)
        if match:
            layers.setdefault(int(match.group(1)), []).append(node.name)
    return layers[max(layers)] if layers else []


def make_int8(src: Path, dst: Path, n_calib: int) -> None:
    images = sorted(CALIB_DIR.glob("*.jpg"))[:n_calib]
    if not images:
        raise FileNotFoundError(f"Aucune image de calibration dans {CALIB_DIR}")
    print(f"  Calibration INT8 sur {len(images)} images d'entrainement...")
    prepared = ensure_opset(src)
    head = detection_head_nodes(prepared)
    print(f"  Tete de detection laissee en precision normale ({len(head)} noeuds)")
    try:
        quantize_static(
            model_input=str(prepared),
            model_output=str(dst),
            calibration_data_reader=ImageCalibrationReader(prepared, images),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=head,
        )
    finally:
        if prepared != src:
            prepared.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Mesures : latence et precision
# ----------------------------------------------------------------------
def measure_latency(model_path: Path) -> float:
    images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
    images = (images * (N_RUNS // len(images) + 1))[:N_RUNS]
    model = YOLO(str(model_path), task="detect")
    for img in images[:N_WARMUP]:
        model.predict(str(img), device="cpu", verbose=False)
    times = []
    for img in images:
        start = time.perf_counter()
        model.predict(str(img), device="cpu", verbose=False)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.mean(times))


def measure_map(model_path: Path, label: str):
    model = YOLO(str(model_path), task="detect")
    res = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        batch=1,
        device="cpu",
        plots=False,
        verbose=False,
        project=str(PROJECT_ROOT / "runs" / "quantization"),
        name=label,
        exist_ok=True,
    )
    return float(res.box.map50), float(res.box.map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--calib-images", type=int, default=64)
    args = parser.parse_args()

    if not FP32_PATH.exists():
        raise FileNotFoundError(f"Modele introuvable : {FP32_PATH} (lancer dvc pull)")

    variants = [("FP32", FP32_PATH)]

    print("[1/3] Conversion FP16...")
    if make_fp16(FP32_PATH, FP16_PATH):
        variants.append(("FP16", FP16_PATH))
    else:
        print("  FP16 impossible avec cette version des outils : variante ignoree.")

    if not args.skip_int8:
        print("[2/3] Quantification INT8...")
        try:
            make_int8(FP32_PATH, INT8_PATH, args.calib_images)
            variants.append(("INT8", INT8_PATH))
        except Exception as e:
            print(f"  INT8 impossible : {str(e)[:200]}")

    print("[3/3] Mesures sur CPU (taille, latence, mAP sur le jeu de test)...")
    rows = []
    for label, path in variants:
        print(f"  - {label}")
        try:
            latency = measure_latency(path)
            map50, map5095 = measure_map(path, label.lower())
        except Exception as e:
            print(f"    mesure impossible pour {label} : {str(e)[:200]}")
            continue
        rows.append(
            {
                "variante": label,
                "taille_mb": round(size_mb(path), 2),
                "latence_ms": round(latency, 1),
                "mAP50_test": round(map50, 3),
                "mAP50-95_test": round(map5095, 3),
            }
        )

    base = rows[0]
    print("\n" + "=" * 72)
    print("RESULTATS DE LA QUANTIFICATION (CPU, jeu de test)")
    print("=" * 72)
    print(f"{'Variante':<10}{'Taille':<12}{'Latence':<13}{'mAP50':<10}{'mAP50-95':<10}")
    for r in rows:
        print(
            f"{r['variante']:<10}{r['taille_mb']:<12}{r['latence_ms']:<13}"
            f"{r['mAP50_test']:<10}{r['mAP50-95_test']:<10}"
        )
    for r in rows[1:]:
        print(
            f"\n{r['variante']} vs FP32 : taille x{base['taille_mb'] / r['taille_mb']:.2f} "
            f"plus petite, latence x{base['latence_ms'] / r['latence_ms']:.2f}, "
            f"delta mAP50 = {r['mAP50_test'] - base['mAP50_test']:+.3f}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "quantization_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTableau enregistre : {csv_path}")

    try:
        import mlflow

        os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("safevision-ai")
        with mlflow.start_run(run_name="onnx_quantization"):
            for r in rows:
                tag = r["variante"].lower()
                mlflow.log_metric(f"{tag}_size_mb", r["taille_mb"])
                mlflow.log_metric(f"{tag}_latency_ms", r["latence_ms"])
                mlflow.log_metric(f"{tag}_map50_test", r["mAP50_test"])
            mlflow.log_artifact(str(csv_path))
        print("Resultats logges dans MLflow (run 'onnx_quantization').")
    except Exception as e:
        print(f"(MLflow non disponible : {e})")


if __name__ == "__main__":
    main()
