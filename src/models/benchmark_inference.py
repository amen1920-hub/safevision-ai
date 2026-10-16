"""
benchmark_inference.py
========================
Compare la latence d'inference entre le modele PyTorch (.pt) et sa version
exportee en ONNX, sur un echantillon d'images de test. Les resultats sont
affiches en console ET logges dans MLflow pour archivage.

Prerequis :
    - models/safevision_yolov8s_v1.pt doit exister
    - models/safevision_yolov8s_v1.onnx doit exister (lancer export_onnx.py avant)
    - pip install onnxruntime

Usage :
    python src/models/benchmark_inference.py
"""

import os
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PT_MODEL_PATH = PROJECT_ROOT / "models" / "safevision_yolov8s_v1.pt"
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "safevision_yolov8s_v1.onnx"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "images"

N_WARMUP = 5     # inferences "a blanc" pour chauffer le GPU/CPU (non comptees)
N_RUNS = 30      # nombre d'inferences mesurees


def get_sample_images(n=N_RUNS):
    images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))[:n]
    if len(images) < n:
        # repete les images si le dataset de test est plus petit que N_RUNS
        images = (images * (n // len(images) + 1))[:n]
    return [str(p) for p in images]


# NOTE : onnxruntime-gpu necessite CUDA 13.x / cuDNN 9.x, non disponibles ici
# (PyTorch utilise CUDA 12.1). ONNX Runtime bascule donc automatiquement sur CPU.
# Pour une comparaison equitable (pas GPU vs CPU), on force aussi PyTorch sur CPU.
DEVICE = "cpu"


def benchmark_pytorch(images):
    model = YOLO(str(PT_MODEL_PATH))

    # warmup
    for img in images[:N_WARMUP]:
        model.predict(img, device=DEVICE, verbose=False)

    times = []
    for img in images:
        start = time.perf_counter()
        model.predict(img, device=DEVICE, verbose=False)
        times.append((time.perf_counter() - start) * 1000)  # en ms

    return times


def benchmark_onnx(images):
    # On reutilise le wrapper Ultralytics, qui sait aussi charger un .onnx
    # et applique le meme pre/post-traitement -> comparaison equitable.
    model = YOLO(str(ONNX_MODEL_PATH))

    for img in images[:N_WARMUP]:
        model.predict(img, device=DEVICE, verbose=False)

    times = []
    for img in images:
        start = time.perf_counter()
        model.predict(img, device=DEVICE, verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    return times


def summarize(name, times):
    arr = np.array(times)
    mean_ms = arr.mean()
    std_ms = arr.std()
    fps = 1000.0 / mean_ms
    print(f"\n--- {name} ---")
    print(f"Latence moyenne : {mean_ms:.2f} ms (+/- {std_ms:.2f} ms)")
    print(f"Latence min/max : {arr.min():.2f} ms / {arr.max():.2f} ms")
    print(f"FPS equivalent  : {fps:.1f}")
    return {"mean_ms": mean_ms, "std_ms": std_ms, "min_ms": arr.min(), "max_ms": arr.max(), "fps": fps}


def main():
    images = get_sample_images()
    print(f"Benchmark sur {len(images)} images ({N_WARMUP} de warmup non comptees)")
    print(f"Device utilise pour les deux modeles : {DEVICE.upper()} "
          f"(comparaison equitable, onnxruntime-gpu non disponible sur cette machine)")

    print("\n[1/2] Benchmark PyTorch (.pt)...")
    pt_times = benchmark_pytorch(images)
    pt_stats = summarize("PyTorch (.pt)", pt_times)

    print("\n[2/2] Benchmark ONNX Runtime (.onnx)...")
    onnx_times = benchmark_onnx(images)
    onnx_stats = summarize("ONNX Runtime (.onnx)", onnx_times)

    speedup = pt_stats["mean_ms"] / onnx_stats["mean_ms"]

    print("\n" + "=" * 60)
    print("RESUME COMPARATIF")
    print("=" * 60)
    print(f"{'Modele':<20}{'Latence moy.':<18}{'FPS':<10}")
    print(f"{'PyTorch':<20}{pt_stats['mean_ms']:.2f} ms{'':<10}{pt_stats['fps']:.1f}")
    print(f"{'ONNX Runtime':<20}{onnx_stats['mean_ms']:.2f} ms{'':<10}{onnx_stats['fps']:.1f}")
    print(f"\nSpeedup ONNX vs PyTorch : {speedup:.2f}x")

    # --- Log dans MLflow pour archivage ---
    try:
        import mlflow
        os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("safevision-ai")
        with mlflow.start_run(run_name="onnx_benchmark"):
            mlflow.log_metric("pytorch_latency_ms", pt_stats["mean_ms"])
            mlflow.log_metric("pytorch_fps", pt_stats["fps"])
            mlflow.log_metric("onnx_latency_ms", onnx_stats["mean_ms"])
            mlflow.log_metric("onnx_fps", onnx_stats["fps"])
            mlflow.log_metric("onnx_speedup", speedup)
        print("\nResultats logges dans MLflow (run 'onnx_benchmark').")
    except Exception as e:
        print(f"\n(MLflow non disponible pour logger ce benchmark : {e})")


if __name__ == "__main__":
    main()
