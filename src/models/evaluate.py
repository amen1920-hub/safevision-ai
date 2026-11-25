"""
evaluate.py
===========
Evalue le modele de reference (YOLOv8n) et le modele avance (YOLOv8s) sur les
jeux de validation et de TEST, puis enregistre les metriques globales et par
classe (CSV) ainsi que les graphiques d'Ultralytics (matrice de confusion...).

Le jeu de test n'a servi ni a l'entrainement ni au choix du modele : ses
resultats sont l'estimation honnete des performances. Ne pas modifier le
modele apres les avoir vus.

Usage :
    python src/models/evaluate.py            # evaluation + fichiers CSV
    python src/models/evaluate.py --mlflow   # journalise aussi dans MLflow
                                             # (le serveur MLflow doit tourner)
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "data" / "processed" / "data.yaml"
OUT_DIR = PROJECT_ROOT / "reports" / "evaluation"

BASELINE_WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs_safevision"
    / "baseline_yolov8n_v2"
    / "weights"
    / "best.pt"
)
MODELS = {
    "baseline_yolov8n": BASELINE_WEIGHTS,
    "advanced_yolov8s": PROJECT_ROOT / "models" / "safevision_yolov8s_v1.pt",
}
SPLITS = ["val", "test"]


def evaluate(name, weights, split):
    """Evalue un modele sur un split et retourne (resume, lignes par classe)."""
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=640,
        batch=8,
        workers=4,
        plots=True,
        project="runs_safevision",
        name=f"eval_{name}_{split}",
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    summary = {
        "model": name,
        "split": split,
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    per_class = []
    for i, class_id in enumerate(box.ap_class_index):
        per_class.append(
            {
                "model": name,
                "split": split,
                "class": metrics.names[int(class_id)],
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "AP50": float(box.ap50[i]),
                "AP50-95": float(box.ap[i]),
            }
        )
    print(f"  -> graphiques et matrice de confusion : {metrics.save_dir}")
    return summary, per_class


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def log_to_mlflow(summaries):
    import mlflow

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("safevision-ai")
    for s in summaries:
        with mlflow.start_run(run_name=f"eval_{s['model']}_{s['split']}"):
            mlflow.log_param("model", s["model"])
            mlflow.log_param("split", s["split"])
            for key in ("mAP50", "mAP50-95", "precision", "recall"):
                mlflow.log_metric(key.replace("-", "_"), s[key])
    print("\nResultats journalises dans MLflow.")


def print_report(summaries, per_class_rows):
    print("\n" + "=" * 70)
    print(
        f"{'Modele':<20}{'Split':<7}{'mAP50':>8}{'mAP50-95':>10}{'Prec.':>8}{'Rappel':>8}"
    )
    for s in summaries:
        print(
            f"{s['model']:<20}{s['split']:<7}{s['mAP50']:>8.3f}"
            f"{s['mAP50-95']:>10.3f}{s['precision']:>8.3f}{s['recall']:>8.3f}"
        )

    test_rows = [r for r in per_class_rows if r["split"] == "test"]
    names = [n for n in MODELS if any(r["model"] == n for r in test_rows)]
    classes = sorted({r["class"] for r in test_rows})
    print("\nAP50 par classe (jeu de test) :")
    print(f"{'Classe':<16}" + "".join(f"{n:>20}" for n in names))
    for c in classes:
        cells = ""
        for n in names:
            found = [
                r["AP50"] for r in test_rows if r["class"] == c and r["model"] == n
            ]
            cells += f"{found[0]:>20.3f}" if found else f"{'-':>20}"
        print(f"{c:<16}{cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow", action="store_true", help="journaliser dans MLflow")
    args = parser.parse_args()

    summaries, per_class_rows = [], []
    for name, weights in MODELS.items():
        if not weights.exists():
            print(f"IGNORE {name} : fichier introuvable ({weights})")
            continue
        for split in SPLITS:
            print(f"\n=== {name} sur '{split}' ===")
            summary, per_class = evaluate(name, weights, split)
            summaries.append(summary)
            per_class_rows.extend(per_class)

    if not summaries:
        print("Aucun modele evalue.")
        return

    write_csv(OUT_DIR / "summary.csv", summaries)
    write_csv(OUT_DIR / "per_class.csv", per_class_rows)
    print_report(summaries, per_class_rows)
    print(f"\nFichiers CSV : {OUT_DIR}")

    if args.mlflow:
        log_to_mlflow(summaries)


if __name__ == "__main__":
    main()
