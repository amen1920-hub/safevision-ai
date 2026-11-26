"""
train.py
=========
Point d'entree unique pour entrainer l'un des deux modeles. Le suivi (parametres,
metriques, poids) est enregistre automatiquement dans MLflow par Ultralytics.

Usage :
    python src/models/train.py --model baseline
    python src/models/train.py --model advanced
    python src/models/train.py --model advanced --name advanced_yolov8s_test

ATTENTION : avec le nom par defaut, un nouvel entrainement ecrase le dossier
des resultats du run existant. Utilise --name pour un essai.

Le serveur MLflow doit tourner dans un autre terminal (http://127.0.0.1:5000).
Le chemin de data.yaml et les hyperparametres sont definis dans architecture.py.
"""

import argparse
import os
import sys
from pathlib import Path

from architecture import MODEL_CONFIGS, build_model, get_train_args

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "safevision-ai")


def check_data() -> bool:
    """Verifie les donnees (voir src/data/dataset.py) avant de lancer l'entrainement."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from dataset import run_checks

    return run_checks()


def main():
    parser = argparse.ArgumentParser(description="Entrainement SafeVision AI")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    parser.add_argument(
        "--name", default=None, help="Nom du run (defaut : celui du modele)"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="Ne pas verifier les donnees avant"
    )
    args = parser.parse_args()

    if not args.skip_check and not check_data():
        print("Verification des donnees echouee : entrainement annule.")
        sys.exit(1)

    train_args = get_train_args(args.model, args.name)
    print(f"Entrainement '{args.model}' - run '{train_args['name']}'")
    model = build_model(args.model)
    model.train(**train_args)

    print("=" * 60)
    print(f"ENTRAINEMENT '{args.model.upper()}' TERMINE")
    print("=" * 60)
    print("Resultats : voir MLflow sur http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
