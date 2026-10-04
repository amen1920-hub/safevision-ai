"""
register_model.py
==================
Enregistre le modele avance (advanced_yolov8s) dans le MLflow Model Registry,
et lui assigne le statut "Production" (le modele que l'API ira charger).

Avant de lancer ce script, recupere le Run ID du run "advanced_yolov8s" :
- Dans MLflow (http://127.0.0.1:5000), ouvre le run, va dans "Overview",
  le "Run ID" est affiche la (une longue chaine hexadecimale).
- Ou regarde l'URL de la page du run : .../runs/<RUN_ID>/...

Usage :
    python src/models/register_model.py --run-id <RUN_ID>
"""

import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

MODEL_NAME = "safevision-yolo"
ARTIFACT_PATH = "weights/best.pt"  # chemin de l'artefact dans le run


def main(run_id: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    # S'assurer que le modele (le "nom") existe dans le registry
    try:
        client.create_registered_model(MODEL_NAME)
        print(f"Modele '{MODEL_NAME}' cree dans le registry.")
    except Exception:
        print(f"Modele '{MODEL_NAME}' existe deja dans le registry, on continue.")

    # Creer une nouvelle version pointant directement vers l'artefact du run
    # (methode compatible avec les artefacts "bruts", pas seulement les
    # modeles logges via mlflow.<flavor>.log_model)
    source = f"runs:/{run_id}/{ARTIFACT_PATH}"
    model_version: ModelVersion = client.create_model_version(
        name=MODEL_NAME,
        source=source,
        run_id=run_id,
    )
    version = model_version.version
    print(f"Modele enregistre : '{MODEL_NAME}' version {version}")

    # 2. Ajouter une description utile
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=(
            "YOLOv8s fine-tune sur le dataset SafeVision AI (9 classes EPI). "
            "Entraine 80 epochs, augmentation renforcee (mixup, copy_paste). "
            f"mAP50=0.626, mAP50-95=0.360, precision=0.866, recall=0.572. "
            f"Run source MLflow : {run_id}"
        ),
    )

    # 3. Assigner le statut "Production" (compatible anciennes et nouvelles versions de MLflow)
    try:
        # API historique (stages) - fonctionne encore sur de nombreuses versions
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Statut 'Production' assigne a la version {version} (via stages).")
    except Exception as e:
        # API recente (aliases) - remplace les stages dans MLflow 2.9+/3.x
        print(f"Les stages classiques ne sont pas disponibles ({e}), utilisation des alias...")
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="production",
            version=version,
        )
        print(f"Alias 'production' assigne a la version {version} (via alias).")

    print("=" * 60)
    print("MODELE ENREGISTRE DANS LE MODEL REGISTRY")
    print("=" * 60)
    print(f"Nom du modele    : {MODEL_NAME}")
    print(f"Version          : {version}")
    print(f"Consultez : http://127.0.0.1:5000/#/models/{MODEL_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID MLflow du run advanced_yolov8s (visible dans l'URL ou l'onglet Overview)",
    )
    args = parser.parse_args()
    main(args.run_id)
