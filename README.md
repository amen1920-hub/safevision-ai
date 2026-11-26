# SafeVision AI

Détection automatique des équipements de protection individuelle (EPI) sur chantier, en temps réel.
Projet de module *Deep Learning Avancé* (Axe 3 : détection d'objets), cycle ingénieur IA & Data Science.

Une image ou un flux webcam est envoyé à une API FastAPI. Un modèle YOLOv8 (exporté en ONNX) détecte les
personnes et les équipements de sécurité. L'interface React dessine les boîtes, affiche les scores et
déclenche une **alerte sonore et visuelle** quand un équipement manque (classes `NO-…`).

![Interface SafeVision AI](docs/screenshot.png)

## Sommaire

1. [Résultats](#résultats)
2. [Architecture](#architecture)
3. [Démarrage rapide (Docker)](#démarrage-rapide-docker)
4. [Interface](#interface)
5. [API](#api)
6. [Cycle MLOps](#cycle-mlops)
7. [Reproduire les expériences](#reproduire-les-expériences)
8. [Tests et CI](#tests-et-ci)
9. [Structure du dépôt](#structure-du-dépôt)
10. [Limites connues](#limites-connues)

## Résultats

Classes détectées : `Person`, `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Mask`, `NO-Mask`,
`Gloves`, `Safety Cone`. Les classes `NO-…` signalent un équipement **manquant**.

### Baseline contre modèle avancé

Évaluation sur le **jeu de test** (55 images, 708 objets), jamais utilisé pour choisir le modèle.

| Modèle | mAP@50 | mAP@[.5:.95] | Précision | Rappel |
|---|---|---|---|---|
| Baseline : YOLOv8n (30 epochs) | 0,587 | 0,339 | 0,771 | 0,548 |
| **Avancé : YOLOv8s** (80 epochs, mixup, copy_paste) | **0,657** | **0,358** | 0,785 | **0,617** |

Le modèle avancé améliore surtout le mAP@50 et le rappel : il rate moins d'objets. Sa précision progresse
peu. Détail par classe : `reports/evaluation/per_class.csv`.

### Quantification (CPU, jeu de test)

Script : `src/models/quantize_onnx.py`.

| Variante | Taille | Latence | mAP@50 | mAP@[.5:.95] |
|---|---|---|---|---|
| FP32 (en production) | 42,7 Mo | 187 ms | 0,661 | 0,358 |
| FP16 | 21,4 Mo | 187 ms | 0,659 | 0,358 |
| INT8 (statique, tête de détection en FP32) | 17,2 Mo | 137 ms | 0,626 | 0,336 |

Le FP16 divise la taille par 2 sans perte de précision, mais n'accélère pas le calcul sur processeur.
L'INT8 est 1,36 fois plus rapide mais perd 3,5 points de mAP@50 : le FP32 reste donc le modèle servi par
l'API, car pour une application de sécurité, rater un équipement manquant coûte plus cher que quelques
millisecondes. Une première tentative INT8, avec la tête de détection quantifiée, donnait un mAP nul :
les scores étaient écrasés à zéro. La version finale la laisse en précision normale.

Les mAP de ce tableau sont mesurés sur les fichiers ONNX (écart de 0,004 avec l'évaluation PyTorch).

## Architecture

```
Navigateur ──► Frontend React (nginx, :3000) ──► API FastAPI (:8000) ──► ONNX Runtime (YOLOv8s)
                                                        ▲
                                  DVC (données + modèle)  ·  MLflow (suivi + registre)
```

| Couche | Technologie |
|---|---|
| Modèle | YOLOv8 (Ultralytics), export ONNX, ONNX Runtime |
| API | FastAPI asynchrone (inférence dans un pool de threads), Pydantic v2, Swagger sur `/docs` |
| Interface | React, Vite, Tailwind CSS |
| MLOps | DVC, MLflow, Docker Compose multi-stage, GitHub Actions |

## Démarrage rapide (Docker)

Prérequis : Docker Desktop (ou Docker Engine + Compose) et Git.

```bash
git clone https://github.com/amen1920-hub/safevision-ai.git
cd safevision-ai
```

### 1. Récupérer le modèle

Le modèle (44 Mo) n'est pas dans Git : il est versionné avec DVC. Le dossier `models/` doit contenir
`safevision_yolov8s_v1.onnx` avant de lancer l'application.

```bash
pip install dvc dvc-gdrive
dvc pull models/safevision_yolov8s_v1.onnx.dvc
```

Le remote DVC est un Google Drive privé : l'accès doit être autorisé par l'auteur du projet. Sans le
fichier ONNX, l'API démarre mais répond `503` sur `/predict`.

### 2. Lancer l'application

```bash
docker compose up --build
```

- Interface : http://localhost:3000
- API et Swagger : http://localhost:8000/docs

> **Port 8000 déjà utilisé ?** Crée un fichier `.env` à la racine contenant `API_PORT=8001`, puis relance
> `docker compose up --build`. L'API sera alors sur http://localhost:8001. Le frontend est reconstruit
> avec la bonne adresse automatiquement.

Pour arrêter : `Ctrl + C`, puis `docker compose down`.

## Interface

Cinq onglets :

| Onglet | Rôle |
|---|---|
| **Image** | Glisser-déposer, galerie d'images d'exemple, boîtes colorées par classe, barres de score animées, latence client et serveur |
| **Webcam** | Détection en continu avec FPS et latences |
| **Tableau de bord** | Analyses, objets détectés, non-conformités, taux de conformité, détections par classe |
| **Modèle** | Nom, architecture, version, métriques et classes (depuis `/model/metadata`) |
| **MLflow** | Lien vers l'interface MLflow (`http://localhost:5000`) |

Une alerte rouge et un signal sonore se déclenchent à chaque non-conformité détectée. Tous les événements
sont conservés dans l'historique des alertes. Le son se coupe avec le bouton « Son activé ».

## API

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | État du serveur, GPU, modèle chargé |
| GET | `/model/metadata` | Nom, architecture, métriques et classes du modèle |
| POST | `/predict` | Image → détections en JSON (boîtes, classes, scores, latence) |
| POST | `/predict/visualize` | Image → image annotée (JPEG) |

Exemple :

```bash
curl -X POST http://localhost:8000/predict -F "file=@mon_image.jpg"
```

Erreurs : `400` si le fichier n'est pas une image valide, `503` si le modèle n'est pas chargé (fichier ONNX
absent du dossier `models/`).

## Cycle MLOps

### DVC : données et modèles versionnés

Les données et les poids ne sont jamais committés dans Git, seulement les fichiers `.dvc`.

```bash
dvc status          # état des données et du modèle
dvc pull            # récupère données et modèles (accès au remote requis)
```

### MLflow : suivi et registre des modèles

Chaque entraînement journalise ses paramètres, ses métriques et ses artefacts. Les runs sont dans
l'expérience `safevision-ai`.

```bash
mlflow ui --port 5000   # puis ouvrir http://localhost:5000
```

Registre des modèles (`safevision-yolo`) :

| Version | Modèle | Statut |
|---|---|---|
| 1 | YOLOv8s (avancé) | **Production** |
| 2 | YOLOv8n (baseline) | Staging |

Enregistrer un modèle :
```bash
python src/models/register_model.py --run-id <RUN_ID>                    # -> Production
python src/models/register_model.py --run-id <RUN_ID> --stage Staging    # -> Staging
```

## Reproduire les expériences

Prérequis : Python 3.11 et un serveur MLflow lancé (`mlflow ui --port 5000`).

```bash
python -m venv venv && venv\Scripts\activate         # Linux/macOS : source venv/bin/activate
pip install -r requirements.txt

python src/data/filter_classes.py          # filtrage des classes
python src/data/split_by_scene.py          # découpage train/valid/test par scène
python src/data/dataset.py                 # vérifie les splits et l'absence de fuite de données
python src/models/train.py --model baseline    # ou --model advanced
python src/models/evaluate.py              # évaluation validation + test
python src/models/export_onnx.py           # export ONNX
python src/models/benchmark_inference.py   # latence PyTorch contre ONNX
python src/models/quantize_onnx.py         # FP16 / INT8 : taille, latence, mAP
python src/utils/visualize.py              # comparaison visuelle baseline / avancé
```

Le découpage se fait **par scène** : les images extraites d'une même vidéo restent dans un seul split, pour
éviter qu'une scène se retrouve à la fois dans l'entraînement et dans le test. `dataset.py` le vérifie.

> `train.py` écrit dans `runs_safevision/` : relancer un entraînement avec le nom par défaut écrase le run
> existant. Utiliser `--name` pour un essai.

## Tests et CI

```bash
cd backend
pytest              # 14 tests unitaires (mock) + 4 tests d'intégration (vrai modèle ONNX)

black --check src backend
flake8 src backend
cd ../frontend && npm run lint
```

Les tests d'intégration nécessitent le fichier `models/safevision_yolov8s_v1.onnx`.

GitHub Actions exécute automatiquement :

- `.github/workflows/ci-tests.yml` : linting et tests ;
- `.github/workflows/docker-build.yml` : build des images Docker.

## Structure du dépôt

```
safevision-ai/
├── .github/workflows/   # CI : tests/linting et build Docker
├── backend/             # API FastAPI (app/, tests/, Dockerfile multi-stage)
├── frontend/            # Application React (pages/, components/, hooks/, Dockerfile multi-stage)
├── src/
│   ├── data/            # filtrage, découpage par scène, vérification (dataset.py)
│   ├── models/          # architecture, train, évaluation, export ONNX, quantification, registre
│   └── utils/           # visualisation comparative
├── data/                # Données (versionnées avec DVC)
├── models/              # Modèle ONNX / PyTorch (versionnés avec DVC)
├── reports/             # Résultats d'évaluation et de quantification
└── docker-compose.yml
```

## Données

Jeu de données public *Construction Site Safety* (Roboflow Universe), filtré sur 9 classes et découpé par
scène en `train`, `valid` et `test`.

## Limites connues

- **`Gloves`** est quasiment indétectable (rappel 0,07, AP50 0,11 sur le test) : objets petits et peu
  nombreux dans le jeu de données.
- Environ **un tiers** des `NO-Mask` et des `NO-Safety Vest` ne sont pas détectés : l'alerte ne se
  déclenche pas dans ces cas.
- Le jeu de test ne contient que **55 images** : un écart de quelques points est dans le bruit statistique.
- Le modèle ne reconnaît pas l'identité des personnes : une même personne vue dans une image importée puis
  dans la webcam est comptée une fois dans chaque source.
- Les alertes par e-mail, la page d'analyse et le réglage des seuils depuis l'interface ne sont pas
  implémentés.

## Auteurs

Projet réalisé dans le cadre du module Deep Learning Avancé, 3ᵉ année du cycle ingénieur IA & Data Science.
<!-- Ajouter les noms des membres du groupe -->
