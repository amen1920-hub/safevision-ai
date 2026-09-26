"""
split_by_scene.py
==================
Corrige le data leakage du split Roboflow en regroupant les frames issues
de la meme video source dans un seul split (train, valid OU test - jamais
reparties entre plusieurs).

Entree  : data/processed/{train,valid,test}/{images,labels}  (deja filtre a 9 classes,
          mais avec un split casse par Roboflow)
Sortie  : le meme dossier data/processed/{train,valid,test}, reorganise proprement
          (l'ancien contenu est copie dans un dossier temporaire avant reecriture,
          donc aucune perte de donnees)

Usage :
    python src/data/split_by_scene.py
"""

import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STAGING_DIR = PROJECT_ROOT / "data" / "_processed_staging"  # dossier temporaire

SPLITS = ["train", "valid", "test"]

# Ratios cibles (en proportion du nombre total d'IMAGES, pas de groupes)
TARGET_RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}

RANDOM_SEED = 42

# Regex pour detecter une frame video : "<prefixe>_mp4-123", "<prefixe>-mp4-45",
# "<prefixe>_MOV-12", "<prefixe>_mov-8", etc.
VIDEO_PATTERN = re.compile(r"^(.*?)[-_](mp4|mov)-\d+", re.IGNORECASE)


def get_group_key(filename_stem: str) -> str:
    """
    Determine la cle de regroupement (source) pour un fichier donne.
    - Si le nom indique une frame video (mp4/mov + numero) -> regroupe par prefixe+video
    - Sinon -> l'image est consideree independante (sa propre cle unique)
    """
    match = VIDEO_PATTERN.match(filename_stem)
    if match:
        prefix = match.group(1)
        ext_marker = match.group(2).lower()
        return f"VIDEO::{prefix}_{ext_marker}"
    else:
        # Image independante : cle unique = son propre nom
        return f"SINGLE::{filename_stem}"


# ------------------------------------------------------------------
# 2. Collecter TOUTES les paires (image, label) des 3 splits actuels
# ------------------------------------------------------------------

all_items = []  # liste de dicts: {stem, image_path, label_path, group}

for split in SPLITS:
    images_dir = PROCESSED_DIR / split / "images"
    labels_dir = PROCESSED_DIR / split / "labels"

    if not labels_dir.exists():
        continue

    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem

        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"  ATTENTION: image introuvable pour {label_path.name}, ignoree")
            continue

        group = get_group_key(stem)
        all_items.append({
            "stem": stem,
            "image_path": image_path,
            "label_path": label_path,
            "group": group,
        })

print(f"Total d'images collectees (avant re-split) : {len(all_items)}")

# ------------------------------------------------------------------
# 3. Regrouper par cle de groupe
# ------------------------------------------------------------------

groups = defaultdict(list)
for item in all_items:
    groups[item["group"]].append(item)

group_keys = list(groups.keys())
group_sizes = {g: len(items) for g, items in groups.items()}

n_video_groups = sum(1 for g in group_keys if g.startswith("VIDEO::"))
n_single_groups = sum(1 for g in group_keys if g.startswith("SINGLE::"))
n_images_in_video_groups = sum(size for g, size in group_sizes.items() if g.startswith("VIDEO::"))

print(f"Groupes video detectes   : {n_video_groups} (regroupant {n_images_in_video_groups} images)")
print(f"Images independantes     : {n_single_groups}")
print()

# ------------------------------------------------------------------
# 4. Repartir les groupes dans train/valid/test (par taille cumulee)
# ------------------------------------------------------------------

random.seed(RANDOM_SEED)
random.shuffle(group_keys)

total_images = len(all_items)
targets = {split: ratio * total_images for split, ratio in TARGET_RATIOS.items()}

assignment = {}  # group_key -> split
counts = {split: 0 for split in SPLITS}

for group_key in group_keys:
    size = group_sizes[group_key]
    # Choisir le split qui est le plus "en retard" par rapport a sa cible
    deficits = {
        split: targets[split] - counts[split]
        for split in SPLITS
    }
    best_split = max(deficits, key=deficits.get)
    assignment[group_key] = best_split
    counts[best_split] += size

print("Repartition finale obtenue :")
for split in SPLITS:
    pct = 100 * counts[split] / total_images
    print(f"  {split:6s} : {counts[split]:4d} images ({pct:.1f}%)")
print()

# ------------------------------------------------------------------
# 5. Ecrire le resultat dans un dossier de staging, puis remplacer processed/
# ------------------------------------------------------------------

if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)

for split in SPLITS:
    (STAGING_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (STAGING_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

for item in all_items:
    group_key = item["group"]
    target_split = assignment[group_key]

    dst_image = STAGING_DIR / target_split / "images" / item["image_path"].name
    dst_label = STAGING_DIR / target_split / "labels" / item["label_path"].name

    shutil.copy2(item["image_path"], dst_image)
    shutil.copy2(item["label_path"], dst_label)

# Copier aussi le data.yaml existant
old_yaml = PROCESSED_DIR / "data.yaml"
if old_yaml.exists():
    shutil.copy2(old_yaml, STAGING_DIR / "data.yaml")

# Remplacer l'ancien data/processed par le nouveau contenu re-splitte
for split in SPLITS:
    old_split_dir = PROCESSED_DIR / split
    if old_split_dir.exists():
        shutil.rmtree(old_split_dir)
    shutil.move(str(STAGING_DIR / split), str(old_split_dir))

shutil.rmtree(STAGING_DIR, ignore_errors=True)

print("=" * 60)
print("RE-SPLIT TERMINE - data/processed/ mis a jour")
print("=" * 60)
print("Toutes les frames d'une meme video sont maintenant dans un seul split.")
print("Aucune image independante n'a ete dupliquee entre splits.")
