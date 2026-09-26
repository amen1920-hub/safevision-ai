"""
filter_classes.py
==================
Filtre le dataset "Construction Site Safety" (25 classes) pour ne garder
que les classes pertinentes pour la détection d'EPI (SafeVision AI).

Entrée  : data/raw/{train,valid,test}/{images,labels} + data/raw/data.yaml
Sortie  : data/processed/{train,valid,test}/{images,labels} + data/processed/data.yaml

Usage :
    python src/data/filter_classes.py
"""

import os
import shutil
from pathlib import Path

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------

# Racine du projet = 2 dossiers au-dessus de ce script (src/data/ -> racine)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Classes originales du dataset (ordre exact du data.yaml de Roboflow)
ORIGINAL_CLASSES = [
    "Excavator", "Gloves", "Hardhat", "Ladder", "Mask", "NO-Hardhat",
    "NO-Mask", "NO-Safety Vest", "Person", "SUV", "Safety Cone",
    "Safety Vest", "bus", "dump truck", "fire hydrant", "machinery",
    "mini-van", "sedan", "semi", "trailer", "truck and trailer",
    "truck", "van", "vehicle", "wheel loader"
]

# Classes qu'on garde pour SafeVision AI (EPI + personnes)
KEPT_CLASSES = [
    "Person", "Hardhat", "NO-Hardhat", "Safety Vest",
    "NO-Safety Vest", "Mask", "NO-Mask", "Gloves", "Safety Cone"
]

SPLITS = ["train", "valid", "test"]

# ------------------------------------------------------------------
# 2. Construire le mapping ancien_id -> nouvel_id
# ------------------------------------------------------------------

# old_id (index dans ORIGINAL_CLASSES) -> new_id (index dans KEPT_CLASSES)
old_to_new = {}
for old_id, class_name in enumerate(ORIGINAL_CLASSES):
    if class_name in KEPT_CLASSES:
        new_id = KEPT_CLASSES.index(class_name)
        old_to_new[old_id] = new_id

print("Mapping des classes conservées (old_id -> new_id) :")
for old_id, new_id in old_to_new.items():
    print(f"  {old_id:2d} ({ORIGINAL_CLASSES[old_id]:20s}) -> {new_id} ({KEPT_CLASSES[new_id]})")
print()

# ------------------------------------------------------------------
# 3. Traiter chaque split (train / valid / test)
# ------------------------------------------------------------------

stats = {}

for split in SPLITS:
    src_images_dir = RAW_DIR / split / "images"
    src_labels_dir = RAW_DIR / split / "labels"
    dst_images_dir = PROCESSED_DIR / split / "images"
    dst_labels_dir = PROCESSED_DIR / split / "labels"

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    kept_images = 0
    dropped_images = 0
    total_boxes_before = 0
    total_boxes_after = 0

    label_files = sorted(src_labels_dir.glob("*.txt"))

    for label_path in label_files:
        total_images += 1
        image_stem = label_path.stem  # nom sans extension

        # Lire les lignes du label original
        with open(label_path, "r") as f:
            lines = f.readlines()

        total_boxes_before += len(lines)

        # Filtrer + renuméroter les lignes
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            old_class_id = int(parts[0])
            if old_class_id in old_to_new:
                new_class_id = old_to_new[old_class_id]
                new_line = " ".join([str(new_class_id)] + parts[1:])
                new_lines.append(new_line)

        # Si l'image n'a plus aucune annotation valide, on la saute
        if len(new_lines) == 0:
            dropped_images += 1
            continue

        total_boxes_after += len(new_lines)
        kept_images += 1

        # Écrire le nouveau fichier label
        dst_label_path = dst_labels_dir / label_path.name
        with open(dst_label_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        # Copier l'image correspondante (essaye .jpg puis .png)
        src_image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = src_images_dir / f"{image_stem}{ext}"
            if candidate.exists():
                src_image_path = candidate
                break

        if src_image_path is None:
            print(f"  ATTENTION: image introuvable pour le label {label_path.name}")
            continue

        dst_image_path = dst_images_dir / src_image_path.name
        shutil.copy2(src_image_path, dst_image_path)

    stats[split] = {
        "total_images": total_images,
        "kept_images": kept_images,
        "dropped_images": dropped_images,
        "total_boxes_before": total_boxes_before,
        "total_boxes_after": total_boxes_after,
    }

    print(f"[{split}] {total_images} images analysées -> "
          f"{kept_images} conservées, {dropped_images} supprimées (aucune annotation EPI)")
    print(f"[{split}] Bounding boxes : {total_boxes_before} -> {total_boxes_after} "
          f"(après filtrage des classes hors-sujet)")
    print()

# ------------------------------------------------------------------
# 4. Générer le nouveau data.yaml
# ------------------------------------------------------------------

yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(KEPT_CLASSES)}
names: {KEPT_CLASSES}
"""

yaml_path = PROCESSED_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Nouveau data.yaml créé : {yaml_path}")
print()

# ------------------------------------------------------------------
# 5. Résumé final
# ------------------------------------------------------------------

print("=" * 60)
print("RÉSUMÉ DU FILTRAGE")
print("=" * 60)
total_kept = sum(s["kept_images"] for s in stats.values())
total_dropped = sum(s["dropped_images"] for s in stats.values())
for split, s in stats.items():
    print(f"{split:6s} : {s['kept_images']:4d} images conservées "
          f"({s['dropped_images']} supprimées)")
print(f"TOTAL  : {total_kept} images conservées, {total_dropped} supprimées")
print(f"Classes : {len(ORIGINAL_CLASSES)} -> {len(KEPT_CLASSES)}")
print("=" * 60)
