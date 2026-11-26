"""
dataset.py
===========
Description et verification du jeu de donnees (data/processed/).

Pour YOLO, le "pipeline Dataset" est gere par Ultralytics a partir de data.yaml :
ce module ne recharge donc pas les images, il VERIFIE que les donnees sont saines
avant un entrainement :

    - chaque image a son fichier d'annotations ;
    - le nombre d'images et d'objets par classe dans chaque split ;
    - AUCUNE fuite de donnees : un meme nom d'image n'apparait pas dans deux splits,
      et les frames d'une meme video restent dans un seul split.

Usage :
    python src/data/dataset.py
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Meme regle de regroupement que split_by_scene.py : "<prefixe>_mp4-123", "...-mov-8"
VIDEO_PATTERN = re.compile(r"^(.*?)[-_](mp4|mov)-\d+", re.IGNORECASE)


def video_group(stem: str):
    """Cle de la video source d'une frame, ou None pour une image independante."""
    match = VIDEO_PATTERN.match(stem)
    return f"{match.group(1)}_{match.group(2).lower()}" if match else None


def load_class_names(data_yaml: Path | None = None) -> list:
    data_yaml = data_yaml or PROCESSED_DIR / "data.yaml"
    with open(data_yaml, encoding="utf-8") as f:
        return list(yaml.safe_load(f)["names"])


def split_stats(split: str, root: Path | None = None) -> dict:
    """Images, annotations manquantes et nombre d'objets par classe pour un split."""
    root = root or PROCESSED_DIR
    images = sorted(
        p
        for p in (root / split / "images").glob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels_dir = root / split / "labels"
    missing = []
    instances = Counter()
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            missing.append(image.name)
            continue
        for line in label.read_text().splitlines():
            if line.strip():
                instances[int(line.split()[0])] += 1
    return {"images": images, "missing_labels": missing, "instances": instances}


def find_leaks(stats: dict) -> list:
    """Retourne la liste des fuites detectees entre splits (vide si tout est propre)."""
    problems = []
    name_to_split = {}
    video_to_splits = defaultdict(set)
    for split, s in stats.items():
        for image in s["images"]:
            if image.name in name_to_split and name_to_split[image.name] != split:
                problems.append(
                    f"image '{image.name}' presente dans {name_to_split[image.name]} et {split}"
                )
            name_to_split[image.name] = split
            group = video_group(image.stem)
            if group:
                video_to_splits[group].add(split)
    for group, splits in video_to_splits.items():
        if len(splits) > 1:
            problems.append(f"video '{group}' repartie entre {sorted(splits)}")
    return problems


def run_checks(root: Path | None = None) -> bool:
    """Affiche le resume du jeu de donnees. Retourne True si tout est valide."""
    root = root or PROCESSED_DIR
    names = load_class_names(root / "data.yaml")
    stats = {split: split_stats(split, root) for split in SPLITS}
    ok = True

    print(f"{'Split':<8}{'Images':<9}{'Objets':<9}")
    for split, s in stats.items():
        print(f"{split:<8}{len(s['images']):<9}{sum(s['instances'].values()):<9}")
        if not s["images"]:
            print(f"  ERREUR : aucune image dans {split}")
            ok = False
        if s["missing_labels"]:
            print(f"  ERREUR : {len(s['missing_labels'])} image(s) sans annotation")
            ok = False

    print("\nObjets par classe :")
    print(f"{'Classe':<18}" + "".join(f"{split:<8}" for split in SPLITS))
    for i, name in enumerate(names):
        counts = "".join(f"{stats[split]['instances'][i]:<8}" for split in SPLITS)
        print(f"{name:<18}{counts}")

    leaks = find_leaks(stats)
    for problem in leaks:
        print(f"FUITE DE DONNEES : {problem}")
    if leaks:
        ok = False
    else:
        print("\nAucune fuite de donnees detectee entre train / valid / test.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)
