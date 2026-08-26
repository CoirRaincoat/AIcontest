"""
Create image-level ground-truth JSON for a split directory.

The competition evaluates:
    {"image.jpg": 0 or 1}

This helper reads the official train_image.json and keeps only the image names
that appear in a target image folder, such as data/val/images.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_names(image_dir: Path) -> List[str]:
    return sorted(p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image-level GT JSON for a dataset split.")
    parser.add_argument("--labels", default="data/images/train/train_image.json", help="Official image-level label JSON.")
    parser.add_argument("--image_dir", default="data/val/images", help="Split image directory.")
    parser.add_argument("--output", default="outputs/val_gt.json", help="Output JSON path.")
    args = parser.parse_args()

    label_path = Path(args.labels)
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    with label_path.open("r", encoding="utf-8") as f:
        labels: Dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}

    image_names = list_image_names(image_dir)
    missing = [name for name in image_names if name not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} images do not exist in {label_path}: {missing[:5]}")

    split_labels = {name: labels[name] for name in image_names}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(split_labels, f, ensure_ascii=False, indent=2)

    positives = sum(split_labels.values())
    print(f"Saved {len(split_labels)} labels to {output_path}")
    print(f"Positive images: {positives}, negative images: {len(split_labels) - positives}")


if __name__ == "__main__":
    main()
