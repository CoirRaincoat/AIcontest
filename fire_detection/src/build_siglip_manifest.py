from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build image-level fire/no-fire CSV manifests from YOLO labels."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\AI\fire_detection\data"),
        help="YOLO dataset root containing train/ and val/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\AI\fire_detection\siglip_data"),
        help="Directory for train.csv, val.csv, and summary.json.",
    )
    return parser.parse_args()


def build_split(data_root: Path, split: str) -> tuple[list[dict[str, object]], dict[str, int]]:
    images_dir = data_root / split / "images"
    labels_dir = data_root / split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing YOLO split folders: {images_dir} / {labels_dir}")

    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    rows: list[dict[str, object]] = []
    positive = 0
    negative = 0
    missing_labels = 0

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            has_fire = bool(label_path.read_text(encoding="utf-8").strip())
        else:
            has_fire = False
            missing_labels += 1

        label = int(has_fire)
        positive += label
        negative += 1 - label
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "image_name": image_path.name,
                "label": label,
                "split": split,
                "label_path": str(label_path.resolve()),
            }
        )

    stats = {
        "total": len(rows),
        "fire": positive,
        "no_fire": negative,
        "missing_label_files_treated_as_no_fire": missing_labels,
    }
    return rows, stats


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["image_path", "image_name", "label", "split", "label_path"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "data_root": str(args.data_root.resolve()),
        "label_rule": "non-empty YOLO label = 1 fire; empty label = 0 no_fire",
        "splits": {},
    }

    for split in ("train", "val"):
        rows, stats = build_split(args.data_root, split)
        csv_path = args.output_dir / f"{split}.csv"
        write_csv(csv_path, rows)
        summary["splits"][split] = stats
        print(
            f"{split}: total={stats['total']} fire={stats['fire']} "
            f"no_fire={stats['no_fire']} -> {csv_path}"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
