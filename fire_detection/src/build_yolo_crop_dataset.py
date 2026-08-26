from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\AI\fire_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fire/non-fire crops from GT boxes and low-confidence YOLO proposals."
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=ROOT / "siglip_data" / "train.csv",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=ROOT / "siglip_data" / "val.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "crop_data_v2",
    )
    parser.add_argument("--proposal-conf", type=float, default=0.03)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--positive-iou", type=float, default=0.30)
    parser.add_argument("--negative-iou", type=float, default=0.05)
    parser.add_argument("--context", type=float, default=0.15)
    parser.add_argument("--min-crop-size", type=int, default=96)
    parser.add_argument("--negative-tiles-per-image", type=int, default=4)
    parser.add_argument("--max-proposals-per-image", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def load_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != 0:
            continue
        cx, cy, box_width, box_height = map(float, parts[1:5])
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def expanded_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    context: float,
    min_crop_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(
        box_width * (1.0 + 2.0 * context),
        box_height * (1.0 + 2.0 * context),
        float(min_crop_size),
    )
    crop_width = min(float(width), side)
    crop_height = min(float(height), side)
    left = min(max(0.0, center_x - crop_width / 2.0), width - crop_width)
    top = min(max(0.0, center_y - crop_height / 2.0), height - crop_height)
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + crop_width)),
        int(round(top + crop_height)),
    )


def negative_tiles(
    image_name: str,
    width: int,
    height: int,
    count: int,
    min_crop_size: int,
) -> list[tuple[int, int, int, int]]:
    seed = int(hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:16], 16)
    generator = random.Random(seed)
    side = min(width, height, max(min_crop_size, int(round(min(width, height) * 0.35))))
    boxes = []
    for _ in range(count):
        left = generator.randint(0, max(width - side, 0))
        top = generator.randint(0, max(height - side, 0))
        boxes.append((left, top, left + side, top + side))
    return boxes


def deduplicate_proposals(
    proposals: list[dict[str, object]], max_count: int
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for proposal in sorted(proposals, key=lambda item: float(item["confidence"]), reverse=True):
        box = tuple(float(value) for value in proposal["box"])
        if any(box_iou(box, tuple(float(v) for v in kept["box"])) >= 0.80 for kept in selected):
            continue
        selected.append(proposal)
        if len(selected) >= max_count:
            break
    return selected


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. Use a new versioned directory."
        )

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

    split_rows = {
        "train": read_manifest(args.train_manifest),
        "val": read_manifest(args.val_manifest),
    }
    all_rows = split_rows["train"] + split_rows["val"]
    row_by_name = {row["image_name"]: row for row in all_rows}
    if len(row_by_name) != len(all_rows):
        raise ValueError("Image names must be unique across train and val manifests.")

    model_settings = {
        "yolo26m": ROOT
        / "runs"
        / "detect"
        / "runs"
        / "train"
        / "fire_yolo26m_960"
        / "weights"
        / "best.pt",
        "yolo26s": ROOT
        / "runs"
        / "detect"
        / "runs"
        / "train"
        / "fire_yolo26s_960"
        / "weights"
        / "best.pt",
        "yolo26s_aug": ROOT
        / "runs"
        / "detect"
        / "runs"
        / "train"
        / "fire_yolo26s_aug_960"
        / "weights"
        / "best.pt",
    }
    for weights in model_settings.values():
        if not weights.is_file():
            raise FileNotFoundError(weights)

    from ultralytics import YOLO

    proposals_by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    for model_name, weights in model_settings.items():
        print(f"loading {model_name}: {weights}")
        model = YOLO(str(weights))
        for index, row in enumerate(all_rows, start=1):
            results = model.predict(
                source=row["image_path"],
                imgsz=args.imgsz,
                conf=args.proposal_conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is not None:
                for box_index in range(len(boxes)):
                    if int(boxes.cls[box_index]) != 0:
                        continue
                    proposals_by_image[row["image_name"]].append(
                        {
                            "model": model_name,
                            "confidence": float(boxes.conf[box_index]),
                            "box": tuple(float(v) for v in boxes.xyxy[box_index].cpu().tolist()),
                        }
                    )
            if index % 100 == 0 or index == len(all_rows):
                print(f"{model_name}: {index}/{len(all_rows)}")
        model.to("cpu")
        del model
        gc.collect()

    manifest_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    counts: Counter[str] = Counter()
    for split, rows in split_rows.items():
        crop_dir = args.output_dir / split / "images"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source_path = Path(row["image_path"])
            label_path = Path(row["label_path"])
            with Image.open(source_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            ground_truth = load_yolo_boxes(label_path, width, height)
            source_stem = source_path.stem

            for gt_index, gt_box in enumerate(ground_truth):
                crop_box = expanded_box(
                    gt_box, width, height, args.context, args.min_crop_size
                )
                crop_name = f"{source_stem}__gt_{gt_index:02d}_p.jpg"
                crop_path = crop_dir / crop_name
                image.crop(crop_box).save(crop_path, quality=95)
                manifest_rows[split].append(
                    {
                        "image_path": str(crop_path.resolve()),
                        "image_name": crop_name,
                        "label": 1,
                        "split": split,
                        "source_image": row["image_name"],
                        "source_path": row["image_path"],
                        "proposal_type": "gt",
                        "model": "ground_truth",
                        "confidence": 1.0,
                        "max_iou": 1.0,
                        "x1": crop_box[0],
                        "y1": crop_box[1],
                        "x2": crop_box[2],
                        "y2": crop_box[3],
                    }
                )
                counts[f"{split}_gt_positive"] += 1

            proposals = deduplicate_proposals(
                proposals_by_image.get(row["image_name"], []),
                args.max_proposals_per_image,
            )
            for proposal_index, proposal in enumerate(proposals):
                proposal_box = tuple(float(value) for value in proposal["box"])
                max_iou = max(
                    (box_iou(proposal_box, gt_box) for gt_box in ground_truth),
                    default=0.0,
                )
                if max_iou >= args.positive_iou:
                    label = 1
                elif max_iou <= args.negative_iou and int(row["label"]) == 0:
                    label = 0
                else:
                    counts[f"{split}_ambiguous_skipped"] += 1
                    continue
                crop_box = expanded_box(
                    proposal_box, width, height, args.context, args.min_crop_size
                )
                marker = "p" if label == 1 else "n"
                crop_name = (
                    f"{source_stem}__yolo_{proposal_index:02d}_{proposal['model']}_{marker}.jpg"
                )
                crop_path = crop_dir / crop_name
                image.crop(crop_box).save(crop_path, quality=95)
                manifest_rows[split].append(
                    {
                        "image_path": str(crop_path.resolve()),
                        "image_name": crop_name,
                        "label": label,
                        "split": split,
                        "source_image": row["image_name"],
                        "source_path": row["image_path"],
                        "proposal_type": "yolo",
                        "model": proposal["model"],
                        "confidence": proposal["confidence"],
                        "max_iou": max_iou,
                        "x1": crop_box[0],
                        "y1": crop_box[1],
                        "x2": crop_box[2],
                        "y2": crop_box[3],
                    }
                )
                counts[f"{split}_yolo_label_{label}"] += 1

            if int(row["label"]) == 0:
                for tile_index, crop_box in enumerate(
                    negative_tiles(
                        row["image_name"],
                        width,
                        height,
                        args.negative_tiles_per_image,
                        args.min_crop_size,
                    )
                ):
                    crop_name = f"{source_stem}__tile_{tile_index:02d}_n.jpg"
                    crop_path = crop_dir / crop_name
                    image.crop(crop_box).save(crop_path, quality=95)
                    manifest_rows[split].append(
                        {
                            "image_path": str(crop_path.resolve()),
                            "image_name": crop_name,
                            "label": 0,
                            "split": split,
                            "source_image": row["image_name"],
                            "source_path": row["image_path"],
                            "proposal_type": "negative_tile",
                            "model": "none",
                            "confidence": 0.0,
                            "max_iou": 0.0,
                            "x1": crop_box[0],
                            "y1": crop_box[1],
                            "x2": crop_box[2],
                            "y2": crop_box[3],
                        }
                    )
                    counts[f"{split}_negative_tile"] += 1

    fieldnames = [
        "image_path",
        "image_name",
        "label",
        "split",
        "source_image",
        "source_path",
        "proposal_type",
        "model",
        "confidence",
        "max_iou",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    for split, rows in manifest_rows.items():
        manifest_path = args.output_dir / f"{split}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "settings": {
            "proposal_conf": args.proposal_conf,
            "imgsz": args.imgsz,
            "iou": args.iou,
            "positive_iou": args.positive_iou,
            "negative_iou": args.negative_iou,
            "context": args.context,
            "min_crop_size": args.min_crop_size,
            "negative_tiles_per_image": args.negative_tiles_per_image,
            "max_proposals_per_image": args.max_proposals_per_image,
        },
        "source_images": {
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
        },
        "crop_rows": {
            "train": len(manifest_rows["train"]),
            "val": len(manifest_rows["val"]),
        },
        "counts": dict(sorted(counts.items())),
        "manifests": {
            "train": str((args.output_dir / "train.csv").resolve()),
            "val": str((args.output_dir / "val.csv").resolve()),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
