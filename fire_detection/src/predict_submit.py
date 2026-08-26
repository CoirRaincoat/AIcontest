"""
Image-level fire prediction for competition submission.

The task evaluates image-level labels:
    {"image.jpg": 0 or 1}

This script runs a YOLO detector and converts detections to image-level
predictions. If any fire box remains after filtering, the image is labeled 1.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(source: Path) -> List[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Unsupported source: {source}")


def predict_image_level(
    weights: str,
    source: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    min_area: float,
    augment: bool = False,
) -> Dict[str, int]:
    from ultralytics import YOLO

    image_paths = iter_images(Path(source))
    model = YOLO(weights)
    predictions: Dict[str, int] = {}

    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            augment=augment,
            verbose=False,
        )

        has_fire = False
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            h, w = result.orig_shape
            image_area = max(float(h * w), 1.0)
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
                if area_ratio >= min_area:
                    has_fire = True
                    break
            if has_fire:
                break

        predictions[image_path.name] = 1 if has_fire else 0

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image-level fire JSON submission.")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights.")
    parser.add_argument("--source", required=True, help="Image file or image directory.")
    parser.add_argument("--output", default="outputs/submission.json", help="Output JSON path.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Device, for example 0 or cpu.")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable Ultralytics test-time augmentation during prediction.",
    )
    parser.add_argument(
        "--min_area",
        type=float,
        default=0.0,
        help="Minimum detected fire box area ratio. Use a small value such as 0.0001 to suppress tiny noise.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = predict_image_level(
        weights=args.weights,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        min_area=args.min_area,
        augment=args.augment,
    )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    positive = sum(predictions.values())
    print(f"Saved {len(predictions)} predictions to {output_path}")
    print(f"Positive images: {positive}, negative images: {len(predictions) - positive}")


if __name__ == "__main__":
    main()
