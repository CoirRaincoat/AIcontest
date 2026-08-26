"""
Tune image-level thresholds on a validation split.

The detector produces boxes. This script searches confidence and minimum box
area thresholds, then reports image-level Precision/Recall/F1.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def load_gt(path: str) -> Dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def list_images(source: str) -> List[Path]:
    source_path = Path(source)
    if source_path.is_file():
        return [source_path]
    return sorted(p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_detections(weights: str, source: str, imgsz: int, iou: float, device: str) -> Dict[str, List[Tuple[float, float]]]:
    from ultralytics import YOLO

    model = YOLO(weights)
    detections: Dict[str, List[Tuple[float, float]]] = {}
    for image_path in list_images(source):
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=0.001,
            iou=iou,
            device=device,
            verbose=False,
        )
        image_dets: List[Tuple[float, float]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            h, w = result.orig_shape
            image_area = max(float(h * w), 1.0)
            for i in range(len(boxes)):
                if int(boxes.cls[i]) != 0:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()
                score = float(boxes.conf[i])
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
                image_dets.append((score, area_ratio))
        detections[image_path.name] = image_dets
    return detections


def score(gt: Dict[str, int], dets: Dict[str, List[Tuple[float, float]]], conf: float, min_area: float) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for name, y_true in gt.items():
        y_pred = int(any(s >= conf and area >= min_area for s, area in dets.get(name, [])))
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "conf": conf,
        "min_area": min_area,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune image-level thresholds for fire detection.")
    parser.add_argument("--weights", required=True, help="Path to YOLO .pt weights.")
    parser.add_argument("--source", default="data/val/images", help="Validation image directory.")
    parser.add_argument("--gt", default="outputs/val_gt.json", help="Validation image-level GT JSON.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Device, for example 0 or cpu.")
    parser.add_argument("--conf_values", default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--area_values", default="0,0.00005,0.0001,0.0002,0.0005,0.001")
    parser.add_argument("--output", default="outputs/threshold_search.json", help="Full search result JSON.")
    args = parser.parse_args()

    gt = load_gt(args.gt)
    dets = collect_detections(args.weights, args.source, args.imgsz, args.iou, args.device)

    rows = []
    for conf in parse_float_list(args.conf_values):
        for min_area in parse_float_list(args.area_values):
            rows.append(score(gt, dets, conf, min_area))

    rows.sort(key=lambda x: (x["f1"], x["recall"], x["precision"]), reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("Top threshold settings:")
    for row in rows[:10]:
        print(
            f"conf={row['conf']:.3f} min_area={row['min_area']:.5f} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}"
        )
    print(f"Saved full search to {output_path}")


if __name__ == "__main__":
    main()
