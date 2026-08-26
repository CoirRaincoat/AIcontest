from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(r"C:\AI\fire_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescue base image predictions using maximum SigLIP2 crop score."
    )
    parser.add_argument(
        "--base-prediction",
        type=Path,
        default=ROOT
        / "outputs"
        / "val_pred_best_siglip_yolo_dinov3_veto_thr008.json",
    )
    parser.add_argument(
        "--crop-manifest",
        type=Path,
        default=ROOT / "crop_data_v2" / "val.csv",
    )
    parser.add_argument(
        "--crop-scores",
        type=Path,
        default=ROOT
        / "runs_siglip"
        / "siglip2_crop_v2_seed2026"
        / "val_scores.csv",
    )
    parser.add_argument("--crop-threshold", type=float, default=0.97)
    parser.add_argument("--gt", type=Path, default=ROOT / "outputs" / "val_gt.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "val_pred_best_siglip_yolo_dino_crop_rescue_thr097.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, int]:
    return {
        str(name): int(value)
        for name, value in json.loads(path.read_text(encoding="utf-8")).items()
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metrics(gt: dict[str, int], pred: dict[str, int]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for name, truth in gt.items():
        value = pred[name]
        if truth == 1 and value == 1:
            tp += 1
        elif truth == 0 and value == 1:
            fp += 1
        elif truth == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()
    base = load_json(args.base_prediction)
    ground_truth = load_json(args.gt)
    manifest = load_csv(args.crop_manifest)
    score_rows = load_csv(args.crop_scores)
    score_by_crop = {
        row["image_name"]: float(row["score"]) for row in score_rows
    }

    max_score = {name: -1.0 for name in base}
    max_crop = {name: "" for name in base}
    for row in manifest:
        if row["proposal_type"] != "yolo":
            continue
        source_name = row["source_image"]
        score = score_by_crop[row["image_name"]]
        if score > max_score[source_name]:
            max_score[source_name] = score
            max_crop[source_name] = row["image_name"]

    predictions = {
        name: int(base[name] == 1 or max_score[name] >= args.crop_threshold)
        for name in sorted(base)
    }
    changed = [
        {
            "image_name": name,
            "truth": ground_truth[name],
            "crop_score": max_score[name],
            "crop_name": max_crop[name],
        }
        for name in sorted(base)
        if predictions[name] != base[name]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "rule": "base YOLO + full-image SigLIP2 + DINOv3 prediction OR high-confidence crop SigLIP2",
        "crop_threshold": args.crop_threshold,
        "metrics": metrics(ground_truth, predictions),
        "changed_images": changed,
        "prediction_file": str(args.output.resolve()),
    }
    summary_path = args.output.with_name(f"{args.output.stem}_metrics.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
