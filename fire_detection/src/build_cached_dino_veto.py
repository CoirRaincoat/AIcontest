from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(r"C:\AI\fire_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a DINOv3 veto to the selected cached YOLO + SigLIP2 prediction."
    )
    parser.add_argument(
        "--base-prediction",
        type=Path,
        default=ROOT
        / "outputs"
        / "val_pred_best_seed2026_yolo_fusion_thr022.json",
    )
    parser.add_argument(
        "--dino-scores",
        type=Path,
        default=ROOT
        / "runs_dinov3"
        / "dinov3_vitb16_linear_seed42"
        / "val_scores.csv",
    )
    parser.add_argument("--dino-threshold", type=float, default=0.08)
    parser.add_argument("--gt", type=Path, default=ROOT / "outputs" / "val_gt.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "val_pred_best_siglip_yolo_dinov3_veto_thr008.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, int]:
    return {
        str(name): int(value)
        for name, value in json.loads(path.read_text(encoding="utf-8")).items()
    }


def load_scores(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {
            str(row["image_name"]): float(row["score"])
            for row in csv.DictReader(handle)
        }


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
    scores = load_scores(args.dino_scores)
    ground_truth = load_json(args.gt)
    if set(base) != set(scores) or set(base) != set(ground_truth):
        raise ValueError("Image names differ between base prediction, DINOv3 scores and GT.")

    predictions = {
        name: int(base[name] == 1 and scores[name] >= args.dino_threshold)
        for name in sorted(base)
    }
    changed = [name for name in sorted(base) if predictions[name] != base[name]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "rule": "cached best YOLO + SigLIP2 prediction AND DINOv3 score threshold",
        "dino_threshold": args.dino_threshold,
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
