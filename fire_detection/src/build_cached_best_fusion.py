from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(r"C:\AI\fire_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the selected SigLIP2 + YOLO validation prediction from cached outputs."
    )
    parser.add_argument(
        "--siglip-scores",
        type=Path,
        default=ROOT / "runs_siglip" / "siglip2_linear_seed2026" / "val_scores.csv",
    )
    parser.add_argument(
        "--yolo-m",
        type=Path,
        default=ROOT / "outputs" / "val_pred_yolo26m_conf010.json",
    )
    parser.add_argument(
        "--yolo-s",
        type=Path,
        default=ROOT / "outputs" / "val_pred_yolo26s_conf020.json",
    )
    parser.add_argument(
        "--yolo-s-aug",
        type=Path,
        default=ROOT / "outputs" / "val_pred_yolo26s_aug_conf020.json",
    )
    parser.add_argument("--gt", type=Path, default=ROOT / "outputs" / "val_gt.json")
    parser.add_argument("--siglip-threshold", type=float, default=0.22)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs"
        / "val_pred_best_seed2026_yolo_fusion_thr022.json",
    )
    return parser.parse_args()


def load_json_predictions(path: Path) -> dict[str, int]:
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


def calculate_metrics(
    ground_truth: dict[str, int], predictions: dict[str, int]
) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for name, truth in ground_truth.items():
        prediction = predictions[name]
        if truth == 1 and prediction == 1:
            tp += 1
        elif truth == 0 and prediction == 1:
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
    scores = load_scores(args.siglip_scores)
    yolo_m = load_json_predictions(args.yolo_m)
    yolo_s = load_json_predictions(args.yolo_s)
    yolo_s_aug = load_json_predictions(args.yolo_s_aug)
    ground_truth = load_json_predictions(args.gt)

    expected_names = set(ground_truth)
    inputs = {
        "SigLIP2 scores": set(scores),
        "YOLO26m": set(yolo_m),
        "YOLO26s": set(yolo_s),
        "YOLO26s augmented": set(yolo_s_aug),
    }
    for label, names in inputs.items():
        if names != expected_names:
            raise ValueError(
                f"{label} image names differ from ground truth: "
                f"actual={len(names)} expected={len(expected_names)}"
            )

    predictions: dict[str, int] = {}
    detail_rows: list[dict[str, object]] = []
    for name in sorted(expected_names):
        yolo_votes = yolo_m[name] + yolo_s[name] + yolo_s_aug[name]
        siglip_positive = scores[name] >= args.siglip_threshold
        prediction = int(yolo_votes >= 1 and siglip_positive)
        predictions[name] = prediction
        detail_rows.append(
            {
                "image_name": name,
                "truth": ground_truth[name],
                "prediction": prediction,
                "siglip_score": f"{scores[name]:.8f}",
                "siglip_positive": int(siglip_positive),
                "yolo_votes": yolo_votes,
                "yolo26m": yolo_m[name],
                "yolo26s": yolo_s[name],
                "yolo26s_aug": yolo_s_aug[name],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    details_path = args.output.with_name(f"{args.output.stem}_details.csv")
    with details_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)

    metrics = calculate_metrics(ground_truth, predictions)
    summary = {
        "rule": "(YOLO26m conf=0.10 OR YOLO26s conf=0.20 OR YOLO26s_aug conf=0.20) AND SigLIP2 seed=2026",
        "siglip_threshold": args.siglip_threshold,
        "metrics": metrics,
        "prediction_file": str(args.output.resolve()),
        "details_file": str(details_path.resolve()),
    }
    summary_path = args.output.with_name(f"{args.output.stem}_metrics.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
