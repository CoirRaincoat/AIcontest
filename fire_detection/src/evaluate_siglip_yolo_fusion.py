from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SigLIP2 and YOLO image-level fusion.")
    parser.add_argument(
        "--siglip-scores",
        type=Path,
        default=Path(r"C:\AI\fire_detection\runs_siglip\siglip2_linear_v1\val_scores.csv"),
    )
    parser.add_argument(
        "--yolo-pred",
        type=Path,
        default=Path(
            r"C:\AI\fire_detection\outputs\val_pred_fusion_A_m010_s015_saug015_1vote.json"
        ),
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path(r"C:\AI\fire_detection\outputs\val_gt.json"),
    )
    parser.add_argument("--siglip-threshold", type=float, default=0.267)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\AI\fire_detection\outputs"),
    )
    return parser.parse_args()


def load_scores(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["image_name"]): float(row["score"]) for row in rows}


def metrics(gt: dict[str, int], pred: dict[str, int]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for name, truth in gt.items():
        value = int(pred.get(name, 0))
        if truth == 1 and value == 1:
            tp += 1
        elif truth == 0 and value == 1:
            fp += 1
        elif truth == 1 and value == 0:
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


def siglip_predictions(scores: dict[str, float], threshold: float) -> dict[str, int]:
    return {name: int(score >= threshold) for name, score in scores.items()}


def fuse(
    siglip: dict[str, int],
    yolo: dict[str, int],
    rule: str,
    names: list[str],
) -> dict[str, int]:
    if rule == "or":
        return {name: int(siglip.get(name, 0) or yolo.get(name, 0)) for name in names}
    if rule == "and":
        return {name: int(siglip.get(name, 0) and yolo.get(name, 0)) for name in names}
    raise ValueError(f"Unknown fusion rule: {rule}")


def threshold_candidates(scores: dict[str, float]) -> list[float]:
    values = sorted(set(scores.values()))
    candidates = [0.0, 1.0]
    candidates.extend(values)
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    return sorted(set(candidates))


def search(
    gt: dict[str, int],
    scores: dict[str, float],
    yolo: dict[str, int],
    rule: str,
) -> tuple[float, dict[str, float | int], dict[str, int]]:
    names = sorted(gt)
    best_threshold = 0.5
    best_metrics: dict[str, float | int] | None = None
    best_pred: dict[str, int] | None = None
    best_key: tuple[float, float, float] | None = None

    for threshold in threshold_candidates(scores):
        siglip = siglip_predictions(scores, threshold)
        if rule == "siglip":
            pred = {name: siglip.get(name, 0) for name in names}
        else:
            pred = fuse(siglip, yolo, rule, names)
        result = metrics(gt, pred)
        key = (float(result["f1"]), float(result["recall"]), float(result["precision"]))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = result
            best_pred = pred

    assert best_metrics is not None and best_pred is not None
    return best_threshold, best_metrics, best_pred


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt = {name: int(value) for name, value in json.loads(args.gt.read_text(encoding="utf-8")).items()}
    yolo = {
        name: int(value)
        for name, value in json.loads(args.yolo_pred.read_text(encoding="utf-8")).items()
    }
    scores = load_scores(args.siglip_scores)

    gt_names = set(gt)
    if set(scores) != gt_names:
        raise ValueError(f"SigLIP2 names differ from GT: scores={len(scores)} gt={len(gt)}")
    if set(yolo) != gt_names:
        raise ValueError(f"YOLO names differ from GT: yolo={len(yolo)} gt={len(gt)}")

    fixed_siglip = siglip_predictions(scores, args.siglip_threshold)
    fixed_or = fuse(fixed_siglip, yolo, "or", sorted(gt))
    fixed_and = fuse(fixed_siglip, yolo, "and", sorted(gt))

    fixed_results = {
        "siglip": metrics(gt, fixed_siglip),
        "yolo": metrics(gt, yolo),
        "or": metrics(gt, fixed_or),
        "and": metrics(gt, fixed_and),
    }

    save_json(args.output_dir / "val_pred_siglip2_thr0267.json", fixed_siglip)
    save_json(args.output_dir / "val_pred_fusion_siglip2_yolo_or_thr0267.json", fixed_or)
    save_json(args.output_dir / "val_pred_fusion_siglip2_yolo_and_thr0267.json", fixed_and)

    searched_results: dict[str, object] = {}
    for rule in ("siglip", "or", "and"):
        threshold, result, pred = search(gt, scores, yolo, rule)
        threshold_name = f"{threshold:.6f}".replace(".", "p")
        pred_path = args.output_dir / f"val_pred_fusion_{rule}_best_thr{threshold_name}.json"
        save_json(pred_path, pred)
        searched_results[rule] = {
            "threshold": threshold,
            "metrics": result,
            "prediction_file": str(pred_path.resolve()),
        }

    summary = {
        "siglip_scores": str(args.siglip_scores.resolve()),
        "yolo_prediction": str(args.yolo_pred.resolve()),
        "fixed_siglip_threshold": args.siglip_threshold,
        "fixed_threshold_results": fixed_results,
        "searched_threshold_results": searched_results,
        "note": "Threshold search is exploratory on the same 220-image validation set.",
    }
    summary_path = args.output_dir / "siglip2_yolo_fusion_results.json"
    save_json(summary_path, summary)

    print("fixed threshold results:")
    print(json.dumps(fixed_results, ensure_ascii=False, indent=2))
    print("searched threshold results:")
    print(json.dumps(searched_results, ensure_ascii=False, indent=2))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
