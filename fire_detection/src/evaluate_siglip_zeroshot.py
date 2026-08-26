from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import AutoModel, AutoProcessor


POSITIVE_PROMPTS = [
    "a photo showing real visible fire and flames",
    "an image of burning flames",
    "real fire with visible orange or red flames",
    "a fire emergency with visible flames",
]

NEGATIVE_PROMPTS = [
    "a photo with no fire and no visible flames",
    "an ordinary scene without fire",
    "lights, sunset, reflections, or smoke without flames",
    "steam, fog, or smoke but no visible fire",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SigLIP2 zero-shot fire classification.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"C:\AI\fire_detection\siglip_data\val.csv"),
    )
    parser.add_argument(
        "--model",
        default="google/siglip2-base-patch16-384",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"C:\AI\hf_cache\hub"),
        help="Hugging Face hub cache containing the downloaded model.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\AI\fire_detection\outputs"),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    best: dict[str, object] | None = None
    for threshold in np.linspace(0.0, 1.0, 1001):
        result = calculate_metrics(labels, scores, float(threshold))
        key = (result["f1"], result["recall"], result["precision"])
        if best is None:
            best = result
            best_key = key
        elif key > best_key:
            best = result
            best_key = key
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    print(f"loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        args.model,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    model.eval().to(device)

    prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    positive_count = len(POSITIVE_PROMPTS)
    all_scores: list[float] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images = []
        for row in batch_rows:
            with Image.open(str(row["image_path"])) as image:
                images.append(image.convert("RGB"))

        inputs = processor(
            text=prompts,
            images=images,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        logits = outputs.logits_per_image.float()
        positive_logits = logits[:, :positive_count].mean(dim=1)
        negative_logits = logits[:, positive_count:].mean(dim=1)
        scores = torch.sigmoid(positive_logits - negative_logits)
        all_scores.extend(scores.detach().cpu().tolist())

        completed = min(start + args.batch_size, len(rows))
        print(f"processed {completed}/{len(rows)}")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    scores = np.asarray(all_scores, dtype=np.float64)
    metrics_at_05 = calculate_metrics(labels, scores, 0.5)
    best = find_best_threshold(labels, scores)

    score_path = args.output_dir / "siglip2_zeroshot_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["image_name", "image_path", "label", "score", "prediction_best"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, score in zip(rows, scores):
            writer.writerow(
                {
                    "image_name": row["image_name"],
                    "image_path": row["image_path"],
                    "label": row["label"],
                    "score": f"{score:.8f}",
                    "prediction_best": int(score >= float(best["threshold"])),
                }
            )

    result = {
        "model": args.model,
        "cache_dir": str(args.cache_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "images": len(rows),
        "positive_prompts": POSITIVE_PROMPTS,
        "negative_prompts": NEGATIVE_PROMPTS,
        "metrics_at_threshold_0_5": metrics_at_05,
        "best_validation_metrics": best,
        "score_file": str(score_path.resolve()),
    }
    result_path = args.output_dir / "siglip2_zeroshot_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("metrics at threshold=0.5:")
    print(json.dumps(metrics_at_05, ensure_ascii=False, indent=2))
    print("best validation metrics:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"scores -> {score_path}")
    print(f"results -> {result_path}")


if __name__ == "__main__":
    main()
