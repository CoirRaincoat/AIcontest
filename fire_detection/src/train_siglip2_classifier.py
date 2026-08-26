from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fire/no-fire linear classifier on frozen SigLIP2 image features."
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(r"C:\AI\fire_detection\siglip_data\train.csv"),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(r"C:\AI\fire_detection\siglip_data\val.csv"),
    )
    parser.add_argument("--model", default="google/siglip2-base-patch16-384")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"C:\AI\hf_cache\hub"),
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path(r"C:\AI\fire_detection\siglip_data\siglip2_base_384_embeddings"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\AI\fire_detection\runs_siglip\siglip2_linear_v1"),
    )
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Ignore saved embeddings and extract them again.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for row in rows:
        row["label"] = int(row["label"])
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
    return rows


def embedding_paths(directory: Path, split: str) -> tuple[Path, Path]:
    return directory / f"{split}.npz", directory / f"{split}_metadata.json"


def load_saved_embeddings(
    directory: Path,
    split: str,
    expected_rows: int,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    npz_path, metadata_path = embedding_paths(directory, split)
    if not npz_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model") != model_name or metadata.get("rows") != expected_rows:
        return None
    data = np.load(npz_path, allow_pickle=False)
    return data["features"], data["labels"], data["image_names"], data["image_paths"]


def save_embeddings(
    directory: Path,
    split: str,
    model_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    image_names: np.ndarray,
    image_paths: np.ndarray,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path, metadata_path = embedding_paths(directory, split)
    np.savez_compressed(
        npz_path,
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
        image_names=image_names.astype(str),
        image_paths=image_paths.astype(str),
    )
    metadata = {
        "model": model_name,
        "split": split,
        "rows": int(len(labels)),
        "feature_dimension": int(features.shape[1]),
        "normalized": True,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_embeddings(
    rows: list[dict[str, object]],
    split: str,
    processor,
    backbone,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    image_names: list[str] = []
    image_paths: list[str] = []

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = []
        for row in batch_rows:
            with Image.open(str(row["image_path"])) as image:
                images.append(image.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if key in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        }

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = backbone.get_image_features(**inputs)
            else:
                output = backbone.get_image_features(**inputs)

        pooled = output.pooler_output if hasattr(output, "pooler_output") else output
        pooled = F.normalize(pooled.float(), dim=1)
        features.append(pooled.cpu().numpy())
        labels.extend(int(row["label"]) for row in batch_rows)
        image_names.extend(str(row["image_name"]) for row in batch_rows)
        image_paths.extend(str(row["image_path"]) for row in batch_rows)

        completed = min(start + batch_size, len(rows))
        print(f"extract {split}: {completed}/{len(rows)}")

    feature_array = np.concatenate(features, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    name_array = np.asarray(image_names, dtype=str)
    path_array = np.asarray(image_paths, dtype=str)
    save_embeddings(
        output_dir,
        split,
        model_name,
        feature_array,
        label_array,
        name_array,
        path_array,
    )
    return feature_array, label_array, name_array, path_array


class LinearFireHead(nn.Module):
    def __init__(self, feature_dimension: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Dropout(dropout),
            nn.Linear(feature_dimension, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def metrics_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
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
    best_key: tuple[float, float, float] | None = None
    for threshold in np.linspace(0.0, 1.0, 1001):
        result = metrics_from_scores(labels, scores, float(threshold))
        key = (float(result["f1"]), float(result["recall"]), float(result["precision"]))
        if best is None or best_key is None or key > best_key:
            best = result
            best_key = key
    assert best is not None
    return best


def evaluate_head(
    head: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    head.eval()
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)
    with torch.inference_mode():
        logits = head(x)
        loss = criterion(logits, y).item()
        scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return float(loss), scores


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embedding_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(args.train_manifest)
    val_rows = load_manifest(args.val_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")

    cached_train = None if args.rebuild_embeddings else load_saved_embeddings(
        args.embedding_dir, "train", len(train_rows), args.model
    )
    cached_val = None if args.rebuild_embeddings else load_saved_embeddings(
        args.embedding_dir, "val", len(val_rows), args.model
    )

    if cached_train is None or cached_val is None:
        print(f"loading frozen backbone: {args.model}")
        processor = AutoProcessor.from_pretrained(
            args.model,
            cache_dir=str(args.cache_dir),
            local_files_only=True,
        )
        backbone = AutoModel.from_pretrained(
            args.model,
            cache_dir=str(args.cache_dir),
            local_files_only=True,
        )
        backbone.eval().to(device)
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        train_data = cached_train or extract_embeddings(
            train_rows,
            "train",
            processor,
            backbone,
            device,
            args.extract_batch_size,
            args.embedding_dir,
            args.model,
        )
        val_data = cached_val or extract_embeddings(
            val_rows,
            "val",
            processor,
            backbone,
            device,
            args.extract_batch_size,
            args.embedding_dir,
            args.model,
        )
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"using saved embeddings: {args.embedding_dir}")
        train_data = cached_train
        val_data = cached_val

    train_features, train_labels, train_names, train_paths = train_data
    val_features, val_labels, val_names, val_paths = val_data
    feature_dimension = int(train_features.shape[1])
    print(f"feature dimension: {feature_dimension}")

    counts = np.bincount(train_labels, minlength=2).astype(np.float64)
    class_weights = len(train_labels) / (2.0 * np.maximum(counts, 1.0))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"class counts [no_fire, fire]: {counts.astype(int).tolist()}")
    print(f"class weights [no_fire, fire]: {class_weights.tolist()}")

    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.head_batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    head = LinearFireHead(feature_dimension, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = args.output_dir / "best_head.pt"
    history_path = args.output_dir / "training_history.csv"
    history_rows: list[dict[str, object]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total_examples = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_labels)
            total_examples += len(batch_labels)

        train_loss = total_loss / max(total_examples, 1)
        val_loss, val_scores = evaluate_head(
            head, val_features, val_labels, criterion, device
        )
        metrics_05 = metrics_from_scores(val_labels, val_scores, 0.5)
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision_0_5": metrics_05["precision"],
            "recall_0_5": metrics_05["recall"],
            "f1_0_5": metrics_05["f1"],
        }
        history_rows.append(history_row)
        print(
            f"epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"P={metrics_05['precision']:.4f} R={metrics_05['recall']:.4f} "
            f"F1={metrics_05['f1']:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "feature_dimension": feature_dimension,
                    "dropout": args.dropout,
                    "model": args.model,
                    "cache_dir": str(args.cache_dir.resolve()),
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "class_names": {0: "no_fire", 1: "fire"},
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["head_state_dict"])
    final_val_loss, final_scores = evaluate_head(
        head, val_features, val_labels, criterion, device
    )
    metrics_05 = metrics_from_scores(val_labels, final_scores, 0.5)
    best_threshold_metrics = find_best_threshold(val_labels, final_scores)

    prediction_path = args.output_dir / "val_scores.csv"
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["image_name", "image_path", "label", "score", "prediction_best"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        threshold = float(best_threshold_metrics["threshold"])
        for image_name, image_path, label, score in zip(
            val_names, val_paths, val_labels, final_scores
        ):
            writer.writerow(
                {
                    "image_name": str(image_name),
                    "image_path": str(image_path),
                    "label": int(label),
                    "score": f"{float(score):.8f}",
                    "prediction_best": int(float(score) >= threshold),
                }
            )

    result = {
        "method": "frozen SigLIP2 image embeddings + weighted linear classifier",
        "model": args.model,
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "train_counts": {"no_fire": int(counts[0]), "fire": int(counts[1])},
        "class_weights": {"no_fire": float(class_weights[0]), "fire": float(class_weights[1])},
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "best_checkpoint_val_loss": float(checkpoint["val_loss"]),
        "final_val_loss": final_val_loss,
        "metrics_at_threshold_0_5": metrics_05,
        "best_validation_metrics": best_threshold_metrics,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": str(history_path.resolve()),
        "val_scores": str(prediction_path.resolve()),
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("metrics at threshold=0.5:")
    print(json.dumps(metrics_05, ensure_ascii=False, indent=2))
    print("best validation metrics:")
    print(json.dumps(best_threshold_metrics, ensure_ascii=False, indent=2))
    print(f"best checkpoint -> {checkpoint_path}")
    print(f"results -> {result_path}")


if __name__ == "__main__":
    main()
