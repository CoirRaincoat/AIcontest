from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor


ROOT = Path(r"C:\AI\fire_detection")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate competition JSON using the selected YOLO + SigLIP2 fusion."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "submission_best_fusion.json",
    )
    parser.add_argument(
        "--siglip-checkpoint",
        type=Path,
        default=ROOT / "runs_siglip" / "siglip2_linear_seed2026" / "best_head.pt",
    )
    parser.add_argument("--siglip-threshold", type=float, default=0.22)
    parser.add_argument("--siglip-batch-size", type=int, default=8)
    parser.add_argument("--dino-checkpoint", type=Path, default=None)
    parser.add_argument("--dino-threshold", type=float, default=0.08)
    parser.add_argument("--dino-batch-size", type=int, default=8)
    parser.add_argument("--crop-checkpoint", type=Path, default=None)
    parser.add_argument("--crop-threshold", type=float, default=0.97)
    parser.add_argument("--crop-batch-size", type=int, default=8)
    parser.add_argument("--crop-proposal-conf", type=float, default=0.03)
    parser.add_argument("--crop-context", type=float, default=0.15)
    parser.add_argument("--crop-min-size", type=int, default=96)
    parser.add_argument("--crop-max-proposals", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path(r"C:\AI\hf_cache\hub"))
    parser.add_argument(
        "--yolo-m-weights",
        type=Path,
        default=ROOT
        / "runs"
        / "detect"
        / "runs"
        / "train"
        / "fire_yolo26m_960"
        / "weights"
        / "best.pt",
    )
    parser.add_argument(
        "--yolo-s-weights",
        type=Path,
        default=ROOT
        / "runs"
        / "detect"
        / "runs"
        / "train"
        / "fire_yolo26s_960"
        / "weights"
        / "best.pt",
    )
    parser.add_argument(
        "--yolo-s-aug-weights",
        type=Path,
        default=ROOT
        / "runs"
        / "detect"
        / "runs"
        / "train"
        / "fire_yolo26s_aug_960"
        / "weights"
        / "best.pt",
    )
    parser.add_argument("--yolo-m-conf", type=float, default=0.10)
    parser.add_argument("--yolo-s-conf", type=float, default=0.20)
    parser.add_argument("--yolo-s-aug-conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source.resolve()]
    if source.is_dir():
        paths = sorted(
            path.resolve()
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        names = [path.name for path in paths]
        if len(names) != len(set(names)):
            raise ValueError("Source contains duplicate image file names in different folders.")
        return paths
    raise FileNotFoundError(f"Unsupported image source: {source}")


def predict_siglip(
    image_paths: list[Path],
    checkpoint_path: Path,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(checkpoint["model"])
    processor = AutoProcessor.from_pretrained(
        model_name, cache_dir=str(cache_dir), local_files_only=True
    )
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=str(cache_dir), local_files_only=True
    ).eval().to(device)
    head = LinearFireHead(
        int(checkpoint["feature_dimension"]), float(checkpoint["dropout"])
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    scores: dict[str, float] = {}
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
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
                    pooled = output.pooler_output if hasattr(output, "pooler_output") else output
                    logits = head(F.normalize(pooled.float(), dim=1))
            else:
                output = backbone.get_image_features(**inputs)
                pooled = output.pooler_output if hasattr(output, "pooler_output") else output
                logits = head(F.normalize(pooled.float(), dim=1))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        scores.update(
            {path.name: float(score) for path, score in zip(batch_paths, probabilities)}
        )
        print(f"SigLIP2: {min(start + batch_size, len(image_paths))}/{len(image_paths)}")

    head.to("cpu")
    backbone.to("cpu")
    del checkpoint, processor, head, backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def predict_yolo(
    image_paths: list[Path],
    weights: Path,
    confidence: float,
    imgsz: int,
    iou: float,
    min_area: float,
    device: str,
    label: str,
) -> dict[str, int]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    predictions: dict[str, int] = {}
    for index, image_path in enumerate(image_paths, start=1):
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=confidence,
            iou=iou,
            device=device,
            verbose=False,
        )
        result = results[0]
        has_fire = False
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            height, width = result.orig_shape
            image_area = max(float(height * width), 1.0)
            for box_index in range(len(boxes)):
                if int(boxes.cls[box_index]) != 0:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[box_index].cpu().tolist()
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
                if area_ratio >= min_area:
                    has_fire = True
                    break
        predictions[image_path.name] = int(has_fire)
        if index % 25 == 0 or index == len(image_paths):
            print(f"{label}: {index}/{len(image_paths)}")

    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def predict_dinov3(
    image_paths: list[Path],
    checkpoint_path: Path,
    cache_dir: Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    os.environ["HF_HOME"] = str(cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    import timm
    from timm.data import create_transform, resolve_model_data_config

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(checkpoint["model"])
    backbone = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
        cache_dir=cache_dir,
    ).eval().to(device)
    transform = create_transform(
        **resolve_model_data_config(backbone), is_training=False
    )
    head = LinearFireHead(
        int(checkpoint["feature_dimension"]), float(checkpoint["dropout"])
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    scores: dict[str, float] = {}
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = backbone(batch)
            else:
                features = backbone(batch)
            logits = head(F.normalize(features.float(), dim=1))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        scores.update(
            {path.name: float(score) for path, score in zip(batch_paths, probabilities)}
        )
        print(f"DINOv3: {min(start + batch_size, len(image_paths))}/{len(image_paths)}")

    head.to("cpu")
    backbone.to("cpu")
    del checkpoint, transform, head, backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def box_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_proposals(
    proposals: list[dict[str, object]], max_count: int
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for proposal in sorted(
        proposals, key=lambda item: float(item["confidence"]), reverse=True
    ):
        box = tuple(float(value) for value in proposal["box"])
        if any(
            box_iou(box, tuple(float(value) for value in kept["box"])) >= 0.80
            for kept in selected
        ):
            continue
        selected.append(proposal)
        if len(selected) >= max_count:
            break
    return selected


def collect_yolo_proposals(
    image_paths: list[Path],
    model_settings: list[tuple[str, Path]],
    confidence: float,
    imgsz: int,
    iou: float,
    device: str,
    max_proposals: int,
) -> dict[str, list[dict[str, object]]]:
    from ultralytics import YOLO

    proposals = {path.name: [] for path in image_paths}
    for label, weights in model_settings:
        model = YOLO(str(weights))
        for index, image_path in enumerate(image_paths, start=1):
            results = model.predict(
                source=str(image_path),
                imgsz=imgsz,
                conf=confidence,
                iou=iou,
                device=device,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is not None:
                for box_index in range(len(boxes)):
                    if int(boxes.cls[box_index]) != 0:
                        continue
                    proposals[image_path.name].append(
                        {
                            "model": label,
                            "confidence": float(boxes.conf[box_index]),
                            "box": tuple(
                                float(value)
                                for value in boxes.xyxy[box_index].cpu().tolist()
                            ),
                        }
                    )
            if index % 50 == 0 or index == len(image_paths):
                print(f"Crop proposals {label}: {index}/{len(image_paths)}")
        model.to("cpu")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        name: deduplicate_proposals(items, max_proposals)
        for name, items in proposals.items()
    }


def expanded_crop_box(
    box: tuple[float, ...],
    width: int,
    height: int,
    context: float,
    min_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(
        box_width * (1.0 + 2.0 * context),
        box_height * (1.0 + 2.0 * context),
        float(min_size),
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


def predict_crop_siglip(
    image_paths: list[Path],
    proposals: dict[str, list[dict[str, object]]],
    checkpoint_path: Path,
    cache_dir: Path,
    batch_size: int,
    context: float,
    min_size: int,
    device: torch.device,
) -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = str(checkpoint["model"])
    processor = AutoProcessor.from_pretrained(
        model_name, cache_dir=str(cache_dir), local_files_only=True
    )
    backbone = AutoModel.from_pretrained(
        model_name, cache_dir=str(cache_dir), local_files_only=True
    ).eval().to(device)
    head = LinearFireHead(
        int(checkpoint["feature_dimension"]), float(checkpoint["dropout"])
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    path_by_name = {path.name: path for path in image_paths}
    crop_specs = [
        (name, proposal)
        for name, items in proposals.items()
        for proposal in items
    ]
    max_scores = {path.name: -1.0 for path in image_paths}
    for start in range(0, len(crop_specs), batch_size):
        batch_specs = crop_specs[start : start + batch_size]
        crops = []
        for image_name, proposal in batch_specs:
            with Image.open(path_by_name[image_name]) as opened:
                image = opened.convert("RGB")
            crop_box = expanded_crop_box(
                tuple(float(value) for value in proposal["box"]),
                image.width,
                image.height,
                context,
                min_size,
            )
            crop = image.crop(crop_box)
            buffer = io.BytesIO()
            crop.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                crops.append(encoded.convert("RGB").copy())

        inputs = processor(images=crops, return_tensors="pt")
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if key in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        }
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = backbone.get_image_features(**inputs)
                    pooled = output.pooler_output if hasattr(output, "pooler_output") else output
                    logits = head(F.normalize(pooled.float(), dim=1))
            else:
                output = backbone.get_image_features(**inputs)
                pooled = output.pooler_output if hasattr(output, "pooler_output") else output
                logits = head(F.normalize(pooled.float(), dim=1))
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
        for (image_name, _), score in zip(batch_specs, probabilities):
            max_scores[image_name] = max(max_scores[image_name], float(score))
        print(f"Crop SigLIP2: {min(start + batch_size, len(crop_specs))}/{len(crop_specs)}")

    head.to("cpu")
    backbone.to("cpu")
    del checkpoint, processor, head, backbone
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return max_scores


def main() -> None:
    args = parse_args()
    image_paths = iter_images(args.source)
    if not image_paths:
        raise ValueError(f"No supported images found in: {args.source}")

    required_files = [
        args.siglip_checkpoint,
        args.yolo_m_weights,
        args.yolo_s_weights,
        args.yolo_s_aug_weights,
    ]
    if args.dino_checkpoint is not None:
        required_files.append(args.dino_checkpoint)
    if args.crop_checkpoint is not None:
        required_files.append(args.crop_checkpoint)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    torch_device = torch.device(
        "cpu" if args.device.lower() == "cpu" else "cuda"
    )
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU.")
    print(f"Images: {len(image_paths)}")
    print(f"Device: {torch_device}")

    siglip_scores = predict_siglip(
        image_paths,
        args.siglip_checkpoint,
        args.cache_dir,
        args.siglip_batch_size,
        torch_device,
    )
    dino_scores = None
    if args.dino_checkpoint is not None:
        dino_scores = predict_dinov3(
            image_paths,
            args.dino_checkpoint,
            args.cache_dir,
            args.dino_batch_size,
            torch_device,
        )
    yolo_m = predict_yolo(
        image_paths,
        args.yolo_m_weights,
        args.yolo_m_conf,
        args.imgsz,
        args.iou,
        args.min_area,
        args.device,
        "YOLO26m",
    )
    yolo_s = predict_yolo(
        image_paths,
        args.yolo_s_weights,
        args.yolo_s_conf,
        args.imgsz,
        args.iou,
        args.min_area,
        args.device,
        "YOLO26s",
    )
    yolo_s_aug = predict_yolo(
        image_paths,
        args.yolo_s_aug_weights,
        args.yolo_s_aug_conf,
        args.imgsz,
        args.iou,
        args.min_area,
        args.device,
        "YOLO26s augmented",
    )
    crop_scores = None
    if args.crop_checkpoint is not None:
        proposals = collect_yolo_proposals(
            image_paths,
            [
                ("yolo26m", args.yolo_m_weights),
                ("yolo26s", args.yolo_s_weights),
                ("yolo26s_aug", args.yolo_s_aug_weights),
            ],
            args.crop_proposal_conf,
            args.imgsz,
            args.iou,
            args.device,
            args.crop_max_proposals,
        )
        crop_scores = predict_crop_siglip(
            image_paths,
            proposals,
            args.crop_checkpoint,
            args.cache_dir,
            args.crop_batch_size,
            args.crop_context,
            args.crop_min_size,
            torch_device,
        )

    predictions: dict[str, int] = {}
    details: list[dict[str, object]] = []
    for path in image_paths:
        name = path.name
        yolo_votes = yolo_m[name] + yolo_s[name] + yolo_s_aug[name]
        siglip_positive = siglip_scores[name] >= args.siglip_threshold
        dino_positive = (
            True if dino_scores is None else dino_scores[name] >= args.dino_threshold
        )
        base_prediction = yolo_votes >= 1 and siglip_positive and dino_positive
        crop_rescue = (
            False if crop_scores is None else crop_scores[name] >= args.crop_threshold
        )
        prediction = int(base_prediction or crop_rescue)
        predictions[name] = prediction
        details.append(
            {
                "image_name": name,
                "prediction": prediction,
                "siglip_score": f"{siglip_scores[name]:.8f}",
                "siglip_positive": int(siglip_positive),
                "dino_score": "" if dino_scores is None else f"{dino_scores[name]:.8f}",
                "dino_positive": "" if dino_scores is None else int(dino_positive),
                "crop_score": "" if crop_scores is None else f"{crop_scores[name]:.8f}",
                "crop_rescue": "" if crop_scores is None else int(crop_rescue),
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
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    if crop_scores is not None:
        rule = "base full-image fusion OR high-confidence crop rescue"
    elif dino_scores is not None:
        rule = "at least one YOLO positive AND SigLIP2 positive AND DINOv3 positive"
    else:
        rule = "at least one YOLO positive AND SigLIP2 positive"

    metadata = {
        "rule": rule,
        "image_count": len(image_paths),
        "positive_count": sum(predictions.values()),
        "negative_count": len(predictions) - sum(predictions.values()),
        "siglip_checkpoint": str(args.siglip_checkpoint.resolve()),
        "siglip_threshold": args.siglip_threshold,
        "dino": (
            None
            if args.dino_checkpoint is None
            else {
                "checkpoint": str(args.dino_checkpoint.resolve()),
                "threshold": args.dino_threshold,
            }
        ),
        "crop_rescue": (
            None
            if args.crop_checkpoint is None
            else {
                "checkpoint": str(args.crop_checkpoint.resolve()),
                "threshold": args.crop_threshold,
                "proposal_conf": args.crop_proposal_conf,
                "context": args.crop_context,
                "min_size": args.crop_min_size,
                "max_proposals": args.crop_max_proposals,
            }
        ),
        "yolo": {
            "m": {"weights": str(args.yolo_m_weights.resolve()), "conf": args.yolo_m_conf},
            "s": {"weights": str(args.yolo_s_weights.resolve()), "conf": args.yolo_s_conf},
            "s_aug": {
                "weights": str(args.yolo_s_aug_weights.resolve()),
                "conf": args.yolo_s_aug_conf,
            },
            "imgsz": args.imgsz,
            "iou": args.iou,
            "min_area": args.min_area,
        },
        "submission_file": str(args.output.resolve()),
        "details_file": str(details_path.resolve()),
    }
    metadata_path = args.output.with_name(f"{args.output.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
