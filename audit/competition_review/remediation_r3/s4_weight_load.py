"""S4 weight-load test (no inference) — mirrors production loading paths exactly.

Loads (all offline, HF_HUB_OFFLINE=1):
  1. three production YOLO best.pt
  2. SigLIP2 backbone (AutoProcessor + AutoModel) + siglip linear head
  3. DINOv3 backbone (timm create_model) + dino linear head
  4. crop SigLIP2 backbone + crop head
All six weights must load; any failure is a hard STOP (env/backbone-cache mismatch).
"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

ROOT = Path(__file__).resolve().parents[3]
FD = ROOT / "fire_detection"
FD_SRC = FD / "src"
HUB = ROOT / "hf_cache" / "hub"
sys.path.insert(0, str(FD_SRC))

import torch  # noqa
from transformers import AutoModel, AutoProcessor  # noqa
from predict_best_fusion import LinearFireHead  # noqa

os.environ["HF_HOME"] = str(HUB.parent)
os.environ["HF_HUB_CACHE"] = str(HUB)

W = {
    "yolo_m": FD / "runs/detect/runs/train/fire_yolo26m_960/weights/best.pt",
    "yolo_s": FD / "runs/detect/runs/train/fire_yolo26s_960/weights/best.pt",
    "yolo_s_aug": FD / "runs/detect/runs/train/fire_yolo26s_aug_960/weights/best.pt",
    "siglip_head": FD / "runs_siglip/siglip2_linear_seed2026/best_head.pt",
    "dino_head": FD / "runs_dinov3/dinov3_vitb16_linear_seed42/best_head.pt",
    "crop_head": FD / "runs_siglip/siglip2_crop_v2_seed2026/best_head.pt",
}

results = {}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yolo(p):
    from ultralytics import YOLO
    m = YOLO(str(p))
    m.to("cpu")
    del m
    return "ok"


def load_siglip_head(p):
    ck = torch.load(p, map_location=dev, weights_only=False)
    model_name = str(ck["model"])
    proc = AutoProcessor.from_pretrained(model_name, cache_dir=str(HUB), local_files_only=True)
    bb = AutoModel.from_pretrained(model_name, cache_dir=str(HUB), local_files_only=True).eval().to(dev)
    head = LinearFireHead(int(ck["feature_dimension"]), float(ck["dropout"])).to(dev)
    head.load_state_dict(ck["head_state_dict"])
    return model_name, int(ck["feature_dimension"]), int(ck["seed"])


def load_dino_head(p):
    import timm
    from timm.data import create_transform, resolve_model_data_config
    ck = torch.load(p, map_location=dev, weights_only=False)
    model_name = str(ck["model"])
    bb = timm.create_model(model_name, pretrained=True, num_classes=0, cache_dir=HUB).eval().to(dev)
    _ = create_transform(**resolve_model_data_config(bb), is_training=False)
    head = LinearFireHead(int(ck["feature_dimension"]), float(ck["dropout"])).to(dev)
    head.load_state_dict(ck["head_state_dict"])
    return model_name, int(ck["feature_dimension"]), int(ck["seed"])


try:
    for k in ("yolo_m", "yolo_s", "yolo_s_aug"):
        results[k] = load_yolo(W[k])
    results["siglip"] = load_siglip_head(W["siglip_head"])
    results["dino"] = load_dino_head(W["dino_head"])
    results["crop"] = load_siglip_head(W["crop_head"])
    results["device"] = str(dev)
    results["vram_used_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == "cuda" else None
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("S4_ALL_LOADS_OK")
    sys.exit(0)
except Exception as e:
    print(f"S4_LOAD_FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(2)
