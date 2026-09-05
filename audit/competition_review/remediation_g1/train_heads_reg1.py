"""G1 three-head rebuild on the grouped-isolation reg1 splits (after the 3 YOLO formal runs).

Steps (each stops the pipeline on failure, logs kept):
  1. write reg1 image manifests (exact production schema:
     image_path,image_name,label,split,label_path; train=reg1-train, val=calib);
  2. self-contain a 4-line patched COPY of build_yolo_crop_dataset.py under remediation_g1/
     (ROOT -> fire_detection; 3 hardcoded YOLO weight paths -> reg1 runs) + record the diff;
     production file untouched;
  3. build crop dataset from reg1 manifests with reg1 YOLOs (all builder hyperparams default,
     = production crop_data_v2 recipe);
  4. train 3 heads with EXACT production seeds/hyperparams (siglip seed2026, dino seed42,
     crop seed2026; epochs100/patience15/lr1e-3/wd1e-4/dropout0.2/batch 8/64);
     val manifest for early stopping = calib (role-equivalent of production's val; holdout
     sealed) — caveat pre-registered: calib is also G2's diagnostic set, but the MAIN rule is
     frozen and needs no calibration, so no rule-selection circularity arises;
  5. freeze REG1_ASSETS.json (paths + sha256) consumed by G2/G6.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FD = ROOT / "fire_detection"
FD_SRC = FD / "src"
PY = sys.executable
HUB = ROOT / "hf_cache" / "hub"

YOLO_RUNS = {
    "yolo_m": HERE / "formal_runs/reg1_fire_yolo26m_960/weights/best.pt",
    "yolo_s": HERE / "formal_runs/reg1_fire_yolo26s_960/weights/best.pt",
    "yolo_s_aug": HERE / "formal_runs/reg1_fire_yolo26s_aug_960/weights/best.pt",
}
HEADS = [
    {"key": "siglip_head", "script": "train_siglip2_classifier.py",
     "model": "google/siglip2-base-patch16-384", "seed": 2026,
     "train": HERE / "manifests/train.csv", "val": HERE / "manifests/calib.csv",
     "emb": HERE / "embeddings/siglip2_base_384_reg1",
     "out": HERE / "runs_heads/siglip2_linear_reg1_seed2026"},
    {"key": "dino_head", "script": "train_dinov3_classifier.py",
     "model": "hf-hub:timm/vit_base_patch16_dinov3.lvd1689m", "seed": 42,
     "train": HERE / "manifests/train.csv", "val": HERE / "manifests/calib.csv",
     "emb": HERE / "embeddings/dinov3_vitb16_reg1",
     "out": HERE / "runs_heads/dinov3_vitb16_linear_reg1_seed42"},
    {"key": "crop_head", "script": "train_siglip2_classifier.py",
     "model": "google/siglip2-base-patch16-384", "seed": 2026,
     "train": None, "val": None,  # filled from crop_data_reg1
     "emb": HERE / "embeddings/siglip2_base_384_crop_reg1",
     "out": HERE / "runs_heads/siglip2_crop_reg1_seed2026"},
]


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(cmd, tag, extra_env=None):
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "YOLO_CONFIG_DIR": str(HERE / "ultralytics_config"),
           **(extra_env or {})}
    r = subprocess.run([PY] + cmd, capture_output=True, text=True, env=env, cwd=str(FD),
                       timeout=14400)
    (HERE / f"logs_heads_{tag}.log").write_text(
        " ".join(str(c) for c in cmd) + "\n\n" + r.stdout[-40000:] +
        "\n===STDERR===\n" + r.stderr[-20000:], encoding="utf-8")
    print(f"[{tag}] rc={r.returncode}")
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit(f"STOP::{tag} failed rc={r.returncode}")
    return r


def build_manifests():
    gt = {k: int(v) for k, v in json.loads(
        (FD / "data/images/train/train_image.json").read_text(encoding="utf-8")).items()}
    mdir = HERE / "manifests"
    mdir.mkdir(exist_ok=True)
    for split in ("train", "calib"):
        names = (HERE / "split" / f"{split}_images.txt").read_text(encoding="utf-8").split()
        rows = []
        for n in names:
            img = HERE / "dataset" / split / "images" / n
            lbl = HERE / "dataset" / split / "labels" / (Path(n).stem + ".txt")
            rows.append(f"{img},{n},{gt[n]},{split},{lbl}")
        (mdir / f"{split}.csv").write_text(
            "image_path,image_name,label,split,label_path\r\n" + "\r\n".join(rows) + "\r\n",
            encoding="utf-8-sig")
        print(f"manifest {split}: {len(rows)} rows")


def patch_crop_builder():
    tools = HERE / "tools"
    tools.mkdir(exist_ok=True)
    src = (FD_SRC / "build_yolo_crop_dataset.py").read_text(encoding="utf-8")
    dst = tools / "build_yolo_crop_dataset_reg1.py"
    diff = []
    # patch 1: ROOT (script hardcodes the dead C:\AI root); point it at the repo tree
    marker_root = 'ROOT = Path(r"C:\\AI\\fire_detection")'
    if marker_root not in src:
        sys.exit("STOP::crop builder ROOT line not found - patch aborted")
    src = src.replace(marker_root, f'ROOT = Path(r"{FD}")')
    diff.append(("ROOT", marker_root, f'ROOT = Path(r"{FD}")'))
    # patches 2-4: replace the whole model_settings dict (3 hardcoded production
    # weight blocks) with one compact literal pointing at the reg1 runs
    start = src.index("model_settings = {")
    end = src.index("\n    }\n", start) + len("\n    }\n")
    old_block = src[start:end]
    new_block = ('model_settings = {\n'
                 f'        "yolo26m": Path(r"{YOLO_RUNS["yolo_m"]}"),\n'
                 f'        "yolo26s": Path(r"{YOLO_RUNS["yolo_s"]}"),\n'
                 f'        "yolo26s_aug": Path(r"{YOLO_RUNS["yolo_s_aug"]}"),\n'
                 '    }\n')
    src = src[:start] + new_block + src[end:]
    diff.append(("model_settings", old_block.strip(), new_block.strip()))
    dst.write_text(src, encoding="utf-8")
    (tools / "PATCH_DIFF.txt").write_text(
        "Audit COPY only - production file untouched. Patches:\n" +
        "\n".join(f"## {k}\n  - {o}\n  + {n}\n" for k, o, n in diff), encoding="utf-8")
    return dst


def main():
    if not (HERE / "split/SPLIT_FREEZE.json").exists():
        sys.exit("STOP::split freeze missing")
    missing = [p for p in YOLO_RUNS.values() if not p.is_file()]
    if missing:
        sys.exit(f"STOP::reg1 YOLO weights missing (run train_reg1.py formal first): {missing}")
    build_manifests()
    builder = patch_crop_builder()
    crop_dir = HERE / "crop_data_reg1"
    if (crop_dir / "train.csv").exists() and (crop_dir / "val.csv").exists():
        print("SKIP crop_dataset (already built)")
    else:
        run([str(builder),
             "--train-manifest", str(HERE / "manifests/train.csv"),
             "--val-manifest", str(HERE / "manifests/calib.csv"),
             "--output-dir", str(crop_dir)], "crop_dataset")
    HEADS[2]["train"] = crop_dir / "train.csv"
    HEADS[2]["val"] = crop_dir / "val.csv"

    for h in HEADS:
        if (h["out"] / "best_head.pt").exists():
            print(f"SKIP head_{h['key']} (best_head.pt already exists)")
            continue
        run([str(FD_SRC / h["script"]),
             "--train-manifest", str(h["train"]), "--val-manifest", str(h["val"]),
             "--model", h["model"], "--cache-dir", str(HUB),
             "--embedding-dir", str(h["emb"]), "--output-dir", str(h["out"]),
             "--seed", str(h["seed"]), "--rebuild-embeddings"], f"head_{h['key']}")

    assets = {"splits": {s: str(HERE / "split" / f"{s}_images.txt")
                         for s in ("train", "calib", "holdout")},
              "data_yaml": str(HERE / "dataset/data_reg1.yaml"),
              "sha256": {}, "paths": {}}
    for k, p in {**YOLO_RUNS, **{h["key"]: h["out"] / "best_head.pt" for h in HEADS}}.items():
        if not p.is_file():
            sys.exit(f"STOP::expected asset missing: {p}")
        assets["paths"][k] = str(p)
        assets["sha256"][k] = sha256(p)
    (HERE / "REG1_ASSETS.json").write_text(json.dumps(assets, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("REG1_ASSETS.json frozen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
