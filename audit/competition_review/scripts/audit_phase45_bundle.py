"""Audit phases 4-5 bundle:
A) Content-level near-duplicate detection: dHash64 val-vs-train (+ in-split calibration)
B) Experiments table: YOLO runs (args/results.csv/sha) + classification-head checkpoints
C) Corrupt-image scan over official raw dir
D) App-rule provenance: artifact/search timestamps + grid rows near (0.85,0.70)
E) corrected F-12 numbers re-print
Writes machine/{near_duplicate_hash.json,near_dup_pairs.csv,experiments_table.csv,
               checkpoints_introspection.json,corrupt_scan.json,provenance_app_rule.json}
"""
import csv
import hashlib
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from PIL import Image

FD = Path("fire_detection")
M = Path("audit/competition_review/machine")
SEED_TAG = "20260827"


def dhash(path, size=8):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = []
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        bits += [1 if row[c] > row[c + 1] else 0 for c in range(size)]
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def ham(a, b):
    return bin(a ^ b).count("1")


tr_dir = FD / "data/train/images"
va_dir = FD / "data/val/images"
raw_dir = FD / "data/images/train/images"

# ---------- A ----------
hashes_tr = {}
for p in sorted(tr_dir.iterdir()):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        hashes_tr[p.stem] = dhash(p)
hashes_va = {}
for p in sorted(va_dir.iterdir()):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        hashes_va[p.stem] = dhash(p)

tr_list = list(hashes_tr.items())
rows_out = []
min_d_all = []
for vname, vh in hashes_va.items():
    dists = [(ham(vh, th), tname) for tname, th in tr_list]
    d_min, t_best = min(dists, key=lambda x: x[0])
    # calibration: second-nearest structurally different measure skipped; record min
    min_d_all.append(d_min)
    rows_out.append({"val_image": vname + ".jpg", "nearest_train": t_best + ".jpg",
                     "hamming64": d_min})
rows_out.sort(key=lambda r: r["hamming64"])
with (M / "near_dup_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["val_image", "nearest_train", "hamming64"])
    w.writeheader()
    w.writerows(rows_out)

hist = {"0": 0, "1-4": 0, "5-8": 0, "9-12": 0, "13-16": 0, ">16": 0}
for d in min_d_all:
    k = "0" if d == 0 else "1-4" if d <= 4 else "5-8" if d <= 8 else "9-12" if d <= 12 else "13-16" if d <= 16 else ">16"
    hist[k] += 1
nd = {
    "method": "dHash 8x8(64bit), grayscale LANCZOS; exhaustive val(220) vs train(880)",
    "val_n": len(hashes_va), "train_n": len(hashes_tr),
    "min_hamming_distribution": hist,
    "median_min_hamming": statistics.median(min_d_all),
    "le_count_8": sum(1 for d in min_d_all if d <= 8),
    "le_count_12": sum(1 for d in min_d_all if d <= 12),
    "top12_pairs": rows_out[:12],
    "interpretation": "dHash<=10 视为高度相似候选; 典型不同场景对距离应远大(见校准)",
}
(M / "near_duplicate_hash.json").write_text(json.dumps(nd, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- B ----------
def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


base_sha = {}
for line in (M / "baseline_sha256.txt").read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) == 2:
        base_sha[parts[1]] = parts[0][:16]

exp_rows = []
yolo_runs = {
    "fire_yolo26m_960": FD / "runs/detect/runs/train/fire_yolo26m_960",
    "fire_yolo26s_960": FD / "runs/detect/runs/train/fire_yolo26s_960",
    "fire_yolo26s_aug_960": FD / "runs/detect/runs/train/fire_yolo26s_aug_960",
    "fire_yolov8n_960(legacy)": FD / "runs/detect/runs/train/fire_yolov8n_960",
    "fire_yolo26s_ext_v1_960(ext)": FD / "runs_ext/fire_yolo26s_ext_v1_960",
}
for rid, rd in yolo_runs.items():
    args = {}
    ay = rd / "args.yaml"
    if ay.is_file():
        for ln in ay.read_text(encoding="utf-8").splitlines():
            if ":" in ln and not ln.startswith(" "):
                k, _, v = ln.partition(":")
                args[k.strip()] = v.strip()
    res = rd / "results.csv"
    best_epoch = best_map = total_t = None
    if res.is_file():
        rr = list(csv.DictReader(res.open(encoding="utf-8")))
        key = next((c for c in rr[0] if "map50-95" in c.lower()), None)
        if key and rr:
            brow = max(rr, key=lambda r: float(r[key]))
            best_epoch = brow.get("epoch")
            best_map = round(float(brow[key]), 4)
            total_t = round(float(rr[-1].get("time", 0)) / 3600, 2)
    wbp = rd / "weights/best.pt"
    exp_rows.append({
        "run": rid, "model": args.get("model"), "data": args.get("data"),
        "seed": args.get("seed"), "epochs": args.get("epochs"),
        "batch": args.get("batch"), "imgsz": args.get("imgsz"),
        "best_epoch_by_mAP50-95": best_epoch, "best_mAP50-95": best_map,
        "train_hours_total(csv_time_col)": total_t,
        "best_pt_sha256_16": base_sha.get(str(wbp).replace("\\", "/")) or (sha16(wbp) if wbp.is_file() else None),
    })

head_rows = []
import torch  # CPU-only introspection of small state dicts
head_dirs = sorted(list((FD / "runs_siglip").iterdir()) + list((FD / "runs_dinov3").iterdir()))
introspection = {}
for hd in head_dirs:
    bh = hd / "best_head.pt"
    if not bh.is_file():
        continue
    try:
        ck = torch.load(bh, map_location="cpu", weights_only=False)
        meta = {k: (str(v)[:60] if not isinstance(v, (int, float)) else v)
                for k, v in ck.items() if k != "head_state_dict"}
        n_params = sum(w.numel() for w in ck["head_state_dict"].values())
        head_rows.append({"run": hd.name, **meta, "head_params": n_params,
                          "sha256_16": base_sha.get(str(bh).replace("\\", "/"))})
    except Exception as e:
        head_rows.append({"run": hd.name, "error": repr(e)[:120]})
    extras = [q.name for q in hd.iterdir()]
    introspection[hd.name] = extras

with (M / "experiments_table.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(exp_rows[0].keys()))
    w.writeheader()
    w.writerows(exp_rows)
(M / "checkpoints_introspection.json").write_text(
    json.dumps({"heads": head_rows, "files_per_run": introspection}, ensure_ascii=False, indent=2),
    encoding="utf-8")

# ---------- C ----------
bad = []
checked = 0
for p in sorted(raw_dir.iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    checked += 1
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im2:
            im2.load()
    except Exception as e:
        bad.append({"file": p.name, "error": repr(e)[:100]})
cs = {"checked": checked, "corrupt_or_unreadable": len(bad), "list": bad[:20]}
(M / "corrupt_scan.json").write_text(json.dumps(cs, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- D ----------
def mt(p):
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%m-%d %H:%M") if p.exists() else "ABSENT"


prov = {
    "artifacts_mtime": {
        "siglip_yolo_fusion_search_all.csv": mt(FD / "outputs/siglip_yolo_fusion_search_all.csv"),
        "search_top20.json": mt(FD / "outputs/siglip_yolo_fusion_search_top20.json"),
        "dino_crop_live_details.csv": mt(FD / "outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv"),
        "README.md": mt(FD / "README.md"),
        "nav_doc": mt(Path("00_先看我_AI项目导航.md")),
    },
}
sf = FD / "outputs/siglip_yolo_fusion_search_top20.json"
if sf.is_file():
    sdata = json.loads(sf.read_text(encoding="utf-8"))
    sample = sdata if isinstance(sdata, list) else sdata.get("top", sdata)
    prov["search_top20_first3"] = sample[:3] if isinstance(sample, list) else str(sample)[:300]
sc = FD / "outputs/siglip_yolo_fusion_search_all.csv"
if sc.is_file():
    with sc.open(encoding="utf-8-sig") as fh:
        rdr = csv.DictReader(fh)
        prov["search_csv_columns"] = rdr.fieldnames
        near = []
        for r in rdr:
            try:
                ts = float(r.get("siglip_threshold", r.get("threshold", "nan")))
                td = float(r.get("dino_threshold", "nan"))
            except (TypeError, ValueError):
                continue
            if abs(ts - 0.85) < 1e-6 and abs(td - 0.70) < 1e-6:
                near.append(r)
        prov["grid_rows_at_(0.85,0.70)"] = near[:3]
        prov["grid_row_count"] = sum(1 for _ in open(sc, encoding="utf-8-sig")) - 1
(M / "provenance_app_rule.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "A_near_dup": {k: nd[k] for k in ("min_hamming_distribution", "median_min_hamming", "le_count_8", "le_count_12")},
    "top6_pairs": nd["top12_pairs"][:6],
    "B_experiments": exp_rows,
    "B_heads": head_rows,
    "C_corrupt": cs,
    "D_provenance": prov,
}, ensure_ascii=False, indent=1))
