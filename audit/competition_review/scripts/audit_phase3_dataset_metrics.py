"""Audit phase 3: dataset/annotation checks + independent metrics recomputation.
Read-only over production data; writes only under audit/competition_review/machine/.
"""
import json
import hashlib
import csv
import statistics
from pathlib import Path
from collections import Counter

FD = Path("fire_detection")
M = Path("audit/competition_review/machine")
M.mkdir(parents=True, exist_ok=True)
report = {}

# ---------- 1. official annotations ----------
coco = json.loads((FD / "data/images/train/train_coco.json").read_text(encoding="utf-8"))
img_json = json.loads((FD / "data/images/train/train_image.json").read_text(encoding="utf-8"))
raw_dir = FD / "data/images/train/images"
raw_files = sorted(p.name for p in raw_dir.iterdir() if p.is_file())
raw_exts = Counter(Path(f).suffix.lower() for f in raw_files)

coco_imgs = coco.get("images", [])
coco_anns = coco.get("annotations", [])
cats = coco.get("categories", [])
info_keys = [k for k in coco.keys() if k not in ("images", "annotations", "categories")]
coco_names = [im.get("file_name") for im in coco_imgs]
dup_coco_names = sorted(n for n, c in Counter(coco_names).items() if c > 1)
img_ids = [im.get("id") for im in coco_imgs]
dup_ids = len(img_ids) - len(set(img_ids))
ann_imgids = {a.get("image_id") for a in coco_anns}
orphan_anns = ann_imgids - set(img_ids)

size_by_id = {im["id"]: (im.get("width"), im.get("height")) for im in coco_imgs}
bbox_bad = 0
oob = 0
wh_bad = 0
per_img_cnt = Counter()
cat_dist = Counter()
areas = []
for a in coco_anns:
    x, y, w, h = a.get("bbox", [None] * 4)
    if None in (x, y, w, h):
        bbox_bad += 1
        continue
    if w <= 0 or h <= 0:
        wh_bad += 1
        continue
    W, H = size_by_id.get(a.get("image_id"), (None, None))
    if W is not None and (x < -1e-6 or y < -1e-6 or x + w > W + 2 or y + h > H + 2):
        oob += 1
    per_img_cnt[a.get("image_id")] += 1
    cat_dist[a.get("category_id")] += 1
    areas.append(w * h)

cats_simple = [{"id": c.get("id"), "name": c.get("name")} for c in cats]
empty_label_images_count = sum(1 for im in coco_imgs if per_img_cnt[im["id"]] == 0)
empty_label_sample = [im["file_name"] for im in coco_imgs if per_img_cnt[im["id"]] == 0][:8]

val_types = Counter(type(v).__name__ for v in img_json.values())
bad_vals = {k: v for k, v in img_json.items() if v not in (0, 1)}
pos_total = sum(1 for v in img_json.values() if v == 1)

# ---------- 2. F-04 ----------
set_raw, set_coco, set_lbl = set(raw_files), set(coco_names), set(img_json.keys())
extra_raw_vs_coco = sorted(set_raw - set_coco)
missing_on_disk = sorted(set_coco - set_raw)


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


extras_detail = []
for f in extra_raw_vs_coco:
    p = raw_dir / f
    extras_detail.append({"file": f, "bytes": p.stat().st_size, "sha256_16": sha16(p)})

lower_raw = {}
for f in raw_files:
    lower_raw.setdefault(f.lower(), []).append(f)
case_collide = {k: v for k, v in lower_raw.items() if len(v) > 1}

lower_lbl = {}
for k in set_lbl:
    lower_lbl.setdefault(k.lower(), []).append(k)
case_collide_lbl = {k: v for k, v in lower_lbl.items() if len(v) > 1}

# ---------- 3. splits ----------
tr_i = {p.name for p in (FD / "data/train/images").iterdir() if p.is_file()}
va_i = {p.name for p in (FD / "data/val/images").iterdir() if p.is_file()}
va_l_names = sorted(p.name for p in (FD / "data/val/labels").iterdir())
inter_tr_va = sorted(tr_i & va_i)
union_sv = tr_i | va_i
split_not_in_coco = sorted(union_sv - set_coco)
coco_not_in_splits = sorted(set_coco - union_sv)

gt = json.loads((FD / "outputs/val_gt.json").read_text(encoding="utf-8"))
mismatch_gt = {k: [v, img_json.get(k)] for k, v in gt.items() if img_json.get(k) != v}
val_gt_keys_not_official = sorted(set(gt) - set_lbl)

# image-level consistency vs detection boxes
name_by_id = {im["id"]: im["file_name"] for im in coco_imgs}
boxed_pos_violations = []   # has box but label==0
label1_no_box = []          # label==1 but zero boxes (weak-supervision signal only)
for im in coco_imgs:
    lblv = img_json.get(im["file_name"])
    cnt = per_img_cnt[im["id"]]
    if cnt > 0 and lblv == 0:
        boxed_pos_violations.append(im["file_name"])
    if lblv == 1 and cnt == 0:
        label1_no_box.append(im["file_name"])

# label txt <-> image pairing counts
pair_ok = sum(1 for n in tr_i if (FD / "data/train/labels" / (Path(n).stem + ".txt")).is_file())
pair_ok_va = sum(1 for n in va_i if (FD / "data/val/labels" / (Path(n).stem + ".txt")).is_file())

# ---------- 4. metrics recompute ----------
def pr(gt_d, pred_d):
    common = set(gt_d) & set(pred_d)
    tp = sum(1 for n in common if gt_d[n] == 1 and pred_d[n] == 1)
    fp = sum(1 for n in common if gt_d[n] == 0 and pred_d[n] == 1)
    fn = sum(1 for n in common if gt_d[n] == 1 and pred_d[n] == 0)
    tn = sum(1 for n in common if gt_d[n] == 0 and pred_d[n] == 0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {
        "n_common": len(common),
        "missing_from_pred": len(set(gt_d) - set(pred_d)),
        "extra_keys": len(set(pred_d) - set(gt_d)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
    }


gtd = {k: int(v) for k, v in gt.items()}
artifacts = [
    ("outputs/val_pred_best_seed2026_yolo_fusion_thr022.json", "S1_yolo_siglip_thr022"),
    ("outputs/val_pred_best_siglip_yolo_dinov3_veto_thr008.json", "S2_dino_veto_thr008"),
    ("outputs/val_pred_best_siglip_yolo_dino_crop_rescue_thr095.json", "S3_crop_rescue_095"),
    ("outputs/val_pred_best_siglip_yolo_dino_crop_rescue_thr097.json", "S3_crop_rescue_097"),
    ("outputs/val_pred_best_siglip_yolo_dino_crop_live.json", "S4_live_final"),
    ("outputs/val_pred_best_seed2026_yolo_fusion_live.json", "S5_fusion_live_nodino"),
]
recomputed = []
for rel, lab in artifacts:
    try:
        pred = {k: int(v) for k, v in json.loads((FD / rel).read_text(encoding="utf-8")).items()}
        r = pr(gtd, pred)
        r["artifact"] = lab
        mrel = rel.replace(".json", "_metrics.json")
        if (FD / mrel).is_file():
            sm = json.loads((FD / mrel).read_text(encoding="utf-8"))
            stored = sm.get("metrics", sm)
            r["stored_metrics"] = stored
            if isinstance(stored, dict) and all(k in stored for k in ("tp", "fp", "fn", "tn")):
                r["match_stored"] = all(stored[k] == r[k] for k in ("tp", "fp", "fn", "tn"))
                r["stored_precision_recall"] = [stored.get("precision"), stored.get("recall")]
        recomputed.append(r)
    except Exception as exc:  # noqa: BLE001 - audit log
        recomputed.append({"artifact": lab, "error": str(exc)})

det_csv = FD / "outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv"
cols = []
with det_csv.open(encoding="utf-8-sig") as fh:
    cols = csv.DictReader(fh).fieldnames

report.update({
    "coco": {
        "top_level_keys": list(coco.keys()), "other_keys": info_keys,
        "n_images": len(coco_imgs), "n_anns": len(coco_anns), "categories": cats_simple,
        "dup_file_names": dup_coco_names, "dup_image_ids": dup_ids,
        "orphan_ann_image_ids": sorted(orphan_anns)[:10],
        "bbox_missing_fields": bbox_bad, "wh_nonpositive": wh_bad, "bbox_oob_tol2px": oob,
        "images_with_zero_boxes": empty_label_images_count, "zero_box_sample": empty_label_sample,
        "cat_distribution": dict(cat_dist),
        "box_area_median_px": round(statistics.median(areas)) if areas else None,
    },
    "image_label_json": {"n": len(img_json), "value_types": dict(val_types),
                         "non_binary_values": bad_vals, "positive_count": pos_total},
    "raw_images": {"count": len(raw_files), "ext": dict(raw_exts)},
    "f04_extra_raw_vs_coco": extras_detail,
    "f04_coco_missing_on_disk": missing_on_disk[:20],
    "sets": {
        "raw_minus_imagejson": len(set_raw - set_lbl),
        "imagejson_minus_raw": sorted(set_lbl - set_raw)[:10],
        "imagejson_minus_coco": sorted(set_lbl - set_coco)[:10],
        "coco_minus_imagejson": sorted(set_coco - set_lbl)[:10],
        "case_collisions_raw": case_collide, "case_collisions_labels": case_collide_lbl,
    },
    "consistency": {
        "has_box_but_label0": boxed_pos_violations[:30],
        "has_box_but_label0_count": len(boxed_pos_violations),
        "label1_zero_box_count": len(label1_no_box),
        "label1_zero_box_sample": label1_no_box[:10],
    },
    "splits": {
        "train_n": len(tr_i), "val_n": len(va_i),
        "train_val_intersection": inter_tr_va,
        "split_union_not_in_official": split_not_in_coco[:10],
        "official_not_used_in_splits": coco_not_in_splits[:10],
        "train_pairing_ok": pair_ok, "val_pairing_ok": pair_ok_va,
        "val_label_ext_sample": va_l_names[:3],
    },
    "val_gt": {"n": len(gt), "pos": sum(gt.values()),
               "mismatch_vs_official": mismatch_gt,
               "keys_not_official": val_gt_keys_not_official},
    "metrics_recompute": recomputed,
    "live_details_columns": cols,
})
(M / "dataset_and_metrics_audit.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

brief = {
    "k_coco": report["coco"],
    "labels": report["image_label_json"],
    "f04_extras": extras_detail,
    "f04_missing_on_disk": report["f04_coco_missing_on_disk"],
    "sets_note": {k: v for k, v in report["sets"].items() if "sample" not in k and "minus_raw" != k},
    "consistency": report["consistency"],
    "splits": {k: v for k, v in report["splits"].items()},
    "val_gt": report["val_gt"],
    "metrics_recompute": recomputed,
    "detail_cols": cols,
}
print(json.dumps(brief, ensure_ascii=False, indent=1))
