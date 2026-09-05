"""Audit follow-ups:
A) F-04: is raw_fire_relabel_dp_20402(1).jpg a byte-duplicate of raw_fire_relabel_dp_20402.jpg?
B) F-05: reproduce documented "web-app full-image rescue" rule at its RECORDED thresholds
   (SigLIP2>=0.85 AND DINOv3>=0.70) using ONLY existing score columns in
   outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv. No threshold search.
Writes audit/competition_review/machine/followup_f04_f05.json
"""
import csv
import hashlib
import json
from pathlib import Path

FD = Path("fire_detection")
M = Path("audit/competition_review/machine")
out = {}

def sha_full(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()

dup_a = FD / "data/images/train/images/raw_fire_relabel_dp_20402(1).jpg"
dup_b = FD / "data/images/train/images/raw_fire_relabel_dp_20402.jpg"
sa, sb = sha_full(dup_a), sha_full(dup_b)
out["f04"] = {
    "extra_file": dup_a.name,
    "canonical_file": dup_b.name,
    "sha256_extra": sa,
    "sha256_canonical": sb,
    "byte_identical": sa == sb,
    "bytes": [dup_a.stat().st_size, dup_b.stat().st_size],
}

# ---- B ----
gt = {k: int(v) for k, v in json.loads(
    (FD / "outputs/val_gt.json").read_text(encoding="utf-8")).items()}
pred_live = {k: int(v) for k, v in json.loads(
    (FD / "outputs/val_pred_best_siglip_yolo_dino_crop_live.json").read_text(encoding="utf-8")).items()}


def metrics(gt_d, pred_d):
    tp = sum(1 for n in gt_d if gt_d[n] == 1 and pred_d[n] == 1)
    fp = sum(1 for n in gt_d if gt_d[n] == 0 and pred_d[n] == 1)
    fn = sum(1 for n in gt_d if gt_d[n] == 1 and pred_d[n] == 0)
    tn = sum(1 for n in gt_d if gt_d[n] == 0 and pred_d[n] == 0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4)}


rows = list(csv.DictReader((FD / "outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv").open(encoding="utf-8-sig")))
changed_full_rescue = []
pred_app = {}
for r in rows:
    name = r["image_name"]
    base = int(r["prediction"])
    full_rescue = float(r["siglip_score"]) >= 0.85 and float(r["dino_score"]) >= 0.70
    p = 1 if (base or full_rescue) else 0
    pred_app[name] = p
    if p != base:
        changed_full_rescue.append({
            "image_name": name, "truth": gt.get(name),
            "base_pred": base, "new_pred": p,
            "siglip_score": float(r["siglip_score"]), "dino_score": float(r["dino_score"]),
        })

out["f05_app_rule_at_documented_thresholds"] = {
    "rule": "S4_live(prediction=base OR crop>=0.97) OR (SigLIP2>=0.85 AND DINOv3>=0.70)",
    "n_images": len(pred_app),
    "metrics": metrics(gt, pred_app),
    "doc_claim": {"precision": 0.9480, "recall": 0.9939},
    "matches_doc_claim": None,
    "changed_count": len(changed_full_rescue),
    "changed_detail": changed_full_rescue,
}
ma = out["f05_app_rule_at_documented_thresholds"]["metrics"]
out["f05_app_rule_at_documented_thresholds"]["matches_doc_claim"] = bool(
    abs(ma["precision"] - 0.9480) < 5e-4 and abs(ma["recall"] - 0.9939) < 5e-4)

# also verify live JSON == recomputed rule from CSV without full-rescue (sanity)
san = {}
for r in rows:
    san[r["image_name"]] = int(r["prediction"])
out["f05_sanity_live_csv_equals_json"] = san == pred_live

(M / "followup_f04_f05.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

brief = dict(out)
brief["f05_app_rule_at_documented_thresholds"] = dict(brief["f05_app_rule_at_documented_thresholds"])
print(json.dumps(brief, ensure_ascii=False, indent=1))
