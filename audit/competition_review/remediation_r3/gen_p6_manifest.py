"""Generate the frozen P6 12-image manifest per P6_APPROVAL_REQUEST.md §3 (selection rules frozen
before any runtime inference; part of R3 within the authorized unattended pipeline 2026-08-28).

Frozen selection rules (verbatim from the approval request):
  source = frozen fire_detection/outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv
  1) FP side: all 9 FPs, take top-4 by siglip_score DESC
  2) FN side: all 3 FNs (take all)
  3) Boundary band: |siglip-0.22|<=0.03 OR |crop-0.97|<=0.02 (crop only when crop_score>=0),
     ascending by min margin, take first 5 not already selected (distinctness enforced here)

Outputs (new registered dir remediation_r3/p6/, supersedes the old machine/p6 path which was never
created; reason documented in R3 report):
  sample_manifest.json  - frozen list + rules + provenance hashes (written BEFORE any inference)
  p6_gt.json            - official GT labels for exactly the 12 images (full-coverage eval input)

This script is deterministic and reads only; it must be re-runnable byte-identically.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FD = ROOT / "fire_detection"
SRC_CSV = FD / "outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv"
GT_JSON = FD / "data/images/train/train_image.json"
OUT_DIR = ROOT / "audit/competition_review/remediation_r3/p6"

EXPECTED_FP_TOTAL, EXPECTED_FN_TOTAL = 9, 3
N_FP, N_FN, N_BND, N_TOTAL = 4, 3, 5, 12
SIG_TAU, CROP_TAU = 0.22, 0.97
SIG_BAND, CROP_BAND = 0.03, 0.02


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rows = []
    with open(SRC_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "image": r["image_name"],
                "pred": int(r["prediction"]),
                "siglip": float(r["siglip_score"]),
                "crop": float(r["crop_score"]),
            })
    gt = {k: int(v) for k, v in json.loads(GT_JSON.read_text(encoding="utf-8")).items()}

    fps = [r for r in rows if gt[r["image"]] == 0 and r["pred"] == 1]
    fns = [r for r in rows if gt[r["image"]] == 1 and r["pred"] == 0]
    assert len(fps) == EXPECTED_FP_TOTAL, f"FP count {len(fps)} != frozen 9"
    assert len(fns) == EXPECTED_FN_TOTAL, f"FN count {len(fns)} != frozen 3"

    fp_sel = sorted(fps, key=lambda r: (-r["siglip"], r["image"]))[:N_FP]

    def margin(r):
        m = abs(r["siglip"] - SIG_TAU) if abs(r["siglip"] - SIG_TAU) <= SIG_BAND else None
        c = abs(r["crop"] - CROP_TAU) if r["crop"] >= 0 and abs(r["crop"] - CROP_TAU) <= CROP_BAND else None
        cands = [x for x in (m, c) if x is not None]
        return min(cands) if cands else None

    bnd = [r for r in rows if margin(r) is not None]
    bnd = sorted(bnd, key=lambda r: (margin(r), r["image"]))
    picked = {r["image"] for r in fp_sel} | {r["image"] for r in fns}
    bnd_sel = []
    for r in bnd:  # skip any boundary candidate already picked -> distinct 12
        if r["image"] not in picked:
            bnd_sel.append(r)
        if len(bnd_sel) == N_BND:
            break
    assert len(bnd_sel) == N_BND, f"boundary band yielded only {len(bnd_sel)}"

    sel = fp_sel + fns + bnd_sel
    names = [r["image"] for r in sel]
    assert len(set(names)) == N_TOTAL, "duplicate selection"
    assert all(n in gt for n in names), "selected image missing official GT"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "frozen": True,
        "frozen_before_inference": True,
        "source_csv": str(SRC_CSV),
        "source_csv_sha256": sha256(SRC_CSV),
        "gt_json_sha256": sha256(GT_JSON),
        "rules": [
            "FP: all 9 FPs -> top4 by siglip_score desc (tie: name asc)",
            "FN: all 3 FNs",
            "boundary: |s-0.22|<=0.03 or |crop-0.97|<=0.02 (crop>=0 only), min-margin asc, first 5 not already picked",
        ],
        "selection": [
            {"image": r["image"], "bucket": b, "pred": r["pred"], "gt": gt[r["image"]],
             "siglip_score": r["siglip"], "crop_score": r["crop"]}
            for r, b in zip(sel, ["FP"] * N_FP + ["FN"] * N_FN + ["BOUNDARY"] * N_BND)
        ],
        "fixed_params": {"imgsz": 960, "votes_min": 1, "yolo_conf": {"m": 0.10, "s": 0.20, "s_aug": 0.20},
                         "siglip_tau": SIG_TAU, "dino_tau": 0.08, "crop_tau": CROP_TAU,
                         "crop_proposal_conf": 0.03, "crop_dedupe_iou": 0.80, "crop_max": 8},
        "tolerance": "scores reproduce frozen details within +-1e-3; prediction bits must match exactly (12/12)",
    }
    (OUT_DIR / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "p6_gt.json").write_text(
        json.dumps({n: gt[n] for n in names}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK manifest 12 images:", json.dumps(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
