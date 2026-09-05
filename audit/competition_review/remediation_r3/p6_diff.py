"""P6 frozen 12-image reproduction diff vs the frozen 220-image details CSV.

Frozen tolerance (P6_APPROVAL_REQUEST.md §4):
  prediction bits must match EXACTLY (12/12); score columns within +-1e-3.
Score columns compared: siglip_score, dino_score, crop_score, and the yolo vote triplet
(yolo26m / yolo26s / yolo26s_aug). crop_score is -1.0 when crop rescue is disabled.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROZEN = Path("fire_detection/outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv")
NEW = HERE / "p6/p6_pred_details.csv"
MAN = json.loads((HERE / "p6/sample_manifest.json").read_text(encoding="utf-8"))
TOL = 1e-3
SCORE_COLS = ["siglip_score", "dino_score", "crop_score"]
YOLO_COLS = ["yolo26m", "yolo26s", "yolo26s_aug"]


def load(path):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["image_name"]] = r
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    froz = load(FROZEN)
    new = load(NEW)
    sel = [s["image"] for s in MAN["selection"]]
    rows = []
    n_bad = 0
    for name in sel:
        f, n = froz.get(name), new.get(name)
        if n is None:
            print(f"MISSING in new output: {name}")
            n_bad += 1
            continue
        row = {"image": name, "pred_frozen": int(f["prediction"]), "pred_new": int(n["prediction"])}
        diffs = {}
        for c in SCORE_COLS:
            fv, nv = num(f[c]), num(n[c])
            d = None if fv is None or nv is None else abs(fv - nv)
            diffs[c] = d
            row[f"{c}_d"] = d
        for c in YOLO_COLS:
            fv, nv = int(f[c]), int(n[c])
            diffs[c] = abs(fv - nv)
            row[f"{c}_d"] = diffs[c]
        row["pred_ok"] = row["pred_frozen"] == row["pred_new"]
        row["scores_ok"] = all(d is None or d <= TOL for d in diffs.values())
        if not (row["pred_ok"] and row["scores_ok"]):
            n_bad += 1
        rows.append(row)
        flag = "OK " if (row["pred_ok"] and row["scores_ok"]) else "BAD"
        print(f"[{flag}] {name} pred {row['pred_frozen']}->{row['pred_new']} "
              f"siglip_d={diffs['siglip_score']} dino_d={diffs['dino_score']} "
              f"crop_d={diffs['crop_score']} yolo_d={[diffs[c] for c in YOLO_COLS]}")

    n_pred = sum(1 for r in rows if r["pred_ok"])
    n_score = sum(1 for r in rows if r["scores_ok"])
    out = {"n": len(rows), "pred_bits_match": n_pred, "scores_match": n_score,
           "pred_12of12": n_pred == 12, "scores_12of12": n_score == 12, "tolerance": TOL,
           "rows": rows}
    (HERE / "p6/p6_diff.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\nPRED {n_pred}/12   SCORES {n_score}/12   tolerance +-{TOL}")
    print("P6_REPRODUCTION_OK" if (n_pred == 12 and n_score == 12) else "P6_MISMATCH_RECORDED")
    # per P6 approval: any mismatch is recorded, NOT tuned away
    return 0


if __name__ == "__main__":
    sys.exit(main())
