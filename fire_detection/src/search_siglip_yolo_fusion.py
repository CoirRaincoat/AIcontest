from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path


ROOT = Path(r"C:\AI\fire_detection")
OUT = ROOT / "outputs"
GT_PATH = OUT / "val_gt.json"
SCORES_PATH = ROOT / "runs_siglip" / "siglip2_linear_v1" / "val_scores.csv"

MODEL_FILES = {
    "m010": OUT / "val_pred_yolo26m_conf010.json",
    "m015": OUT / "val_pred_yolo26m_conf015.json",
    "s015": OUT / "val_pred_yolo26s_conf015.json",
    "s018": OUT / "val_pred_yolo26s_conf018.json",
    "s020": OUT / "val_pred_yolo26s_conf020.json",
    "sa015": OUT / "val_pred_yolo26s_aug_conf015.json",
    "sa018": OUT / "val_pred_yolo26s_aug_conf018.json",
    "sa020": OUT / "val_pred_yolo26s_aug_conf020.json",
}


def load_scores() -> dict[str, float]:
    with SCORES_PATH.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["image_name"]: float(row["score"]) for row in csv.DictReader(handle)}


def metrics(gt: dict[str, int], pred: dict[str, int]) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for name, truth in gt.items():
        value = int(pred.get(name, 0))
        if truth == 1 and value == 1:
            tp += 1
        elif truth == 0 and value == 1:
            fp += 1
        elif truth == 1 and value == 0:
            fn += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_candidates(scores: dict[str, float]) -> list[float]:
    values = sorted(set(scores.values()))
    candidates = [0.0, 1.0]
    candidates.extend(values)
    candidates.extend((a + b) / 2.0 for a, b in zip(values, values[1:]))
    return sorted(set(candidates))


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    gt = {k: int(v) for k, v in json.loads(GT_PATH.read_text(encoding="utf-8")).items()}
    scores = load_scores()
    preds = {
        key: {k: int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
        for key, path in MODEL_FILES.items()
    }
    names = sorted(gt)
    candidates = threshold_candidates(scores)
    rows: list[dict[str, object]] = []

    for m_key, s_key, sa_key in product(
        ("m010", "m015"),
        ("s015", "s018", "s020"),
        ("sa015", "sa018", "sa020"),
    ):
        yolo_votes = {
            name: preds[m_key].get(name, 0)
            + preds[s_key].get(name, 0)
            + preds[sa_key].get(name, 0)
            for name in names
        }
        for vote_k in (1, 2, 3):
            yolo_pred = {name: int(votes >= vote_k) for name, votes in yolo_votes.items()}
            base = metrics(gt, yolo_pred)
            rows.append(
                {
                    "rule": "yolo_only",
                    "m": m_key,
                    "s": s_key,
                    "s_aug": sa_key,
                    "vote_k": vote_k,
                    "siglip_threshold": None,
                    **base,
                }
            )
            best_by_rule: dict[str, tuple[tuple[float, float, float], dict[str, object], dict[str, int], float]] = {}
            for threshold in candidates:
                siglip = {name: int(scores[name] >= threshold) for name in names}
                for rule in ("and", "or"):
                    if rule == "and":
                        fused = {name: int(siglip[name] and yolo_pred[name]) for name in names}
                    else:
                        fused = {name: int(siglip[name] or yolo_pred[name]) for name in names}
                    result = metrics(gt, fused)
                    key = (float(result["f1"]), float(result["recall"]), float(result["precision"]))
                    current = best_by_rule.get(rule)
                    if current is None or key > current[0]:
                        best_by_rule[rule] = (key, result, fused, threshold)

            for rule, (_, result, fused, threshold) in best_by_rule.items():
                pred_path = OUT / (
                    f"val_pred_search_{rule}_{m_key}_{s_key}_{sa_key}_vote{vote_k}_"
                    f"thr{threshold:.6f}.json"
                )
                save_json(pred_path, fused)
                rows.append(
                    {
                        "rule": rule,
                        "m": m_key,
                        "s": s_key,
                        "s_aug": sa_key,
                        "vote_k": vote_k,
                        "siglip_threshold": threshold,
                        "prediction_file": str(pred_path.resolve()),
                        **result,
                    }
                )

    rows.sort(key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])), reverse=True)
    top_path = OUT / "siglip_yolo_fusion_search_top20.json"
    save_json(top_path, rows[:20])
    csv_path = OUT / "siglip_yolo_fusion_search_all.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Top 20 fusion settings:")
    for i, row in enumerate(rows[:20], 1):
        print(
            f"{i:02d} rule={row['rule']} m={row['m']} s={row['s']} s_aug={row['s_aug']} "
            f"vote={row['vote_k']} siglip={row['siglip_threshold']} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}"
        )
    print(f"top -> {top_path}")
    print(f"all -> {csv_path}")


if __name__ == "__main__":
    main()
