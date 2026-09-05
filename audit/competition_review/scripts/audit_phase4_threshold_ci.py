"""Audit phase 4: threshold scans + bootstrap CIs, using ONLY frozen score columns.
No GPU, no retuning for submission; scans are labeled diagnostic.
Writes: machine/threshold_scan_siglip.csv, machine/bootstrap_ci.json,
        machine/f12_metric_caliber_demo.json, machine/f14_corrupt_scan.json
"""
import csv
import json
import random
from pathlib import Path

FD = Path("fire_detection")
M = Path("audit/competition_review/machine")
SEED = 20260827
B = 10000

gt = {k: int(v) for k, v in json.loads((FD / "outputs/val_gt.json").read_text(encoding="utf-8")).items()}
rows = list(csv.DictReader((FD / "outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv").open(encoding="utf-8-sig")))
names = [r["image_name"] for r in rows]
assert len(names) == len(gt) == 220


def m(gtp, prd):
    tp = fp = fn = tn = 0
    for n in gtp:
        t, p = gtp[n], prd[n]
        tp += t == 1 and p == 1
        fp += t == 0 and p == 1
        fn += t == 1 and p == 0
        tn += t == 0 and p == 0
    P = tp / (tp + fp) if tp + fp else None
    R = tp / (tp + fn) if tp + fn else None
    F = 2 * P * R / (P + R) if (P and R and P + R) else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "P": round(P, 4) if P is not None else None,
            "R": round(R, 4) if R is not None else None,
            "F1": round(F, 4) if F is not None else None}


def m_intersection_caliber(gtp, prd):
    """Replicates src/evaluate_image_level.py L29-43 exactly (common-keys basis)."""
    common = set(gtp) & set(prd)
    gt_c = {n: gtp[n] for n in common}
    pr_c = {n: prd[n] for n in common}
    tp = sum(1 for n in common if gt_c[n] == 1 and pr_c[n] == 1)
    fp = sum(1 for n in common if gt_c[n] == 0 and pr_c[n] == 1)
    fn = sum(1 for n in common if gt_c[n] == 1 and pr_c[n] == 0)
    tn = sum(1 for n in common if gt_c[n] == 0 and pr_c[n] == 0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"n_evaluated": len(common),
            "missing_predictions_silently_excluded": len(set(gt) - set(prd)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "P": round(P, 4), "R": round(R, 4), "F1": round(F, 4)}


pred = {}
for r in rows:
    votes = int(r["yolo_votes"])
    s = float(r["siglip_score"])
    d = float(r["dino_score"])
    c = float(r["crop_score"]) if r["crop_score"] != "" else -1.0
    pred[r["image_name"]] = {"votes": votes, "s": s, "d": d, "c": c}

rule_final = lambda x, ts=0.22, td=0.08, tc=0.97: int((x["votes"] >= 1 and x["s"] >= ts and x["d"] >= td) or x["c"] >= tc)
rule_base = lambda x, ts=0.22, td=0.08: int(x["votes"] >= 1 and x["s"] >= ts and x["d"] >= td)
rule_app = lambda x, ts=0.22, td=0.08, tc=0.97: int(rule_final(x, ts, td, tc) or (x["s"] >= 0.85 and x["d"] >= 0.70))

# ---------- sanity: CSV-reconstructed rules reproduce frozen JSONs ----------
live_json = {k: int(v) for k, v in json.loads((FD / "outputs/val_pred_best_siglip_yolo_dino_crop_live.json").read_text(encoding="utf-8")).items()}
recon_final = {n: rule_final(pred[n]) for n in names}
recon_app = {n: rule_app(pred[n]) for n in names}
sanity = {"csv_recon_equals_S4_json": recon_final == live_json}

fixed = {
    "S3_097=S4_live @documented(0.22,0.08,0.97)": {"counts": m(gt, {n: recon_final[n] for n in names}), "labels": "leak-caveat", },
    "app_rule(+) @documented(+0.85,0.70)": {"counts": m(gt, recon_app)},
    "base_only(0.22,0.08,nocrop)": {"counts": m(gt, {n: rule_base(pred[n]) for n in names})},
}
# fill leak caveat flag
for v in fixed.values():
    v["leak_caveat"] = "VAL_INCLUDES_NEIGHBOR_FRAMES_OF_TRAIN -> optimistic (F-11)"

# ---------- 1-D diagnostic sweeps ----------
taus = sorted({round(v, 4) for x in pred.values() for v in [x["s"]]} | {i / 100 for i in range(101)})
scan_rows = []
best = None
for ts in taus:
    pd_ = {n: rule_final(pred[n], ts) for n in names}
    mm = m(gt, pd_)
    rec = {"tau_siglip": ts, "rule": "final(rescue>=0.97)", **mm}
    rec["is_current"] = abs(ts - 0.22) < 1e-9
    scan_rows.append(rec)
    if mm["F1"] is not None:
        if best is None or mm["F1"] > best["F1"]:
            best = rec
# annotate max-F1 and leaning points
if scan_rows:
    maxf1 = max(r["F1"] for r in scan_rows if r["F1"] is not None)
    ties = [r for r in scan_rows if r["F1"] == maxf1]
    bidx = min(range(len(scan_rows)), key=lambda i: (scan_rows[i]["F1"] is None, -(scan_rows[i]["F1"] or 0)))
    best["is_max_f1"] = True
    # recall-leaning: R>=0.99 max P
    cand_r = [r for r in scan_rows if r["R"] is not None and r["R"] >= 0.99]
    rec_point = max(cand_r, key=lambda r: r["P"]) if cand_r else None
    if rec_point:
        rec_point["is_recall_leaning_Rge0.99maxP"] = True
    # precision-leaning: P>=0.96 max R
    cand_p = [r for r in scan_rows if r["P"] is not None and r["P"] >= 0.96]
    prec_point = max(cand_p, key=lambda r: r["R"]) if cand_p else None
    if prec_point:
        prec_point["is_precision_leaning_Pge0.96maxR"] = True

field_order = []
for _r in scan_rows:
    for k in _r.keys():
        if k not in field_order:
            field_order.append(k)
with (M / "threshold_scan_siglip.csv").open("w", newline="", encoding="utf-8") as fh:
    wr = csv.DictWriter(fh, fieldnames=field_order)
    wr.writeheader()
    wr.writerows(scan_rows)

# secondary coarse scans
dino_scan = []
for td in [i / 100 for i in range(0, 61, 2)]:
    dino_scan.append({"tau_dino": td, **m(gt, {n: int((pred[n]["votes"] >= 1 and pred[n]["s"] >= 0.22 and pred[n]["d"] >= td) or pred[n]["c"] >= 0.97) for n in names})})
crop_scan = []
for tc in [0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
    crop_scan.append({"tau_crop": tc, **m(gt, {n: rule_final(pred[n], 0.22, 0.08, tc) for n in names})})

# ---------- bootstrap ----------
rng = random.Random(SEED)


def boot(indices):
    gt_b = {names[i]: gt[names[i]] for i in indices}
    prd_b = {names[i]: recon_app[names[i]] for i in indices}
    return m(gt_b, prd_b)


Pb, Rb = [], []
degen = 0
for _ in range(B):
    idx = [rng.randrange(len(names)) for _ in range(len(names))]
    r = boot(idx)
    if r["P"] is None or r["R"] is None:
        degen += 1
        continue
    Pb.append(r["P"]); Rb.append(r["R"])


def pct(a, q):
    a = sorted(a); k = (len(a) - 1) * q
    f, c = int(k), min(int(k) + 1, len(a) - 1)
    return a[f] + (a[c] - a[f]) * (k - f)


img_ci = {
    "method": "percentile bootstrap, image-level iid resample",
    "B": B, "rng_seed": SEED, "excluded_degenerate": degen,
    "rule": "documented app rule (@0.22/0.08/0.97 + 0.85/0.70)",
    "P_ci95": [round(pct(Pb, 0.025), 4), round(pct(Pb, 0.975), 4)],
    "R_ci95": [round(pct(Rb, 0.025), 4), round(pct(Rb, 0.975), 4)],
}

import re
SEQ_RE = re.compile(r"^(?P<base>.+?)[-_](?P<seq>\d{2,})$")
family = {}
for n in names:
    mo = SEQ_RE.match(Path(n).stem)
    fam = mo.group("base") if mo else Path(n).stem
    family.setdefault(fam, []).append(n)
fams = list(family)
gfam = {f: sum(gt[x] for x in family[f]) for f in fams}
Pc, Rc = [], []
deg_c = 0
for _ in range(B):
    pick = [rng.choice(fams) for _ in range(len(fams))]
    idx = [names.index(x) for f in pick for x in family[f]]
    r = boot(idx)
    if r["P"] is None or r["R"] is None:
        deg_c += 1
        continue
    Pc.append(r["P"]); Rc.append(r["R"])
cluster_ci = {
    "method": "cluster bootstrap over 5 heuristic sequence families (resample families w/ replacement)",
    "families": {f: {"n": len(family[f]), "pos": gfam[f]} for f in fams},
    "B": B, "excluded_degenerate": deg_c,
    "caveat": "仅5簇=>分布不稳定,区间仅供参考; 族分配为命名启发式,已抽样目检确认近邻同源",
    "P_ci95": [round(pct(Pc, 0.025), 4), round(pct(Pc, 0.975), 4)] if Pc else None,
    "R_ci95": [round(pct(Rc, 0.025), 4), round(pct(Rc, 0.975), 4)] if Rc else None,
}

(M / "bootstrap_ci.json").write_text(json.dumps({
    "sanity_csv_reconstruction": sanity, "fixed_threshold_results": fixed,
    "image_iid_bootstrap": img_ci, "family_cluster_bootstrap": cluster_ci,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- F-12 demo: intersection caliber inflation ----------
k_drop = 5
rng2 = random.Random(SEED)
drops = rng2.sample(range(len(names)), k_drop)
drop_set = {names[i] for i in drops}
sub_gt = dict(gt)                    # ground truth still has all 220
sub_pr = {n: recon_app[n] for n in names if n not in drop_set}  # pred file MISSING exactly the 5
inter_basis = m_intersection_caliber(sub_gt, sub_pr)
with_fn = m(sub_gt, {n: recon_app[n] for n in sub_gt})  # all remaining images counted incl. FN
f12 = {
    "dropped_images": [names[i] for i in drops],
    "dropped_truth_labels": [gt[names[i]] for i in drops],
    "intersection_caliber(evaluate_image_level.py L29-43 replica)": inter_basis,
    "proper_full_coverage_basis(FN counted, no silent drop)": with_fn,
    "note_strict_variant": "build_cached_crop_rescue.py L63 用 prd[name] 直接下标 -> 缺键直接 KeyError 崩溃(另一极端, 非虚高)",
    "verdict": "缺图时交集口径同时抬高P与R(漏检真阳不计FN) — 本现存产物均为完整覆盖故未实际发生,风险属提交期操作面",
}
(M / "f12_metric_caliber_demo.json").write_text(json.dumps(f12, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({"sanity": sanity,
                  "fixed": {k: v["counts"] for k, v in fixed.items()},
                  "scan_points_total": len(scan_rows),
                  "max_f1_tau": best.get("tau_siglip"), "max_f1": best.get("F1"),
                  "recall_leaning": rec_point, "precision_leaning": prec_point,
                  "dino_scan_len": len(dino_scan), "crop_scan_len": len(crop_scan),
                  "img_ci": img_ci, "cluster_ci": {k: v for k, v in cluster_ci.items() if k != "families"},
                  "f12": f12}, ensure_ascii=False, indent=1))
