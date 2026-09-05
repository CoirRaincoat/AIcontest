"""G2 evaluation driver: calibration panel + one-shot holdout open (leak-free discipline).

Subcommands
  chain  --split calib|holdout   run the reg1 production chain on a split via junction view,
                                 then the G3 evaluator against that split's official GT.
  rules  --split calib           diagnostic rule panel from the calib details CSV
                                 (main rule as-executed + pre-registered secondaries).
                                 Writes RULE_PANEL_CALIB.json.
  freeze                          asset+rule freeze gate -> RULE_PANEL_FROZEN.json (must exist
                                 and match REG1_ASSETS sha256s before holdout may open).
  holdout                        ONE-SHOT: refuses if holdout/pred.json already exists;
                                 writes HOLDOUT_OPENED.lock, runs chain, evaluates MAIN rule
                                 (= frozen production rule) + pre-registered secondaries,
                                 bootstrap CI (B=10000, seed=20260828).

Rules of engagement (mandate §7): no F1-based auto threshold selection; the MAIN rule is the
pre-evidenced production fusion rule frozen BEFORE holdout; the app-style rule (0.85/0.70) is
F-15 exploratory only; calib may show curves/Pareto/diagnostic F1 but never "official best".
"""
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
G1 = ROOT / "audit/competition_review/remediation_g1"
R3 = ROOT / "audit/competition_review/remediation_r3"
FD_SRC = ROOT / "fire_detection" / "src"
PY = sys.executable

MAIN_RULE = {"name": "production_fusion_rule_FROZEN", "siglip_tau": 0.22, "dino_tau": 0.08,
             "crop_tau": 0.97, "votes_min": 1,
             "structure": "votes>=1 & s>=tau_s & d>=tau_d, OR crop>=tau_c",
             "provenance": "E014/E002 + P6_APPROVAL_REQUEST.md; unchanged by this pipeline"}
SECONDARY_RULES = [
    {"name": "F15_app_fullimage_rule_EXPLORATORY", "full_image": {"siglip": 0.85, "dino": 0.70},
     "status": "pre-registered exploratory only; never presented as pre-registered main"},
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def assets():
    return json.loads((G1 / "REG1_ASSETS.json").read_text(encoding="utf-8"))


def gt_for(split):
    gt_all = {k: int(v) for k, v in json.loads(
        (ROOT / "fire_detection/data/images/train/train_image.json").read_text(encoding="utf-8")).items()}
    names = (G1 / "split" / f"{split}_images.txt").read_text(encoding="utf-8").split()
    return {n: gt_all[n] for n in names}


def stage_split_images(split):
    """Hardlink ONLY this split's images into a flat staging dir (zero extra disk),
    so the production entry processes exactly that split (not the full 1100)."""
    names = (G1 / "split" / f"{split}_images.txt").read_text(encoding="utf-8").split()
    src_dir = ROOT / "fire_detection/data/images/train/images"
    staged = HERE / split / "staged_images"
    staged.mkdir(parents=True, exist_ok=True)
    import os as _os
    for n in names:
        t = staged / n
        if t.exists():
            continue
        try:
            _os.link(src_dir / n, t)
        except OSError:
            import shutil
            shutil.copy2(src_dir / n, t)
    return staged


def run_chain(split):
    a = assets()
    outdir = HERE / split
    outdir.mkdir(exist_ok=True)
    staged = stage_split_images(split)
    P = a["paths"]
    cmd = [PY, str(FD_SRC / "predict_best_fusion.py"),
           "--source", str(staged),
           "--output", str(outdir / "pred.json"),
           "--siglip-checkpoint", str(P["siglip_head"]),
           "--dino-checkpoint", str(P["dino_head"]),
           "--crop-checkpoint", str(P["crop_head"]),
           "--yolo-m-weights", str(P["yolo_m"]),
           "--yolo-s-weights", str(P["yolo_s"]),
           "--yolo-s-aug-weights", str(P["yolo_s_aug"]),
           "--cache-dir", str(ROOT / "hf_cache/hub")]
    env = {**__import__("os").environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=7200)
    (outdir / "chain_stdout.log").write_text(r.stdout[-30000:] + "\n===STDERR===\n" + r.stderr[-30000:],
                                             encoding="utf-8")
    if r.returncode != 0:
        print(f"CHAIN_FAILED rc={r.returncode} split={split} (see chain_stdout.log)")
        return 3
    gtj = outdir / "gt.json"
    gtj.write_text(json.dumps(gt_for(split), ensure_ascii=False, indent=2), encoding="utf-8")
    ev = subprocess.run([PY, str(FD_SRC / "evaluate_image_level.py"),
                         "--gt", str(gtj), "--pred", str(outdir / "pred.json")],
                        capture_output=True, text=True)
    (outdir / "evaluate_output.txt").write_text(ev.stdout + ev.stderr, encoding="utf-8")
    print(ev.stdout)
    print(f"EVALUATOR_RC={ev.returncode}")
    return ev.returncode


def predict_row(row, rule):
    votes = int(row["yolo26m"]) + int(row["yolo26s"]) + int(row["yolo26s_aug"])
    s = float(row["siglip_score"])
    d = float(row["dino_score"]) if row["dino_score"] != "" else None
    c = float(row["crop_score"]) if row["crop_score"] != "" else None
    base = votes >= rule["votes_min"] and s >= rule["siglip_tau"] and (
        d is None or d >= rule["dino_tau"])
    rescue = c is not None and c >= rule["crop_tau"]
    return int(base or rescue)


def metrics(gt, pred):
    tp = sum(1 for k in gt if gt[k] == 1 and pred.get(k) == 1)
    fp = sum(1 for k in gt if gt[k] == 0 and pred.get(k) == 1)
    fn = sum(1 for k in gt if gt[k] == 1 and pred.get(k) != 1)
    tn = sum(1 for k in gt if gt[k] == 0 and pred.get(k) != 1)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1_diagnostic": f1}


def rules_calib():
    det = HERE / "calib" / "pred_details.csv"
    rows = list(csv.DictReader(open(det, encoding="utf-8-sig")))
    gt = gt_for("calib")
    panel = {"main_rule_as_executed": MAIN_RULE,
             "main_rule_metrics_recomputed": metrics(gt, {r["image_name"]: int(r["prediction"]) for r in rows}),
             "siglip_tau_sweep_DIAGNOSTIC_ONLY": [],
             "note": "sweep is calibration-set diagnostics; NEVER an official-best selector",
             "secondary": {}}
    for tau in [round(0.05 + 0.05 * i, 2) for i in range(19)]:
        rule = {**MAIN_RULE, "siglip_tau": tau, "name": f"sweep_s@{tau}"}
        panel["siglip_tau_sweep_DIAGNOSTIC_ONLY"].append(
            {"siglip_tau": tau, **metrics(gt, {r["image_name"]: predict_row(r, rule) for r in rows})})
    fr = SECONDARY_RULES[0]["full_image"]
    panel["secondary"]["F15_app_fullimage_rule_EXPLORATORY"] = metrics(
        gt, {r["image_name"]: int(float(r["siglip_score"]) >= fr["siglip"]
                                  and float(r["dino_score"]) >= fr["dino"]) for r in rows})
    (HERE / "RULE_PANEL_CALIB.json").write_text(json.dumps(panel, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print("RULE_PANEL_CALIB.json written")
    return 0


def freeze():
    a = assets()
    blob = {"main_rule": MAIN_RULE, "secondary_rules_pre_registered": SECONDARY_RULES,
            "assets_sha256": a["sha256"], "calib_panel": str(HERE / "RULE_PANEL_CALIB.json"),
            "holdout_status": "SEALED", "bootstrap": {"B": 10000, "seed": 20260828,
                                                      "method": "percentile iid, as E024 precedent"}}
    (HERE / "RULE_PANEL_FROZEN.json").write_text(json.dumps(blob, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print("FROZEN main rule + assets before holdout open")
    return 0


def holdout_eval():
    if (HERE / "holdout" / "pred.json").exists() or (HERE / "HOLDOUT_OPENED.lock").exists():
        print("REFUSED::holdout already opened once (lock/pred exists) - one-shot discipline")
        return 2
    fz = json.loads((HERE / "RULE_PANEL_FROZEN.json").read_text(encoding="utf-8"))
    a = assets()
    if fz["assets_sha256"] != a["sha256"]:
        print("REFUSED::assets changed after rule freeze")
        return 2
    (HERE / "HOLDOUT_OPENED.lock").write_text(json.dumps(
        {"opened": True, "main_rule": fz["main_rule"]}, indent=2), encoding="utf-8")
    rc = run_chain("holdout")
    if rc != 0:
        return rc
    rows = list(csv.DictReader(open(HERE / "holdout" / "pred_details.csv", encoding="utf-8-sig")))
    gt = gt_for("holdout")
    main = metrics(gt, {r["image_name"]: predict_row(r, MAIN_RULE) for r in rows})

    rng = random.Random(20260828)
    row_of = {r["image_name"]: r for r in rows}
    keys = sorted(gt)
    n = len(keys)
    ps, rs = [], []
    for _ in range(10000):
        smp = [keys[rng.randrange(n)] for _ in range(n)]
        g = {k: gt[k] for k in smp}
        pr = {k: predict_row(row_of[k], MAIN_RULE) for k in smp}
        m = metrics(g, pr)
        ps.append(m["precision"] or 0.0)
        rs.append(m["recall"] or 0.0)
    ps.sort()
    rs.sort()
    ci = {"precision_95CI": [round(ps[250], 4), round(ps[9750 - 1], 4)],
          "recall_95CI": [round(rs[250], 4), round(rs[9750 - 1], 4)]}

    sec = {}
    for rule in SECONDARY_RULES:
        if "full_image" in rule:
            fr = rule["full_image"]
            sec[rule["name"]] = metrics(gt, {r["image_name"]: int(
                float(r["siglip_score"]) >= fr["siglip"] and float(r["dino_score"]) >= fr["dino"])
                for r in rows})
    out = {"main_rule": MAIN_RULE, "metrics": main, "bootstrap_95CI": ci,
           "secondary_pre_registered": sec, "n": n,
           "statement": ("This result resolves THIS round's content-cluster leakage (grouped "
                         "isolation, d<=8 cross-set edges = 0). It is same-source data, NOT "
                         "true external validation."),
           "holdout_opened_once": True}
    (HERE / "holdout" / "HOLDOUT_RESULT.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"main": main, "CI": ci}, indent=2))
    return 0


def main():
    cmd = sys.argv[1]
    if cmd == "chain":
        return run_chain(sys.argv[sys.argv.index("--split") + 1])
    if cmd == "rules":
        return rules_calib()
    if cmd == "freeze":
        return freeze()
    if cmd == "holdout":
        return holdout_eval()
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
