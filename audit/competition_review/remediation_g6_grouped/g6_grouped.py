"""G6 SURROGATE submission rehearsal with the GROUPED (reg1 retrained) model.

Same canonical-1100 surrogate rehearsal as the original-model G6, but using the reg1
retrained checkpoints from REG1_ASSETS.json, in a NEW unique directory (remediation_g6_grouped/),
strictly separated from the original-model G6 products.

Markings (recorded verbatim):
  GROUPED_MODEL_REG1_RETRAINED
  SURROGATE_REHEARSAL_NOT_OFFICIAL_SUBMISSION

Does NOT auto-promote the new weights over production (decision deferred to the user).
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FD = ROOT / "fire_detection"
FD_SRC = FD / "src"
G1 = ROOT / "audit/competition_review/remediation_g1"
PY = sys.executable
HUB = ROOT / "hf_cache" / "hub"
VALIDATOR = ROOT / "audit/competition_review/scripts/p7_validator.py"
MANIFEST = ROOT / "audit/competition_review/machine/phase7/canonical_manifest.json"
SOURCE = HERE / "surrogate_images"
OUT_JSON = HERE / "submission_surrogate.json"

ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def log(msg):
    print(msg, flush=True)
    with open(HERE / "g6_grouped_progress.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def assets():
    a = json.loads((G1 / "REG1_ASSETS.json").read_text(encoding="utf-8"))
    return a["paths"], a["sha256"]


def stage_images():
    src = FD / "data/images/train/images"
    names = sorted(p.name for p in src.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and "(1)" not in p.stem)
    assert len(names) == 1100, len(names)
    SOURCE.mkdir(parents=True, exist_ok=True)
    for n in names:
        t = SOURCE / n
        if t.exists():
            continue
        try:
            os.link(src / n, t)
        except OSError:
            import shutil
            shutil.copy2(src / n, t)
    (HERE / "surrogate_manifest.json").write_text(json.dumps(
        {"n": len(names), "names": names, "excluded_duplicate": "raw_fire_relabel_dp_20402(1).jpg"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"staged {len(names)} canonical images")


def run_predict(P):
    cmd = [PY, str(FD_SRC / "predict_best_fusion.py"),
           "--source", str(SOURCE), "--output", str(OUT_JSON),
           "--siglip-checkpoint", str(P["siglip_head"]),
           "--dino-checkpoint", str(P["dino_head"]),
           "--crop-checkpoint", str(P["crop_head"]),
           "--yolo-m-weights", str(P["yolo_m"]),
           "--yolo-s-weights", str(P["yolo_s"]),
           "--yolo-s-aug-weights", str(P["yolo_s_aug"]),
           "--cache-dir", str(HUB)]
    log("RUN_PREDICT_START")
    with open(HERE / "predict_stdout.log", "wb") as outf:
        r = subprocess.run(cmd, stdout=outf, stderr=subprocess.STDOUT, env=ENV, cwd=str(ROOT),
                           timeout=7200)
    log(f"RUN_PREDICT_RC={r.returncode}")
    return r.returncode


def run_validator(pred: Path):
    r = subprocess.run([PY, str(VALIDATOR), "validate", "--pred", str(pred),
                        "--manifest", str(MANIFEST), "--root", str(ROOT),
                        "--out", str(HERE / "validator_result.json")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    (HERE / "validator_stdout.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.returncode


def check(pred: Path):
    data = json.loads(pred.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    canon = set(man["canonical_sample_list"])
    keys = list(data.keys())
    return {
        "n_keys": len(keys), "exact_1100": len(keys) == 1100,
        "coverage_exact": set(keys) == canon, "no_extra": set(keys) <= canon,
        "no_missing": canon <= set(keys), "no_duplicate": len(keys) == len(set(keys)),
        "all_int_0_1": all(isinstance(v, int) and v in (0, 1) for v in data.values()),
        "no_nan_inf": all(not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                          for v in data.values()),
        "no_partial_artifact": not any((HERE / f"submission_surrogate_{s}").exists()
                                       for s in ("PARTIAL.json", "failures.json")),
    }


def main():
    (HERE / "g6_grouped_progress.log").write_text("", encoding="utf-8")
    P, S = assets()
    stage_images()
    t0 = time.time()
    if OUT_JSON.exists():
        log("SKIP_PREDICT (already produced)")
        rc = 0
    else:
        rc = run_predict(P)
    import hashlib
    pred_sha = hashlib.sha256(OUT_JSON.read_bytes()).hexdigest() if OUT_JSON.exists() else None
    checks = check(OUT_JSON) if OUT_JSON.exists() else {}
    vrc = run_validator(OUT_JSON) if OUT_JSON.exists() else None
    result = {
        "markers": ["GROUPED_MODEL_REG1_RETRAINED", "SURROGATE_REHEARSAL_NOT_OFFICIAL_SUBMISSION"],
        "disclaimer": "Surrogate rehearsal only. Not an official submission; does NOT auto-promote "
                      "the reg1 weights over production.",
        "model": "GROUPED reg1 retrained chain (content-cluster isolated)",
        "assets_sha256": S,
        "predict_rc": rc, "validator_rc": vrc, "submission_json_sha256": pred_sha,
        "checks": checks, "wall_s": round(time.time() - t0, 1),
    }
    result["pass"] = bool(rc == 0 and vrc == 0 and checks.get("exact_1100")
                          and checks.get("coverage_exact") and checks.get("no_duplicate")
                          and checks.get("all_int_0_1") and checks.get("no_nan_inf")
                          and checks.get("no_partial_artifact"))
    (HERE / "G6_GROUPED_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    log(f"DONE pass={result['pass']}")
    print("G6_GROUPED_" + ("PASS" if result["pass"] else "FAIL"))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
