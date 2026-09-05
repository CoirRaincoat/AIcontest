"""G6 SURROGATE submission rehearsal with the ORIGINAL production model.

Produces, in a NEW unique directory, a canonical-1100 submission JSON via the production entry,
then validates it with p7_validator (no-GT mode). This validates the ENGINEERING submission
chain only. It does NOT lift F-11, does NOT raise the generalization score, and is NOT an
official test-set submission.

Mandatory product markings (recorded verbatim in the result):
  ORIGINAL_MODEL_WITH_KNOWN_LEAKAGE
  SURROGATE_REHEARSAL_NOT_OFFICIAL_SUBMISSION
"""
import csv
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
PY = sys.executable
HUB = ROOT / "hf_cache" / "hub"
VALIDATOR = ROOT / "audit/competition_review/scripts/p7_validator.py"
MANIFEST = ROOT / "audit/competition_review/machine/phase7/canonical_manifest.json"
SOURCE = HERE / "surrogate_images"
OUT_JSON = HERE / "submission_surrogate.json"

ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "HF_HUB_DISABLE_TELEMETRY": "1"}


def log(msg):
    print(msg, flush=True)
    with open(HERE / "g6_original_progress.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S") + " " + msg + "\n")


def run_predict():
    cmd = [PY, str(FD_SRC / "predict_best_fusion.py"),
           "--source", str(SOURCE), "--output", str(OUT_JSON),
           "--siglip-checkpoint", str(FD / "runs_siglip/siglip2_linear_seed2026/best_head.pt"),
           "--dino-checkpoint", str(FD / "runs_dinov3/dinov3_vitb16_linear_seed42/best_head.pt"),
           "--crop-checkpoint", str(FD / "runs_siglip/siglip2_crop_v2_seed2026/best_head.pt"),
           "--yolo-m-weights", str(FD / "runs/detect/runs/train/fire_yolo26m_960/weights/best.pt"),
           "--yolo-s-weights", str(FD / "runs/detect/runs/train/fire_yolo26s_960/weights/best.pt"),
           "--yolo-s-aug-weights", str(FD / "runs/detect/runs/train/fire_yolo26s_aug_960/weights/best.pt"),
           "--cache-dir", str(HUB)]
    log("RUN_PREDICT_START")
    # binary redirect avoids the GBK-decode crash on ultralytics progress-bar glyphs
    with open(HERE / "predict_stdout.log", "wb") as outf:
        r = subprocess.run(cmd, stdout=outf, stderr=subprocess.STDOUT, env=ENV, cwd=str(ROOT),
                           timeout=7200)
    log(f"RUN_PREDICT_RC={r.returncode}")
    return r.returncode


def run_validator(pred: Path) -> tuple[int, str]:
    r = subprocess.run([PY, str(VALIDATOR), "validate", "--pred", str(pred),
                        "--manifest", str(MANIFEST), "--root", str(ROOT),
                        "--out", str(HERE / "validator_result.json")],
                       capture_output=True, text=True)
    (HERE / "validator_stdout.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.returncode, (r.stdout + r.stderr)


def check_submission(pred: Path) -> dict:
    data = json.loads(pred.read_text(encoding="utf-8"))
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    canon = set(man["canonical_sample_list"])
    keys = list(data.keys())
    checks = {
        "n_keys": len(keys),
        "n_canonical": len(canon),
        "exact_1100": len(keys) == 1100 and len(canon) == 1100,
        "coverage_exact": set(keys) == canon,
        "no_extra_keys": set(keys) <= canon,
        "no_missing_keys": canon <= set(keys),
        "no_duplicate_keys": len(keys) == len(set(keys)),
        "all_values_int_0_or_1": all(isinstance(v, int) and v in (0, 1) for v in data.values()),
        "no_nan_inf": all(not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                          for v in data.values()),
    }
    # illegal types / out-of-range
    illegal = [k for k, v in data.items() if not (isinstance(v, int) and v in (0, 1))]
    checks["illegal_types"] = illegal
    # PARTIAL / failure artifacts must be absent
    checks["no_partial_artifact"] = not any(
        (HERE / f"submission_surrogate_{s}").exists() for s in ("PARTIAL.json", "failures.json"))
    return checks


def main():
    if not (HERE / "g6_original_progress.log").exists():
        (HERE / "g6_original_progress.log").write_text("", encoding="utf-8")
    t0 = time.time()
    if OUT_JSON.exists():
        log("SKIP_PREDICT (submission_surrogate.json already produced; skipping re-run)")
        rc = 0
    else:
        rc = run_predict()
    checks = {}
    vrc = None
    if rc == 0 and OUT_JSON.exists():
        checks = check_submission(OUT_JSON)
        vrc, vout = run_validator(OUT_JSON)
    else:
        vrc, vout = run_validator(OUT_JSON) if OUT_JSON.exists() else (None, "")

    # reproducibility: rerun determinism is already evidenced by R3 (G3 220-image E019-exact);
    # here record input->output determinism by re-hashing the produced artifact vs source list.
    import hashlib
    pred_sha = hashlib.sha256(OUT_JSON.read_bytes()).hexdigest() if OUT_JSON.exists() else None
    man_sha = json.loads((HERE / "surrogate_manifest.json").read_text(encoding="utf-8"))

    result = {
        "markers": ["ORIGINAL_MODEL_WITH_KNOWN_LEAKAGE",
                    "SURROGATE_REHEARSAL_NOT_OFFICIAL_SUBMISSION"],
        "disclaimer": ("Engineering submission-chain rehearsal only. Does NOT lift F-11, does "
                       "NOT raise generalization score, is NOT an official test-set submission."),
        "model": "ORIGINAL production weights (known data leakage)",
        "predict_rc": rc,
        "validator_rc": vrc,
        "submission_json_sha256": pred_sha,
        "checks": checks,
        "surrogate_manifest": man_sha,
        "wall_s": round(time.time() - t0, 1),
        "n_images_staged": man_sha["n"],
    }
    result["pass"] = bool(
        rc == 0 and vrc == 0 and checks.get("exact_1100") and checks.get("coverage_exact")
        and checks.get("no_duplicate_keys") and checks.get("all_values_int_0_or_1")
        and checks.get("no_nan_inf") and checks.get("no_partial_artifact")
        and checks.get("no_extra_keys") and checks.get("no_missing_keys")
        and not checks.get("illegal_types"))
    (HERE / "G6_ORIGINAL_RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    log(f"DONE pass={result['pass']} pred_rc={rc} validator_rc={vrc}")
    print("G6_ORIGINAL_" + ("PASS" if result["pass"] else "FAIL"))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
