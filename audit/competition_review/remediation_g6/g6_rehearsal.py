"""G6 submission rehearsal on a SURROGATE list (official test set absent — never faked).

What it proves (mechanism only):
  1. production entry -> submission-format JSON over a 1100-image surrogate view (junction,
     zero copies);
  2. p7_validator no-GT structural validation PASSes on the produced submission;
  3. naming / exact-coverage (canonical 1100 keys, no extras) / exit codes / atomic writes;
  4. PARTIAL protection: a corrupt COPY injected into the surrogate dir yields exit 3,
     *_PARTIAL.json + *_failures.json, main submission name absent, validator still rejects.

Everything is labeled: SURROGATE REHEARSAL — 不等于正式提交就绪 (official test list absent).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R3 = ROOT / "audit/competition_review/remediation_r3"
G1 = ROOT / "audit/competition_review/remediation_g1"
FD = ROOT / "fire_detection"
FD_SRC = FD / "src"
PY = sys.executable
CANON = FD / "data/images/train/images"


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def assets():
    aj = G1 / "REG1_ASSETS.json"
    if aj.exists():
        a = json.loads(aj.read_text(encoding="utf-8"))
        return {k: a["paths"][k] for k in ("siglip_head", "dino_head", "crop_head",
                                           "yolo_m", "yolo_s", "yolo_s_aug")}, "reg1_retrained_chain"
    raise SystemExit("STOP::REG1_ASSETS.json missing - G6 runs after G1 (no fake fallback)")


def chain(source: Path, out: Path, tag: str, w):
    cmd = [PY, str(FD_SRC / "predict_best_fusion.py"),
           "--source", str(source), "--output", str(out),
           "--siglip-checkpoint", str(w["siglip_head"]), "--dino-checkpoint", str(w["dino_head"]),
           "--crop-checkpoint", str(w["crop_head"]),
           "--yolo-m-weights", str(w["yolo_m"]), "--yolo-s-weights", str(w["yolo_s"]),
           "--yolo-s-aug-weights", str(w["yolo_s_aug"]),
           "--cache-dir", str(ROOT / "hf_cache/hub")]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=14400)
    (HERE / f"rehearsal_{tag}.log").write_text(
        r.stdout[-40000:] + "\n===STDERR===\n" + r.stderr[-20000:], encoding="utf-8")
    return r.returncode


VALIDATOR = ROOT / "audit/competition_review/scripts/p7_validator.py"
MANIFEST = ROOT / "audit/competition_review/machine/phase7/canonical_manifest.json"


def validate(pred: Path, tag: str):
    r = subprocess.run([PY, str(VALIDATOR), "validate", "--pred", str(pred),
                        "--manifest", str(MANIFEST), "--root", str(ROOT),
                        "--out", str(HERE / f"validator_{tag}.json")],
                       capture_output=True, text=True)
    (HERE / f"validator_{tag}.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    return r.returncode, (r.stdout + r.stderr)[-2000:]


def main():
    w, chain_kind = assets()
    steps = {}
    view = HERE / "surrogate_images_view"
    if not view.exists():
        subprocess.run(["cmd", "/c", "mklink", "/J", str(view), str(CANON)], capture_output=True)

    # 1. full surrogate rehearsal run
    out = HERE / "submission_surrogate.json"
    rc = chain(view, out, "full", w)
    steps["full_chain_rc"] = rc
    steps["full_chain_kind"] = chain_kind
    pred = json.loads(out.read_text(encoding="utf-8"))
    canon_names = {p.name for p in CANON.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and "(1)" not in p.stem}
    steps["coverage"] = {"keys": len(pred), "canonical": len(canon_names),
                         "exact_match": set(pred) == canon_names}
    vrc, vtxt = validate(out, "full")
    steps["validator_rc_full"] = vrc

    # 2. corrupt-injection protection probe (copies only; originals untouched)
    probe = HERE / "probe_corrupt_view"
    probe.mkdir(exist_ok=True)
    bad = probe / "corrupt_injected.jpg"
    src = next(p for p in sorted(CANON.iterdir()) if p.suffix.lower() == ".jpg" and "(1)" not in p.stem)
    if not bad.exists():
        raw = src.read_bytes()
        bad.write_bytes(raw[: len(raw) // 2])  # truncated COPY of a surrogate image
    rc2 = chain(probe, HERE / "submission_probe.json", "probe", w)
    steps["probe_rc"] = rc2
    steps["probe_partial_artifacts"] = {
        "partial_json_exists": (HERE / "submission_probe_PARTIAL.json").exists(),
        "failures_json_exists": (HERE / "submission_probe_failures.json").exists(),
        "main_name_absent": not (HERE / "submission_probe.json").exists()}
    vrc2, _ = validate(HERE / "submission_probe_PARTIAL.json", "probe_partial")
    steps["validator_rc_partial"] = vrc2

    steps["surrogate_disclaimer"] = ("SURROGATE REHEARSAL on the 1100-image canonical view; "
                                     "official test list absent; 不等于正式提交就绪")
    (HERE / "REHEARSAL_RESULT.json").write_text(json.dumps(steps, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(json.dumps(steps, ensure_ascii=False, indent=2))
    ok = (steps["full_chain_rc"] == 0 and steps["coverage"]["exact_match"]
          and steps["validator_rc_full"] == 0 and steps["probe_rc"] == 3
          and all(steps["probe_partial_artifacts"].values()) and vrc2 != 0)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
