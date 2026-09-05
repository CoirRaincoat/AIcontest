"""p7_validate_artifacts.py — 用独立校验器核验现存预测产物(阶段7)。

两条模式:
  A) val子集(220) + 官方 train_image.json 限制版作为GT(非项目 val_gt.json, 独立取数;
     另做两者的逐键一致性检查) → 严格口径复算 + 与存储 *_metrics.json 对账 + 与
     阶段3审计冻结格口(E019)三方一致才算 MATCH。
  B) 同一产物对 canonical 1100 全集 → 预期 FAIL(missing=880), 证明其为内部val产物
     而非可提交件。

输出: machine/phase7/existing_artifacts_validation.json (sort_keys, 无时间戳)
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p7_validator as V  # noqa: E402

M7 = V.AUDIT_DIR / "machine" / "phase7"
OUTS = V.FD / "outputs"
manifest = json.loads((M7 / "canonical_manifest.json").read_text(encoding="utf-8"))
subset = manifest["anchors"]["internal_val220_members"]

official_labels = {k: int(v) for k, v in json.loads(
    (V.FD / "data/images/train/train_image.json").read_text(encoding="utf-8")).items()}
gt_restrict = {n: official_labels[n] for n in subset}
gt_file = M7 / "official_gt_val220_restricted.json"
gt_file.write_text(json.dumps(gt_restrict, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")

# 独立性对账: 项目自己的 val_gt.json 必须与官方限制版逐键一致
vg = json.loads((OUTS / "val_gt.json").read_text(encoding="utf-8"))
vg_int = {k: int(v) for k, v in vg.items()}
identity = {"keys_equal": set(vg_int) == set(subset),
            "values_equal": vg_int == gt_restrict,
            "n": len(vg_int)}
if not (identity["keys_equal"] and identity["values_equal"]):
    raise SystemExit("val_gt.json 与官方标签限制版不一致 — 停止, 上报")

# 阶段3冻结的期望格口(E019) —— 第三方锚点
FROZEN_EXPECT = {
    "val_pred_best_seed2026_yolo_fusion_thr022": {"tp": 161, "fp": 10, "fn": 4, "tn": 45,
                                                  "P": 0.9415, "R": 0.9758},
    "val_pred_best_siglip_yolo_dinov3_veto_thr008": {"tp": 161, "fp": 9, "fn": 4, "tn": 46,
                                                     "P": 0.9471, "R": 0.9758},
    "val_pred_best_siglip_yolo_dino_crop_rescue_thr095": {"tp": 163, "fp": 10, "fn": 2, "tn": 45,
                                                          "P": 0.9422, "R": 0.9879},
    "val_pred_best_siglip_yolo_dino_crop_rescue_thr097": {"tp": 162, "fp": 9, "fn": 3, "tn": 46,
                                                          "P": 0.9474, "R": 0.9818},
    "val_pred_best_siglip_yolo_dino_crop_live": {"tp": 162, "fp": 9, "fn": 3, "tn": 46,
                                                 "P": 0.9474, "R": 0.9818},
}


def extract_stored_metric(path: Path):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    found = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = k.lower()
                if isinstance(v, (int, float)) and (lk.startswith("precision") or lk.startswith("recall")):
                    found.setdefault(lk, v)
                elif isinstance(v, dict):
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(data)
    return found or None


artifacts = FROZEN_EXPECT.keys() | {"val_gt.json"}
results = {}
all_match = True
for stem in sorted(FROZEN_EXPECT):
    pred_p = OUTS / f"{stem}.json"
    res_a, code_a = V.validate(pred_p, manifest, gt_path=gt_file, subset="internal_val220")
    met = res_a.get("metrics") or {}
    strict = met.get("strict_full_coverage_metrics", {})
    counts = {k: met.get(k) for k in ("present_tp", "present_fp", "present_fn", "tn",
                                      "missing_predictions_total")}
    recomp = {"tp": counts["present_tp"], "fp": counts["present_fp"], "fn": counts["present_fn"],
              "tn": counts["tn"],
              "P": round(strict["precision"], 4), "R": round(strict["recall"], 4)}
    frozen_ok = all(recomp[k] == FROZEN_EXPECT[stem][k] for k in ("tp", "fp", "fn", "tn", "P", "R"))
    sidecar = extract_stored_metric(OUTS / f"{stem}_metrics.json")
    sidecar_ok = True
    if sidecar:
        sp = [v for k, v in sidecar.items() if k.startswith("precision")]
        sr = [v for k, v in sidecar.items() if k.startswith("recall")]
        sidecar_ok = any(abs(round(p, 4) - recomp["P"]) < 5e-5 for p in sp) and \
            any(abs(round(r, 4) - recomp["R"]) < 5e-5 for r in sr)
    row = {
        "mode_A_val_subset": {
            "verdict": res_a["verdict"], "exit_code": code_a,
            "structural_errors": len(res_a["errors"]),
            "recomputed_strict": recomp,
            "frozen_expect_E019": FROZEN_EXPECT[stem],
            "matches_frozen_stage3": bool(frozen_ok),
            "stored_sidecar_file": f"{stem}_metrics.json",
            "stored_sidecar_values": sidecar,
            "matches_stored_sidecar": bool(sidecar_ok) if sidecar else "no_sidecar",
            "coverage": res_a.get("coverage"),
        },
        "mode_B_full_1100": {},
    }
    # 模式B: 无GT结构校验 vs 1100
    mbres, mbc = V.validate(pred_p, manifest, gt_path=None)
    cov_b = mbres.get("coverage", {})
    expected_fail = (mbc == V.EXIT_FAIL and cov_b.get("universe") == 1100
                     and cov_b.get("parsed_keys") == 220 and cov_b.get("missing") == 880)
    row["mode_B_full_1100"] = {
        "verdict": mbres["verdict"], "exit_code": mbc, "coverage": cov_b,
        "expected_fail_missing880_confirmed": bool(expected_fail),
        "interpretation": "内部val产物≠提交件; 若官方测试集尺寸未知, 提交时须以其清单为准重新校验",
    }
    row["aggregate_row_verdict"] = "OK" if (
        res_a["verdict"] == "PASS" and code_a == 0 and frozen_ok and
        sidecar_ok is not False and expected_fail) else "REVIEW"
    if row["aggregate_row_verdict"] != "OK":
        all_match = False
    results[stem] = row

doc = {
    "tool_version": V.TOOL_VERSION,
    "gt_source_independence_note":
        "GT=官方 train_image.json 按 internal_val220_members 限制(project val_gt.json仅作一致性对照)",
    "gt_identity_check_vs_project_val_gt_json": identity,
    "canonical_n": manifest["n_canonical"],
    "anchors_verified_by_validator_on_every_load": ["train_coco.json_sha256", "train_image.json_sha256"],
    "artifacts": results,
    "summary": {
        "rows_total": len(results),
        "rows_OK": sum(1 for r in results.values() if r["aggregate_row_verdict"] == "OK"),
        "strict_recompute_all_match_frozen_and_stored": all_match,
        "full_universe_expected_fail_confirmed_for_all": all(
            r["mode_B_full_1100"]["expected_fail_missing880_confirmed"] for r in results.values()),
    },
}
(M7 / "existing_artifacts_validation.json").write_text(
    json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
print(json.dumps({"gt_identity": identity, **doc["summary"],
                  "rows": {k: v["aggregate_row_verdict"] for k, v in results.items()}},
                 ensure_ascii=False))
sys.exit(0 if all_match else 1)
