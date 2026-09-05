"""p7_validator.py — SF-2026-01 独立CPU提交校验器（阶段7，仅审计目录）。

独立性声明:
  * 指标/解析逻辑为本文件全新实现，不 import 生产 src/* 任何函数;
  * canonical 清单来自官方附件 train_coco.json(1100) 现场冻结并哈希锚定;
  * 唯一外部依赖: Pillow(仅 preflight-images 子命令需要)。
确定性: 输入文件以 SHA-256 锚定; 一切遍历 sorted(); 输出 json.dumps(sort_keys=True);
        产物内不含时间戳 → 同机同输入重复运行字节一致(Python 版本记录于 meta)。

口径策略(本工具):
  官方数字 = 全集分母; 缺失预测 = 硬错误, 不静默跳过;
  同时给出两个诊断列: "strict_if_missing_as_negative"(缺失按负类保守计) 与
  "production_intersection_diagnostic"(镜像 evaluate_image_level.py 交集口径用于展示差异,
   零分母沿用其静默0.0语义) —— 该列永远不得作为对外成绩。
  P=tp/(tp+fp), R=tp/(tp+fn); 分母0 → None + flag, 绝不冒充综合分。

退出码: 0=通过; 1=校验失败(含结构/覆盖/指标前提); 2=致命IO或用法错误。preflight-images
        发现坏图亦为1(带逐文件明细), 不允许无提示崩溃。

用法见同目录 VALIDATOR_README.md / 文末 CLI。
"""
import hashlib
import json
import sys
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parents[1]
PROJ_ROOT = AUDIT_DIR.parents[1]
FD = PROJ_ROOT / "fire_detection"
EXCLUDED_REL = "fire_detection/data/images/train/images/raw_fire_relabel_dp_20402(1).jpg"
IMG_EXT = {".jpg", ".jpeg", ".png"}
TOOL_VERSION = "p7-validator-1.0"

EXIT_OK, EXIT_FAIL, EXIT_FATAL = 0, 1, 2


# ---------------------------------------------------------------- helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_constant(name):
    raise ValueError(f"forbidden JSON constant literal: {name}")


def _no_dup_pairs(pairs):
    keys = [k for k, _ in pairs]
    dups = sorted({k for k in keys if keys.count(k) > 1})
    if dups:
        raise ValueError(f"duplicate JSON key(s): {dups}")
    return dict(pairs)


def parse_json_strict(raw: bytes):
    """Returns (obj|None, error_dict|None). Rejects: bad UTF-8, BOM, duplicate keys
    (anywhere), NaN/Infinity literals (plain json.loads would silently accept these)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, {"code": "not_utf8", "message": repr(e)[:120]}
    if text.startswith("﻿"):
        return None, {"code": "utf8_bom_present",
                      "message": "leading U+FEFF BOM not allowed"}
    try:
        obj = json.loads(text, object_pairs_hook=_no_dup_pairs, parse_constant=_reject_constant)
    except ValueError as e:
        msg = str(e)
        code = "duplicate_json_key" if msg.startswith("duplicate JSON key") else \
            "forbidden_constant" if msg.startswith("forbidden JSON constant") else "json_unparsable"
        return None, {"code": code, "message": msg[:200]}
    return obj, None


def value_error_code(v):
    """Classify a top-level value against contract {filename: int 0|1}."""
    if isinstance(v, bool):
        return ("boolean_not_int", f"expected int 0|1, got bool {v}")
    if isinstance(v, float):
        return ("float_not_int", f"expected int 0|1, got float {v!r} (e.g. 1.0 is invalid)")
    if isinstance(v, int):
        if v not in (0, 1):
            return ("value_out_of_range", f"int must be 0 or 1, got {v}")
        return None
    if v is None:
        return ("null_value", "null is not a valid label")
    if isinstance(v, str):
        return ("string_value", f"label must be int, got string {v!r}")
    return ("unsupported_value_type", type(v).__name__)


def key_error_code(k):
    """Classify a submission filename key."""
    if not isinstance(k, str):
        return ("key_not_string", repr(k))
    if ("/" in k) or ("\\" in k) or k in (".", "..") or ".." in k:
        return ("path_traversal_or_separator", f"key {k!r} contains path syntax")
    if k != Path(k).name or Path(k).name == "":
        return ("key_not_bare_filename", k)
    if k.strip() != k or k == "":
        return ("key_whitespace", repr(k))
    return None


def confusion(canonical_names, gt, pred):
    """Full-universe confusion. pred values must already be validated ints;
    names absent from pred are counted as missing separately (NOT silently)."""
    tp = fp = fn = tn = missing_pos = missing_neg = 0
    missing_names = []
    for n in canonical_names:
        t = gt[n]
        if n not in pred:
            missing_names.append(n)
            if t == 1:
                missing_pos += 1
            else:
                missing_neg += 1
            continue
        p = pred[n]
        tp += (t == 1 and p == 1)
        fp += (t == 0 and p == 1)
        fn += (t == 1 and p == 0)
        tn += (t == 0 and p == 0)
    present = len(canonical_names) - len(missing_names)

    def pr(tp_, fp_, fn_):
        p_den, r_den = tp_ + fp_, tp_ + fn_
        # F1 由整数格口直接计算 2TP/(分母和), 避免用已舍入的P/R二次舍入
        F = None if (not p_den or not r_den or (p_den + r_den) == 0) else \
            round(2 * tp_ / (p_den + r_den), 6)
        flags = []
        if not p_den:
            flags.append("precision_undefined_no_predicted_positive")
        if not r_den:
            flags.append("recall_undefined_no_positive_truth")
        return {"precision": round(tp_ / p_den, 6) if p_den else None,
                "recall": round(tp_ / r_den, 6) if r_den else None,
                "f1_diagnostic": F,
                "undefined_flags": flags}

    strict = pr(tp, fp, fn + missing_pos)      # missing treated as negative (conservative)
    inter_names = [n for n in canonical_names if n in pred]
    inter = {}
    itp = sum(1 for n in inter_names if gt[n] == 1 and pred[n] == 1)
    ifp = sum(1 for n in inter_names if gt[n] == 0 and pred[n] == 1)
    ifn = sum(1 for n in inter_names if gt[n] == 1 and pred[n] == 0)
    p_den, r_den = itp + ifp, itp + ifn
    inter = {"n_common": len(inter_names),
             # production semantics: silent 0.0 on zero denominators — DIAGNOSTIC ONLY
             "precision_production_semantics": (itp / p_den) if p_den else 0.0,
             "recall_production_semantics": (itp / r_den) if r_den else 0.0}
    return {
        "present_tp": tp, "present_fp": fp, "present_fn": fn, "tn": tn,
        "missing_predictions_total": len(missing_names),
        "missing_true_positive": missing_pos, "missing_true_negative": missing_neg,
        "strict_full_coverage_metrics": strict,
        "production_intersection_diagnostic_DO_NOT_REPORT": inter,
        "missing_names_sorted": sorted(missing_names)[:50],
    }


# ---------------------------------------------------------------- manifest
def freeze(root: Path | None = None, out_dir: Path | None = None) -> dict:
    root = root or PROJ_ROOT
    out_dir = out_dir or (AUDIT_DIR / "machine" / "phase7")
    out_dir.mkdir(parents=True, exist_ok=True)
    fd = root / "fire_detection"
    coco_p = fd / "data/images/train/train_coco.json"
    lab_p = fd / "data/images/train/train_image.json"
    raw_d = fd / "data/images/train/images"
    errors = []
    coco = json.loads(coco_p.read_text(encoding="utf-8"))
    canon_from_coco = sorted(im["file_name"] for im in coco["images"])
    labels = {k: int(v) for k, v in json.loads(lab_p.read_text(encoding="utf-8")).items()}
    if set(labels) != set(canon_from_coco):
        errors.append({"code": "labels_vs_coco_keyset_mismatch",
                       "n_lab_only": len(set(labels) - set(canon_from_coco)),
                       "n_coco_only": len(set(canon_from_coco) - set(labels))})

    disk = sorted(p.name for p in raw_d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXT)
    excluded_name = Path(EXCLUDED_REL).name
    expected_disk = sorted(canon_from_coco + [excluded_name])
    extra_on_disk = sorted(set(disk) - set(expected_disk))
    gone_from_disk = sorted(set(expected_disk) - set(disk))
    if extra_on_disk:
        errors.append({"code": "unexpected_extra_files_on_disk_fail_closed", "files": extra_on_disk})
    if gone_from_disk:
        errors.append({"code": "canonical_files_missing_from_disk", "files": gone_from_disk[:20]})

    def case_collide(names):
        m = {}
        for n in names:
            m.setdefault(n.lower(), []).append(n)
        return {k: v for k, v in m.items() if len(v) > 1}

    cc = case_collide(canon_from_coco)
    if cc:
        errors.append({"code": "canonical_case_insensitive_collision", "collisions": cc})
    exc_path = root / EXCLUDED_REL
    excluded_sha = sha256_file(exc_path) if exc_path.is_file() else None
    canon_sha_of_same_content = sha256_file(raw_d / "raw_fire_relabel_dp_20402.jpg")
    excl_record = {
        "relative_path": EXCLUDED_REL.replace("\\", "/"), "sha256": excluded_sha,
        "reason": "F-04已结案: Windows '(1)' 复制残留, 与 canonical "
                  "raw_fire_relabel_dp_20402.jpg 字节相同(SHA-256相等), 不属官方集合; 显式排除而非模糊去重",
        "duplicate_of_canonical_file": "raw_fire_relabel_dp_20402.jpg",
        "duplicate_of_canonical_sha256": canon_sha_of_same_content,
        "byte_identical": excluded_sha == canon_sha_of_same_content and excluded_sha is not None,
    }

    val_d = fd / "data/val/images"
    tr_d = fd / "data/train/images"
    val_members = sorted(p.name for p in val_d.iterdir() if p.suffix.lower() in IMG_EXT)
    tr_members = sorted(p.name for p in tr_d.iterdir() if p.suffix.lower() in IMG_EXT)
    if set(val_members) & set(tr_members):
        errors.append({"code": "val_train_overlap"})
    if set(val_members) | set(tr_members) != set(canon_from_coco):
        errors.append({"code": "splits_do_not_tile_canonical"})

    manifest = {
        "tool_version": TOOL_VERSION,
        "definition": "canonical=官方附件train_coco.json全部images.file_name, 排序冻结",
        "n_canonical": len(canon_from_coco),
        "canonical_sample_list": canon_from_coco,
        "excluded_files": [excl_record],
        "case_sensitive_contract": True,
        "anchors": {
            "train_coco.json_sha256": sha256_file(coco_p),
            "train_image.json_sha256": sha256_file(lab_p),
            "n_labels_positive": sum(labels.values()),
            "internal_val220_members": val_members,
            "internal_val220_sha256_of_concat": hashlib.sha256(
                "\n".join(val_members).encode()).hexdigest(),
            "gt_rule": "image-level label == 1 iff该图在train_image.json为1"
        },
        "freeze_errors": errors,
    }
    (out_dir / "canonical_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    ok = not errors
    print(json.dumps({"freeze_ok": ok, "n_canonical": len(canon_from_coco),
                      "excluded_byte_identical": excl_record["byte_identical"],
                      "errors": errors}, ensure_ascii=False))
    return {"ok": ok}


def load_manifest(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("tool_version") != TOOL_VERSION:
        raise SystemExit(f"{EXIT_FATAL}: manifest produced by {m.get('tool_version')}, need {TOOL_VERSION} -> refreeze")
    for fname, key in [("data/images/train/train_coco.json", "train_coco.json_sha256"),
                       ("data/images/train/train_image.json", "train_image.json_sha256")]:
        cur = sha256_file(FD / fname)
        if cur != m["anchors"][key]:
            raise SystemExit(f"{EXIT_FATAL}: input drift — {fname} sha256 changed since freeze ({cur} != {m['anchors'][key]})")
    exc = Path(PROJ_ROOT) / m["excluded_files"][0]["relative_path"]
    if exc.is_file():
        if sha256_file(exc) != m["excluded_files"][0]["sha256"]:
            raise SystemExit(f"{EXIT_FATAL}: excluded file content drifted: {exc}")
    elif m["excluded_files"][0]["sha256"] is not None:
        pass  # file later removed from disk is acceptable IF disk re-scan passes below
    return m


def _display_rel(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(PROJ_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def validate(pred_path: Path, manifest: dict, gt_path: Path | None = None,
             subset: str | None = None) -> tuple[dict, int]:
    pred_path = pred_path.resolve()
    if gt_path is not None:
        gt_path = gt_path.resolve()
    res = {"tool": TOOL_VERSION, "verdict": "PASS", "errors": [], "warnings": [],
           "pred_input": {"relative_path": _display_rel(pred_path),
                          "size_bytes": pred_path.stat().st_size,
                          "sha256": sha256_file(pred_path)}}
    canonical = manifest["canonical_sample_list"]
    universe = canonical

    if subset:
        anchors = manifest["anchors"]
        members = anchors.get(subset, anchors.get(subset + "_members"))
        if members is None:
            raise SystemExit(f"{EXIT_FATAL}: unknown subset key {subset!r}; "
                             f"available={sorted(k for k in anchors if k.endswith('_members'))}")
        if isinstance(members, str):
            raise SystemExit(f"{EXIT_FATAL}: subset {subset!r} anchor is not a list")
        if not set(members) <= set(canonical):
            res["errors"].append({"code": "subset_not_within_canonical"})
        universe = sorted(members)

    gt = None
    if gt_path is not None:
        res["gt_input"] = {"sha256": sha256_file(gt_path)}
        gobj, err = parse_json_strict(gt_path.read_bytes())
        if err:
            res["errors"].append({"stage": "gt_parse", **err})
        else:
            if not isinstance(gobj, dict):
                res["errors"].append({"stage": "gt", "code": "gt_top_level_not_object"})
            else:
                bad = [key_error_code(k) for k in gobj if key_error_code(k)]
                if bad:
                    res["errors"].append({"stage": "gt_keys",
                                          "list": [{"code": c, "message": m} for c, m in bad[:10]]})
                universe_lower = {u.lower(): u for u in universe}
                if set(map(str.lower, map(str, gobj))) != set(universe_lower) or \
                        any(universe_lower.get(str(k).lower()) != k for k in gobj):
                    res["errors"].append({"code": "gt_keyset_mismatch_vs_universe",
                                          "n_gt": len(gobj), "n_universe": len(universe)})
                else:
                    gt = {}
                    for k, v in gobj.items():
                        vc = value_error_code(v)
                        if vc:
                            res["errors"].append({"stage": "gt_values", "key": k,
                                                  "code": vc[0], "message": vc[1]})
                            gt = None
                            break
                        gt[k] = v

    obj, err = parse_json_strict(pred_path.read_bytes())
    if err:
        res["errors"].append({"stage": "pred_parse", **err})
        obj = None

    checked_pred = {}
    if obj is not None:
        if not isinstance(obj, dict):
            res["errors"].append({"code": "top_level_not_object",
                                  "actual_type": type(obj).__name__,
                                  "contract": '{"<filename>": <int 0|1>, ...}'})
        else:
            canon_lower = {}
            collide = {}
            for c in canonical:
                low = c.lower()
                if low in canon_lower:
                    collide.setdefault(low, []).append(canon_lower[low])
                canon_lower[low] = c
            if collide:
                res["errors"].append({"code": "canonical_case_collision_internal", "collisions": collide})

            extra_exact, case_notes = [], []
            seen_lower = {}
            for k, v in obj.items():
                ke = key_error_code(k)
                if ke:
                    res["errors"].append({"stage": "pred_key", "key": k,
                                          "code": ke[0], "message": ke[1]})
                    continue
                ve = value_error_code(v)
                if ve:
                    res["errors"].append({"stage": "pred_value", "key": k,
                                          "code": ve[0], "message": ve[1]})
                    continue
                if k.lower() in seen_lower and seen_lower[k.lower()] != k:
                    res["errors"].append({"code": "pred_case_collision_between_keys",
                                          "keys": sorted([seen_lower[k.lower()], k])})
                seen_lower[k.lower()] = k
                checked_pred[k] = v
                if k in canonical_set(manifest):
                    continue
                if k.lower() in canon_lower:
                    case_notes.append({"pred_key": k, "canonical_spelling": canon_lower[k.lower()],
                                       "hint": "大小写不敏感匹配但精确匹配失败——Windows盘符上会指向同一文件, 违反精确契约"})
                else:
                    extra_exact.append(k)
            missing_exact = sorted(set(universe) - set(checked_pred))
            # coverage enforced in every mode (structure-only included): missing = fail closed
            if case_notes:
                res["errors"].append({"code": "case_insensitive_key_mismatch", "detail": case_notes[:20]})
            if extra_exact:
                res["errors"].append({"code": "extra_samples", "files": sorted(extra_exact)[:50],
                                      "count": len(extra_exact)})
            # duplicates as samples == duplicate JSON keys (already caught at parse) + exact repeats impossible in dict
            res["coverage"] = {"universe": len(universe), "parsed_keys": len(checked_pred),
                               "missing": len(missing_exact), "extra": len(extra_exact)}
            if missing_exact:
                res["errors"].append({"code": "missing_predictions_fail_closed",
                                      "count": len(missing_exact), "first_missing": missing_exact[:50]})

    metrics_block = None
    fatal_metric_prereqs = {"top_level_not_object", "not_utf8", "json_unparsable",
                            "forbidden_constant", "duplicate_json_key"}
    if gt is not None and isinstance(obj, dict) and not any(
            e.get("stage") in ("gt_keys", "gt_values") or e.get("code") in fatal_metric_prereqs
            for e in res["errors"]):
        # Missing predictions are already a hard error above; metrics are still computed
        # BOTH ways so reviewers can see the exact impact (strict = missing-as-negative).
        cm = confusion(universe, gt, checked_pred)
        metrics_block = cm
        prod_r = cm["production_intersection_diagnostic_DO_NOT_REPORT"]["recall_production_semantics"]
        strict_r = cm["strict_full_coverage_metrics"]["recall"]
        cm["recall_inflation_trap_delta_PRODUCTION_minus_STRICT"] = (
            None if (strict_r is None) else round(prod_r - strict_r, 6))
        if cm["missing_predictions_total"]:
            res["warnings"].append({
                "code": "metrics_computed_despite_errors_for_diagnosis_only",
                "note": "官方成绩一列请勿引用；strict 列把缺失当负类是给评审看影响面的下界"})
    res["metrics"] = metrics_block
    if not obj:
        res["coverage"] = res.get("coverage") or {"universe": len(universe), "parsed_keys": 0,
                                                  "missing": len(universe), "extra": 0}
    res["verdict"] = "PASS" if not res["errors"] else "FAIL"
    exit_code = EXIT_OK if res["verdict"] == "PASS" else EXIT_FAIL
    return res, exit_code


_CANON_SET_CACHE = {}


def canonical_set(manifest):
    tid = id(manifest)
    if tid not in _CANON_SET_CACHE:
        _CANON_SET_CACHE[tid] = frozenset(manifest["canonical_sample_list"])
    return _CANON_SET_CACHE[tid]


# ---------------------------------------------------------------- preflight images
def preflight_images(dir_path: Path) -> tuple[dict, int]:
    try:
        from PIL import Image
    except ImportError:
        out = {"tool": TOOL_VERSION, "check": "preflight-images",
               "fatal": "Pillow unavailable in this interpreter", "corrupt": []}
        return out, EXIT_FATAL
    files = sorted(p for p in dir_path.rglob("*") if p.is_file())
    corrupt, total_img = [], 0
    for p in files:
        if p.stat().st_size == 0:
            corrupt.append({"file": str(p.relative_to(dir_path)), "error": "zero-byte file"})
            continue
        if p.suffix.lower() not in IMG_EXT:
            continue
        total_img += 1
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im2:
                im2.load()
        except Exception as e:  # noqa: BLE001 — validator must report ANY decode failure
            corrupt.append({"file": str(p.relative_to(dir_path)),
                            "exception_class": type(e).__name__, "message": str(e)[:160]})
    out = {"tool": TOOL_VERSION, "check": "preflight-images",
           "directory": str(dir_path.relative_to(PROJ_ROOT)).replace("\\", "/"),
           "files_seen": len(files), "images_checked": total_img,
           "corrupt_count": len(corrupt), "corrupt": corrupt}
    return out, (EXIT_OK if not corrupt else EXIT_FAIL)


# ---------------------------------------------------------------- CLI
def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return EXIT_FATAL
    cmd = argv[1]
    if cmd == "freeze":
        args = argv[2:]
        custom_root = Path(args[args.index("--root") + 1]) if "--root" in args else None
        freeze(custom_root)
        return EXIT_OK
    if cmd == "validate":
        kw = {"--pred": None, "--manifest": None, "--gt": None, "--subset": None,
              "--out": None, "--root": None}
        for i, a in enumerate(argv[2:]):
            if a in kw:
                kw[a] = argv[3 + i]
        if not kw["--pred"] or not kw["--manifest"]:
            print("usage: validate --pred FILE --manifest MANIFEST [--gt GT] [--subset anchors_key] [--out OUT]")
            return EXIT_FATAL
        root_override = kw["--root"]
        global FD
        if root_override:
            FD = Path(root_override) / "fire_detection"
        manifest = load_manifest(Path(kw["--manifest"]))
        res, code = validate(Path(kw["--pred"]), manifest,
                             gt_path=Path(kw["--gt"]) if kw["--gt"] else None,
                             subset=kw["--subset"])
        js = json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1)
        if kw["--out"]:
            Path(kw["--out"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kw["--out"]).write_text(js, encoding="utf-8")
        print(js[:2000])
        print(f"\nVERDICT={res['verdict']} errors={len(res['errors'])}", file=sys.stderr)
        return code
    if cmd == "preflight-images":
        kw = {"--dir": None, "--out": None}
        for i, a in enumerate(argv[2:]):
            if a in kw:
                kw[a] = argv[3 + i]
        if not kw["--dir"]:
            print("usage: preflight-images --dir DIR [--out OUT]")
            return EXIT_FATAL
        res, code = preflight_images(Path(kw["--dir"]))
        js = json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1)
        if kw["--out"]:
            Path(kw["--out"]).write_text(js, encoding="utf-8")
        print(js[:2000])
        print(f"\ncorrupt_count={res['corrupt_count']}", file=sys.stderr)
        return code
    print(__doc__)
    return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main(sys.argv))
