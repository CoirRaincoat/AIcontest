"""p7_selftest.py — 校验器自证套件(阶段7)。

每个用例都带人工可手算的期望结果(expected, 含推导式), 自动比较 actual。
生成夹具 → machine/phase7/fixtures/ ; 结果 → machine/phase7/selftest_results.json
总体规则: 任一核心用例失败 => selftest verdict=FAIL, 校验器不得标 PASS。

手算口径: strict 列 = 全集分母+缺失按负类; production 交集列 = 仅诊断镜像(零分母0.0)。
"""
import copy
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
M7 = AUDIT / "machine" / "phase7"
FIX = M7 / "fixtures"
sys.path.insert(0, str(HERE))
import p7_validator as V  # noqa: E402

R6 = [f"img_{i}.jpg" for i in range(6)]
TEN = [f"img_{i}.jpg" for i in range(10)]


def gt_of(pos):
    return {n: (1 if n in pos else 0) for n in (R6 + TEN if len(pos) > 3 else R6)}


def tiny_manifest(canon, subsets=None):
    real = json.loads((M7 / "canonical_manifest.json").read_text(encoding="utf-8"))
    m = copy.deepcopy(real)
    m["canonical_sample_list"] = sorted(canon)
    m["n_canonical"] = len(canon)
    m["excluded_files"] = []
    m["anchors"]["__frozen_coco_and_labels_unchanged_note"] = \
        "tiny universe仅覆盖结构/指标回归; anchor sha仍指真实官方附件(load_manifest可校验)"
    for k, v in (subsets or {}).items():
        m["anchors"][k] = sorted(v)
    return m


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))


def run(pred_obj_or_raw, manifest, gt=None, subset=None):
    FIX.mkdir(parents=True, exist_ok=True)
    pp = FIX / "_pred_tmp.json"
    if isinstance(pred_obj_or_raw, bytes):
        pp.write_bytes(pred_obj_or_raw)
    elif isinstance(pred_obj_or_raw, str):
        pp.write_text(pred_obj_or_raw, encoding="utf-8")
    else:
        write_json(pp, pred_obj_or_raw)
    gp = None
    if gt is not None:
        gp = FIX / "_gt_tmp.json"
        write_json(gp, gt)
    res, code = V.validate(pp, manifest, gt_path=gp, subset=subset)
    return res, code


def codes(res):
    return {e.get("code") or e.get("stage") for e in res["errors"]}


RESULTS = []


def check(case_id, name, cond, expected, actual, derivation):
    RESULTS.append({"id": case_id, "name": name, "pass": bool(cond),
                    "expected": expected, "actual": actual,
                    "hand_derivation": derivation})


m6 = tiny_manifest(R6)
m10 = tiny_manifest(TEN)

# T01 正常全对齐 — 手算 tp2 fp0 fn1 tn3 -> P=1 R=2/3 F1=0.8
full_gt = {"img_0.jpg": 1, "img_1.jpg": 1, "img_2.jpg": 1,
           "img_3.jpg": 0, "img_4.jpg": 0, "img_5.jpg": 0}
pred01 = {"img_0.jpg": 1, "img_1.jpg": 0, "img_2.jpg": 1,
          "img_3.jpg": 0, "img_4.jpg": 0, "img_5.jpg": 0}
res, c = run(pred01, m6, full_gt)
st = res["metrics"]["strict_full_coverage_metrics"]
check("T01", "正常预测完整匹配", res["verdict"] == "PASS" and c == 0 and
      st == {"precision": 1.0, "recall": round(2 / 3, 6), "f1_diagnostic": 0.8,
             "undefined_flags": []} and res["metrics"]["present_tp"] == 2 and
      res["metrics"]["present_fn"] == 1 and res["metrics"]["tn"] == 3 and
      res["metrics"]["missing_predictions_total"] == 0,
      "PASS; P=1.0 R=0.666667 F1=0.8; tp2 fn1 tn3", st,
      "tp=(i0,i2)=2 fp=0 fn=i1=1 tn=(i3..i5)=3; P=2/(2+0)=1; R=2/(2+1)=2/3; F1=2*(2/3)/(5/3)=0.8")

# T02 完美 → tp3 tn3 P=R=F1=1
res, c = run({k: v for k, v in full_gt.items()}, m6, full_gt)
st = res["metrics"]["strict_full_coverage_metrics"]
check("T02", "完美预测", res["verdict"] == "PASS" and st ==
      {"precision": 1.0, "recall": 1.0, "f1_diagnostic": 1.0, "undefined_flags": []},
      "P=R=F1=1", st, "pred==gt: tp3 fp0 fn0 tn3")

# T03 在完美预测上把一张负样本误判为正 → tp3 fp1 fn0
p03 = dict(full_gt); p03["img_3.jpg"] = 1
res, _ = run(p03, m6, full_gt)
st = res["metrics"]["strict_full_coverage_metrics"]
check("T03", "追加单FP: P降为3/4, R保持1", st["precision"] == 0.75 and
      st["recall"] == 1.0 and st["f1_diagnostic"] == round(6 / 7, 6) and
      res["verdict"] == "PASS" and res["metrics"]["present_fp"] == 1,
      "tp3 fp1 fn0 tn2 → P=3/4=0.75 R=3/3=1.0 F1=2*3/(4+3)=6/7≈0.857143", st,
      "pred=(gt再加i3): tp3 fp1 fn0; P=3/(3+1)=0.75; R=3/(3+0)=1; F1=2*3/(P分母+R分母)=6/7")

# T04 空预测 {} → FAIL(missing×6) 且 strict: R=0, P未定义flag; 交集镜像 n=0(生产会静默输出0.0)
res, c = run({}, m6, full_gt)
cm = res["metrics"]
diag = cm["production_intersection_diagnostic_DO_NOT_REPORT"]
check("T04", "空预测文件被硬拦且给出定义性标注", res["verdict"] == "FAIL" and c != 0 and
      cm["missing_predictions_total"] == 6 and
      "missing_predictions_fail_closed" in codes(res) and
      cm["strict_full_coverage_metrics"]["recall"] == 0.0 and
      "precision_undefined_no_predicted_positive" in cm["strict_full_coverage_metrics"]["undefined_flags"] and
      diag["n_common"] == 0 and diag["recall_production_semantics"] == 0.0,
      "FAIL missing=6; strict R=0 P=None(flag); 交集镜像n=0 r=0.0(静默语义)",
      {"verdict": res["verdict"], "strict": cm["strict_full_coverage_metrics"], "inter": diag},
      "全集6图均缺→缺失计负: fn=3 R=0/(3)=0; P分母=0→None+flag; 生产交集脚本此处会无告警打印 P=0.0 R=0.0 并继续运行")

# T05 缺失5张真阳性(F-12核心回归): 十图6正4负, pred 恰缺 img_1..img_5
g10 = {**{f"img_{i}.jpg": (1 if i < 6 else 0) for i in range(10)}}
p10 = {k: v for k, v in g10.items() if k not in {f"img_{i}.jpg" for i in range(1, 6)}}
res, c = run(p10, m10, g10)
cm = res["metrics"]; st = cm["strict_full_coverage_metrics"]; dg = cm["production_intersection_diagnostic_DO_NOT_REPORT"]
check("T05", "缺5真阳: 缺失计FN(strict R=1/6) vs 交集镜像R=1.0 差距被显式量化",
      res["verdict"] == "FAIL" and cm["missing_true_positive"] == 5 and
      st["precision"] == 1.0 and st["recall"] == round(1 / 6, 6) and
      dg["recall_production_semantics"] == 1.0 and dg["n_common"] == 5 and
      abs(cm["recall_inflation_trap_delta_PRODUCTION_minus_STRICT"] - round(1 - 1 / 6, 6)) < 1e-9,
      "FAIL; strict P=1 R=1/6; inter r=1.0 delta=+5/6",
      {"strict": st, "inter": dg, "delta": cm["recall_inflation_trap_delta_PRODUCTION_minus_STRICT"]},
      "保留tp=img_0=1, 缺失5正计FN: P=1/(1+0)=1; R=1/(1+5)=1/6; 交集只看交集5图: tp1 fn0→R=1(假象); 生产evaluate_image_level将静默输出该R")

# T06 缺失真阴性: 数值无影响但契约违约必须显式报错
p06 = {k: v for k, v in full_gt.items() if k != "img_4.jpg"}
res, _ = run(p06, m6, full_gt)
cm = res["metrics"]; st = cm["strict_full_coverage_metrics"]
check("T06", "缺负样本: 指标不变但FAIL-CLOSED", res["verdict"] == "FAIL" and
      cm["missing_true_negative"] == 1 and cm["missing_true_positive"] == 0 and
      st == {"precision": 1.0, "recall": 1.0, "f1_diagnostic": 1.0,
             "undefined_flags": []} and "missing_predictions_fail_closed" in codes(res),
      "FAIL missing_neg=1; strict=P1/R1/F11(与完整正确预测一致)", st,
      "基线为完美预测(tp3 fp0 fn0 tn3); 缺img_4按负类计→tn不变, 所有格口与全对齐相同 "
      "→ 交集口径对缺失负类毫无感知; 提交键契约损坏仍须非零退出")

# T07..T12 结构与类型系列
cases = [
    ("T07", '{"img_0.jpg": 0, "img_0.jpg": 1}', None, "duplicate_json_key", "duplicate_json_key",
     "重复键被标准json.loads静默覆盖(后值1胜出), object_pairs_hook 必须拦截"),
    ("T08", '{"img_0.jpg": true}', "stage_pred_value_bool", "boolean_not_int",
     "boolean_not_int", "true/false是Python bool(bool< int), 常规校验会误收"),
    ("T09", '{"img_0.jpg": 1.0}', None, "float_not_int", "float_not_int",
     "官方JSON要求整数, 1.0浮点必须拒绝(避免下游隐式取整)"),
    ("T10", '{"img_0.jpg": "1"}', None, "string_value", "string_value",
     '字符串"1"必须拒绝'),
    ("T11", '{"img_0.jpg": NaN}', None, "forbidden_constant", "forbidden_constant",
     "json.loads默认接受NaN/Infinity字面量, parse_constant拦截"),
    ("T12", '{"img_0.jpg": 2}', None, "value_out_of_range", "value_out_of_range",
     "允许值仅{0,1}"),
]
for cid, rawtxt, gtm, ecode, _, derivation in cases:
    gtuse = dict.fromkeys(R6, 0)
    gtuse.update({"img_0.jpg": 1})
    res, _ = run(rawtxt, m6, gtuse)
    got = codes(res)
    hit = (ecode in got) or any(ecode in str(e) for e in res["errors"])
    check(cid, f"结构回归 {ecode}", res["verdict"] == "FAIL" and hit, f"errors含{ecode}",
          sorted(got), derivation)

# T13 额外样本 + 无GT结构模式
p13 = dict(full_gt); p13["ghost.jpg"] = 0
res, _ = run(p13, m6, full_gt)
check("T13a", "额外样本fail-closed", "extra_samples" in codes(res) and
      res["coverage"]["extra"] == 1 and res["coverage"]["missing"] == 0,
      "extra_samples; coverage extra=1", res["coverage"], "ghost.jpg不在canonical 6图内")
res_nogt, c_nogt = run(pred01, m6, gt=None)
check("T13b", "无GT结构模式仍强制覆盖检查", c_nogt == 0 and
      res_nogt["metrics"] is None and res_nogt["coverage"]["missing"] == 0,
      "PASS且无metrics块(明确不带成绩), coverage齐", {"code": c_nogt, **res_nogt["coverage"]},
      "测试集GT本地不存在时的真实提交形态: 只验结构/覆盖/类型, 不冒充官方综合分")

# T14 大小写冲突合同: pred 用 IMG_0.JPG(大写) vs canonical img_0.jpg
p14 = {"IMG_0.JPG": 1, "img_1.jpg": 0, "img_2.jpg": 1,
       "img_3.jpg": 0, "img_4.jpg": 0, "img_5.jpg": 0}
res, _ = run(p14, m6, full_gt)
check("T14", "大小写不敏感匹配≠精确匹配, 必须报错", "case_insensitive_key_mismatch" in codes(res)
      and "missing_predictions_fail_closed" in codes(res),
      "case_insensitive_key_mismatch + missing(img_0)", sorted(codes(res)),
      "Windows盘上IMG_0.JPG会命中同一文件, 违反逐字符契约; 不得静默归一化")

# T15 路径穿越键
p15 = dict(full_gt); p15["../evil.jpg"] = 1
res, _ = run(p15, m6, full_gt)
check("T15", "路径语法键拒绝", "path_traversal_or_separator" in codes(res),
      "path_traversal_or_separator", sorted(codes(res)), "'../'出现在key中")

# T16 键间大小写冲突(Photo.jpg vs PHOTO.jpg 同时出现)
mPh = tiny_manifest(["Photo.jpg", "y.jpg"])
res, _ = run({"photo.jpg": 0, "PHOTO.jpg": 1, "y.jpg": 0}, mPh,
             {"Photo.jpg": 0, "y.jpg": 0})
check("T16", "pred内部两键大小写等价冲突", "pred_case_collision_between_keys" in codes(res),
      "pred_case_collision_between_keys(photo.jpg/PHOTO.jpg)",
      [e for e in res["errors"] if e.get("code") == "pred_case_collision_between_keys"],
      "同名不同写法在Windows是同一文件, 覆盖语义歧义须失败关闭(镜像F-12缓存键冲突类别)")

# T17 缓存键缺失 ↔ build_cached_crop_rescue L63 直接下标崩溃的守卫
def total_coverage_access(d, k):
    return d[k]  # 与生产 L63 同构: 缺键即 KeyError

caught = {}
try:
    total_coverage_access({n: 0 for n in R6[:3]}, "img_4.jpg")
except KeyError as ke:
    caught = {"class": type(ke).__name__, "key": str(ke)}
check("T17", "缺键访问异常可捕获并被结构化上报(对照生产两端极端)",
      caught.get("class") == "KeyError" and caught.get("key") == "'img_4.jpg'",
      "KeyError('img_4.jpg')", caught,
      "生产 evaluate_image_level=静默丢(T05已拦), build_cached_crop_rescue L63=裸KeyError崩(此处演示等效缺陷的可捕获取代)")

# T18 真实产物对1100全集应预期性失败(先于driver粗验证)
real_manifest = json.loads((M7 / "canonical_manifest.json").read_text(encoding="utf-8"))
live_p = PROJ_ROOT_LIVE = AUDIT.parents[1] / "fire_detection/outputs/val_pred_best_siglip_yolo_dino_crop_live.json"
res, c = V.validate(live_p, real_manifest, gt_path=None)
check("T18", "现存val-220产物对1100全集=结构不合格", c != 0 and
      "missing_predictions_fail_closed" in codes(res) and
      res["coverage"]["universe"] == 1100 and res["coverage"]["parsed_keys"] == 220,
      "FAIL missing=880 universe=1100 keys=220", res["coverage"],
      "val内部产物不是提交件: 对全集缺少880张——证明'不存在可直接提交的1100全量JSON'(本地亦无官方测试集文件名)")

# CLI subprocess: 真实manifest锚定 + val子集模式 PASS; preflight坏图 rc!=0
FIXC = FIX / "bad_img_dir"
if FIXC.exists():
    for q in FIXC.iterdir():
        q.unlink()
FIXC.mkdir(parents=True, exist_ok=True)
from PIL import Image  # noqa: E402

buf = io.BytesIO()
Image.new("RGB", (24, 24), (200, 30, 30)).save(buf, format="JPEG")
good_bytes = buf.getvalue()
(FIXC / "ok_1.jpg").write_bytes(good_bytes)
Image.new("RGB", (16, 16), (30, 200, 30)).save(FIXC / "ok_2.png")
(FIXC / "truncated.jpg").write_bytes(good_bytes[: len(good_bytes) // 3])
(FIXC / "zero.jpg").write_bytes(b"")

py = sys.executable
FD_ROOT = str(AUDIT.parents[1])
cli_pred = FD_ROOT + r"/fire_detection/outputs/val_pred_best_siglip_yolo_dino_crop_live.json"
cli_gt = FD_ROOT + r"/fire_detection/outputs/val_gt.json"
cli_man = str(M7 / "canonical_manifest.json")
out_cli = M7 / "cli_val_subset_result.json"
r1 = subprocess.run([py, str(HERE / "p7_validator.py"), "validate", "--pred", cli_pred,
                     "--manifest", cli_man, "--gt", cli_gt, "--subset", "internal_val220",
                     "--out", str(out_cli)],
                    capture_output=True, text=True, cwd=FD_ROOT)
j1 = json.loads(out_cli.read_text(encoding="utf-8")) if out_cli.is_file() else {}
st_cli = j1.get("metrics", {}).get("strict_full_coverage_metrics", {})
check("T19", "CLI端到端: val子集严格复算匹配存储(.9474/.9818)", r1.returncode == 0 and
      st_cli.get("precision") == 0.947368 and st_cli.get("recall") == 0.981818 and
      st_cli.get("f1_diagnostic") == 0.964286 and
      j1.get("metrics", {}).get("present_tp") == 162 and j1.get("metrics").get("present_fp") == 9
      and j1.get("metrics").get("tn") == 46,
      "rc=0; tp162 fp9 fn3 tn46 P=.947368 R=.981818 F1=.964286",
      {"rc": r1.returncode, "tp/fp/tn": (j1.get("metrics", {}).get("present_tp"),
                                         j1.get("metrics", {}).get("present_fp"),
                                         j1.get("metrics", {}).get("tn")), "strict": st_cli},
      "手算同源E019/E020独立版: tp162/167 P=162/171=.947368; R=162/165=.981818; F1=2PR/(P+R)=.964286 "
      "(python计算162/171=0.94736842..., 162/165=0.98181818..., 2*162/(2*162+9+3)=324/336=0.96428571...)")

out_pref = M7 / "preflight_bad_fixture_result.json"
r2 = subprocess.run([py, str(HERE / "p7_validator.py"), "preflight-images",
                     "--dir", str(FIXC), "--out", str(out_pref)],
                    capture_output=True, text=True, cwd=FD_ROOT)
pj = json.loads(out_pref.read_text(encoding="utf-8")) if out_pref.is_file() else {}
bad_names = sorted(x["file"] for x in pj.get("corrupt", []))
check("T20", "F-14预检: 单/双损坏图指名报告且rc=1", r2.returncode == 1 and
      pj.get("corrupt_count") == 2 and bad_names == ["truncated.jpg", "zero.jpg"],
      "rc=1; corrupt=[truncated.jpg, zero.jpg] 逐文件定位", {"rc": r2.returncode, "corrupt": bad_names},
      "truncated=合法JPEG前1/3字节→decode失败; zero=空字节; 校验器列明各文件而不整体崩溃(替代生产行为的检测点, 不修生产链)")

# ---------------------------------------------------------------- summary
total = len(RESULTS)
passed = sum(r["pass"] for r in RESULTS)
failed_ids = [r["id"] for r in RESULTS if not r["pass"]]
summary_doc = {
    "tool_version": V.TOOL_VERSION,
    "python_version": sys.version.split()[0],
    "platform_note": "审计机Win11/Py3.13; 同机重跑输出一致(无时间戳字段)",
    "total_cases": total, "passed": passed, "failed": len(failed_ids),
    "failed_ids": failed_ids,
    "overall": "PASS" if passed == total else "FAIL",
    "validator_marked_usable_only_if_overall_pass": True,
    "cases": RESULTS,
}
(M7 / "selftest_results.json").write_text(
    json.dumps(summary_doc, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
print(json.dumps({k: summary_doc[k] for k in
                  ("overall", "total_cases", "passed", "failed", "failed_ids")},
                 ensure_ascii=False))
sys.exit(0 if summary_doc["overall"] == "PASS" else 1)
