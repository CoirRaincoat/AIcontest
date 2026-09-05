# VALIDATOR_README.md — 独立CPU提交校验器使用说明（阶段7）

工具: `scripts/p7_validator.py`（核心） · `scripts/p7_selftest.py`（自证21例） · `scripts/p7_validate_artifacts.py`（现存产物核验）
依赖: 仅标准库 + Pillow（仅 `preflight-images` 需要）。不 import 生产 src/*，指标为全新实现。

## 确定性运行命令（审计机 Py3.13.5, Win11；输出无时间戳、sort_keys，同机重跑字节一致）

```powershell
cd C:\Users\CoirRaincoat\PyCharmMiscProject\AIcontest\AI

# 0) 冻结/刷新 canonical 清单（官方 train_coco.json 1100 图；含对第1101文件的显式排除记录）
python audit/competition_review/scripts/p7_validator.py freeze

# 1) 自证套件（任一失败 => overall=FAIL，校验器不得标可用）
python audit/competition_review/scripts/p7_selftest.py            # rc 0=PASS 1=FAIL

# 2) 校验任意提交件（真实场景：无 GT 时自动进入"仅结构"模式，绝不冒充官方综合分）
python audit/competition_review/scripts/p7_validator.py validate ^
  --pred <SUBMISSION.json> ^
  --manifest audit/competition_review/machine/phase7/canonical_manifest.json ^
  [--gt <GT.json>] [--subset internal_val220] [--out <result.json>]

# 3) 图像目录预检（F-14 检测点：逐文件指名损坏项，绝不让单图坏图静默通过）
python audit/competition_review/scripts/p7_validator.py preflight-images --dir <DIR> [--out <result.json>]

# 4) 现存产物双模式核验（val子集严格复算 + 对1100全集预期性失败演示）
python audit/competition_review/scripts/p7_validate_artifacts.py   # rc 0=全部OK
```

退出码: **0**=PASS · **1**=校验失败（结构/覆盖/坏图，附逐条 error 明细）· **2**=致命 IO/用法错。
每次 `validate` 前都会复核清单锚点 sha256（train_coco.json / train_image.json / 被排除文件）——输入漂移即拒绝运行。

## 口径契约（与生产实现的差异 = 本工具存在的理由）

| 规则 | 本校验器 | 生产 evaluate_image_level.py |
|---|---|---|
| 分母 | 官方全集，缺失预测→硬错误 fail-closed | gt∩pred 交集，缺失仅打印[F-12] |
| 缺失真阳 | 报错 + strict 列按负类计入 FN 展示影响面 | 静默消失 → R 虚高(+3.0pp/缺5张)[E026] |
| 交集口径 | 仅作诊断列 `production_intersection_diagnostic_DO_NOT_REPORT` | 即为正式口径 |
| 零分母 | None + `undefined_flags` 显式标注 | 静默取 0.0 |
| 综合分 | 不生成任何综合分（官方未规定公式） | n/a |

错误码速查: `duplicate_json_key`（object_pairs_hook 递归检测）、`forbidden_constant`（NaN/Infinity 字面量）、
`boolean_not_int` / `float_not_int` / `string_value` / `null_value` / `value_out_of_range`、
`missing_predictions_fail_closed` / `extra_samples` / `case_insensitive_key_mismatch` /
`pred_case_collision_between_keys` / `path_traversal_or_separator` / `not_utf8` / `utf8_bom_present` /
`unexpected_extra_files_on_disk_fail_closed` / `canonical_case_insensitive_collision`。

## 第1101张的显式排除（非模糊去重）

- 相对路径: `fire_detection/data/images/train/images/raw_fire_relabel_dp_20402(1).jpg`
- SHA-256: `50264a1ddb0fbeb0103ad194bb25c7c2…`（完整值冻结于 machine/phase7/canonical_manifest.json）
- 与 canonical `raw_fire_relabel_dp_20402.jpg` 字节相同（SHA256 相等），Windows "(1)" 复制残留[F-04已结案]
- 排除逻辑为**精确名单匹配**；凡出现其他多余/缺失/大小写冲突文件一律 fail-closed（T13/T14/T18 演示）

## 产物索引

- 清单: machine/phase7/canonical_manifest.json（1100 样本表 + 锚哈希 + val220 子集）
- 自证: machine/phase7/selftest_results.json（T01–T20，每例含手算推导 actual-vs-expected 自动比对）
- 产物核验: machine/phase7/existing_artifacts_validation.json
- 夹具: machine/phase7/fixtures/bad_img_dir/（ok_1.jpg ok_2.png truncated.jpg zero.jpg —— 由 p7_selftest.py 现场确定性重建）

## 能力边界（防止误读）

1. 结构校验能力通过 ≠ 端到端提交就绪：本机不存在官方测试集，也无任何 1100 键的预测 JSON（已全库扫描证实），真正的提交文件必须等测试集清单后用 `validate` 无GT模式现场校验。
2. `preflight-images` 是提交前筛查点，不修改也不弥补生产推理链"单图损坏中止整批且零输出"[F-14] 的行为。
3. 大小写契约按逐字符精确匹配执行（即使 Windows 盘大小写不敏感也不归一化放行）。
