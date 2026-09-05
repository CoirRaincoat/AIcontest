# CHANGELOG.md — 整改阶段R1（G3+G4）实际变更记录

日期: 2026-08-27 · 授权: 仅G3/G4 · Git HEAD基线: `874de91` · 执行者: 首席算法审计员(本会话)

## 变更文件清单

| # | 文件 | 动作 | 修改前 SHA-256 | 修改后 SHA-256 |
|---|---|---|---|---|
| 1 | `fire_detection/src/evaluate_image_level.py` | 重写实现(58→~230行) | `73f996fc…88f7a3` | `6b776165…741f37` |
| 2 | `fire_detection/src/predict_best_fusion.py` | 外科手术式修改(688→~780行) | `e6ba9e7a…da4a08` | `092bae45…90f2647` |
| 3 | `fire_detection/src/batch_safety.py` | **新增**(纯标准库, ~140行) | —(新文件) | `5403634b…22c5d2aa` |

完整哈希见 `backups/R1_baseline.sha256` / `backups/R1_post_patch.sha256`；统一diff见 `diffs/g3_*.patch`、`diffs/g4_*.patch`。

## G3 evaluate_image_level.py — 修改函数明细

- `load_binary_json`(重写): 合同解析——UTF-8读取; `object_pairs_hook` 全文档拒绝重复键; `parse_constant` 拒绝 NaN/±Infinity 字面量; 顶层必须object; 值必须非bool的int∈{0,1}(true/1.0/"1"/null/越界各自给出机器可读reason); 违规抛 `ContractError`。
- 新增 `value_problem(v)` / `ContractError` / `_object_pairs` / `_reject_constant`。
- 新增 `evaluate(gt,pred)->dict`: 官方全集口径(TP/FP/FN/TN + strict P/R + F1诊断列 + 缺失正/负计数 + `intersection_basis_UNSAFE_DO_NOT_REPORT`镜像 + `recall_inflation_trap_delta` + `undefined_flags` + `complete_coverage`); 零分母返回None而非0.0。
- 新增 `contract_problems(gt,pred)`: 缺失→`missing_predictions_fail_closed`、多余→`extra_predictions_fail_closed`(带名单)。
- `main`(重写): 合同错误→stderr明细+退出码2; 非全覆盖→打印STRICT结果+明确"INCOMPLETE"警告+交集陷阱量化行+FATAL名单, 退出码2, **不输出可误读成绩块**; 全覆盖→五行输出与旧版逐字节一致(Images evaluated/TP..TN/Precision/Recall/F1), 退出码0。
- CLI契约不变: `--gt`/`--pred` required。

## G4 predict_best_fusion.py — 修改函数明细

- 模块头: 新增 `import sys` + `sys.path.insert` + `import batch_safety as bs`。
- `predict_siglip` / `predict_dinov3`(签名+批循环): 新增 `failures` 参数(默认None自建); 装载段逐图 try/except(`load_preprocess`)跳坏图; 推理段整段 try/except(`inference`), 失败时该批每张可用图各记一条后整批跳过(模型调用天然批级); 可用图 zip 评分, 进度打印保留。
- `predict_yolo`(重写循环): 逐图逻辑移入 `_score_one` 闭包, 经 `bs.run_items_isolated` 逐图隔离(`inference`), 进度计数保留(含失败图)。
- `collect_yolo_proposals`: 内层逐图改 `_collect_one` + `run_items_isolated`; 收集结果按名回填; 进度行为简化为每模型一行 `Crop proposals {label}: N/N`。
- `predict_crop_siglip`: 逐 proposal 装载/裁剪/编码 try/except(`load_preprocess`)仅弃该proposal(分数保持-1.0即不触发rescue); 推理段同siglip批级登记。
- `main`: 创建共享 `failures` 日志并传入全部五个阶段; 融合循环对缺上游分数的图 `record_failure(stage="fusion")` 后 continue(不再KeyError); 输出纪律——有失败: 不写正式提交名, 经 `bs.write_partial_products` 原子写 `<stem>_PARTIAL.json`/`_PARTIAL_details.csv`/`_failures.json`, 返回3; 无失败: 文件名/内容结构不变, JSON/CSV/metadata 三件套改原子写, 返回0; details为空不再 IndexError。入口尾改 `raise SystemExit(main())`。
- **策略零漂移守卫**: 13项argparse默认(τ_sig=0.22, τ_dino=0.08, τ_crop=0.97, conf .10/.20/.20, imgsz=960, iou=0.45, min_area=0.0, crop-ctx/min/max)经测试AST逐一断言未变(s01)。

## 未变更项(审计承诺)

生产数据/权重/既有outputs/冻结指标/阈值与融合规则/既有审计报告内容(仅按惯例追加状态行与索引) — 全部未动。`git status` 足迹恰为: M×2(上表) + ?? batch_safety.py + ?? audit/。

## 测试与验证

- `tests/test_g3_g4.py`: **30/30 PASS**(29 unittest + 1 change-scope守卫), rc=0, 重跑字节一致。方法构成: unit×13 / cli×5 / static_source_check×6 / integration_unit×1 + scope×1。
- 验证级别声明(强制): **单元级验证**。predict_best_fusion.py 因 torch/ultralytics 未安装无法在本机导入, 其正确性由 batch_safety 真实执行 + 源码结构断言承载; 未运行任何真实端到端GPU推理(YOLO/VLM前向), 该项验证属 G5/G7, 未批未做。

## 回滚方式

```
git checkout -- fire_detection/src/evaluate_image_level.py fire_detection/src/predict_best_fusion.py
rm fire_detection/src/batch_safety.py
# 或直接: git stash / git checkout 874de91 -- fire_detection/src/
# 校验: sha256sum 两文件应等于 backups/R1_baseline.sha256 中的值
```
