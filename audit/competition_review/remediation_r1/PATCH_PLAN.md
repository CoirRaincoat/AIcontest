# PATCH_PLAN.md — 整改阶段R1执行计划（G3/G4，仅此两项）

授权: 用户2026-08-27批准 · 范围=两个生产源文件的定向修复+相关测试 · 其余决策门(G1/G2/G5/G6/G7/P6延期)全部未批。
原则: 最小侵入、无阈值/模型/策略变化、正常路径字节级行为兼容、失败路径fail-loud。

## 基线(修改前实测)

- Git HEAD: `874de91aafdd18b30e083ee8801e570936953799` (main)；工作区干净(仅审计目录未跟踪, 属本会话产物, 无外来代码改动)。
- `fire_detection/src/evaluate_image_level.py` SHA256: `73f996fcf5391c5d29a7f51c30c01cfe7910e35ac182e245bcbab3d53288f7a3`（58行）
- `fire_detection/src/predict_best_fusion.py` SHA256: `e6ba9e7ae8646fd1a4ecef54e7b1466e741aa6a577f9261f01b53bcf6fda4a08`（687行）
- 关键函数位置(原): evaluate L20-31 main+交集口径主体; predict: iter_images L111, predict_siglip L127(batch循环L149-175), predict_yolo L186(L200-225逐图), predict_dinov3 L235(L267-285), collect_yolo_proposals L327(L341-366), predict_crop_siglip L408(L439-477), 主融合循环 L591-619, 输出写盘 L621-629/679-682(非原子)。
- 备份: remediation_r1/backups/*.pre_r1.py + backups/R1_baseline.sha256

## G3 设计（evaluate_image_level.py 重写实现层、保留CLI契约）

1. `load_binary_json`: 升级为合同解析 —— UTF-8; `object_pairs_hook` 拒绝重复键(递归生效); `parse_constant` 拒绝 NaN/Infinity 字面量; 顶层必须是object; 每个值必须 **非bool 的 int ∈ {0,1}**（拒绝 true/false/1.0/"1"/null/越界）；违规抛 `ContractError`(载 `ValueError`)。
2. 新 `evaluate(gt, pred) -> dict`: 官方口径=以 GT 键全集为分母——
   - 缺失预测: 记入 manifest(missing_names/missing_true_positive/…), 且诊断列 strict 把缺失按负类计 FN(tp/(tp+fn+缺失正)), **绝不静默消失**;
   - 多余预测: 列出并按契约判错(extra 不参与任何格口);
   - 零分母: None+"undefined_flag", 不再伪装 0.0;
   - 同时输出 `intersection_basis_UNSAFE_DO_NOT_REPORT` 镜像列(仅用于展示旧缺陷差异)与 `recall_inflation_trap_delta`;
   - 另导出 `contract_problems(gt,pred)` 供调用方/测试获取错误清单。
3. `main()`: 载入任一合同错误→打印明细并 SystemExit(2); 评估完成后若存在 missing/extra → 打印严格结果+明确警告行("INCOMPLETE...不构成有效提交成绩")并 **退出码2**; 全覆盖时打印内容与旧版逐字符等价(TP/FP/FN/TN/Precision/Recall/F1 同格式同值, 含 Images evaluated 行), 退出码0。
4. 不新增第三方依赖; 文件仍可被测试直接 import。

## G4 设计（predict_best_fusion.py + 新增 stdlib 辅助模块）

1. 新增 `fire_detection/src/batch_safety.py`(纯标准库, 可独立单测): 
   - `record_failure(failures,file,stage,exc,index)` 标准化失败条目(file/stage∈{load_preprocess,inference,fusion}/exception_type/message截断/index)+控制台可见输出;
   - `run_items_isolated(items,worker,*,stage,failures,key_of)` 通用逐图隔离器(yolo与crop-proposal两处真实逐图循环直接复用);
   - `atomic_write_text/json(path,text|obj)` 经 `<name>.tmp` + `os.replace` 原子落盘;
   - `write_batch_failures_manifest(output_stem_parts...)`; 常量 `EXIT_PARTIAL_FAILURES=3`(区别于G3的2)。
2. 五个打分/提框函数增加可选参数 `failures`(默认None时自建局部表, 主流程传入共享列表):
   - siglip/dino 批装载入段: 逐图 try/except(load_preprocess)跳过坏图并把可用图并入批; 推理段整体 try/except(inference), 失败时该批每张可用图各记一条(模型调用天然整批受影响, 逐图登记保证清单可诊断);
   - predict_yolo / collect_yolo_proposals: 改用 run_items_isolated 逐图隔离;
   - crop: 逐 proposal 装载裁剪(load_preprocess)失败仅弃该 proposal; 推理段同批处理。
3. 主融合循环(L591区): 任一上游分数字典缺该图名 → record_failure(stage="fusion")并 continue(其余图继续), 不再 KeyError 中止。
4. 输出纪律:
   - 无失败: 路径/文件名/内容结构与旧行为一致(signature/metadata键不变), 但三件套(JSON/CSV/metadata)改原子写入; 返回0;
   - 有失败: **主提交文件名不写**(防半成品冒充完整提交) —— 改写 `<stem>_PARTIAL.json`(predictions只含成功图) + `<stem>_PARTIAL_details.csv`(若有成功行) + `<stem>_failures.json` 清单; metadata 增加 status/failed_image_count/failed_images_file 仅在partial模式注入; 打印 INCOMPLETE 批头; **返回3**;
   - 成功细节空列表时不再因 details[0] 触发 IndexError。
5. 入口尾改为 `raise SystemExit(main())` 使包装脚本($LASTEXITCODE)可感知3。阈值/权重/imgsz等 argparse 默认与融合规则表达式一律不动(由静态断言守卫)。

## 测试设计（remediation_r1/tests/test_g3_g4.py, 单进程可重复, 全部手算期望）

- 方法标注: `unit`=直接执行生产模块代码(evaluate_image_level/batch_safety 无重依赖可导入); `cli`=子进程运行修复后的评测器(真实生产入口); `static_source_check`=结构断言(predict_best_fusion 因缺 ultralytics 无法导入, 如实标注单元级/静态级, **非端到端**)。
- 覆盖映射: 用户9类要求逐一对应 T-G3-01..07(g01-g07含CLI全量回归val_gt×live=.9474/.9818/.9643) 与 T-G4-01..12(h01-h09+s01-s04)。partial产物 → p7_validator 必拒(h09)。
- 已知边界如实声明: YOLO/VLM真实前向未跑(G5/G7未批); mock仅替代异常源。

## 回滚方式

```
git checkout -- fire_detection/src/evaluate_image_level.py fire_detection/src/predict_best_fusion.py
rm fire_detection/src/batch_safety.py
# 校验: sha256sum 两文件应回到本文件“基线”节的哈希
```

风险与回退触发: 若正常小样回归打印与旧版不一致或引入任何阈值漂移(静态断言失败) → 立即回滚并在报告披露。
