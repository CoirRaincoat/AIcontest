# COMMANDS.md — 实际执行命令日志（追加式）

格式: [UTC+8 时间] 命令摘要 | 退出码 | 耗时 | 输出要点

## 阶段 0

- 01:05 `ls -la` 根目录 + `du -sh */` | 0 | <5s | fire_detection=2.4G, external_datasets=3.0G, hf_cache=1.8G
- 01:06 `git rev-parse/branch/status/remote` | 0 | <1s | HEAD=874de91, main, 干净
- 01:07 `uname/python --version/df/powershell RAM+CPU` | 0 | <10s | Win11 build26200; Py3.13.5; C盘剩8.1G; 15.7G RAM; i7-14650HX×24
- 01:07 `nvidia-smi --query-gpu...` | 0 | <2s | RTX 5060 Laptop 8151MiB(空闲5441MiB), drv573.24, cc12.0
- 01:08 `python - importlib.metadata 探测16包` | 0 | <3s | 缺 ultralytics/transformers/timm/gradio/safetensors/onnx*/tensorrt
- 01:09 `find 扩展名统计 / find -printf 最大文件` | 0 | ~20s | 58,561文件; Top32大文件表
- 01:10 `find *.json 元数据定位` | 0 | ~15s | train_coco.json=505,090B; train_image.json=40,046B（均位于 data/images/train/）
- 01:12 `-d 存在性探测×5` | 0 | <2s | ANACONDA2 envs、C:\AI、部署应用目录全部 MISSING
- 01:13 `find 分目录图像计数` | 0 | ~30s | raw=1101(!), train=880, val=220, ext_v1=2880, crop_v2=5281; outputs 完整清单
- 01:14 Read 导航文档 + README.md + grep 权重默认值/阈值 | 0 | <5s | 详见 E012/E013
- 01:20 `sha256sum 225关键文件` > machine/baseline_sha256.txt | 0 | ~90s | 成功225行
- 01:21 `find pt/safetensors 清单` > machine/weights_inventory.txt | 0 | ~10s | 32条目

## 阶段 1–3（2026-08-27 01:3x–02:0x）

- Read predict_best_fusion.py(688行全读) / predict_best_fusion_crop.ps1 / evaluate_image_level.py / build_cached_crop_rescue.py / coco_to_yolo.py | 0 | — | E014-E018
- Glob **/*siglip_yolo_dino* → 12个文件存在(修正F-05首轮误判) | 0
- `python audit_phase3_dataset_metrics.py`(审计目录内脚本,只读生产数据) | 0 | ~40s | machine/dataset_and_metrics_audit.json + 控制台摘要
- `python audit_phase3_followup.py` | 0 | ~5s | machine/followup_f04_f05.json: 重复副本SHA256一致; 应用规则@记录阈值复现 P=.948 R=.9939
- `python audit_phase3_split_leakage.py` v1(regex过贪婪,弃用) → 内联重写两轮 | 0 | ~10s | machine/split_leakage_audit.json: 98.2%≤10帧差
- cat build_manifest.json; ls external_datasets/DFire; find license/readme→无 | 0 | E022/F-09
- grep args.yaml ×4 runs | 0 | E022 最终链纯官方数据
- 写入: REQUIREMENTS_MATRIX.md / FINDINGS.md / STATE.md / EVIDENCE_INDEX.md / machine/stage123_summary.json

## 阶段 4–5（2026-08-27 02:1x–03:0x，用户批准①②④ + ③暂缓转审批单）

- `python audit_phase4_threshold_ci.py` | 0 | ~3min | bootstrap_ci.json(CI两组)+threshold_scan_siglip.csv(277点)+f12_quant demo(经两轮修正语义后定稿)
- `python audit_phase45_bundle.py` | 0 | ~2min | near_duplicate_hash.json(22对d=0/85≤8) + experiments_table.csv + checkpoints_introspection.json + corrupt_scan.json(0/1101坏图) + provenance_app_rule.json(无dino轴/无0.85,0.70行/mtime链)
- `python audit_phase45_f11_split.py` | 0 | ~40s | f11_proposed_split.json: 773簇, 922/178, 挪84张, 重划后交叉d≤8=0(跨侧max=55), 边界带32对
- 修正记录: F-12 demo 首版"proper coverage"未模拟缺失→重算为严格FN口径(R=.9636)；split脚本v1 regex弃用改内容哈希；bundle_console GBK mojibake→文件UTF8正确
- 写入: METRICS.md / F11_SPLIT_PROPOSAL.md / P6_APPROVAL_REQUEST.md / machine/stage45_summary.json + 更新 FINDINGS(F-11/F-12/F-14/N3/N5/+F-15)、STATE、EVIDENCE_INDEX(E024–E031)、REQUIREMENTS_MATRIX 注记

## 阶段 7（用户决议C: 跳过阶段6, 无pip/无GPU; 纯CPU独立校验器）

- `p7_validator.py freeze` | 0 | ~8s | canonical_manifest.json: 1100样本冻结+排除文件SHA256(50264a1d…, 与canonical字节相同)+锚哈希+val220子集
- `p7_selftest.py` | 首轮 FAIL×4(T03/T06期望值笔误、T15错误码形状、T19子集键名) → 修正 validator 错误对象结构(code/message 规范化)、subset 键解析、两处手算期望 | 复跑 **21/21 PASS** rc=0 | ~20s
- `p7_validate_artifacts.py` | 0 | ~15s | existing_artifacts_validation.json: 5产物行OK(严格复算==E019==sidecar三方一致); mode-B 全部 missing=880 预期失败; gt_identity=true
- determinism 复跑比对 existing_artifacts_validation.json 字节一致=True; 全库JSON扫描≥1100键 = NONE
- 写入: VALIDATOR_README.md / machine/stage7_summary.json + 更新 STATE、FINDINGS(F-12/F-14阶段7栏+新增N6)、EVIDENCE_INDEX(E032–E034)、REQUIREMENTS_MATRIX(R4/R10增注)

## 阶段 9–10（用户批准: 执行9后直入10; 阶段6永久跳过保持未验证、阶段8不授权实测）

- 会话记录检索(grep transcript) | 0 | <1min | 校回冻结评分原文: 成熟度0–5×权重公式、硬门槛6条全文、上限5条全文 —— 未凭记忆重述规则
- 纯写作产物(零GPU/零网络): DEFENSE_QA.md(10×6) / FINAL_AUDIT_REPORT.md / EXECUTIVE_SUMMARY.md / REMEDIATION_ROADMAP.md(P0–P3+G1–G7门) / machine/final_audit_summary.json
- 同步: STATE(8=未验证记U / 9–10 COMPLETE+最终判定行)
- 审计终止态: 全部授权边界未被突破; 无生产文件改动; audit目录外零写入

## 规则

后续每个阶段的命令在执行后追加；失败命令也记录。MAX_SINGLE_COMMAND_MINUTES=30，超限前须先申请。

## 整改阶段 R1（用户批准: 仅G3/G4生产修复; G1/G2/G5/G6/G7未批, P6延期并G7）

- 基线: HEAD==874de91 复核; `git status --porcelain`=仅?? audit/(本会话产物, 无外来改动→不触发暂停); 目标文件SHA256×2记录并备份(backups/*.pre_r1.py)
- `python -m py_compile`×3 + AST函数映射 | 0 | <2s | COMPILE_OK
- Edits: evaluate_image_level.py 重写(G3) / 新增 batch_safety.py / predict_best_fusion.py 外科修改×9处(G4)
- `git diff`×2 → diffs/g3_*.patch(282行) g4_*.patch(573行); `sha256sum`→backups/R1_post_patch.sha256; git status 足迹恰=M×2+?? batch_safety.py+?? audit/
- `python remediation_r1/tests/test_g3_g4.py` 首轮 26/30 → 修正3处**测试期望错误**(c01选错产物:E019链头对应crop_live件; g04手算P=0.0非0.5; g05严格R缺正时=0.0非None)+计数口径 → **30/30 PASS rc=0**, 重跑TEST_RESULTS.json字节一致
- 同步: PATCH_PLAN/CHANGELOG/TEST_RESULTS/machine/remediation_r1_summary + FINDINGS(F-12/F-14→FIXED(code)待端到端) + EVIDENCE_INDEX(E035–E037) + STATE
- 审计边界: 生产树改动=授权内2改1增; 数据/权重/既有outputs/冻结指标零触碰

## 整改阶段 R2-A（用户批准: G5只读预检与实施方案; 不建环境不联网不安装）

- 冻结: HEAD/状态/三文件SHA复核=与R1一致, 无外来改动(probes/r1_freeze.txt)
- 探针: nvidia-smi×2 / py -0p / where / conda探测 / df -h / Get-Volume / pkg_scan.py×2解释器(仅importlib.metadata) / torch version.py文件读取(+cpu直证) / du定向(fire_detection 2.4G, external 3.0G, hf_cache 1.8G, venv 3.4G) / pip cache info(136.6MB) / RAM 15.7G
- 证据恢复: requirements.txt(唯一清单, 无transformers/timm!) / README环境节(conda siglip_learn已灭失) / 训练二进制日志grep -a(横幅8.4.87·py3.11.15·torch2.8.0+cu128·5070Ti他机) / ckpt_introspection(model id) / weights_inventory(1.84G推理权重在盘) / git历史(仅初始提交)
- 首轮du超时2min(venv site-packages)→300s重试成功; 无其他失败命令
- 写入: remediation_r2a/{ENV_INVENTORY,ENV_OPTIONS,STORAGE_BUDGET,VERIFICATION_SEQUENCE,DEPENDENCY_EVIDENCE.json} + machine/remediation_r2a_summary.json + EVIDENCE_INDEX(E038–E040) + STATE
- 边界: 零安装/零联网/零删除/零CUDA初始化/零权重加载; 未写remediation_r2a外新产物
