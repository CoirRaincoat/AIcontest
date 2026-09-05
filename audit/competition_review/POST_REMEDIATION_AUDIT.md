# POST_REMEDIATION_AUDIT.md — 整改后复审（2026-08-28，无人值守流水线）

> 原报告 `FINAL_AUDIT_REPORT.md` 保持不变。本文件按**原冻结评分规则**重评（零事后权重/门槛改动），
> 仅端到端实证的发现升级 FIXED(code)→CLOSED。无官方测试集 JSON，**不解除提交门槛**，
> **不预测排名、不生成官方综合分**。

## 1. 环境可复现性（原 F-01）→ RESOLVED

- 新建隔离环境 `C:\fire_envs\sf2026_min`（Python 3.11.9 / torch 2.8.0+cu128 / torchvision 0.23.0+cu128 /
  ultralytics 8.4.87 / transformers 5.16.1 / timm 1.0.28），验收门 17/17 PASS（含 sm_120、
  CUDA 12.8、FP16 matmul+conv、生产入口可导入、C:≥70GB）。
- `frozen_requirements.txt` + torch wheel sha256 归档 → 环境可复现，F-01 的"本机不可运行"缺口闭合。

## 2. G3 / G4 端到端关闭（原 F-12 / F-14）→ CLOSED

- **G3 全集口径**：220 图全链重跑 → `evaluate_image_level.py` 输出与 E019 **逐字节一致**
  （TP162 FP9 FN3 TN46 P.9474 R.9818 F1.9643）。R1 的重写评测器在真实前向上闭合。
- **G4 坏图隔离**：截断副本注入 → rc=3、PARTIAL 仅含正常图、failures 含 {file,stage,exception_type}、
  主提交名未写、p7_validator 拒收(exit 1)。R1 的 batch_safety 故障隔离在真实前向上闭合。
- 据此，F-12/F-14 由 FIXED(code) 升级为 **CLOSED(端到端)**。

## 3. P6 冻结 12 图复现

- **预测位 12/12 精确一致**；分数 11/12 在 ±1e-3，1 图 dino 漂移 2e-3（FP16 跨 GPU 数值效应，
  阈值 0.08 远未触及，预测不受影响）。按审批单「不一致只记录不调参」，原始分数差已存
  `remediation_r3/p6/p6_diff.json`。H1 运行时等价在预测层闭合。

## 4. G7 延迟 / 显存

- 冷启动 59.08s；暖 12 图 28.22s（均摊 2.35s/图）；单图全链上界 19.71s。
- 峰值显存：torch 分配 1776.9 MiB / nvidia-smi 5181 MiB；峰值温度 59°C。无 OOM。

## 5. F-11 分组隔离（原数据泄漏）→ 划分冻结，重训被预算门停止

- **已完成**：dHash 并查集逐字复算与冻结 f11 全量一致（773 簇/1468 对/683 单簇/922-178/84 挪/32 边界对）；
  三集 698/198/146 + 隔离 58，**跨集 d≤8=0**；门槛全过；`SPLIT_FREEZE.json` + per-image SHA 冻结。
- **未完成**：三模型正式重训投影 9.7h **超过 8h 预算**，按「投影超则正式训练前停止」条款未启动。
  **F-11 数据泄漏仍未解决**（重训未发生，权重仍为全量 1100 图训练的历史产物）。

## 6. G2 / G6 → 未执行（依赖 G1）

- G2 无泄漏评估、G6 提交彩排均依赖 G1 重训产物（`REG1_ASSETS.json` 未生成）→ **BLOCKED**。
- 两个驱动（`g2_eval.py` / `g6_rehearsal.py`）已就绪，批准放宽预算/换机后可直接复用。

## 7. 重评结论

- **判定维持 NOT READY**。内部就绪度仍 **39/100**：绑定封顶条款「未解决数据泄漏 ≤39」未解除
  （G1 重训未发生）。
- 非绑定链改善：`无端到端推理 ≤49` 封顶条款**已解除**（R3 实证端到端可运行）；`无独立可信 val ≤69`
  仍生效（G1/G2 未完成，性能仍标注"未充分验证"）。
- **不解除提交门槛**（无官方测试集 JSON）；**不生成排名/官方综合分**。

## 8. 恢复入口与需人工决策

- 恢复：读 `machine/unattended_pipeline_status.json`；G1 证据 `remediation_g1/G1_TRAINING_BLOCKED.md`；
  换机/扩预算后 `python train_reg1.py --model {m|s|s_aug} --mode formal`。
- 需人工决策：(1) 是否放宽 8h 预算或在 5070 Ti 机重训；(2) G6 是否允许以原始生产模型做
  G1 无关的 surrogate 彩排（机制验证，已具备条件）。
