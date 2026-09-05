# R3 端到端复验 + P6 + G7 报告（2026-08-28，无人值守流水线）

环境: `C:\fire_envs\sf2026_min`（Python 3.11.9 / torch 2.8.0+cu128 / torchvision 0.23.0+cu128 /
ultralytics 8.4.87 / transformers 5.16.1 / timm 1.0.28），GPU = RTX 5060 Laptop (sm_120, 8151 MiB)。
全链 `HF_HUB_OFFLINE=1`。生产代码零改动：因 `predict_best_fusion.py:26` 硬编码死路径
`ROOT=C:\AI\fire_detection`（F-01/F-02），全部 6 个权重/checkpoint/cache 路径经 CLI 显式传入仓库路径。

## 结果总表

| 步骤 | 门 | 结果 | 判定 |
|---|---|---|---|
| S0 | 权重 SHA-256 | 6 个生产权重全部 == machine/baseline_sha256.txt | PASS |
| S1 | P6 冻结清单 | 12 图清单+gt 在推理前落盘（含来源 CSV SHA-256） | PASS |
| S3 | 模块导入 | ultralytics 8.4.87 / transformers 5.16.1 / timm 1.0.28；dinov3.lvd1689m 在注册表+缓存 | PASS |
| S4 | 五类权重加载 | 3×YOLO + SigLIP头 + DINO头 + crop头 全部离线加载；VRAM ~1433MiB | PASS |
| S5 | 单图推理 | rc=0，JSON 恰 1 键 int 值 | PASS |
| S6 | G4 坏图注入 | rc=3；PARTIAL 仅含正常图；failures 含 {file,stage,exception_type}；主提交名未写；validator 拒收(exit 1) | **CLOSED(端到端)** |
| S7 | G3 全集口径 | 220 图全链 rc=0；评测器输出 == E019 逐字节一致 (TP162 FP9 FN3 TN46 P.9474 R.9818 F1.9643) | **CLOSED(端到端)** |
| S8 | P6 冻结 12 图 | 预测位 12/12 精确一致；分数 11/12 在 ±1e-3，1 图 dino 漂移 2e-3 | PASS(见下) |
| S9 | G7 计时显存 | 冷启动 59.08s；暖 12 图 28.22s；均摊 2.35s/图；峰值显存 torch 1777MiB / nvidia-smi 5181MiB；峰值温度 59°C | PASS |
| S10 | p7_validator | preflight 源图 1101 文件 corrupt_count=0 | PASS |

## P6 复现结论（S8 详细）

- **预测位 12/12 完全一致**（0/1 标签逐图精确复现冻结产物）。
- 分数列：11/12 在 ±1e-3 内（多数精确到 0.0）；**唯一偏差** `raw_fire_relabel_dp_20564.jpg`
  的 dino_score 冻结 0.76437330 → 本次 0.76636505（Δ=1.99e-3，约 2×容差）。
- **归因**：dino 阈值 0.08，该图 0.764/0.766 远在决策边界之外，预测不受影响；该漂移为
  FP16 autocast 跨 GPU 数值效应（历史产物产自 5070 Ti Laptop，本次 5060 Laptop + torch 2.8.0+cu128）。
- **处置**：按 P6 审批单「不一致只记录不调参」，已记录原始分数差（p6/p6_diff.json），
  未做任何配置改动去追平。H1（运行时等价）在**预测层**闭合。

## G7 延迟/显存（S9 详细）

- 冷启动（含导入+权重加载+12图推理，子进程）：**59.084s**
- 暖端到端（子进程，OS 页缓存暖）：median **59.08s**（冷/暖几乎相同 —— 权重加载主导，页缓存增益小）
- 暖进程内阶段分解 12 图：**28.221s**（均摊 **2.352s/图**）
- 阶段中位数(s)：SigLIP 7.720 / DINO 2.793 / YOLO-m 1.411 / YOLO-s 1.271 / YOLO-s_aug 1.237 /
  提案收集 3.959 / crop 9.829
- 单图全链上界（每阶段重载权重）：median **19.712s**
- 峰值显存：torch 分配 **1776.9 MiB**；nvidia-smi 采样峰值 **5181 MiB**（含保留/碎片）
- 峰值温度 **59°C**（远低于 87°C 中止线）；三次子进程无 OOM

## G4 坏图注入证据（S6 详细）

- 损坏副本 = 截断 val 图至 50% 字节（仅审计目录副本，原图未动）。
- 失败记录 9 条全部指向同一坏图 `corrupt_truncated.jpg`，分阶段隔离：
  load_preprocess(OSError×2) / inference(IndexError×6) / fusion(RuntimeError×1)。
- PARTIAL.json 仅含 `normal_copy.jpg`；`corrupt_pred.json`（主提交名）未创建；
  p7_validator 对 PARTIAL 退出码 1（fail-closed 拒收）。

## 声明

- 上述为**端到端复验**（真实前向 + 评测器 + 校验器），非 R1 的单元级验证。据此，
  R1 的 G3/G4 修复由 FIXED(code) 升级为 **CLOSED(端到端)**。
- G3 全集复验在 220 图 val 上与 E019 逐字节一致，证明重跑确定性在预测层成立。
- 未做任何阈值/规则/权重改动；未联网下载任何权重（HF 缓存 + 本地 best.pt 全程离线）。
