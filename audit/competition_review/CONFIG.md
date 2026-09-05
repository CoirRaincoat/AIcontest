# CONFIG.md — 审计配置

- audit_id: SF2026-01-AUDIT-20260827-R1
- 开始时间: 2026-08-27 01:11 (+08:00)
- 项目根目录（审计对象）: `C:\Users\CoirRaincoat\PyCharmMiscProject\AIcontest\AI`
- 审计目录: `./audit/competition_review/`

## 参数（仓库内无既有审计配置，采用任务指定默认值）

| 参数 | 值 |
|---|---|
| AUDIT_MODE | READ_ONLY |
| ALLOW_WRITES | 仅 `./audit/competition_review/` |
| ALLOW_PRODUCTION_EDITS | false |
| ALLOW_DATA_MUTATION | false |
| ALLOW_MODEL_TRAINING | false |
| ALLOW_NETWORK_DOWNLOAD | false |
| ALLOW_PACKAGE_INSTALL | false |
| ALLOW_EXISTING_TESTS | true |
| ALLOW_EXISTING_INFERENCE | true（输出必须重定向到审计目录；若需写入项目目录则先请求批准） |
| MAX_SINGLE_COMMAND_MINUTES | 30 |
| MAX_AUDIT_DISK_GB | 5 |
| SAMPLE_SEED | 20260827 |
| TARGET_PRECISION / TARGET_RECALL / TARGET_LATENCY_MS / TARGET_MODEL_SIZE_MB | null（未知，未设官方阈值） |
| DEPLOYMENT_TARGET | unknown |

## 审计机环境（2026-08-27 探测）

- OS: Windows 11 Home China, build 26200 (MINGW64/bash 3.6.6)
- Python: 3.13.5 — `C:\Users\CoirRaincoat\PyCharmMiscProject\.venv`（当前 PATH 默认）
- 关键包: torch==2.12.1, torchvision==0.27.1, opencv-python==4.12.0.88, numpy==2.4.5,
  pillow==12.0.0, scikit-learn==1.8.0, huggingface_hub==1.17.0
- 未安装: ultralytics, transformers, timm, gradio, onnx/onnxruntime/tensorrt（导入探测）
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB（探测时空闲 5441 MiB），驱动 573.24, 计算能力 12.0
- CPU/RAM: i7-14650HX，24 逻辑核 / 15.7 GB
- 磁盘 C: 仅剩 8.1 GB 可用（98% 已用）→ 审计全程禁止大文件落盘
- Git: HEAD 874de91aafdd18b30e083ee8801e570936953799 (main)，工作区干净（不含被 .gitignore 排除的数据/权重/缓存）

## 环境风险备注

文档记载的生产环境 `C:\ANACONDA2\envs\yolo_learn`、`C:\ANACONDA2\envs\siglip_learn`、原项目位置 `C:\AI`、部署应用目录
`C:\Users\<姓名>\Desktop\fire_detection\deployment\fire_demo` 在本机**均不存在**（见 E010）。本审计无法直接假设生产推理可运行。

## 已获批例外

无。尚未执行任何高成本动作（未训练、未联网下载、未安装依赖、未改生产文件）。

## 待批准事项（后续阶段，如需要将逐项申请）

1. 阶段4 若需运行真实 YOLO 融合推理复算验证指标：其输出目录参数如不能完全重定向进审计目录，则申请批准。
2. 阶段7 smoke test 需要一个合成小图片目录与独立输出目录，全部建在审计目录内，预计 <50MB —— 拟视为已允许范围。
3. 不计划任何网络下载或包安装；如缺失依赖阻断验证，将改用缓存产物离线复核并记录为 U/C 级证据。
