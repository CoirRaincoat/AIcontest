# BASELINE_MANIFEST.md — 冻结基线（2026-08-27 01:11 +08:00）

## Git 锚点

- HEAD: `874de91aafdd18b30e083ee8801e570936953799`，branch main，唯一提交 "Initial commit: fire detection project (code + model weights)"
- remote: https://github.com/CoirRaincoat/AIcontest.git
- 工作区: 干净（`git status --porcelain` = 0 条）
- 注意: `.gitignore` 排除了数据/缓存目录与部分产物 → git 无法锚定下列未版本化资产，以下用哈希+大小+mtime 锚定。

## 版本化关键资产 SHA-256

完整清单: `machine/baseline_sha256.txt`（225 行），覆盖：
根 README/导航文档、`.gitignore`、`fire_detection/README.md`、`requirements.txt`、
`data/data.yaml`、`configs/fire_config.yaml`、官方标注两 JSON、`src/*.py` 全部源码、
`scripts/*` 全部脚本、`outputs/*.json|*.csv` 验证产物、
最终推理链 6 权重：3×YOLO26 best.pt（m/s/s_aug）+ 3×best_head.pt（siglip_linear_seed2026 / dinov3_vitb16_linear_seed42 / siglip2_crop_v2_seed2026）。

## 未版本化但关键的资产（由 inventory 锚定）

`machine/weights_inventory.txt`（32 个 .pt/.safetensors）要点：

| 资产 | 大小 | mtime |
|---|---|---|
| hf_cache SigLIP2-base-384 safetensors | 1432.4 MB | 07-20 22:25 |
| hf_cache DINOv3 vit_base_patch16 safetensors | 326.7 MB | 07-21 15:54 |
| 根预训练 yolo26n/s/m.pt | 5.3/19.5/42.2 MB | 07-05~07-15 |
| 最终 YOLO26m_960 best.pt / last.pt | 各 42.0 MB | 07-15 21:16 |
| 最终 YOLO26s_960 best.pt / last.pt | 各 19.4 MB | 07-12 11:48 |
| 最终 YOLO26s_aug_960 best.pt / last.pt | 各 19.4 MB | 07-16 00:08 |
| runs/detect/runs/detect/**嵌套** fire_yolo26m_aug_960 best/last | 各 42.0 MB | 07-16 12:28 |
| runs_ext/fire_yolo26s_ext_v1_960 best.pt | 19.4 MB | 07-17 14:29 |
| 旧 yolov8n epoch0-40 + yolo26n_960-2 等历史权重 | ~120 MB | 07-05~07-11 |

结构异常记录: `runs/detect/runs/detect/runs/train/fire_yolo26m_aug_960` 为**双嵌套重复训练目录**，
而文档所述 m_aug 权重路径为 `runs/detect/runs/train/fire_yolo26m_aug_960`（该标准位置不存在 m_aug；见阶段2核实 predict_best_fusion.py 实际默认值指向哪一个）。

## 官方标注与原始数据

| 文件 | 大小 | 备注 |
|---|---|---|
| `fire_detection/data/images/train/train_coco.json` | 505,090 B | 检测框标注（哈希已冻结） |
| `fire_detection/data/images/train/train_image.json` | 40,046 B | 图像级标注（哈希已冻结） |
| 原始图片目录 `data/images/train/images` | **1101 张** jpg/png/jpeg | ⚠ 官方口径为 1100，差异待阶段3定位 |

## 内部划分（计数，阶段3深入）

- train 880 imgs / val 220 imgs（labels 目录为同名 txt）
- data_ext_v1/train/images = 2880 imgs（含外部数据）
- crop_data_v2 = 5281 imgs（局部裁剪训练数据）

## 已有实验产物（冻结在 baseline_sha256.txt 中）

`outputs/` 下: `val_gt.json`(8005B), 多组 threshold_search*.json, `val_pred_best_seed2026_yolo_fusion_thr022_metrics.json`,
`val_pred_best_seed2026_yolo_fusion_live_details.csv` 等。注意导航文档引用的文件名是
`val_pred_best_siglip_yolo_dino_crop_*` 系列 —— 该命名在当前 outputs 清单中**不存在**（文档与磁盘不一致，待阶段4核实是否被覆盖/改名）。

## 全库统计（不含 .git）

58,561 个文件。扩展名 Top: jpg 32,384 / txt(YOLO标签) 25,731 / json 200 / png 85 / csv 34 / pt 30 / py 24 / ps1 14 / yaml 12 / log 7 / npz 6。
最大五文件全部为骨干权重或最终 YOLO 权重（1.43GB SigLIP2 … 19MB 级）。

## 已证实缺失的文档记载依赖（本机）

`C:\ANACONDA2\envs\{yolo_learn,siglip_learn}`、`C:\AI`、部署应用目录 `C:\Users\<姓名>\Desktop\fire_detection\deployment\fire_demo` — 均不存在。
