# CONTENT_SOURCES.md — PPT 内容证据映射

> 每一页内容所依据的项目文件，方便核对事实。事实以源码、配置、正式运行产物与冻结状态文件为准。

## 第 1 页 · 封面

- 标题「面向监控场景的火情识别系统」、副标题「基于目标检测与视觉语义模型的多模型融合方案」
- 依据：`fire_detection/README.md`、`00_先看我_AI项目导航.md`（项目编号 SF-2026-01、多模型融合方案）
- 作者/学校/指导教师：项目内无相关事实，设置为占位命令 `\ProjectAuthor` 等（`待填写`）
- Logo：模板引用 `pic/Westlake_University_Logo.png` 不存在，已删除引用（不伪造）

## 第 2 页 · 项目背景

- 监控场景火情防范需求、火焰/烟雾形态多变
- 依据：`fire_detection/README.md`（监控场景火情识别）、`audit/competition_review/REQUIREMENTS_MATRIX.md`（R1 图像输入/R3 图像级火/无火输出）
- 可信度：背景性表述，由项目主题与任务书要求支撑

## 第 3 页 · 项目目标与主要功能

- 「输入—处理—输出」：读取图像 → 检测/语义/融合 → 图像级 0/1 + JSON
- 依据：`fire_detection/src/predict_best_fusion.py`（main 流程）、`REQUIREMENTS_MATRIX.md`（R1/R3/R4）
- 图像级 0/1 语义：`train_coco.json` 类别名 "fire"（正类=存在火焰）

## 第 4 页 · 系统总体架构（TikZ 自绘）

- 流程节点：监控图像 → 预处理 → 目标检测（YOLO）→ 候选区域裁剪 → 视觉语义分类（SigLIP/DINO）→ 局部复核 → 多模型决策融合 → 图像级判断 → JSON 生成与校验
- 依据：`fire_detection/src/predict_best_fusion.py`（predict_siglip/predict_yolo/predict_dinov3/collect_yolo_proposals/predict_crop_siglip 与融合段）
- 架构图为自绘矢量图，无项目外素材

## 第 5 页 · 数据处理与模型训练

- 两类标注（目标框 + 图像级）、数据集划分、图像增强、多模型训练
- 依据：`fire_detection/README.md`（train_coco.json 与 train_image.json 区别）、`src/coco_to_yolo.py`、`src/train.py`、`src/train_siglip2_classifier.py`、`src/train_dinov3_classifier.py`、`runs/detect/runs/train/*/args.yaml`（epochs/batch/imgsz/增强）
- 训练产物：`fire_detection/runs/`、`runs_siglip/`、`runs_dinov3/`

## 第 6 页 · 核心模型与协同方式

- YOLO（定位候选）、SigLIP/DINO（视觉语义）、局部复核、决策融合
- 依据：`fire_detection/README.md`（三 YOLO + SigLIP2 + DINOv3 + 局部 SigLIP2）、`src/predict_best_fusion.py`

## 第 7 页 · 推理与输出流程

- 扫描→预处理→检测/分类→融合→图像级判断→原子写 JSON→独立校验
- 依据：`src/predict_best_fusion.py`（iter_images、各 predict 函数、main 融合与原子写段）、`src/batch_safety.py`（原子写）、`audit/competition_review/scripts/p7_validator.py`

## 第 8 页 · 工程可靠性设计（正面功能表述）

- 逐图异常隔离、批处理不中断、不完整结果标记、原子写入、输入预检查、输出校验、环境冻结、模型配置管理
- 依据：`src/batch_safety.py`（run_items_isolated / write_partial_products / atomic_write）、`src/evaluate_image_level.py`（严格解析）、`audit/competition_review/scripts/p7_validator.py`、`audit/competition_review/remediation_r2b/frozen_requirements.txt`、`MODEL_CANDIDATE_REGISTRY.json`（权重 SHA 注册）
- 注意：以"正面功能"表述，不出现审计编号（F-xx/Gxx/Rxx）

## 第 9 页 · 项目成果 + 预测示例图

- 成果清单：完整链路、GPU 环境复现、权重/配置管理、JSON 生成、回滚方案
- 依据：`audit/competition_review/FINAL_PROJECT_STATUS.md`（环境已复现、链路打通）、`MODEL_CANDIDATE_REGISTRY.json`（权重 SHA 管理）、`PRODUCTION_SWITCH_AND_ROLLBACK.md`（回滚方案）
- 图片：3 张来自项目现有 YOLO 检测输出 `fire_detection/runs/detect/predict/`，缩放至 assets/（保持宽高比）：
  - `assets/pred_example1.jpg` ← `20250526_firesmoke_00640.jpg`
  - `assets/pred_example2.jpg` ← `20260116_fs_sjt_00865.jpg`
  - `assets/pred_example3.jpg` ← `raw_fire_relabel_dp_20488.jpg`
- 注意：图片为真实检测输出，非伪造截图；不展示 Precision/Recall/F1 等指标

## 第 10 页 · 总结与后续工作

- 总结（完整系统、完整链路、兼顾识别与可靠性）与后续方向（更多场景、实时视频、轻量化）
- 依据：`FINAL_PROJECT_STATUS.md` 的完成项与"尚未完成"项（云端/可视化/外部验证等后续方向）
- 注意：不宣称已完成官方测试，不写成风险清单

## 第 11 页 · 结束页

- 「谢谢！欢迎交流与讨论」，无事实依赖
