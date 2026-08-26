# 监控场景火情识别算法系统

项目编号：`SF-2026-01`  
项目根目录：`C:\AI\fire_detection`

这是一个面向比赛的图像级火情识别项目。系统使用 YOLO26 目标检测模型提供火焰候选框，再结合 SigLIP2、DINOv3 和局部 SigLIP2 做整图判断与小目标补救。

进入 `C:\AI` 后，建议先阅读：

```text
C:\AI\00_先看我_AI项目导航.md
```

## 当前状态

- 官方训练资源：1100 张图片。
- 当前划分：880 张训练图、220 张验证图。
- 类别数：1 类，类别 `0` 为 `fire`。
- 最终 YOLO 模型：YOLO26m、YOLO26s、YOLO26s_aug。
- 网页应用：已完成，入口在桌面项目的 `deployment\fire_demo`。
- 测试集：以比赛平台正式发布为准。

当前应用验证结果：

```text
Precision = 0.9480
Recall    = 0.9939
F1        = 0.9704
```

该结果是在当前 220 张验证图上得到的，不等于官方测试集成绩。

## 目录速查

```text
C:\AI\fire_detection
├── data\                    官方数据和 YOLO 格式数据
├── data_ext_v1\             官方数据与外部数据融合版本
├── crop_data_v2\            局部候选框分类训练数据
├── external_datasets\       外部数据集位于 C:\AI\external_datasets
├── runs\                    YOLO 训练记录和权重
├── runs_siglip\             SigLIP2 分类头和训练记录
├── runs_dinov3\             DINOv3 分类头和训练记录
├── outputs\                 验证预测、阈值搜索和指标
├── src\                     Python 源码
├── scripts\                 PowerShell 训练、推理和提交脚本
├── siglip_data\             SigLIP2 图像级数据清单
├── dinov3_data\             DINOv3 图像级数据清单
├── hf_cache\                位于 C:\AI\hf_cache，保存骨干模型缓存
└── configs\                 旧版基础训练配置
```

## 数据和标签

### 原始官方资源

```text
C:\AI\fire_detection\data\images\train\images
C:\AI\fire_detection\data\images\train\train_coco.json
C:\AI\fire_detection\data\images\train\train_image.json
```

- `train_coco.json`：目标检测框标签，适合转换后训练 YOLO。
- `train_image.json`：图像级 0/1 标签，格式接近比赛测试答案。

### YOLO 训练数据

```text
C:\AI\fire_detection\data\train\images
C:\AI\fire_detection\data\train\labels
C:\AI\fire_detection\data\val\images
C:\AI\fire_detection\data\val\labels
C:\AI\fire_detection\data\data.yaml
```

YOLO 标签格式：

```text
class_id center_x center_y width height
```

坐标是相对于图片宽高归一化后的数值。空的 `.txt` 表示没有目标框。图片和标签必须同名，例如：

```text
abc.jpg
abc.txt
```

注意：数据标签可能存在争议。`raw_fire_relabel_dp_20411.txt` 当前为空，但原图中有疑似火焰区域。遇到类似样本，应先记录和复核，不要直接批量修改标签。

## 最终模型权重

### YOLO26

```text
C:\AI\fire_detection\runs\detect\runs\train\fire_yolo26m_960\weights\best.pt
C:\AI\fire_detection\runs\detect\runs\train\fire_yolo26s_960\weights\best.pt
C:\AI\fire_detection\runs\detect\runs\train\fire_yolo26s_aug_960\weights\best.pt
```

### 分类头

```text
C:\AI\fire_detection\runs_siglip\siglip2_linear_seed2026\best_head.pt
C:\AI\fire_detection\runs_dinov3\dinov3_vitb16_linear_seed42\best_head.pt
C:\AI\fire_detection\runs_siglip\siglip2_crop_v2_seed2026\best_head.pt
```

分类头文件很小，因为完整骨干模型保存在：

```text
C:\AI\hf_cache
```

不要删除 `hf_cache`，本机离线运行 SigLIP2 和 DINOv3 需要它。

预训练起点权重 `yolo26n.pt`、`yolo26s.pt`、`yolo26m.pt` 在项目根目录。训练完成后用于推理的是训练目录中的 `weights\best.pt`，不是根目录的预训练权重。

## 如何看训练结果

以 YOLO26m 为例：

```text
C:\AI\fire_detection\runs\detect\runs\train\fire_yolo26m_960
```

重点文件：

- `args.yaml`：实际训练参数。
- `results.csv`：每轮损失、Precision、Recall、mAP。
- `results.png`：训练曲线。
- `confusion_matrix.png`：混淆矩阵。
- `weights\best.pt`：验证表现最好的权重。
- `weights\last.pt`：最后一轮权重，主要用于断点续训。

比赛评估是图像级 Precision 和 Recall，所以不能只看 YOLO 的框级 mAP。最终图像级验证结果在：

```text
C:\AI\fire_detection\outputs\val_gt.json
C:\AI\fire_detection\outputs\val_pred_best_siglip_yolo_dino_crop_live_details.csv
C:\AI\fire_detection\outputs\val_pred_best_siglip_yolo_dino_crop_rescue_thr097_metrics.json
```

## 当前融合规则

基础融合：

```text
base = (YOLO26m OR YOLO26s OR YOLO26s_aug)
       AND 全图 SigLIP2
       AND DINOv3
```

网页应用额外使用两个补救分支：

```text
全图补救：全图 SigLIP2 >= 0.85 且 DINOv3 >= 0.70
局部补救：局部 SigLIP2 >= 0.97

final = base OR 全图补救 OR 局部补救
```

验证集上，该规则为 F1=`0.9704`。网页应用已经包含全图补救规则，但旧的批量脚本仍需在正式测试集发布前同步该规则，再作为最终提交推理入口。

## 本地可视化应用

应用目录：

```text
C:\Users\薇尔莉特\Desktop\fire_detection\deployment\fire_demo
```

重要文件：

- `app.py`：上传图片、融合推理、标注框和分数展示。
- `start_local.ps1`：本机启动脚本。
- `requirements.txt`：应用依赖。
- `README.md`：Hugging Face Spaces 配置。
- `DEPLOYMENT_GUIDE.md`：云端部署说明。

本地地址：

```text
http://127.0.0.1:7860
```

本地融合推理使用 `siglip_learn` 环境。云端不能读取本机的 `C:\AI`，部署时需要一并提供 `deployment\fire_demo\models` 中的六个自定义权重，并准备 SigLIP2、DINOv3 的骨干模型缓存或下载权限。

## 比赛提交

细则明确的答案格式是 JSON：以图片名为键，以 `0/1` 为值。

```json
{
  "xxx1.jpg": 0,
  "xxx2.jpg": 1
}
```

测试集发布后：

1. 将测试图片放入英文路径，例如 `C:\AI\fire_detection\test_images`。
2. 使用最终批量融合脚本进行推理。
3. 检查 JSON 图片名数量与测试集一致。
4. 检查所有值只能是整数 `0` 或 `1`。
5. 上传 JSON，并按平台要求提交模型、推理代码和云端可视化地址。

批量推理入口：

```text
C:\AI\fire_detection\scripts\predict_best_fusion_crop.ps1
```

批量推理核心源码：

```text
C:\AI\fire_detection\src\predict_best_fusion.py
```

`best.pt` 本身不是图像级 JSON 答案。最终答案必须由测试集推理生成。

## Python 环境

当前主要环境：

```text
C:\ANACONDA2\envs\yolo_learn
C:\ANACONDA2\envs\siglip_learn
```

- `yolo_learn`：YOLO 训练和普通检测。
- `siglip_learn`：SigLIP2、DINOv3、融合推理和 Gradio 应用。

项目根目录的 `requirements.txt` 是基础项目依赖说明。实际融合应用的依赖以 `deployment\fire_demo\requirements.txt` 为准。

## 进一步学习时的入口

- 学数据转换：`src\coco_to_yolo.py`。
- 学 YOLO 训练：`src\train.py` 和训练目录内的 `args.yaml`。
- 学图像级评估：`src\evaluate_image_level.py`、`src\tune_image_threshold.py`。
- 学 SigLIP2：`src\train_siglip2_classifier.py`。
- 学 DINOv3：`src\train_dinov3_classifier.py`。
- 学局部补救：`src\build_yolo_crop_dataset.py`、`src\build_cached_crop_rescue.py`。
- 学融合推理：`src\predict_best_fusion.py`。

清理记录和完整路径索引见：

```text
C:\AI\00_先看我_AI项目导航.md
```
