# PROJECT_MAP.md — 项目结构、入口与数据流（阶段0草稿，基于文档+文件枚举；待阶段2代码级验证）

## 自述架构（来源 E001/E002，B级）

```
输入图像
  └─ YOLO26m_960 / YOLO26s_960 / YOLO26s_aug_960（检测火焰框, 置信度阈值见 E012）
       └─ 全图 SigLIP2(linear head on siglip2-base-384) + DINOv3(vit-b16 linear head) 逐图打分
            ├─ base = (任一YOLO有框) AND SigLIP2≥τs AND DINOv3≥τd
            ├─ 局部补救: 裁剪候选框 → 局部SigLIP2 ≥ 0.97 → 正
            └─ (仅网页应用另加) 全图补救: SigLIP2≥0.85 AND DINOv3≥0.70   ← 与批量脚本不同步(F-06)
                 └─ 图像级 0/1 → 提交 JSON + 可视化(Gradio)
```

## 目录/入口索引（实际文件已枚举核实）

| 类别 | 文件 | 自述作用 |
|---|---|---|
| 数据转换 | src/coco_to_yolo.py | 官方COCO→YOLO txt |
| 训练(YOLO) | src/train.py; scripts/train_gpu_yolov8{s,n}_960.ps1 | YOLO训练/续训 |
| 训练(分类头) | src/train_siglip2_classifier.py; train_dinov3_classifier.py | 骨干冻结+线性头 |
| 缓存构建 | src/build_cached_{best_fusion,crop_rescue,dino_veto}.py; build_siglip_manifest.py | 用 npz 嵌入离线复算融合 |
| 批量推理(提交口径) | src/predict_best_fusion.py ← scripts/predict_best_fusion{,_crop,_dino}.ps1 | 生成图像级JSON |
| 评测 | src/evaluate_image_level.py; tune_image_threshold.py; evaluate*.py | 指标/阈值搜索 |
| 可视化 | deployment/fire_demo（**仓库外且本机缺失**, F-03） | Gradio 演示 |
| 验证产物 | outputs/val_gt.json + threshold_search*×8 + val_pred_best_seed2026_yolo_fusion_* | 已有验证预测与指标 |

## 数据树

```
data/images/train/{images=1101, train_coco.json, train_image.json}   官方原始
data/{train,val}/{images,labels}                                     880/220 YOLO格式
data_ext_v1/train/images=2880                                        官方+DFire外部混合
crop_data_v2=5281                                                    局部裁剪训练数据
siglip_data / dinov3_data                                            图像级清单+npz嵌入缓存
hf_cache                                                             SigLIP2/DINOv3骨干(离线必需)
runs*/ runs_siglip/ runs_dinov3                                      权重与训练记录
```

## 未验证点（阶段2-4）

- 生产推理链在本机无记录环境的可运行性（F-01）；predict_best_fusion.py 默认权重确切路径（含 m_aug 的真实位置, F-07）
- 训练/推理预处理一致性；嵌入缓存链与在线推理是否等价
- 验证集 val_gt.json 与官方 train_image.json 的从属关系与互斥性
