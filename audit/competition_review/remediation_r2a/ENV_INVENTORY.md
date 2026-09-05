# ENV_INVENTORY.md — 本机环境只读盘点（R2-A）

日期: 2026-08-27 · 授权: 仅读取/盘点/小型审计文档 · 零安装/零联网/零删除/零CUDA初始化/零权重加载
原始探针输出: `probes/`（r1_freeze / gpu / interpreters / pkg_scan_results / disk_and_caches / sizes_ram）+ `probes/pkg_scan.py`（仅 importlib.metadata，不导入包本体）

## 0. R1 状态冻结（先决检查 — 通过）

- Git HEAD `874de91aafdd18b30e083ee8801e570936953799`（main），无新外来改动。
- 足迹恰为 R1 授权范围: `M evaluate_image_level.py`、`M predict_best_fusion.py`、`?? batch_safety.py`、`?? audit/`。
- R1 三文件 SHA-256 复核一致: `6b776165…` / `092bae45…` / `5403634b…`（全文见 probes/r1_freeze.txt）。
- 本阶段未触碰这三个文件。

## 1. 主机 / GPU

| 项 | 值 | 证据 |
|---|---|---|
| OS | Windows 11 Home 10.0.26200 | pkg_scan（platform） |
| CPU / RAM | i7-14650HX（24线程）/ 15.7 GB | sizes_ram.txt |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU（Blackwell, **compute capability 12.0 = sm_120**） | nvidia-smi |
| 驱动 | 573.24（WDDM），驱动支持 CUDA ≤ **12.8** | nvidia-smi 头行 "CUDA Version: 12.8" |
| 显存 | 8151 MiB 总量 / 4981 MiB 当前空闲（桌面占用约2.8G） | nvidia-smi query |
| 兼容结论（本地可确认） | sm_120 无旧架构预编译核 → torch 必须为 **cu128 系构建**（CUDA ≥12.8）；驱动 573.24 满足 cu128 运行时要求 | A级（nvidia-smi 直证 + Blackwell 公开架构事实）；具体 wheel 版本×Py版本组合需联网核实 |

## 2. 解释器与包管理器

| 候选 | 路径 | Python | 结论 |
|---|---|---|---|
| A: 当前默认 `python`（PyCharm venv） | `C:\Users\CoirRaincoat\PyCharmMiscProject\.venv\Scripts\python.exe` | 3.13.5 | torch==2.12.1**+cpu**、torchvision==0.27.1+cpu、numpy 2.4.5、pillow 12.0、opencv 4.12、huggingface-hub 1.17.0、pandas 3.0.3、matplotlib 3.10.9；**缺** ultralytics/transformers/timm/safetensors/gradio。**不可用于GPU推理（+cpu构建）**，且按授权不得污染 |
| B: 系统 Python313 | `C:\Users\CoirRaincoat\AppData\Local\Programs\Python\Python313\python.exe` | 3.13.5 | torch==2.12.0**+cpu**（无torchvision），其余核心包全缺。同样不可用 |
| WindowsApps python.exe | 存根（Store alias） | — | 非真实解释器，忽略 |
| conda / mamba | **未安装**（where 无结果, command not found） | — | 无 conda 环境可复用 |
| py 启动器登记 | 仅 `-V:3.13` 一项 | — | **本机不存在 Python 3.11**；若走3.11路线需下载解释器 |

- **两个 torch 的 +cpu 属性为文件级直证**: `torch/version.py` 内 `__version__='2.12.x+cpu'`, `cuda=None`, 且 `torch/lib` 无任何 cudart/cublas/cudnn DLL（未 import torch）。
- 结论: **不存在无需安装即可使用的GPU环境**（问题6答案=否）。

## 3. 历史环境（README 声明的）现状

- README.md L222-232: 融合推理/Gradio 历史环境为 conda `C:\ANACONDA2\envs\siglip_learn`（及 yolo_learn）→ **本机 C:\ANACONDA2 不存在**（阶段0已探明, 本次未复扫以避免大目录递归）。历史环境已不可用。
- 训练日志横幅（A级, outputs/logs/resume_yolov8n_960.out.log）:
  `Ultralytics 8.4.87  Python-3.11.15  torch-2.8.0+cu128  CUDA:0 (RTX 5070 Ti Laptop GPU, 12227MiB)`
  → 历史训练环境指纹: **Python 3.11 + torch 2.8.0+cu128 + ultralytics 8.4.87**。注意该日志 GPU 为 **5070 Ti**，即该环境属另一台机器，不能证明 siglip_learn 的包组合，但直接证明"本项目历史曾在 cu128 线运行"。
- 生产包装器 `scripts/predict_best_fusion_crop.ps1` 硬编码 `C:\ANACONDA2\envs\siglip_learn\python.exe` 与 `C:\AI\...` → 二者本机均已缺失（F-01/F-02 已知问题），新环境必须以**显式CLI参数+环境变量**方式启动（见 ENV_OPTIONS.md §8，无需改生产代码）。

## 4. 磁盘与占用

| 盘 | 总量 | 剩余 | 备注 |
|---|---|---|---|
| C: (Windows-SSD) | 300.0 GB | **98.1 GB** | 阶段0时仅8.1G→用户已清理；为唯一可行安装目标 |
| D: (Data) | 174.7 GB | **11.7 GB**（94%占） | **不足以承载环境**（见 STORAGE_BUDGET 判 BLOCKED） |

| 对象 | 占用 | 说明 |
|---|---|---|
| fire_detection/ | 2.4 GB | 代码+runs+outputs+数据 |
| external_datasets/ | 3.0 GB | ext_v1 等（非提交链） |
| hf_cache/（仓库根） | 1.8 GB | **含推理所需全部骨干**: siglip2-base-patch16-384 safetensors 1432.4MB + timm vit_base_patch16_dinov3.lvd1689m 326.7MB |
| 现有 venv | 3.4 GB | torch-cpu 2.12.1 为大头 |
| pip 缓存 | 136.6 MB（http-v2）+ 0 wheel | 位于 `c:\users\…\appdata\local\pip\cache` |
| 推理权重合计（P6/G7 直接所需） | ≈ **1.84 GB 已在盘** | 上两骨干 + 3×YOLO best.pt(42/19.4/19.4MB) + 3个线性头(~30KB) → **零权重下载** |

## 5. 明确未做的事（授权边界）

未安装/升级/卸载任何包；未联网；未删除/清理任何缓存；未改 PATH/注册表/系统Python；未 import torch、未初始化 CUDA、未加载任何权重；未递归扫描无关大目录；未写 remediation_r2a 之外的新产物。
