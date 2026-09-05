# ENV_OPTIONS.md — 最小推理复验环境方案（R2-A 设计稿，本阶段不执行）

原则: 只建**最小推理环境**（P6/G7足够），训练/UI依赖一律暂缓；完全隔离、零污染、可整体回滚；所有体积为估算（禁止联网核实期），精确值以批准后实际安装时的实测为准。

## 方案对比

| | 方案 M1（推荐）: 独立 Python 3.11 venv | 方案 M2: 复用现有 3.13 venv | 方案 D: 装到 D: 盘 |
|---|---|---|---|
| Python | 3.11.x（历史环境同线, 日志A级证据） | 3.13.5（现成解释器） | 同M1 |
| 与现有venv隔离 | 完全（新解释器+新venv, 不装 --system-site-packages） | ✗ 违反"不得污染现有venv" | 同M1 |
| GPU | 需装 torch 2.8.0+cu128（替换其+cpu思路） | 现有torch是+cpu, 必须重装 → 与"不污染"冲突 | 同M1 |
| 判定 | **可行** | **否决**（既污染又需先卸载cpu torch） | **否决/BLOCKED**（11.7G 余量低于门槛） |

## M1 方案要素（对应用户十条要求）

1. **建议 Python 版本**: 3.11.x（证据: 历史训练横幅 `Python-3.11.15`）。本机无 3.11 → 需批准下载（python.org 安装器用户级安装 或 uv 托管解释器，均无需管理员/注册表权限的方式优先）。
2. **PyTorch 组合**: `torch==2.8.0+cu128`（历史直证版本）+ `torchvision` 官方配对版（推定 0.23.0，**UNKNOWN 需联网核实**）＋ index `https://download.pytorch.org/whl/cu128`。驱动 573.24 满足 cu128；sm_120 原生支持。
3. **最小依赖集**: `ultralytics==8.4.87`（历史直证）、`transformers`（版本UNKNOWN→装当前稳定版后立即 `pip freeze` 固化归档）、`timm`（同前，须含 dinov3.lvd1689m 定义）、随依赖带入的 safetensors / huggingface-hub / numpy / opencv-python / Pillow / tqdm / pyyaml。ultralytics 会被动带入 matplotlib/pandas/scipy 等，无需单独声明。
4. **暂缓项**: gradio、pycocotools、seaborn、loguru、onnx*、accelerate —— 均非 P6/G7 必需（gradio 甚至源码不在仓库）。
5. **绝对安装路径**: `C:\fire_envs\` 下，推荐 `C:\fire_envs\py311\`（解释器）与 `C:\fire_envs\sf2026_min\`（venv）。C: 现余 98.1G，为唯一满足余量的盘；**D: 判 BLOCKED**。仓库根 hf_cache 复用为模型缓存（1.8G 已在盘，零权重下载）。
6. **隔离性**: 与 `PyCharmMiscProject\.venv`（3.13/torch-cpu）完全隔离——不同解释器、不同 venv 目录、无 `--system-site-packages`、不写 PATH（激活仅会话内）、不改注册表。卸载=删两个目录。
7. **精确安装命令草案**（执行需另行批准）:
```powershell
# (a) 获取 Python 3.11 —— 用户级、无注册表污染优先
#     选项1: uv (单文件工具):  winget install --user astral-sh.uv ; uv python install 3.11
#     选项2: python.org 3.11.x 安装器, 勾选仅当前用户, InstallWebDriver/AddToPath 均不勾
# (b) 建 venv（完全隔离）
C:\fire_envs\py311\python.exe -m venv C:\fire_envs\sf2026_min
C:\fire_envs\sf2026_min\Scripts\python.exe -m pip install --upgrade pip
# (c) torch cu128 线（--no-cache-dir 抑制缓存峰值; 估算下载 2.7–3.2GB）
C:\fire_envs\sf2026_min\Scripts\pip.exe install --no-cache-dir torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# (d) 推理侧最小集（transformers/timm 装当前稳定版, 版本UNKNOWN已在DEPENDENCY_EVIDENCE声明）
C:\fire_envs\sf2026_min\Scripts\pip.exe install --no-cache-dir "ultralytics==8.4.87" transformers timm
# (e) 立即固化 A 级版本记录（存 audit/competition_review/remediation_r2a/frozen_requirements.txt）
C:\fire_envs\sf2026_min\Scripts\pip.exe freeze
```
8. **体积估算**（标≈者为估算）: 下载 ≈2.8–3.5GB（torch cu128 wheel 为主）；安装后环境 ≈**7–9GB**（Windows cu128 wheel 将 CUDA DLL 打包进 torch/lib）；`--no-cache-dir` 下临时峰值 ≈ 安装中多占一份解压中内容 ≈ **峰值总占用 ≈ 10–11.5GB**；pip 全局缓存增量 ≈0（用了 --no-cache-dir；若不加则 +2.8–3.2GB 缓存峰值）。现有 pip 缓存仅 136.6MB 不清理。
9. **失败回滚**: 整体删除 `C:\fire_envs\`（或仅 sf2026_min）即回滚；过程不触碰现有 venv/系统 Python/仓库/hf_cache；下载失败无残留（--no-cache-dir）。任何步骤失败按 VERIFICATION_SEQUENCE 停止。
10. **仍需联网核实的结论**: ① torchvision↔torch 2.8.0 精配对版本；② transformers/timm 的可用稳定版与 dinov3.lvd1689m 支持下限；③ py3.11 + cu128 wheel 在源的现势存在性；④ ultralytics 8.4.87 在 py3.11 的依赖解析现势；⑤ wheel 实际字节数（本文件体积估算的校准）。——全部标记 UNKNOWN，安装时实测并冻结记录。

## 启动适配（零生产代码改动）

生产脚本 ROOT 硬编码 `C:\AI`、包装器 ps1 硬编码已灭失的 conda 路径 —— 新环境以**显式参数**运行 `predict_best_fusion.py`：
- 全部权重/缓存/阈值均有 CLI 参数（`--yolo-*-weights`、`--siglip-checkpoint`、`--dino-checkpoint`、`--crop-checkpoint`、`--cache-dir`、`--source`、`--output`…）；
- 环境变量: `HF_HOME=<仓库>\hf_cache`、`HF_HUB_CACHE=<仓库>\hf_cache\hub`、`HF_HUB_OFFLINE=1`、`YOLO_CONFIG_DIR`/`MPLCONFIGDIR` 指向可写临时目录（沿袭原 ps1 的做法）；
- R1 后产出纪律: 失败→`*_PARTIAL.json`+退出码3；成功→正式JSON+退出码0，接 p7_validator 复核。
