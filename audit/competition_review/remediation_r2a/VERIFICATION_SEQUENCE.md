# VERIFICATION_SEQUENCE.md — 环境建立后的最小验证顺序（R2-A 设计稿）

规则: **逐门推进，任一门失败立即停止**，记录失败现场（命令/输出/退出码）后上报，不得扩大运行规模、不得跳门、不得"顺手多跑"。每门产物写 `audit/competition_review/remediation_r2a/verify/stepNN_*.json`（由批准后的执行阶段生成）。所有命令使用新环境绝对路径解释器，全程 HF_HUB_OFFLINE=1。

| 门 | 内容 | 命令形态 | 通过判据 | 失败动作 |
|---|---|---|---|---|
| S1 | 解释器身份与包版本核验 | `<env>\python -V`; `<env>\pip freeze`; `importlib.metadata` 扫描 | Python=3.11.x; freeze 与 frozen_requirements.txt 一致; 无包落在本机其他环境（`pip -V` 路径含 sf2026_min） | 停 |
| S2 | torch CUDA 可用性 | `<env>\python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"` | `(2.8.0+cu128, 12.8, True, 'RTX 5060 Laptop GPU', (12,0))` —— 首次允许的 CUDA 初始化（属批准的验证行为） | 停（记录驱动/内核报错） |
| S3 | 模块仅导入测试 | `<env>\python -c "import ultralytics, transformers, timm, safetensors"`（不加载权重） | 全部导入成功并打印各自 `__version__`；timm 侧确认含 dinov3.lvd1689m 条目 | 停（记录缺失/版本冲突） |
| S4 | 权重加载测试（CPU/GPU 各一次 load, 不推理） | 依次 `YOLO(<3×best.pt>)`; `AutoModel.from_pretrained(google/siglip2…, local_files_only=True)`; `timm.create_model('hf-hub:timm/vit_base_patch16_dinov3.lvd1689m', pretrained=True)`; 三线性头 `load_state_dict` | 五类权重全部加载成功；全程未访问网络 | 停（骨干缓存不匹配→记录并上报, 不得联网重下） |
| S5 | 单张正常图推理 | 从 val 取1张: `predict_best_fusion.py --source <1张> --output <r2a/verify/step05.json> --device 0` + 全部显式参数 | 退出码0；JSON 恰含该图键；行数/字段与 R1 契约一致；p7 结构检查无错误 | 停（首败即停: 记录堆栈+是否 G4 记录了失败） |
| S6 | 单张损坏图真实故障注入（**G4 端到端复验门**） | 构造截断jpg+正常jpg各1张同跑同上 | 退出码3；`*_PARTIAL.json` 仅含正常图；`*_failures.json` 记录 {file, stage, exception_type}；**主提交名文件不存在**；随后对该 PARTIAL 产物跑 p7_validator 必须 FAIL(missing) | 停 |
| S7 | **G3 全集口径端到端复验门** | 用 S5 产物对官方220 val_gt 跑 `evaluate_image_level.py`（预期非全覆盖→退出码2+STRICT诊断）；再用220全量图完整跑一遍（12图版: P6子集gt）做全覆盖对照 | 非全覆盖: 退出码2、缺图进名单、strict R 不虚高；全覆盖: 五行输出、与 R1 评测器一致 | 停 |
| S8 | P6 冻结 12 图复现 | 原审批单 P6_APPROVAL_REQUEST.md 的 12 图清单+冻结参数；输出至 r2a/verify/step08*；评测对照该单预期 | 复现输出与审批单预期一致（A/B/C 决议按当时约定）；若不一致只记录不调参 | 停 |
| S9 | G7 延迟与显存 | 12图×`time.perf_counter` 分段(YOLO/VLM/crop/IO)+`nvidia-smi --query-gpu=memory.used` 采样 | latency_profile.json 落盘；无 OOM；数值仅记录不评价 | 停 |
| S10 | p7_validator 终检 | `p7_validator.py validate`（无GT模式结构校验）对 S5/S8 正式产物; preflight-images 对源图集 | 全部 PASS；纳入 EVIDENCE_INDEX | 收尾归档 |

## 停止条款（硬性）

- S1–S4 任一失败 ⇒ 环境问题，禁止进入推理门。
- S5/S6 首个失败 ⇒ 禁止继续 S7+；如果是 G4/G3 端到端复验失败，R1 的 FIXED(code) 状态立即降级回 OPEN 并通报。
- 任何门出现：盘余量跌破 20 GB、显卡驱动报错/崩溃、疑似网络访问（离线模式被绕过）⇒ 立即终止全部后续门。
- 每门完成即时归档证据后再进下一门；会话中断可从 last-passed 门+1 恢复。
