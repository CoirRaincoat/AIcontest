# P6_APPROVAL_REQUEST.md — 阶段6 最小验证（小批量真实GPU推理）审批单

状态: **待审批（BLOCKED，未执行任何推理）** · 日期 2026-08-27 · 审计 SF2026-01-AUDIT-20260827-R1
原则: 输出仅写 `audit/competition_review/machine/p6/`；不改任何生产文件/权重/阈值；单命令≤30分钟。

## 1. 目的与假设（要检验什么）

- H1 (端到端等价): 在真实前向中，批量链对抽样图像逐张复现冻结分数列(±1e-3)与0/1标签 → 把"CSV重建复算成功[A级]"升级为"运行时等价也已证实"，闭合 F-01 留下的 R2 运行时缺口的最小部分。
- H2 (FP 归因): 抽样假阳性图应为视觉易混淆负类(暖色光源/烟雾状蒸汽等)，支持 F-13 的先验敏感性讨论。
- H3 (FN 归因): 假阴性应集中于小目标/远距火焰被 YOLO 漏检或 DINO≥0.08 否决门挡掉，为 F-08 提供闭环证据。
- 若 H1 复现失败 → 立即上报差异明细并停止；不尝试修改任何配置去"追平"。

## 2. 入口 / 配置 / 权重（含 SHA-256 前16位，均已冻结）

- 入口: `fire_detection/src/predict_best_fusion.py`（只读调用，输出路径用参数重定向审计目录，不使用其 C:\AI 默认值）
- 固定参数: imgsz=960; YOLO conf m=0.10/s=0.20/s_aug=0.20 votes≥1; SigLIP τ=0.22; DINOv3 τ=0.08; crop 补救 τ=0.97(proposal conf≥0.03, IoU dedupe 0.80, max8) —— 与 E014/E002 文档一致
- 权重(SHA256-16, 见 machine/experiments_table.csv):
  yolo26m=`788478d0…` yolo26s=`2115c79c…` yolo26s_aug=`7c6a37fd…`; siglip/dino/crop 三头+骨干经 runs_siglip/runs_dinov3 introspection 与 hf_cache 锁定[E031]

## 3. 样本清单与选样规则（现在冻结，防事后挑样）

来源=frozen `outputs/val_pred_best_siglip_yolo_dino_crop_live_details.csv`，共12张：
1. FP侧: 全部9个FP按 siglip_score 降序取前4；
2. FN侧: 全部3个FN全取；
3. 边界带: |s−0.22|≤0.03 或 |crop−0.97|≤0.02，按最小裕度升序取前5。
生成脚本连同固定文件名清单先落盘(`machine/p6/sample_manifest.json`)再运行。

## 4. 预期输出（全部写入 machine/p6/）

- `p6_pred.json` —— 官方提交格式 `{filename: int 0|1}` 逐图恰一次
- `p6_details.csv` —— 与冻结 details 同列(score 列含 vote/s/d/c)
- `p6_diff.json` —— 对比冻结标签：期望 12/12 一致；任一不一致即记录原始分数差并告警
- 不产生截图以外的新权重、不缓存覆盖（HF/HOME/缓存环境变量在运行器内显式指向临时目录）

## 5. 资源估算（RTX 5060 Laptop 8151MiB，当前空闲 ~5441MiB [E009阶段0]）

| 项 | 估计 | 依据 |
|---|---|---|
| GPU 显存峰值 | <4GB(B级估) | 同框架训练曾在本卡完成(args/results 为证)；推理由此更低 |
| 单命令耗时 | <30min(目标<10min) | 12图×(3×YOLO@960+2 VLM全图+≤8 crop头)，上界按10s/图 |
| 磁盘增量 | <200MB | 仅JSON/CSV/清单 |
| 训练? | 无 | 明确不训练、不扫参 |

## 6. 输出路径与回滚

- 全部产物: `audit/competition_review/machine/p6/**`
- 回滚 = 删除该目录；生产树零接触(Git 干净态可校验 `git status --porcelain` 应无变化)

## 7. 依赖缺失与 F-01 处理计划

本 venv 缺 ultralytics/transformers/timm/gradio[E004]；ps1 引用的三个 conda 环境(C:\ANACONDA2\*)与 C:\AI 本机不存在[E009]。选项:
- A(推荐): 授权一次性 `pip install ultralytics transformers timm`(需联网授权; torch 2.12.1 已存在无需重装)；安装记录进 audit 目录日志，不动 requirements.txt 以外的任何文件
- B: 用户提供一个已装齐依赖的本机 python 路径(此前检测不存在，或环境在另一台机器——则本票移交那台机器执行)
- C: 两项皆不可行 → 阶段6维持 BLOCKED，直接进阶段7(JSON 校验器+合成 smoke，纯CPU无需批准)

## 决议栏(留空待填)

- 批准项: ☐A ☐B ☐C　附加条件:
- 审批时间/备注:
