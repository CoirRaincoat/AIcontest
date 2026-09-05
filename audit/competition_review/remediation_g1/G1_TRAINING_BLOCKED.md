# G1 正式重训 — 预算门停止（BLOCKED，2026-08-28）

## 判定：PROJECTED_OVER_BUDGET → 正式训练不启动

按授权「预算 ≤8h GPU wall clock；dry-run 单轮计时投影，投影超则正式训练前停止」条款，三模型
分组隔离重训在正式训练前被**预算门停止**。

## 计时证据（A 级实测）

| 模型 | 干跑 per-epoch (s) | 历史稳态 per-epoch (s, 5070Ti/880图) | 预期轮数 | 投影 |
|---|---|---|---|---|
| yolo26m (batch4) | **172.3**（results.csv 实测） | 100.1（8009s/80） | 80 | 3.83h |
| yolo26s (batch8) | **134.8**（results.csv 实测） | 56.1（4149s/74） | 74（早停 patience20） | 2.77h |
| yolo26s_aug (batch8) | ~139（按 s+3% aug 估） | 58.1（4651s/80） | 80 | 3.09h |
| **合计** | | | | **9.69h** |

- 干跑均为 1 轮（epochs=1）、指标未读、仅验证路径/保存/OOM。
- 即使计入 close_mosaic 末 10 轮提速（~15%）与 s 早停（~8%），修正后仍 ≈ **8.8h > 8h**。
- 根因：RTX 5060 Laptop 训练吞吐约为历史 5070 Ti 的 **1.7–2.4 倍慢**（预算移动端 GPU），
  且 reg1 三集（698+198）虽比历史 880 小 0.79×，仍不足以把三模型 80 轮压回 8h。

## 未做 / 未触碰

- **未启动任何正式训练**（无 formal run 发生；dryrun 产物在 remediation_g1/dryrun/ 保留）。
- 未为压预算而降低 epochs/imgsz、未换小模型、未合并模型、未跳过任一模型 —— 这些都会改动
  冻结的生产训练配置，属禁止项。
- 未因指标差重跑；未覆盖旧 runs；未删任何缓存/文件。

## 已冻结且可复用（后续若获预算/换机授权）

- `remediation_g1/split/SPLIT_FREEZE.json` + `per_image_sha256.json`（三集 698/198/146 + 隔离 58，跨集 d≤8=0）
- `remediation_g1/dataset/data_reg1.yaml` + junction 视图 + 三模型 args.yaml 配置差异
- `train_reg1.py`（dryrun/formal 驱动 + 8h 投影门 + 监控线程）——formal 模式随时可复用，仅需授权放宽预算或迁移到 5070 Ti 机。

## 下游影响

- G2（reg1 校准+holdout 评估）、G6（reg1 提交彩排）依赖 G1 产物 → **同被阻塞**。
- R2-B（环境）与 R3（端到端/P6/G7）已完成，不受影响。

## 恢复入口

读 `remediation_g1/TRAIN_RECORD_m_dryrun.json` + `TRAIN_RECORD_s_dryrun.json`（含投影）；
`logs_train_*_dryrun.log` 为计时现场；批准后对每个模型跑
`python train_reg1.py --model {m|s|s_aug} --mode formal`。
