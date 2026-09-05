# FINAL_PROJECT_STATUS.md — 终局状态（2026-08-28）

> 封存完成。不再开展任何训练、推理或生产修改。以下为终局状态。

## 终局状态

| 项 | 状态 |
|---|---|
| 工程环境 | **已复现**（py3.11.9 / torch2.8.0+cu128 / ultralytics8.4.87 / transformers5.16.1 / timm1.0.28，验收门 17/17） |
| G3 / G4 | **端到端 CLOSED**（G3 220 图全链 == E019 逐字节；G4 坏图隔离 rc3+PARTIAL+validator 拒收） |
| LEGACY_ORIGINAL_WITH_KNOWN_LEAKAGE | **工程可运行，但无可信无泄漏泛化指标**（F-11 CONFIRMED/UNRESOLVED；P=.9474 仅可复算不可作无偏证据）；保持当前默认生产模型 |
| REG1_GROUPED_RC | **工程可运行且有内容簇隔离指标**（holdout P=.8462/R=.9340 + 95%CI；F-11 CLOSED_FOR_REG1）；状态 = **RELEASE_CANDIDATE** |
| 正式提交 | **等待官方测试集**（无官方测试集 JSON，提交就绪未解除） |
| 生产权重 | **尚未切换**（LEGACY 仍为默认；REG1 未晋升） |
| 当前项目状态 | **READY_FOR_EXTERNAL_PROMOTION_TEST**（非正式 SUBMISSION_READY） |

## 就绪度（原冻结规则重评，非官方综合分）

- **LEGACY_ORIGINAL_WITH_KNOWN_LEAKAGE**：裸分 49 → 数据泄漏封顶 → **39/100**（NOT READY，泄漏未解决）
- **REG1_GROUPED_RC**：**57/100**（解除泄漏/端到端/独立 val 封顶；仍标"未充分验证"：同源非外部）

## 封存动作（已执行）

- 原生产权重（LEGACY）零删除/零移动/零覆盖/零重命名。
- REG1 权重独立存放于 `remediation_g1/formal_runs/` + `remediation_g1/runs_heads/`。
- 双候选已登记入 `MODEL_CANDIDATE_REGISTRY.json`（12 权重 SHA + 环境指纹 + 配置 + 阈值 + G6 + 指标 + 限制）。
- 未做任何再次训练、调参、改阈值、改种子。

## 待用户决策（唯一）

- 是否晋升 REG1_GROUPED_RC 覆盖 LEGACY（需经 PROMOTION_GATE_PLAN.md 的 A 或 B 门槛；
  切换流程见 PRODUCTION_SWITCH_AND_ROLLBACK.md）。当前**不晋升**。

## 封存产物索引

- MODEL_CANDIDATE_REGISTRY.json
- CANDIDATE_COMPARISON.md
- PROMOTION_GATE_PLAN.md
- PRODUCTION_SWITCH_AND_ROLLBACK.md
- machine/final_candidate_status.json
- 早期审计：FINAL_AUDIT_REPORT.md、POST_REMEDIATION_AUDIT.md、POST_G1_REMEDIATION_AUDIT.md
