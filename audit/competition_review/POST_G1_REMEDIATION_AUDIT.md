# POST_G1_REMEDIATION_AUDIT.md — G1/G2 整改后复审（2026-08-28）

> 保留原 `FINAL_AUDIT_REPORT.md` 与 `POST_REMEDIATION_AUDIT.md` 不变。本文件按**原冻结评分规则**
> 重评（门槛零改动），仅端到端实证的发现升级。无官方测试集 JSON → **不解除提交门槛**、
> **不预测排名、不生成官方综合分**。

## 1. 原始模型工程提交链（原模型 G6）

- **PASS**：canonical 1100 图 → 提交 JSON → p7_validator(no-GT) 通过。
- 检查全过：恰好 1100 项、无缺失/额外/重复键、全 int 0/1、无 NaN/Inf、无 PARTIAL/failure。
- 标记 `ORIGINAL_MODEL_WITH_KNOWN_LEAKAGE` + `SURROGATE_REHEARSAL_NOT_OFFICIAL_SUBMISSION`。
- 结论：工程提交链健全，**不解除 F-11**、**不提升泛化评分**。

## 2. G1 是否完成 —— 完成

| 模型 | 轮数 | 耗时 | best.pt SHA-256（前16） |
|---|---|---|---|
| yolo26m | 80 | 6448s | 39e6d16c65be7858 |
| yolo26s | 74（早停） | 3612s | aaa492d36f3ac9e4 |
| yolo26s_aug | 80 | 4074s | 28520644f85195f0 |
| siglip 头 | — | — | c7ae93114e7a693f |
| dino 头 | — | — | 8b8005f8f43f1d82 |
| crop 头 | — | — | b57ee5f86115a31f |

- 累计 GPU 墙钟 **3.93h**（≤12h 预算）；无恢复、无中止、无 NaN/OOM/温度告警。
- 三 YOLO 严格沿用历史 args.yaml（仅 data/project/name/model 覆盖）；三头沿用生产
  seed/hyperparam（siglip/dino/crop = 2026/42/2026）。头训练样本 698（164neg+534pos）✓。

## 3. F-11 是否具备解除条件 —— 解除（6/6）

1. 三模型只使用 train ✓（YOLO 用 train.txt=698，头用 train.csv=698）
2. calibration/final 未进入训练 ✓（calib 仅作 val/早停，holdout 从未用）
3. quarantine 未违规使用 ✓（58 张未入任何集合）
4. checkpoint+规则在开启 final 前冻结 ✓（REG1_ASSETS.json + RULE_PANEL_FROZEN.json）
5. final 只评估一次 ✓（HOLDOUT_OPENED.lock）
6. 全程无跨集 dHash≤8 连接 ✓（cross_le8=0）

## 4. grouped final holdout 结果（G2，一次性）

- **主规则（冻结生产融合规则）**：TP=99 FP=18 FN=7 TN=22，**P=0.8462 / R=0.9340 / F1=0.8879**
- 95% CI（iid bootstrap B=10000, seed=20260828）：P[0.7949, 0.8974] R[0.90, 0.9706]
- F-15 应用全图补救（**exploratory**）：TP56 FP0 FN50 TN40，P=1.0 / R=0.5283
- 校准集诊断面板：RULE_PANEL_CALIB.json（sweep 仅诊断，未用于选阈）

## 5. 新模型工程提交链（grouped G6）—— PASS

- canonical 1100 图 → 提交 JSON（sha a39fee81a90199aa）→ p7_validator 通过，检查全过。
- 标记 `GROUPED_MODEL_REG1_RETRAINED` + `SURROGATE_REHEARSAL_NOT_OFFICIAL_SUBMISSION`。
- 新权重**未**覆盖生产权重；是否晋升由用户另决。

## 6. 重评结论（原冻结规则）

- **绑定封顶「存在未解决的明确数据泄漏 ≤39」解除**（F-11 六条件满足 → 泄漏已隔离）。
- **「无可运行端到端推理 ≤49」解除**（R3 端到端闭合）。
- **「无独立可信验证 ≤69」部分解除**：holdout 为内容簇隔离的独立可信评估（d≤8 跨集=0、
  规则冻结、一次性开启、CI 报告），但**同源数据、非外部来源** → 性能仍标"未充分验证"。
- **「提交 JSON 不合规 ≤59」部分解除**：G6 证明 1100 键合法 JSON 可产出+校验通过，
  但**无官方测试集** → 提交就绪仍不成立。
- **「仅文档声明维度 L≤1」**：部署效率经 G7 实测部分解除；完整系统/可视化/云端仍文档级。

**内部就绪度**：原 39/100（数据泄漏封顶）→ **裸分 42/100 起**，且部署效率(G7)/可复现性(环境冻结
+端到端)/鲁棒性(G4闭合+重训)等维度有上调证据；按"不生成官方综合分"原则不擅自给出精确新综合分。
**判定维持 NOT READY**（无官方测试集、同源非外部验证）。

## 7. 新模型是否值得晋升 —— 建议由用户决定，不自动晋升

- 无泄漏 holdout P=0.846 低于原污染 val P=0.947（符合隔离后回落的诚实预期）。
- 新权重是内容簇隔离训练产物，泛化上限未被外部验证；是否替换生产权重需用户基于
  官方测试集或业务权衡另决。

## 8. 恢复入口与需人工决策

- 恢复：`machine/g1_controller_status.json`、`machine/g1_g2_final_summary.json`、
  `remediation_g1/REG1_ASSETS.json`。
- 唯一需人工决策：**是否晋升 grouped 新模型覆盖生产权重**（其余步骤已全部完成）。
