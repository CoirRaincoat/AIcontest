# STATE.md — 审计状态（可恢复入口）

- audit_id: SF2026-01-AUDIT-20260827-R1
- 基线: Git HEAD 874de91aafdd18b30e083ee8801e570936953799 (main)；非版本化资产以 machine/baseline_sha256.txt + weights_inventory.txt 锚定
- 最近更新: 2026-08-27 整改阶段R2-A完成(G5只读预检与方案, 未建环境), 暂停待批准联网/安装(命令与预算见remediation_r2a/ENV_OPTIONS.md)

## 阶段状态

| 阶段 | 状态 |
|---|---|
| 0 预检与基线冻结 | COMPLETE |
| 1 合规矩阵 | COMPLETE（R5/R6=FAIL(仓库内无实物)，R1/R2/R4/R7/R9=PARTIAL，R3/R10=PASS） |
| 2 架构/数据流/可运行性 | COMPLETE（静态闭环；运行时验证本机受阻=F-01） |
| 3 数据集与标注审计 + 官方口径复算 | COMPLETE（E019/E020: 六产物全部复算吻合） |
| 4 深度指标分析(CI/阈值诊断/F-12/F-14/事后性) | COMPLETE（METRICS.md + E024–E028） |
| 5 模型与实验可信度 | COMPLETE（E030/E031 实验账本与自省） |
| 6 鲁棒性与错误分析(GPU推理) | **BLOCKED**（用户决议C: 暂不执行; P6审批单存档备future A/B/C 或永久搁置） |
| 7 提交JSON独立校验器+自证+产物核验 | **COMPLETE**（p7_validator.py 21/21 PASS + 5产物三方对账 OK + canonical 1100冻结; E032–E034） |
| 8 部署效率测量 | **未验证(决议: 暂不授权)** — 记U, 声明不可作数 |
| 9 复现/安全/答辩风险(反向质询) | **COMPLETE**（DEFENSE_QA.md: 10主题×6字段, 含每项最小补充实验） |
| 10 评分与最终报告 | **COMPLETE**（FINAL_AUDIT_REPORT + EXECUTIVE_SUMMARY + REMEDIATION_ROADMAP + machine/final_audit_summary.json; 冻结评分规则经会话记录校回, 原封套用零修改） |

> **最终判定: NOT READY · 内部就绪度 39/100**（裸分42 → 被"未解决数据泄漏≤39"条款封顶；无端到端推理≤49、无独立可信val≤69为非绑定链但69条同时给性能打上"未充分验证"标签；第5/7维受"仅声明L≤1"上限）。审计全程授权边界零突破，生产树零改动。

## 整改阶段R1（2026-08-27, 仅G3+G4获批准）

| 项 | 结果 |
|---|---|
| G3 全集口径评测器 | **FIXED(code)**: evaluate_image_level.py 重写; 缺5真阳反例 strict R=1/6(不再虚高), fail-closed 退出码2, 合同解析拒重复键/NaN/类型陷阱, 全覆盖输出与旧版逐字节一致, 现存产物复算==E019 [E035] |
| G4 单图故障隔离 | **FIXED(code)**: 新增batch_safety.py + predict_best_fusion.py 五阶段逐图隔离/融合缺分记录/PARTIAL产物纪律(退出码3, 主提交名不写)/原子写; 13项策略默认AST冻结未漂移 [E036] |
| 测试 | 30/30 PASS(单元级: unit13+cli5+static6+integration1+scope1), 重跑字节一致 [E037] |
| 现有指标一致性 | C01: 新评测器CLI复算 crop_live 产物 = TP162 FP9 FN3 TN46 P.9474 R.9818 F1.9643 == E019 ✔ |
| 未端到端验证 | predict_best_fusion 本机不可导入(缺torch/ultralytics, 属G5); 真实前向逐图隔离未跑(属G7/P6); 正式JSON彩排未做(属G6) — **本阶段为单元级验证, 不构成端到端就绪声明** |
| 改动足迹 | M evaluate_image_level.py + M predict_best_fusion.py + ?? batch_safety.py(恰为授权范围); 回滚= `git checkout 874de91 -- fire_detection/src/ && rm fire_detection/src/batch_safety.py` |
| 产物 | remediation_r1/{PATCH_PLAN,CHANGELOG,TEST_RESULTS,tests,fixtures,diffs,backups} + machine/remediation_r1_summary.json |

> 评分影响: F-12/F-14 机制缺陷已修, 但**39/100封顶不变**——绑定条款为未解决数据泄漏(G1未批)与端到端不可运行(G5未批); 两项修复待端到端复验后方可在复评中体现。

## 整改阶段R2-A（2026-08-27, G5只读预检 — 未创建任何环境）

- **可复用环境=无**: 双Python3.13.5的torch均+cpu(文件级直证); conda不存在; 历史siglip_learn/C:\AI已灭失。
- **GPU**: 5060 Laptop sm_120 + 驱动573.24(CUDA≤12.8) ⇒ 唯一可行线=torch cu128系。
- **历史指纹(A级)**: `ultralytics 8.4.87 · py3.11.15 · torch 2.8.0+cu128`(日志横幅, 属他机5070 Ti)。
- **UNKNOWN(不猜)**: transformers/timm精确版本(仓库清单未列+环境灭失)、torchvision配对、wheel现势——批准联网后实测并pip freeze固化。
- **方案M1**: py3.11 venv @ `C:\fire_envs\sf2026_min`(完全隔离), torch2.8.0+cu128+ultralytics8.4.87+transformers/timm稳定版, `--no-cache-dir`; 权重零下载(hf_cache 1.8G全在盘); 峰值≈10–11.5G, C:余98.1G⇒PASS; D: 11.7G⇒**BLOCKED**。
- **验证顺序**: S1版本→S2 CUDA→S3导入→S4权重→S5单图→S6损坏图(G4端到端复验)→S7全集口径(G3复验)→S8 P6冻结12图→S9 G7计时→S10 validator终检; 首败即停。
- 待批准: 联网(py.org/uv + pytorch cu128 index + pypi)、ENV_OPTIONS.md §7安装命令、目标目录C:\fire_envs\、磁盘预算峰值≤12G。

## 阶段4–5 关键结果速览

1. **固定阈值面板(A级)**: base .9471/.9758；+crop@0.97 =批量链 .9474/.9818(F1 .9643)；+应用全图补救 .9480/.9939——全部与存储一致并加"泄漏偏乐观"强制标注。
2. **95%CI**(B=10000 seed20260827): iid P[.9245,.9730] R[.9896,1.0]；5族cluster P[.8852,1.0] R[.9872,1.0](仅5簇,弱)。
3. **阈值敏感性(E024)**: 277点扫描——当前 τs=0.22 恰为本验证集 argmax-F1 ⇒ 印证"同集选优"；偏P候选 τs=.5542→P.9613/R.903；final规则内无 R≥.99 点。**扫描仅诊断，禁止用于提交择优**。
4. **F-11 升级(E025)**: 内容级 dHash——22张 val↔train 距离0，38.6%≤8 ⇒ 泄漏从"模式"变"硬证据"；命名启发式降级(目检证伪)。重划提案 machine/f11_proposed_split.json(922/178、挪84图、重划后 d≤8 重叠=0)+重训成本表(F11_SPLIT_PROPOSAL.md §6)，均未执行。
5. **F-12 量化(E026)**: 缺5真阳预测 ⇒ 交集口径 R=.9938 vs 严格口径 .9636(+3.0pp虚高,零告警)；现存产物全覆盖故历史数字未受影响；build_cached 链缺键即崩。0损坏图(E027)旁证 F-14 属前瞻风险。
6. **F-15 新增(E028)**: 应用全图补救(0.85/0.70)在仓库无预先固定痕迹(网格产物无dino轴、无该行;README晚于产物)⇒ 标注"同集事后候选规则"，非造假指控而是报告口径问题。
7. **阶段5**: YOLO26m/s/s_aug measured 2.22/1.15/1.29 GPU·h、mAP50-95 .3479/.3482/.3645[E030]；三头 ckpt 自带元数据与代码全对上[E031]；SigLIP头种子敏感性低(val_loss .314–.329)；ext_v1(.3786)外部数据不在融合链；legacy args.yaml 再现第二个人名路径(N5)。

## 正在运行

无。

## 阶段7 关键结果速览

1. 独立CPU校验器 `p7_validator.py` 交付: 指标/解析全新实现, canonical 1100 自官方 train_coco.json 冻结(锚哈希每次运行复核); 第1101文件以精确路径+SHA256(50264a1ddb0fbeb0…)显式排除; 多余/缺失/大小写冲突一律 fail-closed。
2. 自证 **21/21 PASS**: F-12 全机制(缺真阳R虚高strict R=1/6 vs 交集镜像1.0/空预测/重复键/NaN/true/1.0/"1"/越界/键大小写冲突/缺键崩溃等价物)逐项转成必拦用例, 均含手算推导。
3. 现存5个预测产物三方对账 OK: 校验器独立复算 == E019冻结格口 == 存储sidecar(.9415/.9471/.9422/.9474/.9474系); mode-B 对1100全集全部预期失败(missing=880)——它们是内部val产物而非提交件。
4. **不存在可提交JSON**(全库无≥1100键)[E034]: 结构能力通过≠端到端提交就绪; 正式提交件须测试集到位后现场产出并复核。
5. F-14 检测点落地(preflight-images 夹具逐文件指名 rc=1), 生产链未修改未掩盖。

## 正在运行

无。

## 下一条具体动作（候选, 待用户选择）

- A: 阶段9 复现/安全/答辩风险静态审计(纯CPU无需批准): 个人路径泄露总清单+脱敏建议、DFire许可文本核查方案、requirements锁定草案、答辩问答风险底稿。
- B: 阶段10 最终评分与综合报告(各维度打分表+READY判定稿)。
- C: 重启P6(12图GPU小批)或阶段8时延测量 —— 需重新授权依赖/GPU。

## 待用户确认的问题

1. F11_SPLIT_PROPOSAL.md §6 重训是否单独立项(GPU≈5–6h级)。
2. P6 是否永久搁置(计算层等价已由三方对账闭合, 运行时层缺口仅存于F-01描述内)。

## 恢复指令

读 STATE → 核对 HEAD==874de91 且 machine/baseline_sha256.txt 抽样一致 → 按"下一条具体动作"继续; 阶段7入口 machine/stage7_summary.json; 校验器用法 VALIDATOR_README.md; 证据 EVIDENCE_INDEX E001–E034。
