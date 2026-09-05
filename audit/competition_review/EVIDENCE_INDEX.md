# EVIDENCE_INDEX.md — 证据索引（阶段0-3）

等级: A=实测/独立复算/原始文件可定位；B=配置/日志/文档互证；C=仅有声明；U=无法验证。

| ID | 等级 | 内容 / 位置 | 支持的结论 |
|---|---|---|---|
| E001 | B | 00_先看我_AI项目导航.md 全文 | 自述融合方案/划分/争议标签/应用与批量规则差异 |
| E002 | B | fire_detection/README.md 全文 | 融合阈值(0.22/0.08/0.97, 应用补救0.85/0.70)与成绩自述 |
| E003 | A | git HEAD 874de91 main 干净 | 基线锚点 |
| E004 | A | CONFIG.md 环境探测 | 本机依赖缺失清单 |
| E005/E006 | A | 扩展名统计; machine/weights_inventory.txt | 库构成; 权重全景(m_aug双嵌套异常) |
| E007/E008 | A | 标注文件定位; machine/baseline_sha256.txt(225行) | 官方标注存在; 关键资产哈希冻结 |
| E009 | A | 存在性探测 MISSING×5 | C:\AI、conda envs、部署目录本机不存在 |
| E010 | A | 分目录计数 raw1101/train880/val220/ext2880/crop5281 | 规模与F-04线索 |
| E011→已修正 | A | outputs 完整清单(head截断导致首轮误判) | val_pred_best_siglip_yolo_dino_crop_* 系列实际存在 |
| E012 | A | grep 默认权重/阈值 | predict_best_fusion.py 默认值与 build_cached_* 阈值 |
| E013 | C→升级为A | 见E020 | 文档成绩自述 |
| **E014** | A | src/predict_best_fusion.py 全读(L20,64 硬编码; L589-629 输出; L594-602 规则) | 推理链路、JSON生成、无per-image异常隔离 |
| **E015** | A | scripts/predict_best_fusion_crop.ps1 全读 | 提交包装器: conda python 绝对路径、HF离线环境变量、参数透传 |
| **E016** | A | src/evaluate_image_level.py 全读(L29-43) | 指标=交集口径+零分母静默0(F-12事实1/2) |
| **E017** | A | src/build_cached_crop_rescue.py 全读(L63直接下标) | 缓存链实现与其指标函数行为差异 |
| **E018** | A | src/coco_to_yolo.py 全读(L105-118) | seed42 单图随机划分, 无视频分组 |
| **E019** | A | machine/dataset_and_metrics_audit.json | COCO结构1100img/1445ann/cat=fire/1处越界±2px/236零框图==负类数；image级int严格binary正864；划分互斥全覆盖且 pairing 100%；val_gt 与官方完全一致；**S1–S5 六产物 P/R 独立复算与存储 metrics 全部一致(match=true)**：S1 .9415/.9758, S2 .9471/.9758, S3@095 .9422/.9879, S3@097=S4_live .9474/.9818, S5 .9415/.9758 |
| **E020** | A | machine/followup_f04_f05.json | F-04重复副本字节相同(SHA256一致)；按文档固定阈值0.85/0.70 复现“应用全图补救”: TP164 FP9 FN1 TN46 → **P=.9480 R=.9939 F1=.9704 与文档声称精确一致**(仅救回2张均为真阳性)；CSV↔JSON sanity一致 |
| **E021** | A | machine/split_leakage_audit.json | 5序列族覆盖100%样本；val 216/220(98.2%)在train有≤10序号差近邻(中位差1) [F-11] |
| **E022** | A | data_ext_v1/build_manifest.json + 各run args.yaml | DFire+800火/-1200负 seed42 类映射规则明确；**最终三YOLO均 data/data.yaml(纯官方)**; ext模型(seed42)不在融合链 |
| E023 | B | 导航§6/README 应用补救描述 | 与E020复算互相印证 |
| **E024** | A | machine/bootstrap_ci.json + threshold_scan_siglip.csv(277点, dino×31, crop×11) + scripts/audit_phase4_threshold_ci.py | 固定阈值面板复算、iid/5族cluster 95%CI(P[.9245,.9730]/[.8852,1.0], R[.9896,1.0]/[.9872,1.0])；τs=0.22=argmaxF1(.9643)；偏P点.5542；仅诊断用途 |
| **E025** | A | machine/near_duplicate_hash.json + near_dup_pairs.csv (+ bundle脚本) | 内容级泄漏: val↔train dHash64 距离0共22对, ≤8共85张(38.6%), 中位最近距离12 → F-11 硬证据 |
| **E026** | A | machine/f12_metric_caliber_demo.json | 缺5真阳预测时交集口径R=.9938 vs 全覆盖口径.9636(+3.0pp虚高零告警)；现存产物全覆盖未受影响 → F-12 |
| **E027** | A | machine/corrupt_scan.json | 官方raw 1101张 PIL verify+load 全部通过(0损坏) → F-14 属前瞻风险非现存缺陷 |
| **E028** | A | machine/provenance_app_rule.json (mtime链 + search_all.csv列结构剖析) | 网格无dino维、无(0.85,0.70)行、产物07-21早于README/导航07-22+ ⇒ 应用全图补救规则无预先固定痕迹 → 新增F-15 |
| **E029** | A | scripts/audit_phase45_f11_split.py + machine/f11_proposed_split.json + f11_moved_files.csv | 内容簇重划提案: 773簇(d≤8连通), 880/220→922/178, 挪84图; 重划后交叉d≤8=0(max跨侧55)、边界带9–12余32对待人工复核; 分布统计完整 → F11_SPLIT_PROPOSAL.md |
| **E030** | A | machine/experiments_table.csv (runs args.yaml+results.csv 挖掘) | YOLO26m/s/s_aug best ep80/54/74, mAP50-95 .3479/.3482/.3645, 训练时长(time列)2.22/1.15/1.29 GPU·h; ext_v1 .3786 不在融合链; legacy v8n args 含第二个人名桌面路径(N5/F-02) |
| **E031** | A | machine/checkpoints_introspection.json (CPU torch.load 自省) | SigLIP/DINOv3/crop 头 ckpt 内嵌 model id/feature_dim768/dropout0.2/seed/best_epoch/val_loss/class_names{0:no_fire,1:fire}; 权重↔配置↔代码一致性A级; SigLIP头5种子val_loss∈[.314,.329]低敏感 |
| **E032** | A | scripts/p7_validator.py + p7_selftest.py + machine/phase7/selftest_results.json | 独立CPU校验器(不复用生产指标代码)21用例全PASS(T01–T20): 手算期望自动比对；F-12机制(缺真阳R虚高/空预测/重复键/NaN/类型陷阱/键冲突)全部转为必拦用例; F-14坏图夹具逐文件指名rc=1; CLI端到端val子集复算 P=.947368/R=.981818/F1=.964286 与E019三方一致 |
| **E033** | A | machine/phase7/existing_artifacts_validation.json + canonical_manifest.json | 5个现存预测JSON严格口径(官方train_image.json限制版GT, 非项目val_gt.json)复算与E019冻结格口及存储sidecar全部一致; mode-B对1100全集全部预期失败(missing=880); gt_identity keys/values逐键相等 |
| **E034** | A | p7_validate_artifacts.py 全库扫描输出 + determinism hash比对(existing_artifacts_validation.json 重跑字节一致) | 不存在≥1100键的JSON(无可用提交件); 校验器确定性(无时间戳/sort_keys/输入锚哈希复核)成立 |

## 阶段7 小结

- 校验器自身质量由21个带手算期望的自证用例保证(overall=PASS才允许标可用); canonical=官方附件冻结(排除项为精确名单+SHA256锚定, 50264a1ddb0fbeb0…)。
- 独立第三方链闭合: 官方附件 → 冻结清单 → 独立指标实现 → 三方对账(E019审计版 / 存储sidecar / 本校验器)全部吻合。

## 阶段4–5 可信度小结

- 新升A: 泄漏内容级量化(E025)、CI与敏感性全景(E024)、F-12机制量化(E026)、坏图扫描(E027)、事后性证据链(E028)、重划提案可复算(E029)、实验账本(E030/E031)。
- 明确排除: E021命名族口径降级为粗分组参考(目检证伪其可比性, 但保留供cluster bootstrap)。

## 证据可信度小结（阶段性）

- 已证实(A): 数据完整性/标注一致性/划分规模/指标复算全对账/文档两大成绩数字可复现/泄漏模式量化。
- 高概率(B): 外部数据未进入最终提交链(args.yaml直证, 但分类头训练数据构成未逐行核对——siglip_data/dinov3_data 清单基于官方集待验)。
- 待核实: F-06 parity、F-08 泛化外推幅度、分类头训练数据纯净度。
- 无法验证(U): 网页应用运行时行为(DAFire许可原文, 部署实物)。

## 整改阶段R1 (2026-08-27, G3+G4)

- **E035 [A,生产补丁]**: G3全集口径评测器落地 — evaluate_image_level.py 重写(SHA 73f996fc…→6b776165…); 缺5真阳反例 strict R=1/6, 交集镜像1.0+delta量化, 退出码2 fail-closed; 合同解析拒重复键/NaN/类型陷阱; 全覆盖stdout与旧版逐字节一致(C05)。文件: remediation_r1/diffs/g3_evaluate_image_level.patch。
- **E036 [A,生产补丁]**: G4单图故障隔离落地 — 新增batch_safety.py(纯标准库, 5403634b…); predict_best_fusion.py 五阶段逐图隔离+融合循环缺分记录+PARTIAL产物纪律(主提交名不写, *_failures.json, 退出码3)+三件套原子写(SHA e6ba9e7a…→092bae45…); 13项策略默认AST冻结未漂移(s01)。文件: remediation_r1/diffs/g4_predict_best_fusion.patch。
- **E037 [A,测试]**: R1验收套件 30/30 PASS(29 unittest+1改动范围守卫), rc=0, 重跑字节一致; C01=新评测器CLI复算现存live产物==E019(TP162 FP9 FN3 TN46 P.9474 R.9818 F1.9643); h09=p7_validator拒收G4风格partial产物。**验证级别=单元级**: predict_best_fusion因缺torch/ultralytics不可导入, 由batch_safety真实执行+源码结构断言承载; 真实端到端未验证(待G5/G7)。文件: remediation_r1/TEST_RESULTS.json, machine/remediation_r1_summary.json。

## 整改阶段R2-A (2026-08-27, G5只读预检; 零安装/零联网)

- **E038 [A,环境]**: 本机盘点 — RTX 5060 Laptop cc12.0(sm_120)+驱动573.24(CUDA≤12.8)⇒需cu128系torch; 双Python3.13.5的torch均**+cpu**(version.py cuda=None+零CUDA DLL, 未import); 无conda; 本机无3.11; C:余98.1G/D:余11.7G(BLOCKED); hf_cache 1.8G含推理全部骨干(权重零下载)。文件: remediation_r2a/ENV_INVENTORY.md + probes/。
- **E039 [A/B,依赖]**: 历史环境指纹=训练日志横幅 `Ultralytics 8.4.87 · Python-3.11.15 · torch-2.8.0+cu128`(属RTX 5070 Ti他机); transformers/timm版本本地零证据(仓库requirements未列+siglip_learn灭失)=UNKNOWN, 禁猜; 方案M1=py3.11+torch2.8.0+cu128@C:\fire_envs(峰值≈10–11.5G, 余量PASS)。文件: DEPENDENCY_EVIDENCE.json, ENV_OPTIONS.md, STORAGE_BUDGET.md。
- **E040 [设计]**: 十门验证顺序(S6=G4端到端复验门, S7=G3端到端复验门, S8=P6冻结12图), 逐门停止首败即停。文件: VERIFICATION_SEQUENCE.md; 汇总 machine/remediation_r2a_summary.json。
