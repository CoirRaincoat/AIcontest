# R3_PLAN.md — 端到端复验 + P6 + G7 执行序（2026-08-28，无人值守流水线授权）

前置: R2-B 环境验收门 `remediation_r2b/verify_env_gate.py` 全过 (ENV_GATE_RESULT.json all_pass=true)。
顺序执行，**任一步硬失败即停止**，保留日志/退出码，不跳步、不重跑追结果。

| # | 动作 | 命令（env=C:/fire_envs/sf2026_min/Scripts/python.exe） | 通过判据 |
|---|---|---|---|
| 0 | 权重 SHA 复核 | `certutil -hashfile <5个生产权重> SHA256` 对照 machine/weights_inventory.txt | 全部一致 |
| 1 | P6 冻结清单生成（推理前） | 任意 py: `remediation_r3/gen_p6_manifest.py` | 12图清单+gt 落盘 ✅(已完成 2026-08-28) |
| 2 | 单图正常推理 (S5) | `predict_best_fusion.py --source <1张val图> --output remediation_r3/single/single_pred.json --dino-checkpoint … --crop-checkpoint … --cache-dir <repo>/hf_cache/hub` + 全部显式参数, env HF_HUB_OFFLINE=1 | rc=0, JSON 恰1键, 契约无误 |
| 3 | 全链最小冒烟 (12图) | 同上 `--source remediation_r3/p6 --output remediation_r3/p6/p6_pred_smoke.json` | rc=0, 12键 |
| 4 | P6 冻结复现 (S8) | 步骤3的正式跑: `--output remediation_r3/p6/p6_pred.json` + `_details.csv`；diff 脚本对照冻结 details ±1e-3 + 标签 12/12 | 12/12 位一致, 分数±1e-3；不一致只记录不调参 |
| 5 | G3 全集口径端到端 (S7) | (a) 步骤2单图产物 vs 220 官方 gt → 预期 rc=2 + INCOMPLETE + strict 口径; (b) 220全图全链跑 `--source fire_detection/data/val/images --output remediation_r3/full220/val_pred_reg0.json` → `evaluate_image_level.py` vs val_gt.json | (a) rc=2 缺图入名单; (b) rc=0 五行输出 == E019 (TP162 FP9 FN3 TN46 P.9474 R.9818 F1.9643) |
| 6 | G4 真实坏图注入 (S6) | 审计目录内构造 2 张截断/损坏副本 + 1 张正常图 → 全链 | rc=3; *_PARTIAL.json 仅正常图; *_failures.json 有 {file,stage,exception_type}; 主提交名不存在; p7_validator 拒收 PARTIAL |
| 7 | G7 计时 (S9) | `remediation_r3/g7_latency.py` | latency_profile.json 落盘; 无 OOM; 数值只记录 |
| 8 | p7_validator 终检 (S10) | 对步骤4/5(b) 正式产物 validate；preflight-images 对源图集 | 全 PASS |

坏图制作纪律: 仅在 remediation_r3/corrupt/ 内生成副本并截断，**严禁触碰原始数据**。
核心门(步骤2/4/5/6)任一失败 → 不进入 G1，降级记录 R1 状态。

产物: R3_REPORT.md + P6_G7_END_TO_END_REPORT.md + 本表各步骤日志/JSON/退出码 + EVIDENCE_INDEX 增补。
