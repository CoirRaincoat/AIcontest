# G1→G2→新模型G6→最终复审 汇报（≤30行）

1. **最终停止点**：全部步骤完成，无硬失败，安全结束（本轮 G1 三模型+三头、G2、新模型 G6 全部跑完）。
2. **原模型 G6**：PASS（1100 键精确覆盖，p7_validator rc=0，标记 ORIGINAL_MODEL_WITH_KNOWN_LEAKAGE）。
3. **三模型训练状态/耗时/SHA**：m=80轮/6448s/39e6d16c65be7858；s=74轮(早停)/3612s/aaa492d36f3ac9e4；s_aug=80轮/4074s/28520644f85195f0（另三头 siglip=c7ae9311、dino=8b8005f8、crop=b57ee5f8）。
4. **累计 GPU 训练时间**：3.93h（预算 12h，余量充足；实际 per-epoch 远快于干跑首轮）。
5. **是否发生恢复**：无（resume_ledger 空；中途仅头重建因缺 sklearn 依赖失败一次，补装后重跑，属依赖补齐非训练恢复）。
6. **grouped holdout（一次性，n=146）**：TP99 FP18 FN7 TN22；P=0.8462 R=0.9340 F1=0.8879；95%CI P[0.7949,0.8974] R[0.90,0.9706]。
7. **primary vs exploratory**：primary=冻结生产融合规则（P.8462/R.9340）；exploratory=F-15应用全图补救（P1.0/R.5283，仅诊断，禁止择优冒充）。
8. **F-11 是否解除**：解除（6/6 条件满足：仅train训练/calib仅val/holdout仅一次/checkpoint+规则先冻结/隔离区未用/跨集d≤8=0）。
9. **新模型 G6**：PASS（1100 键，validator rc=0，标记 GROUPED_MODEL_REG1_RETRAINED，未覆盖生产权重）。
10. **新评分+仍生效硬门槛**：绑定封顶「数据泄漏≤39」解除 → 裸分42起，G7/端到端/环境/重训提供上调证据（不擅自给精确综合分）；仍生效：无官方测试集（提交就绪未解除）、同源非外部（性能标"未充分验证"）、完整系统/可视化/云端仍文档级。
11. **是否建议晋升**：不自动晋升；无泄漏 P=0.846 低于原污染 P=0.947（隔离后诚实回落），是否替换生产权重由用户决定。
12. **恢复入口+唯一人工决策**：入口 machine/g1_g2_final_summary.json + remediation_g1/REG1_ASSETS.json + g1_controller_status.json；唯一决策=是否晋升 grouped 新模型覆盖生产权重。
