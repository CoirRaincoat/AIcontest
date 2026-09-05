# PRODUCTION_SWITCH_AND_ROLLBACK.md — 可回滚切换方案（仅方案，不实际切换）

> 当前不执行任何切换。本方案用于未来若获用户批准晋升 REG1_GROUPED_RC 时的安全流程。

## 核心禁令

- **禁止**通过直接覆盖同名权重完成晋升（即绝不把 REG1 的 best.pt 覆盖到 LEGACY 的
  `fire_detection/runs/...` 路径，反之亦然）。
- 切换 = 修改生产推理入口的**权重路径引用**（或部署一个指向候选权重的独立入口），
  而非移动/覆盖权重文件本身。
- 任何切换前先全量备份 manifest + SHA。

## 原子切换方式（候选 A：路径重定向，推荐）

1. 生产推理入口的权重路径不写死为单一路径，而是读取一个**候选指针文件**
   `active_candidate.json`（内容如 `{"candidate_id": "...", "weights_base": "..."}`）。
2. 切换时用**原子写**更新指针：写 `active_candidate.json.tmp` → `os.replace` 到正式名。
3. 六类权重（3 YOLO + 3 head）的路径由指针 + 注册表路径联合解析，**一次切换六者一致**，
   杜绝 LEGACY/REG1 的 YOLO 与分类头混装。

## 切换前后检查（强制门）

- **切换前**：
  1. 备份当前 `active_candidate.json` 与当前六权重 SHA（读 MODEL_CANDIDATE_REGISTRY.json）
  2. 校验目标候选六权重 SHA == 注册表冻结值（全部 12 个 SHA 逐一对上）
  3. 校验数据划分/配置 SHA 未漂移
- **切换后**：
  4. 重新读指针，断言六权重路径一致归属同一 candidate_id
  5. 六权重 SHA 复核 == 注册表

## 冒烟测试（切换后、对外前）

6. 单张正常图推理 → rc=0、JSON 恰 1 键、契约正确
7. 单张损坏图 → rc=3、PARTIAL+failures、主提交名不写（G4 复验）
8. canonical 1100 surrogate → p7_validator(no-GT) PASS（G6 复验）

## 失败回滚

9. 任一门失败 → 立即把 `active_candidate.json` 原子回滚到备份值
10. 回滚后重跑冒烟测试确认恢复到 LEGACY
11. 全程不触碰权重文件本体，故回滚零风险、零数据迁移

## 混装防护（LEGACY/REG1 的 YOLO 与分类头）

- 六权重作为一个原子组切换（YOLO_m/s/s_aug + siglip/dino/crop head 同时切换），
  从不单独切换某几个。
- 推理入口按 `active_candidate.json` 的 candidate_id 一次性解析六路径；若解析出的
  六路径分属两个 candidate（SHA 与注册表不符），立即 fail-closed 拒绝运行。
- 每次推理启动打印 candidate_id + 六权重 SHA 前 16 位，写入 metadata 供事后对账。

## 记录

- 每次切换动作（谁、何时、从哪个 candidate 到哪个、前后 SHA）写入
  `machine/switch_ledger.json`（append-only）。
