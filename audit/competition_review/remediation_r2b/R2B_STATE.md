# R2B_STATE.md — 环境建立运行日志（无人值守流水线）

## 2026-08-28

- 解释器: python-3.11.9 (python.org, sha256 5ee42c4e…5fdde 校验) → `C:\fire_envs\py311` 用户级静默安装（AddToPath=0, Launcher=0, 退出码0, 不触碰 PATH/注册表 launcher）
- venv: `C:\fire_envs\sf2026_min`（pip 26.2.1）
- 事件1: 首次 `pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128`（--no-cache-dir）在 3461.4MB 主 wheel 下载至 ~470MB 时被**运行器终止**（后台任务 ~55 分钟寿命上限；pip 部分下载被丢弃，无 --no-cache-dir 可续传）。retry_ledger.network_transient = 1/2
- **方法偏离（登记）**: 改用 8 路分段 Range 下载同一官方 wheel（同源 download.pytorch.org/whl/cu128，HEAD 证实 Content-Length=3461420395、Accept-Ranges: bytes），脚本 `dl_torch_parts.sh` 幂等可续传；拼装后校验字节数与 SHA-256，再 `pip install <本地wheel>`。带宽探测：单连接 0.2–1.9 MB/s 波动，并行连接各自独立 ⇒ 8 路并行
- 事件2(探测): 三条并行 20MB Range 探测 2/3 成功（1 条因 shell 后台链引号解析失败退出，属脚本问题非网络失败），实测并行连接带宽独立
- 下载目标目录: `C:\fire_envs\dl_torch\`（C:\fire_envs 已登记；wheel 拼装+校验后安装，安装完成后保留 wheel 作为冻结凭证）
- 环境验收门: `verify_env_gate.py`（17 项）待 torch+其余包装完后执行
- 后续包: `pip install --no-cache-dir ultralytics==8.4.87 transformers timm`（pypi 官方源；装后立即 pip freeze + 哈希冻结；≤2 次有证据的版本修正）

## 现场保全

- pip 首次安装日志: `logs/pip_torch_cu128.log`（保留，作为事件1证据）
- 运行器终止通知原文已存会话记录；pip 进程死亡经 Get-Process 复核

## 下载完成（事件2 收尾）
- 8路分段 Range 下载全部到位，脚本在运行器清理前已完成：CURL_PHASE_RC=0, ASSEMBLY_OK, 字节数 3461420395 == HEAD Content-Length 精确匹配
- 组装 wheel zip 完整性 testzip=OK（11512 成员）
- torch wheel SHA-256: 34c55443aafd31046a7963b63d30bc3b628ee4a704f826796c865fdfd05bb596
- 来源: download.pytorch.org/whl/cu128（官方，HTTPS 每段 TLS 校验 + 字节精确拼接）
