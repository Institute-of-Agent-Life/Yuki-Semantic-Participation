# Yuki Semantic Participation

Yuki 的群聊语义观察与自主参与控制器。它接收 Host 已入账、已授权的事件，用 Jev
解释邀请、续聊和讨论边界，维护参与状态，并提出供 Host 接纳的自主机会。
本库不持有 QQ 凭据，不发送消息，不创建或运行主 Agent。

## 当前决策流程

1. 新观察只以真实 `human` 输入为焦点，上下文可包含真实人类与 SELF 发言。
   后台记忆种子扫描、seed 正文及新 recall/contact 候选已经删除；主 Agent 仍可按需查记忆。
2. 已有有效关系可以快速接话，Main 不等 Jev；同一真实输入可在原队列并行复核。
   `participation_view` 是纯查询，接纳不冒充语义确认，观察结果不二次启动同一 Main。
   自身 quiet 和真实语义纠正各自能收起原快路，join/stay 或发送不能互相覆盖放行。
3. 同人自动续聊不等待复核额度。已唯一匹配的普通续聊在后台只保留最新待评，
   新邀请与未匹配来源独立保留；统一 3/12/20 秒稀疏调度，不因已经活跃而加密。
4. 明确邀请/续答由 Host 提升为普通入场，`advance(include_addressed=False)` 避免同来源再开 SELF。
   非请求式加入和无来源 `intrinsic` 机会沿[连续概率模型](docs/autonomous-evolution.md)采样。
   实际自主 Work、计算、公开表达与真实接收反馈分别影响后续机会。
5. Host 核验来源、停止边界、唯一 owner、generation、身份和授权，再交给原 Main。
   行动、显式发送和 `NO_REPLY` 都由 Main 决定；恢复依据原执行身份与真实回执。

已有真人焦点的请求冻结局部 engage 和可选 SELF 整体意愿，帮助 Jev 区分退出后的新邀请
与他人交流。意愿不是聊天发言、真人 stop 或授权，也不因更新状态而额外调用模型。
当前投影为 `semantic-state-v3`，rubric 为 `v6-zh-6`，默认 Jev 模型为 `jev-1.13.0`。
缺状态保持未知，无消费者的 mood 已退出新自报。

`JevObserver` 的 `identity` 回调在编码每个新请求时提供当前角色的 `display_name` 与
显式 `aliases`，固定问句中的 `{{name}}` 由同一次身份读取展开；动态候选和原聊天不改写。改名不重置 Controller，
不新增观察；已经编码的请求和已有结果保持原件。`invite_yuki`、`yuki_participation` 等
仍是固定协议键，不表示角色必须叫 Yuki，也不声明其性别。

参与查询不再重算未使用的 `UnitParticipation.belief`。自主评分与历史折叠在一次调用内
聚合原动作，复用已解析的当前检查点，避免随单元数反复解码整个状态；
真实概率公式、quiet、来源撤回和原运行回执保持原语义，没有增加缓存或接话门槛。

## 仓库边界

- 本库负责观测协议、参与事实与纯查询、稀疏续聊复核、连续机会采样和快照。
- Host 负责 canonical 身份、权限、唯一接纳、Work、Main、工具和投递。关系或意愿不授予执行权限。
- `ScopedEvent.source_order` 和普通 `SelfReport.source` 只在本地保留原来源顺序，
  同时间输入不按 ID 字符串或结果到达顺序排序；这些管理字段不发送给 Jev。
- 旧 recall/contact 类型只供原已接纳运行和回执读取，不再排新观察或采样。
  旧自报、废复核字段升级及恢复边界见[宿主协议](docs/protocol.md)。

## 开发与回放

需要 Python 3.12 或更新版本：

~~~sh
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
python -m scripts.replay_sparse_review --output /tmp/sparse-review.json
~~~

[三版本离线对照](docs/evidence/sparse-review-2026-10-22.json)使用原 pin、仅合并队列、
完整稀疏方案的实际 ObservationSession 和合成 observer，比较真实 `evaluate` 启动数、
积压、到期和语义刹车延迟。90 条持续输入为 20→6→5 次观察；换题刹车为 12→16→20 秒。
减少观察密度有真实延迟取舍，不能据此保证线上 token 成本、模型准确率或 Main 速度。
原跨度 4 报告只保留历史证据；当前不再有等待复核额度的门槛。

此前 30 天合成负载、旧概率参数和真实合成 API 探针保留在 `docs/evidence/`，
只证明其报告所记版本，不作为当前发言率、缓存命中率或社交质量预测。
当前实现、定向验证与待完成验收见[实施状态](docs/implementation.md)。
测试和回放不会自动调用付费 API、主模型或 QQ。
