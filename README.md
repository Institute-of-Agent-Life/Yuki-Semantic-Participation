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
3. 未核验快接按原输入计数。默认 `continuation_review_span=4`，达到后新输入回到原语义入口；
   明确请求、原 Work 输入和已选入原 Main 的合法补充保持原入口。它不是发送配额。
4. 明确邀请/续答由 Host 提升为普通入场，`advance(include_addressed=False)` 避免同来源再开 SELF。
   非请求式加入和无来源 `intrinsic` 机会沿[连续概率模型](docs/autonomous-evolution.md)采样。
   实际自主 Work、计算、公开表达与真实接收反馈分别影响后续机会。
5. Host 核验来源、停止边界、唯一 owner、generation、身份和授权，再交给原 Main。
   行动、显式发送和 `NO_REPLY` 都由 Main 决定；恢复依据原执行身份与真实回执。

已有真人焦点的请求冻结局部 engage 和可选 SELF 整体意愿，帮助 Jev 区分退出后的新邀请
与他人交流。意愿不是聊天发言、真人 stop 或授权，也不因更新状态而额外调用模型。
当前投影为 `semantic-state-v3`，rubric 为 `v6-zh-5`，默认 Jev 模型为 `jev-1.13.0`。
缺状态保持未知，无消费者的 mood 已退出新自报。

参与查询不再重算未使用的 `UnitParticipation.belief`。自主评分与历史折叠在一次调用内
聚合原动作，复用已解析的当前检查点，避免随单元数反复解码整个状态；
真实概率公式、quiet、来源撤回和原运行回执保持原语义，没有增加缓存或接话门槛。

## 仓库边界

- 本库负责观测协议、参与事实与纯查询、有限快路复核、连续机会采样和快照。
- Host 负责 canonical 身份、权限、唯一接纳、Work、Main、工具和投递。关系或意愿不授予执行权限。
- `ScopedEvent.source_order` 和普通 `SelfReport.source` 只在本地保留原来源顺序，
  同时间输入不按 ID 字符串或结果到达顺序排序；这些管理字段不发送给 Jev。
- 旧 recall/contact 类型只供原已接纳运行和回执读取，不再排新观察或采样。
  旧自报、旧未知复核进度及恢复边界见[宿主协议](docs/protocol.md)。

## 开发与回放

需要 Python 3.12 或更新版本：

~~~sh
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
mypy src
python -m scripts.replay_fastpath_review --output /tmp/fastpath-review.json
~~~

默认跨度 4 的选择依据是[有限快路回放](docs/evidence/fastpath-review-replay-2026-10-15.json)：
192 条固定 active/inactive 队列对照使用真实 Controller、合成语义和虚拟时钟，
复用五份真实请求的耗时量级；输入 90 秒后另排空 22 秒，观察窗口合计 112 秒。
这说明有限外推与及时续聊的取舍，不证明线上最优值、模型准确率、Main 实际延迟或群聊效果。
后台真人复核可能新增 Jev 请求，需与种子删除的降量分别计量，不能预报净成本下降。

此前 30 天合成负载、旧概率参数和真实合成 API 探针保留在 `docs/evidence/`，
只证明其报告所记版本，不作为当前发言率、缓存命中率或社交质量预测。
当前实现、定向验证与待完成验收见[实施状态](docs/implementation.md)。
测试和回放不会自动调用付费 API、主模型或 QQ。
