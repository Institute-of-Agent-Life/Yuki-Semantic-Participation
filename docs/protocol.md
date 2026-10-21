# 宿主与控制器协议

以下为独立库现行接口。Host 固定依赖、CI、合并、部署与真实 QQ 效果分别核验；
Yuki 的接线与授权见其 `docs/architecture/semantic-participation.md`。

## 身份、观测与讨论单元

一个 Controller/ObservationSession 只属于一个 `(conversation_id, generation)`。
Host 完成 canonical 入库、来源版本、可见性和授权后提供 ScopedEvent；平台消息凭据留在接入层，
引用与业务所有权使用原内部来源，target 只是语义对象，不授予 Person 私有读取权限。

`ScopedEvent.kind` 只表示真实 human/self，新 `request_observation` 仅接受 human 焦点及 conversation。
真实 SELF 发言可作上下文和原表达锚点，不作新观察焦点；后台 Memory 不再构造 seed。
thread、target、unit_options 及其中的 new 均由 Host 提供，Jev 只能选既有组合或 unknown，
Controller 复核完整分布、唯一赢家及其选项，不让模型制造身份或讨论 ID。
名字提及只提高观察优先级，不等同 @ 或授权。明确邀请且楼层留给当前 Agent 时，
即使旧讨论归属 unknown，仍可采用 Host 已给出且核验的新讨论单元。

`ScopedEvent.mentions` 接收 Host 的原生提及快照：kind、display_name、原文位置和可证明的内部
Account／Assignment／Person／Presence。顺序、重复、未知目标及未知位置均保留；正文中的普通
`@文字` 不生成此事实。提及不改 ScopedEvent.target 或讨论单元，它们仍表示原参与对象。
Jev 新请求提供原位 mentions：人物沿同作者的当前人物投影比较，账号仅用请求内 aN 关联，
Assignment／Presence ID 留在本地原 Snapshot。kind=yuki 明确提到当前 Agent，member 和 all 保原语义。
未知账号不需要 Person，也不据此创建身份或授权。无提及字段的旧 Snapshot 表示未提供，
不补查、不重写原已付请求；快照与独立存储的 serde 保原 ID 和原名称。

Jev 使用原生 `model/state/questions → answers/usage`，默认模型 `jev-1.13.0`，
rubric `v6-zh-5`。请求使用中性 eN/tN、原文、作者、关系、引用、相对时间和省略说明。
scope、generation、revision、请求序号等本地管理字段不发送。
Snapshot 保留 Host 实际提供的材料，DTO 不另加实验字节或条目帽；实际 Provider 容量错误正常返回，
字节不是 token，token 只记录真实 usage。

`JevObserver(..., identity=...)` 要求 Host 提供同步身份回调，返回当前角色的
`display_name` 与显式 `aliases`。编码每个新请求时读取一次，放入 `state.agent_identity`，
只作为结合原生提及、引用与上下文识别对象的资料，不是扮演角色或修改评分规则的指令。
本库不加载人格文件，不推断性别，不由名字创建身份或权限。
`invite_yuki`、`extend_yuki`、`ask_yuki_stop`、`floor_state.yuki` 和 `yuki_participation`
保留原协议含义；这里的 yuki 是标签，不是当前角色的名字。
评分选项与控制语义未变，rubric 仍为 `v6-zh-5`，原已应用事实与 Controller 状态不重置。
后续新观察使用调用时的身份参考；改名本身不排观察、不改 Snapshot，不改写已编码请求或已有结果。

interaction、information、floor 是新资格的必需维度，缺失或非法保留诊断并计失败；
合法 unknown 不计失败。明确邀请和楼层已确定时，信息新旧或旧讨论归属 unknown 不单独否决邀请。
其它 unknown 不生成新资格或用户停止；辅助 boundary 缺失不等于整个 Provider 故障。

## 冻结的参与意愿

原队列取出真实焦点后，`Controller.freeze_observation` 冻结当时合法意愿及本地复核位置。
新投影为 `semantic-state-v3`，外发结构示例为：

~~~json
{
  "projection_revision": "semantic-state-v3",
  "yuki_participation": {
    "scope_intent": {"engage": "stay", "source_seconds_from_focus": -120},
    "units": [
      {"thread": "t2", "target": "已有目标", "engage": "quiet", "source_seconds_from_focus": -30}
    ]
  }
}
~~~

这只是结构示例，不是真实群聊抓包。局部 engage 对应原讨论/对象，SELF 整体意愿单独限定 scope。
歧义焦点相关的 unit_options 各自映射，不按同作者、quiet 或活跃度强选一个单元。
thread 与原选项共用 tN；相对时间沿原报告来源时间，不把响应完成时间伪装成来源时间。
没有合法报告则省略，不默认 stay；quiet 仍可查询与投影，不能因非活跃而过滤丢失。

rubric 明确这是当前 Agent 自身意愿，不是聊天成员发言、真人邀请、停止要求或授权。
quiet 不阻止识别真人重新邀请，join/stay 不证明真人仍在和当前 Agent 交流。
Jev 解释真实输入，后端决定参与；意愿变化本身不新排观察、不补问“是否应该 quiet”。
在途自报变化不改写 Snapshot、请求字节或 trace，不 veto 原结果；真实来源修订与撤回继续核验。
Observer 必须返回与实际提交相同的 Snapshot。

`source_order`、`SelfReport.source`、run/request/response、完整 basis 与 `unit_reviews`
留在本地作来源、先后及回写证明；外发只有相关 engage、作用范围及相对时间。
旧真人 Snapshot 缺这些字段时表示未提供，不补造旧意愿，不重写已付请求。

## 接纳与执行

~~~text
Host canonical 入账、版本与授权
  → observe_committed_event / session.observe（仅记录）
  → participation_view（纯查询）
  → Host 显式 request_observation → evaluate_due
  → Host 唯一 selector 及 epoch
  → advance(include_addressed=False, intrinsic_allowed=...)
  → 保存 State/observer checkpoint
  → Host 原子接纳 proposal（同 ID 返回同 run）
  → 原 WorkScheduler / Main / 工具与发送
  → 原 run 的 Feedback → 保存 State
~~~

Proposal 是机会，不是发送命令或工具许可。Host 仍核验来源、停止、Space/Presence、
scope/generation、owner/epoch 与真实权限，原 Work 容量沿原调度处理。
慢观察不占全局接纳锁，写事务不等待模型/网关或扫描历史。

`pending` 保留未得到 Host 答案的提议。先按原 proposal_id 查询：已接纳接回原 run，
结果未知不删除、不换 ID 或重提；Host 证实未接纳才调用 `discard_unaccepted_proposal`。
真实接受后让出 pending 槽，原已接纳 Work 的等待与恢复由 Host 管理，不能让旧槽封住后续机会。
`include_addressed=False` 避免真人邀请同时产生 SELF；它不删除原观察或未知接纳。

master off 是 off；显式关闭 semantic 才使用 Host legacy。缺 key、观测故障或 unknown 不切 proposer，
不伪造观察。401/403 原退避为 300–900 秒，其它传输/限流/服务端故障为 30–180 秒；
422 丢弃该次无效请求，没有新鲜真人材料不做空探测。旧 configuration_valid 读取后丢弃。
last_failure 只留类别、HTTP 状态、时间和可信来源，不保存异常正文或密钥。

## 普通参与与有限快路

`participating_units` 返回真实保留状态，包含 quiet/closed，Host 的 active 消费者才排除它们。
`participation_actor_ids` 只读取原 checkpoint 中真人单元的 actor/target ID，供 Host 批量准备
当前身份关系；不推导参与、计数、语义复核或概率，不改原事件与检查点。来源页 hydrate 不为
收集人物 ID 调用完整 `participating_units`；原事件正文退休后仍可从保留单元准备这些身份。
`participation_view` 返回来源有效性、真实解析、候选/匹配及 needs_observation，不排队、不消费、不采样。
`UnitParticipation` 只包含参与事实和快捷匹配需要的字段；未使用的 `belief` 展示字段已删除。
真实自主概率仍由 `Controller.belief` 按原观察、输入、表达和衰减计算。
一次参与查询、意愿冻结或接纳写入只解析一次当前 participation checkpoint；
自主评分和退休折叠按讨论/对象一次聚合动作，不为每个单元重新解析或遍历全部事实。
这些 typed 读取和索引只在当前同步调用内复用，不跨状态变更保存，也不缓存权限或新增有效期。
Host 默认 new 单元不冒充真实 Jev 解释；自身群表达也不能让全部群成员自动参与。
单元必须有真实表达或 join/stay，并同时保有未被语义否定的关系依据，才能用于快捷匹配。

Host 以 `UnitBinding(scope, unit, actor, basis)` 关联普通接纳。单元保留首次建立的来源，
input_ref/last_at 独立前进，每次表达和 hint 保留自己的来源，不无限累积历轮祖先依赖。

- `observe_unit_input(binding, ref, continuation=False)` 登记真实输入并消费原事件。Host 只对
  原普通 continuation 传 True，按同一来源记一次快接；接纳不伪造观察或真实表达。
- `observe_unit_hint` 保存可选 join/stay/quiet，限定原 unit，不制造用户 stop。
  旧报告不能覆盖更新意愿，也不重建被撤销的来源。无报告不补问。
- `observe_unit_expression` 按原 effect/run 关联真实 message，compute/tool 不冒充发言；
  多条真实物理锚点不重复累计逻辑表达。晚到回执照实入账，不重开已退役关系。

`continuation_review_span` 默认 4，计同一讨论/对象未经语义确认的快捷接纳输入，
不是消息条数、冷却时间或每日配额。Main 快接不等 Jev，原队列可并行复核真实焦点；
匹配或已接纳不再让 needs_observation 自动 False，但仍需显式排队，不保证每条消息均评分。

自身 quiet 和 other_exchange/转题/真实停止分别收起原快路，不能以 OR 互相放行。
自身 stay、发送、unknown、缺维度或观察故障都不续确认额度；达到跨度的新输入回原语义入口。
真实新邀请保留原即时入口，不是全群禁言。正向确认只推进原焦点的位置，
不能因后续 C/D 出现在 context 或 HTTP 期间到达就批量清零。
hydrate、重复回调、steer/排队竞争、重启不重复计数；原最小接纳身份仍用于晚到报告排序。

Host 在未接纳队列的 promotion/selected writer 核 `continuation_current`，正文 revision
没变也要消费最新退出事实。原 Main 合法补充的 `check_budget=False` 只跳新 activation 的跨度，
仍核验来源、quiet 和语义资格；原 Work 输入沿原匹配/接纳，不被 K 误伤。
已接纳 Main、预算和真实发送不因纠正倒退，不重跑或重发已交付结果。

普通 `SelfReport.source` 指向原输入，`ScopedEvent.source_order` 是 Host 提供的原内部顺序
（Yuki 用账本 row.id），并随 SeenSource/最小接纳事实保留。
同时间的报告与观察按可证明的原来源顺序处理，不解析 opaque ID 字符串或按 HTTP 到达时间猜；
缺证明保持未知。旧 B 否定可收起未确认 C 借用的旧依据，不能覆盖较新已确认邀请；
旧 activation 的晚 stay 不能覆盖较新 quiet，真正较新的同时间报告仍可正常更新。
冷恢复后 Host 重读同一来源的明确顺序时，仅补原 SeenSource 元数据，不将它记成新事件，
也不改写已有 event、队列或在途 Snapshot。旧 hint 缺 source 时，只能由其原 hint_basis 中唯一同时间来源
恢复比较依据，无法唯一证明时继续未知，不能按到达顺序补造。

## 种子退役与原运行恢复

后台 Memory 扫描/游标、新 seed/recall/contact 排评、seed_fit、claim/attempt/content fingerprint 已删除。
conversation 熟悉度、真实 SELF 来源、Main 按需 Memory、自省与 Dream 保留。
RECALL/CONTACT 枚举只读历史提议，不进入新观察或采样。

旧 State/observer checkpoint 在严格校验前清退 seed 正文、未接纳候选及种子派生键，
不清空真实 human/self、意愿、原运行或回执，也不把种子搬到下一次 Jev context。
旧已接纳 recall/contact 保留原 proposal/run 映射；pending 即使到期也先查 Host 接纳，
不能在查询前丢掉恢复 W、公开暴露 E、真实接收 R 及计算所需的原件。
旧 run/普通 binding 的 Memory 依赖由 Host 按明确 fact ID 核 active、verified、有效时间、
原群/可见 SELF 范围与至少一条当前可读证据，不恢复候选扫描或借目标人扩权。

`SelfDelta` 仅含可选 engage。无消费者 mood 已删除；旧 reader 只清退该字段，
保留原报告身份与合法 engage，mood-only 不创建空状态。输出尾段由 Host 在可见发送前剥离，
无效尾段不重试主模型，子 Agent 不代报主回合。

`participation_v1` 保留原 namespace 与 SnapshotStore CAS。旧 unit 缺 `fast_inputs`
表示未知，不当作 0 或新满额，真实新语义确认才建立已知进度。
正文最多 600 秒、256 条/256 KiB，裁剪不等于交流自动过期；必要 seen/source_order、
retired_refs、当前意愿、最小表达及复核依据随原单元保留。
真实修改/撤回继续撤销来源，quiet 与实际关闭不因裁剪或重启解除。
64 单元/1024 来源身份是原资源边界，优先清退旧派生关系，不保留无限正文或拒收新业务。
缺退休证明不凭任意 seen 指纹复活关系；旧 reader 不认识的数据不作为新资格。
未知扩展 namespace 保持原件且禁止写回。既有 `continuation_current` 单次读取仍可按原版本、
actor、来源修订和退休证明识别原单元；这保留已有读合同，不把未知扩展普遍当作新资格。

## 升级与回退

升级同时固定库修订与 Host lock，保存数据库、原协议/工具文件和挂载配置的完整备份。
仅清退三个种子参数 independent_source_factor、recall_source_decay_seconds、contact_source_decay_seconds，
其它实际覆盖保留，任意未知字段仍严格校验；新复核跨度沿原参数加载，不重置预算或采样事实。

旧付费协议和 trace 保留原件，新请求使用新投影/rubric；历史重构必须用其原版本，
不把当前编码结果称为抓到的历史 HTTP 原字节。新意愿或复核状态不制造兼容重跑或补发。

回退优先使用兼容新格式的修复镜像。已有新输入、执行或投递事实时，不能恢复旧数据库覆盖它们。
固定旧 reader 读写 namespace 的专项测试不等于任意旧镜像都可安全回退。
SnapshotStore 是同步单线程 SQLite CAS，不跨线程共享连接，不在事务内等外部工作；
它的容量检查仍在写事务内聚合 payload 大小，不声称事务中完全无扫描。
本库不实现 QQ、远程发送、多租户认证或独立 Agent Runtime。
