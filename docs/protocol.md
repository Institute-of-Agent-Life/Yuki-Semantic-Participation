# 宿主与控制器协议

以下为独立库的现行接口。Host 固定依赖、部署与真实 QQ 社交效果须按 Host 交付记录分别核验。
Yuki 接线与权限的现行说明见其 `docs/architecture/semantic-participation.md`。

## 身份、观测与讨论单元

每个 `Controller` / `ObservationSession` 只属于一个 `(conversation_id, generation)`。
创建新 generation 使用新的实例及持久 key；旧响应不能进入新实例。
Host 先完成 canonical 入库、可见性和授权，再传递有版本的 `ScopedEvent`。
平台消息凭据只在接入层解析引用，之后 `reply_to` 使用可信内部来源引用，不靠正文或 QQ 消息号重建身份。

`thread`、`target` 和歧义候选组合由 Host 给出。Jev 只能从既有 `unit_options` 或 unknown 中选择，
不能制造 Person、讨论 ID 或授权对象；`new` 也必须是 Host 预先提供的选项。
适配器映射结果后，Controller 再核验完整分布、唯一赢家及其所属选项，原 Snapshot 不被改写。
target 是语义对象，不能成为读取私人资料的 principal。

Host 可以对包含机器人称呼的焦点标记 `observation_priority`，使其在稀疏队列中先交给 Jev；
此标记不是 `@`、回复授权或候选资格。Jev 若判断为明确邀请/续聊且楼层留给 Yuki，
即便旧讨论 `unit_selection=unknown`，Controller 也只可采用 Host 已给出的 `new` 单元，
且须核对该单元的 thread 和作者目标；其它歧义继续保持未知。
明确邀请的合格观测可由 Host 提升为普通 USER_MESSAGE；非请求式候选按当前状态的连续机会率采样，
不等待积累门槛。采样只控制新自主 Work 的机会，不限制已接纳 Work 内的消息条数。
Host 对新的 proposal 仍执行来源、
停止边界、代际、权限和 Main Agent 接纳检查。已有 Work 的执行容量由 Host 原调度处理。

Jev 请求采用紧凑语义投影：中性局部引用、原文、作者角色、讨论与对象关系、引用关系、相对时间和省略说明。
scope、generation、revision、sequence 等本地管理字段留在原 Snapshot 中校验，不发送给模型。
快照保留完整焦点、Host 提供的上下文与引用锚点；DTO 不按字符串或条目数量截断，
适配器不叠加实验字节上限或裁剪输入。实际 Provider 容量错误沿原请求失败返回，
字节不是 token，token 只记录 Provider usage。

interaction、information、floor 是新资格所需的必需维度；缺失或非法时保留诊断并计健康失败。
合法 unknown 不计失败；若交际行为和楼层均明确表示邀请 Yuki，信息新旧或旧讨论归属 unknown 不单独否决邀请。其余缺乏明确交际行为或楼层的 unknown 不生成资格或停止边界；软状态仍按完整概率分布更新。
单个辅助 boundary 缺失不等于整个 Provider 故障。明确停止和重新邀请的作用范围均由 boundary 观测约束。

## 接纳与执行

```text
Host canonical 入库、来源版本与授权检查
  → controller.observe_committed_event(ScopedEvent)（或 session.observe，仅记上下文）
  → controller.participation_view(event, now)（纯查询）
  → 必要时 session.request_observation(event.ref)
  → await session.evaluate_due(now, active=...)
  → Host 唯一 selector 更新 off / legacy / semantic 及 epoch
  → controller.advance(now, controller_epoch=..., host_available=..., intrinsic_allowed=...,
                       include_addressed=False)
  → 保存 State（含 observer_checkpoint）
  → Host submit_proposal；同 proposal_id 返回同 run 或明确拒绝
  → 原有 WorkScheduler / Main Agent / 工具与发送链
  → controller.observe_run_feedback(Feedback)
  → 保存 State
```

`host_available` 是宿主当前的接纳事实，不由 Jev 决定。Proposal 不是发送授权。
Host 再核验 scope/generation、owner/epoch、来源版本与可见性、真实停止、Space 和 Presence。
支持是 Controller 解释机会的资料，不是 Main Agent 的工具许可或第二次评分门槛。
`pending` 只保留尚未得到 Host 回答的 proposal；收到真实接纳反馈后让出该槽。
已接纳 run 的执行、等待和恢复由 Host 的原 Work 与持久回执管理，控制器保留其反馈与来源关联，
不以一个未清空的接纳槽阻止后续合法机会。
慢观测不占全局接纳锁；原子写事务不等待 Jev、主模型、网关，也不扫描历史。
同一 proposal 由原持久记录返回同 run；Host 不把其它 proposal 曾考虑过的来源当作接纳禁令。

`include_addressed` 默认 True 兼容旧调用；当前 Host 设 False，已观察的真人邀请不再同时产生
新 SELF 入场。它只过滤候选，不删除原观察或忙碌后待处理的来源。旧 pending 先以原 proposal ID
查询 Host 接纳记录；已接纳或外部结果未知不得删除或换 ID。Host 证实未接纳后可调用
`discard_unaccepted_proposal`，只退役该 pending/proposal，保留原候选与观察供普通入场。

## 普通参与查询与反馈

`Controller.participating_units(now)` 返回已有普通 unit 和实际 SELF 表达的 unit/锚点。
`participation_view(event, now)` 返回 `source_valid`、`current_resolved`、`current_unit`、
`addressed`、`matched_unit`、`candidates`、`ambiguous`、`needs_observation`；查询不排队、
不修改快照、消耗事件或采样。已记录事件有效只证明来源存在；`current_resolved` 还需要真实
Jev 观察，Host 默认解析出的 new unit 不能冒充观察。新真人引用群 SELF 表达时可提供候选锚点，
不能据此把全部群成员当成已参与；个人目标、真实关闭、局部 quiet 和歧义继续约束匹配。
`matched_unit` 还须有真实 message 表达关联或明确自身 join/stay，不能仅凭收到 @ 并接纳、
但尚无表达且 NO_REPLY 就自证参与。`expressed` 是原效果事实，不是 E 分数门槛。
已绑定输入被同源真实观察重释为别的 unit 或 unknown 歧义时，旧 unit 不再用于匹配，
但原入场和表达映射保留；查询不会为修正状态再调用 Jev。

Host 将已接纳普通轮绑定为 `UnitBinding(scope, unit={thread,target}, actor, basis)`，
`basis` 可含实际观察的焦点及上下文依赖。每条引用须仍为当前有效版本：

同 scope、unit 和 actor 的参与单元保留首次建立时的 `binding`，最新输入单独更新
`input_ref` 和 `last_at`。后续入场不累计所有前轮来源；每次真实表达的关联和自身意愿的
`hint_basis` 仍保留该次完整绑定。建立来源或最新输入失效时不能继续匹配，某次意愿的
独有上下文失效只撤销该意愿，不改写实际表达回执。

- `observe_unit_input(binding, event_ref)` 登记真实入场并消费原事件。它建立 H，不伪造 Jev 观察或互惠 E。
- `observe_unit_hint(binding, SelfReport)` 登记可选 join/stay/quiet，仅作用于该 unit。quiet 是自身意愿，
  不写用户 stop；join/stay 不解除真实关闭。它只更新仍存在的真实入场 unit，晚到意愿不重建
  已重释或退役的参与关系。无 hint 不补问；原 run/request/response 防重。
- `observe_unit_expression(binding, run_ref, Effect, anchor=...)` 关联真实 message 效果，
  按原 effect ID 去重；`actual_targets` 保留传输事实，不把 group 改成人。一个逻辑效果的多条
  真实物理锚点可逐条补登记，不重复计表达。compute/tool 不能冒充 expression。
  晚到真实回执保留原效果关联，但不据此重建已退役的 unit；新普通入场继续由原真实输入建立。

来源修改、撤回和 generation 切换使对应绑定/意愿失效。普通轮复用 Host 原执行和发送，
不制造 SELF proposal、Work 或第二个 executor；库不决定执行权限。

Yuki 的唯一 selector 覆盖 legacy 和 semantic。master off 始终是 off；显式关闭 semantic
才使用 legacy。Jev 缺 key 或观测故障不改变 proposer，真实语义反馈及活动继续演化，
不伪造新观测。degraded 仅供诊断，真实成功才清除；原队列与有界退避负责新鲜输入的恢复。
401/403 退避 300–900 秒；传输、限流与服务端错误退避 30–180 秒。422 丢弃该次请求，
不反复重试原输入或封停整个 scope。没有真实新鲜输入时不做空探测。旧检查点的
configuration_valid 字段读取后丢弃，不再成为永久禁试条件；不清除控制器反馈或已接纳 Work。
原检查点 last_failure 仅保存错误类别、HTTP 状态码、时间和可信内部来源引用，
不保留异常正文或凭据。当前没有生产 shadow 模式配置。

接纳产生正式 SELF initiative run，与唯一 `initiative:<run_id>` Work 绑定。SELF 不借用最近一位发言者权限，
没有人类 user/person principal；当前社交读写范围是本群授权历史、群可见及公开 SELF 记忆和本群普通发送。
持久工作区、终端与计算继续使用原工具执行边界，不能借此获得私聊或宿主管理权限。
它继续使用主 Agent 固定提示词及工具声明，执行处检查权限，不建立第二套 Agent Runtime。
最终文字默认内部返回；公开表达走显式发送工具，NO_REPLY 合法，调度器不自动补一句收尾。

## 来源、记忆种子与撤回

Host 发现来源撤回或版本变化，先调用 `observe_source_change`，再送入已授权的新版本。
解释及其上下文依赖和参与绑定一并失效。context 出现在快照中不代表它已被作为焦点评分。
原 `predict_continuation` 及 predicted 新候选已退役；可信引用关系用于纯参与查询，
不取代待评语义，也不能绕过结束、对象或 generation 边界。
重新开放必须有独立、范围一致的明确邀请证据，不能由沉默衰减或自身发言制造。

Yuki 的 memory-only seed 来自 active、verified、当前可读且 lineage 合法的群/SELF 记忆。
读取使用固定本轮时间上界和 `(change_time, id)` 前行游标，`change_time=max(updated_at, valid_from)`；
即使一页没有合法 lineage 也推进，未来生效材料到期后才进入，旧材料不按时钟回绕重投。
真正新增 evidence 同事务更新 fact 的变更时间，重复 evidence 不更新。每轮读取与 lineage 查询有界，
Host 持久化游标和已接纳来源；contact target 不提供 Person 私有记忆访问权。

## 回执、自报与恢复

`Feedback.effects` 区分 `message`、`compute`、`tool`。只有真实发送回执更新自身发言痕迹；
发送尝试、文件写入和计算完成都不等于发言。分条消息按逻辑表达 ID 去重，`actual_targets` 取真实目标。
Yuki 的工具证据严格为 event 或 initiative run 二选一；静默工具回合通过独立增量回执进入 SELF 自省，
不伪造聊天事件。错误、NO_REPLY 和运行中断也不会触发独立的罐头对外回复。

恢复使用原 State：队列、请求序号、健康和退避都持久化。在途 Jev 评分没有外部业务效果，
只能重排仍新鲜且获准的材料；在途 Host 提议则先按 proposal_id 查接纳结果，不能换 ID 重提。
已接纳 run 按原 Work、预算、执行 ID、回执和原 Presence 续跑。owner/epoch 变化只影响新接纳，
不重建旧 Work；generation 失效仍中止旧执行。迟到效果可审计，但不能复活终态或重复发消息。

`SelfDelta` 是可选自报。Yuki 已接入 `<yuki-state>` 尾段剥离，并按实际 run/request/response 防重；
在公开文字、语音和后续历史前移除控制尾段。解析失败不要求主模型补交，子 Agent 不能冒充主回合，
mood 不产生他人语义证据。

参与扩展使用 `State.host_checkpoint.participation_v1` version 1 namespace，不新增顶层 State 字段。
其它 Host 键保持原样；固定旧 b9 reader 可读取、恢复并保存此 namespace，新 reader 不将未知版本
当成参与证据，也不覆盖未知版本。namespace 随原 SnapshotStore CAS 持久化，未保存的自身意愿
允许丢失，不能因此重跑主模型或已发送效果。原事件正文及语义解释仍按 600 秒、256 条及
256 KiB 的有限 replay 窗口整理，旧数值贡献进入既有 belief baseline；这不是参与关系的
十分钟到期规则。已建立 unit 保留原 `SeenSource(revision, at)` 小身份、最新输入与当前意愿
依赖、一个已退休的真实表达锚点，以及最后一个原 message 效果回执。它不保留旧正文或
伪造新观察，也不恢复旧候选；普通 `_valid` 仍只承认当前原始事件。

仅原 unit 内在合法整理时记录到 `UnitState.retired_refs` 的引用可在正文退出后复用这些身份，
同时要求 `SeenSource` 同版本且未撤销。默认空证明不能凭其它 stop 留下的 seen 升格。
最新实际解释变更对象或 unknown 歧义时立即退役对应派生 unit，不能因解释退出窗口忘掉冲突。来源修改/撤回
继续撤销对应参与依据；Host 仍须用原账本核验来源版本。真实关闭与局部 quiet 不因整理
或重启解除。参与单元仍最多 64 个，来源指纹沿用 1024 个资源界限；满额时优先退役旧派生
unit，而非保留无限正文或封停业务。固定旧 b9 reader 可保存 namespace，但可能裁掉
不认识的退休依据；新 reader 此后视为未知，不凭 namespace 或旧 SourceRef 复活参与关系。

SnapshotStore 是单线程同步 SQLite CAS 存储，不跨线程共享连接，不在事务内等待外部工作。
其容量检查仍在写事务内聚合已有 payload 大小，不宣称事务内完全无扫描。
公网服务、多租户身份验证、远程发送和独立 Agent Runtime 均不在本库内。
