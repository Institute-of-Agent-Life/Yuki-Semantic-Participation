# 自主 Work 的连续概率模型

本文件对应 `dynamics.py` 与 `Controller.advance()`。独立库只提出机会；Host 接纳、授权、
运行 Main Agent 和发送。Host 可将已观察的明确邀请与有效直接续答提升为普通入场，
并设 `advance(include_addressed=False)`，使同来源不再额外提出 SELF proposal。
库默认 True 兼容旧 caller。普通持续参与查询和 unit 自报不经过下面的自主采样；
proposal 不等于一次模型调用或公开发送。

新语义来源只来自真实 human conversation，context 可含真实 human/self；后台 Memory 种子、
recall/contact 新机会和相关 fingerprint/claim 已退役，无来源 intrinsic 保留。
Main 按需查询 Memory、自省与 Dream 不因此退出，旧已接纳运行继续按原事实恢复。

## 输入与状态

真人事件形成短时活动 `q`、来源候选使用的慢群情境 `ℓ`，以及无来源机会使用的
真人活动量 `H`。`H` 是每条真人消息的 48 小时指数核之和；它不计 Yuki 自己的消息。
SELF 的结构化 `engage` 形成 `d`，按 1800 秒回中性。无消费者的 `mood` 已从新自报退出，
旧读取只清退 mood，保留原 engage 与报告身份。真实 `NO_REPLY` 形成 `n`；确认计算形成 `f`。
普通局部 engage 不向下方 source_rate/pressure 再增加一份权重，也不把局部 quiet 当作全群开关。
已有真人焦点的 Jev 请求冻结 scope 正确的局部/SELF 意愿，作为解释上下文；意愿更新本身不购买观察。

每个确实启动主 Agent 的自主 Work 在首次确认模型请求时形成一个 `WorkPulse`。
同一 Work 的多次请求、工具调用或任意条消息都只贡献一个脉冲。
脉冲形成 1800 秒的快时标 `W_f` 和 21600 秒的慢时标 `W_s`。
该 Work 有确认的公开发送时，记录第一次发送时间；没有确认的发送、仅有 proposal、
接纳前失败和 `uncertain` 都不构成公开暴露。一个 Work 发送多条消息仍只形成一次暴露。

Host 用真实内部出站事件 ID 将消息锚定到原 run。Jev 对后续真人消息的概率判断
形成接收记录；有锚点的接续、确认或新邀请是正反馈，明确停止是负反馈。
无真实锚点或 Jev 无法确认时保持未知。`E` 表示公开表达仍未被接住的程度，
从发送后平滑上升，再按 43200 秒时标自然消退；`R` 为接收记录的 21600 秒衰减和。
回复不靠一个二值条件把 `E` 突然清零。

```text
W_f(t) = Σ_work exp(-(t-t_work)/1800)
W_s(t) = Σ_work exp(-(t-t_work)/21600)
E(t) = Σ_public_work (1-positive_reception_work)
       · (1-exp(-(t-t_public)/1200)) · exp(-(t-t_public)/43200)
R(t) = tanh(Σ_observation reception_score · exp(-(t-t_observation)/21600))
z = 0.2 + 0.8d - 0.5q - 0.3f - 0.8n - 1.8W_f - 0.25W_s - 0.55E + 0.5R
H(t) = Σ_human exp(-(t-t_i)/172800)
A(t) = H(t)/(H(t)+80)
s = t - t_last_human
G(s) = (1-exp(-s/600)) · exp(-s/64800)
λ_intrinsic = A(t) · G(s) · sigmoid(z) / 600
P(本轮机会 | Δt) = 1-exp(-λ_total Δt)
```

没有真人活动证据时，无来源机会率为零；活跃群安静下来后，`G` 先平滑上升，
之后随沉默继续衰减。`H` 同时衰减，长期无人说话的群不会因为冷却期结束而重新变活跃。
这里没有固定静默时长、累计门槛或每日配额。合法非请求来源以 Jev 支持、
楼层、注意和群情境构成各自的非负机会率，同一轮与无来源率竞争。
来源有效期、撤回、停止边界和 Host 的唯一接纳仍是事实约束，概率不授予权限。

每次可用调度区间用持久的样本序号、scope、epoch 生成可重放伪随机数。
重启后以恢复时刻重新开始采样，停机期间不积压假想机会。不可用、已有待接纳机会或
明确邀请时也不累积自主抽样时间。单次 Work 内没有消息数限制和重新采样。

## 普通快路、稀疏复核与双退出

普通快接不经过上方自主概率率，也不等待 Jev 逐条批准。复核跨度及其纯计数已删除，
Host 可把同一真人焦点排入原队列旁路复核。唯一匹配的普通续聊只保留最新待评，
保最早待评时刻，独立邀请和未匹配来源分别保留；统一复用原较稀疏 3/12/20 秒档。
没有新真人材料就没有周期调用，quiet/发送/保存状态不单独购买评分。

自身 quiet 和真实语义纠正各自收起原快路；真实新邀请仍能唤起。
未知不更新实际单元 review，stay/send、未知或故障不抹掉已有否定。
原接纳来源 continuation_refs、source_order 与 SelfReport.source 保留真实身份和顺序，
不再决定额度。已接纳 Work、预算与发送回执保持原合同。

[三版本离线对照](evidence/sparse-review-2026-10-22.json)实际运行原 pin、仅合并队列与完整新方案的
ObservationSession，用合成 observer 计真实 evaluate 启动。90 条持续输入为 20→6→5 次，
稀疏 3 条均为 3 次；换题到应用否定为 12→16→20 秒。降低采样密度确有延迟与覆盖取舍，
到期和丢弃逐项记录。它不证明真实模型准确率、Main 速度、缓存或线上成本。
旧有限跨度报告只留历史证据，不再指导当前额度配置。

## 快照与验证

旧 `g` 恢复、`p` 发言惩罚、`N_Y/N_H` 比例、净值正数门槛及长期沉寂底率
均不参与决策。新 `H` 在现有快照中缺失时由 Host 对当前代次最近五个活动衰减时标的已入账真人事件
时间按小时聚合，做一次有界、只读的近似初始化；逐条事件继续从现有短时窗口接入，
不重写历史聊天或补采样。小时桶用桶内首末事件时间的中点估计贡献，保留真实最后真人时间。
初始化后 `H` 随快照压缩为带时间的迹线；真实计算、短时活动、来源群情境及
`NO_REPLY` 保留独立职责。
工作脉冲按当前 Work/暴露最大时标的 12 倍、至少三天保留。接收明细保留至多两天且
至多 2048 项，较早贡献压成带时间戳的衰减基线，公开 Work 的正向接收程度
单独保留，避免活跃群的快照无界增长。

`tests/test_idle_autonomy.py` 验证活跃群冷场与长期沉寂的连续差异、Work 与消息数分离、锚定接收反馈、
以及快照恢复。`scripts/replay_autonomous_evolution.py` 在旧 30 天合成资料上分别模拟
每次 Work 启动后 `NO_REPLY` 与公开发出后无人回应，比较自主机会分布；模拟发送不进入
真实 SocialService，也没有真人接收观测。锚定回复和迟到回执由定向测试覆盖。
旧底率版本与下列旧 30 天回放证据保留在 `docs/evidence/`，只对应各自报告所记输入、参数与源码。
以下历史计数不作为本次种子删除和新增后台复核后的总量预测。
先前 14400 秒基率公式在两群各 30 天合成轨迹中，仅模型 `NO_REPLY` 的情形提出 1,576 次机会，
其中无来源 25 次、最后真人消息超过六小时 12 次；每个 Work 都模拟公开发送且无人回应时，
分别为 1,551、7、5 次。对应报告见[仅计算回执](evidence/autonomous-activity-window-no-reply.json)
与[无人回应的公开发送](evidence/autonomous-activity-window-unanswered.json)。高活跃轨迹有大量
真人来源机会，公开发送后无回应也会压低后续无来源率。这些数字只衡量合成机会，
不代表真实 QQ 发言数。
该次调参采用 600 秒基率、0.2 基础倾向、600 秒冷场上升时标与 80 活动半饱和值，当前默认继续保留这些系数。
600 秒不是固定唤醒周期；快慢 Work 密度、未回应暴露及接收反馈的权重保持。
配对回放可以通过 `scripts/replay_autonomous_evolution.py --parameters profile.json` 指定
完整参数；报告保存实际生效参数。相同资料、脚本和样本序号的配对结果如下：

| 历史模拟结果 | 调参前总机会 / 无来源 / 超六小时 | 该次调参后总机会 / 无来源 / 超六小时 |
| --- | --- | --- |
| 仅计算后 `NO_REPLY` | 1576 / 25 / 12 | 1857 / 304 / 124 |
| 每个 Work 公开发送且无人回应 | 1551 / 7 / 5 | 1627 / 84 / 47 |

两群各 30 天共 15952 条合成真人消息，6267 次合成观察，41 个超过六小时的空白。
无来源机会增加约 12 倍，模拟公开无人回应仍将 304 次机会压到 84 次；总模拟
Work 分别增加约 18% 和 5%。最大最后真人消息间隔分别约 43.3 和 41.4 小时，
这来自曾有活动的群冷场长尾，并不赋予长期沉寂群固定突破率。
该次报告见[仅计算](evidence/autonomous-observer-recovery-aggressive-no-reply.json)和
[无人回应](evidence/autonomous-observer-recovery-aggressive-unanswered.json)，配对基准见
[先前仅计算](evidence/autonomous-observer-recovery-previous-no-reply.json)和
[先前无人回应](evidence/autonomous-observer-recovery-previous-unanswered.json)。
新回放证据与真实 QQ 行为分别验收。
真实群聊效果须与 Host 回执及 Jev 观察另行验证。

## 参数与安全升级

`AutonomyParameters` 收拢非请求自主机会的基率、压力权重、来源率、迹线增量与时标。独立库的来源基率默认分母为 45 秒；
Host 可能有自己的 30 秒默认或真实挂载覆盖，以实际加载参数为准，不把分母当固定等待周期。
Host 可在每轮采样前换入经过校验的整份参数；配置不写入快照，不重写已接纳的 Work、
回执或抽样序号。改变时标会从当前保留的迹线和脉冲继续演化，已经按旧时标裁剪的
远古证据不会重新出现。确定性协议边界、权限和语义观测 Rubric 不属于此配置。

退役参数 independent_source_factor、recall_source_decay_seconds、contact_source_decay_seconds
和 continuation_review_span 只在旧配置 reader 清退，其余未知字段仍严格校验。Host 升级时保留其余实际覆盖，
不拿新示例整份替换线上调参。冷恢复保留原预算、采样与真实效果，不补停机期间的抽样。

旧已接纳 recall/contact 的原 proposal/run、Work/compute、公开暴露和接收反馈继续恢复，
不因删除种子生成器而把真实 W/E/R 清空。旧 seed 正文和未接纳派生队列不再进入新请求。
已经产生新输入、工作或投递后，回退使用兼容新格式的修复版本，不能用旧数据库覆盖新事实。
