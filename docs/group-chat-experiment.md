# 群聊合成负载与验收边界

本文保留早期合成负载与当时回放事实；现行种子退役、意愿投影和有限快路见[宿主协议](protocol.md)与[当前模型](autonomous-evolution.md)。
旧报告数字不作为当前源码或线上效果预测。

## 输入来源

`fixtures/group-chat-workload-v2-full.json.gz` 是两条各 30 天的合成轨迹。
服务器上的只读 SQLite 聚合只用于校准消息量、间隔、时段、成员集中度、@、引用、
媒体和长度分布；原始群号、成员 ID、正文和单条时间戳均未进入仓库。
两条轨迹含 14,488 与 1,464 条真人形状消息，另有 8,653 条外生 Yuki 形状消息；
后者只是环境上下文，不是本次控制器发出的消息。
[校准报告](evidence/group-chat-workload-v2-calibration.json)记录生成结果。

正文、话题关系与 `act/information/floor/boundary` 标签由开发者编写或生成。
这些标签不是独立人工盲标，也不是 Jev 的真实预测。因此回放可以检查模型状态、
时序和来源边界，不能估计真实中文判断准确率或群友感受。

## 当前连续回放

运行：

```sh
python -m scripts.replay_autonomous_evolution --fixture fixtures/group-chat-workload-v2-full.json.gz --output /tmp/autonomous-replay.json
```

每条轨迹用一个控制器跨日推进；有活动来源时按 2 秒、空档按 60 秒虚拟 tick。
每个 proposal 由记录器反馈 `NO_REPLY`，不调用 Jev、主 Agent 或 QQ 网关。
这样可比较**机会形成**，不能把 proposal 当作实际发送。

| 指标 | 初版参数 | 调整后参数 |
| --- | ---: | ---: |
| 合成真人消息 | 15,952 | 15,952 |
| 总 proposal | 2,679 | 3,568 |
| 无来源 proposal | 118 | 1,007 |
| 距上一条真人消息的最远间隔 | 约 32 分钟 | 约 5 小时 52 分钟 |
| 超过 6 小时的静默窗口 | 41 | 41 |
| 其中出现 proposal 的窗口 | 0 | 0 |
| 实际 QQ 发送 | 0 | 0 |

[初版报告](evidence/autonomous-evolution-v1-replay.json)和
[调整后报告](evidence/autonomous-evolution-v2-replay.json)保留夹具与源码哈希。
长时安静来自连续衰减与思考成本，不来自一个“6 小时禁言”条件。
五次来源 proposal 落在尚未评分的同讨论结束或停止输入之后，须由 Host 的
待评分新入站屏障在接纳前阻断。

## 历史实验与尚缺证据

`group-chat-workload-v1-*`、`group-chat-workload-v2-replay.json` 和
`group-chat-idle-wake-v1.json` 记录较早的随机强度、短有效期与逐日重置实验；
其时机数字不代表当前连续模型。`scripts/replay_shadow.py` 保留旧 Host fallback
评分器的工程对照，不是本库仍在运行旧 rate/hazard 模型。

当前已有有限的合成 Jev HTTP 探针，尚无独立中文盲标、相同真实输入的隔离 Host
影子实验或获准真实群的行为验收。真实系统的接纳、主 Agent `NO_REPLY`、工具、
实际发送与群友回应须分别记录，才能判断新机会是否有用。
