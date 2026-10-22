"""Build and replay synthetic group-chat workloads from private metadata aggregates.

Build reads only the output of profile_group_metadata.py. Replay calls the real
ObservationSession/Controller with fixture answers and a recording-only executor
that models a charged Work with NO_REPLY or an unanswered send. Neither command
calls Jev, the Main Agent, a database, or QQ.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
from pathlib import Path

from yuki_participation.autonomy_parameters import DEFAULT_AUTONOMY_PARAMETERS, AutonomyParameters
from yuki_participation.controller import Controller
from yuki_participation.models import (
    Choice,
    Effect,
    Feedback,
    Observation,
    Scope,
    ScopedEvent,
    SourceRef,
)
from yuki_participation.rubric import CRITERIA, REVISION
from yuki_participation.session import ObservationSession

ROOT = Path(__file__).resolve().parents[1]
ACTS = (
    "other_exchange",
    "open_group",
    "acknowledge",
    "invite_yuki",
    "extend_yuki",
    "close_topic",
    "ask_yuki_stop",
    "unknown",
)
TEXTS = {
    "other_exchange": (
        "你们刚才说的周末安排怎么样？",
        "这个我跟小林说一下。",
        "我先去吃饭，等会回来。",
    ),
    "open_group": ("有人知道这个问题怎么处理吗？", "大家看看这个思路行不行。", "我把进度贴一下。"),
    "acknowledge": ("收到，谢谢。", "嗯嗯", "好，我看到了。"),
    "invite_yuki": (
        "Yuki，这一步应该怎么做？",
        "有没有人能帮我看下，Yuki也可以。",
        "刚才的问题还没解决，你有什么建议？",
    ),
    "extend_yuki": ("那第二步呢？", "你刚才说的那个具体怎么操作？", "我试了，但结果不太一样。"),
    "close_topic": ("已经解决了，谢谢大家。", "这件事先到这里吧。"),
    "ask_yuki_stop": ("Yuki先别接这个话题了。",),
    "unknown": ("[一张没有文字说明的图片]", "这个？"),
}


def provenance(fixture: dict[str, object]) -> dict[str, object]:
    content = json.dumps(fixture, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sources = (
        "src/yuki_participation/controller.py",
        "src/yuki_participation/dynamics.py",
        "src/yuki_participation/scheduling.py",
        "src/yuki_participation/session.py",
        "scripts/group_chat_experiment.py",
    )
    return {
        "fixture_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "rubric_revision": REVISION,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
    }


def event(
    key: str,
    at: float,
    author: str,
    thread: str,
    text: str,
    act: str,
    *,
    mention: bool = False,
    reply_to: str | None = None,
    direct: bool = False,
    kind: str = "human",
    target: str | None = None,
) -> dict[str, object]:
    return {
        "id": key,
        "at": round(at, 2),
        "author": author,
        "target": target or author,
        "thread": thread,
        "kind": kind,
        "text": text,
        "act": act,
        "information": "refine"
        if act == "extend_yuki"
        else "new"
        if act in {"invite_yuki", "open_group"}
        else "social_ack",
        "floor": "yuki"
        if act in {"invite_yuki", "extend_yuki"}
        else "open"
        if act == "open_group"
        else "other",
        "boundary": "target_thread" if act in {"close_topic", "ask_yuki_stop"} else "unknown",
        "mentions_bot": mention,
        "reply_to": reply_to,
        "direct": direct,
    }


def focused_scenes() -> list[dict[str, object]]:
    return [
        {
            "id": "no_at_continuation",
            "purpose": "direct @ entry, then unmentioned replies and a later fresh invitation",
            "events": [
                event(
                    "a1",
                    1000,
                    "A",
                    "topic-a",
                    "@Yuki 帮我看看这个设置",
                    "invite_yuki",
                    mention=True,
                    direct=True,
                ),
                event(
                    "s1",
                    1004,
                    "SELF",
                    "topic-a",
                    "先检查设置项。",
                    "unknown",
                    kind="self",
                    target="A",
                    reply_to="a1",
                ),
                event("a2", 1020, "A", "topic-a", "那第二步呢？", "extend_yuki", reply_to="s1"),
                event("b1", 1032, "B", "topic-b", "今天晚上几点开始？", "other_exchange"),
                event("a3", 1048, "A", "topic-a", "我按你说的改了，但还是报错", "extend_yuki"),
                event(
                    "a4",
                    1170,
                    "A",
                    "topic-a",
                    "刚才的问题还没解决，你再帮我看一下？",
                    "invite_yuki",
                ),
            ],
        },
        {
            "id": "closure_and_reopen",
            "purpose": "thanks alone, explicit closure, unrelated chatter, scoped reopening",
            "events": [
                event("c1", 2000, "A", "topic-c", "Yuki能帮我理一下步骤吗？", "invite_yuki"),
                event("c2", 2012, "A", "topic-c", "谢谢", "acknowledge"),
                event("c3", 2030, "A", "topic-c", "已经解决了，谢谢", "close_topic"),
                event("d1", 2040, "B", "topic-d", "我想问个别的问题", "open_group"),
                event("c4", 2050, "A", "topic-c", "我先去忙了", "other_exchange"),
                event("c5", 2200, "A", "topic-c", "新情况出现了，Yuki再帮我看看", "invite_yuki"),
            ],
        },
        {
            "id": "burst_and_ambiguous_floor",
            "purpose": "many short fragments, parallel speakers and a message for someone else",
            "events": [
                event("e1", 3000, "A", "topic-e", "有人知道日志在哪？", "open_group"),
                event("e2", 3002, "B", "topic-f", "阿明，文件发你了", "other_exchange"),
                event("e3", 3004, "A", "topic-e", "Yuki你能看下吗", "invite_yuki"),
                event("e4", 3005, "C", "topic-f", "收到", "acknowledge"),
                event("e5", 3007, "B", "topic-f", "我给阿明解释一下", "other_exchange"),
                event("e6", 3010, "A", "topic-e", "我找到文件了，下一步呢", "extend_yuki"),
                event("e7", 3014, "C", "topic-f", "谢谢", "acknowledge"),
                event("e8", 3022, "A", "topic-e", "好了，问题解决了", "close_topic"),
            ],
        },
        {
            "id": "stop_and_silence",
            "purpose": "explicit stop must persist through silence and later unrelated activity",
            "events": [
                event("f1", 4000, "A", "topic-g", "Yuki帮我看看", "invite_yuki"),
                event("f2", 4020, "A", "topic-g", "Yuki先别接这个话题了", "ask_yuki_stop"),
                event("f3", 4200, "B", "topic-h", "今天的安排有变化吗", "open_group"),
                event("f4", 4500, "A", "topic-g", "我回来了", "other_exchange"),
            ],
        },
    ]


def inverse_quantile(profile: dict[str, object], u: float) -> float:
    q = profile["interarrival_seconds"]
    assert isinstance(q, dict)
    points = [
        (0.0, 0.5),
        (0.1, float(q["p10"])),
        (0.25, float(q["p25"])),
        (0.5, float(q["p50"])),
        (0.75, float(q["p75"])),
        (0.9, float(q["p90"])),
        (0.99, float(q["p99"])),
        (1.0, float(q["p99"]) * 2),
    ]
    for (left_p, left_v), (right_p, right_v) in zip(points, points[1:], strict=False):
        if u <= right_p:
            fraction = (u - left_p) / (right_p - left_p)
            return round(
                math.exp(
                    math.log(max(left_v, 0.1)) * (1 - fraction)
                    + math.log(max(right_v, 0.1)) * fraction
                ),
                2,
            )
    raise AssertionError("quantile interval not found")


def choose_act(rng: random.Random, mention: bool) -> str:
    # Deliberate sensitivity assumption, not measured semantic prevalence.
    weights = (
        (0.12, 0.04, 0.06, 0.51, 0.19, 0.02, 0.01, 0.05)
        if mention
        else (0.43, 0.2, 0.13, 0.08, 0.08, 0.03, 0.01, 0.04)
    )
    return rng.choices(ACTS, weights=weights)[0]


def quota_flags(count: int, share: float, rng: random.Random) -> list[bool]:
    selected = min(count, max(0, round(count * share)))
    flags = [True] * selected + [False] * (count - selected)
    rng.shuffle(flags)
    return flags


def weighted_people(count: int, weights: list[float], rng: random.Random) -> list[str]:
    expected = [count * weight for weight in weights]
    quotas = [math.floor(value) for value in expected]
    for index in sorted(
        range(len(weights)), key=lambda item: expected[item] - quotas[item], reverse=True
    )[: count - sum(quotas)]:
        quotas[index] += 1
    people = [f"P{index + 1}" for index, quota in enumerate(quotas) for _ in range(quota)]
    rng.shuffle(people)
    return people


def ambient_scene(
    profile: dict[str, object], rank: int, rng: random.Random, count: int
) -> dict[str, object]:
    authors = int(profile["active_human_authors"])
    top1 = float(profile["top_1_author_share"])
    top3 = float(profile["top_3_author_share"])
    weights = [top1, (top3 - top1) / 2, (top3 - top1) / 2]
    weights.extend([(1 - top3) / (authors - 3)] * (authors - 3))
    people = weighted_people(count, weights, rng)
    result: list[dict[str, object]] = []
    at = 1000.0
    previous_by_thread: dict[str, str] = {}
    mentions = quota_flags(count, float(profile["bot_at_segment_share"]), rng)
    media_flags = quota_flags(count, float(profile["media_segment_share"]), rng)
    quote_flags = quota_flags(count, float(profile["platform_reply_share"]), rng)
    quantile_levels = [(index + 0.5) / (count - 1) for index in range(count - 1)]
    rng.shuffle(quantile_levels)
    for index in range(count):
        if index:
            at += inverse_quantile(profile, quantile_levels[index - 1])
        person = people[index]
        thread = rng.choices(("topic-1", "topic-2", "topic-3"), weights=(0.6, 0.25, 0.15))[0]
        mention = mentions[index]
        act = choose_act(rng, mention)
        is_media = media_flags[index]
        text = rng.choice(TEXTS[act])
        if is_media:
            text = (
                "[图片或语音，合成占位；无原始媒体内容]"
                if act == "unknown"
                else text + " [媒体占位]"
            )
        reply = previous_by_thread.get(thread) if quote_flags[index] else None
        key = f"g{rank}-{index + 1}"
        result.append(event(key, at, person, thread, text, act, mention=mention, reply_to=reply))
        previous_by_thread[thread] = key
    return {
        "id": f"server_shape_rank_{rank}",
        "purpose": (
            "synthetic traffic timing and actor concentration shaped by private server aggregates"
        ),
        "events": result,
    }


def build(profile_path: Path, *, seed: int) -> dict[str, object]:
    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    if (
        profile.get("kind") != "private_metadata_aggregate_only"
        or len(profile.get("groups", [])) < 2
    ):
        raise ValueError("expected at least two eligible private aggregate groups")
    rng = random.Random(seed)
    scenes = [
        ambient_scene(profile["groups"][0], 1, rng, 120),
        ambient_scene(profile["groups"][1], 2, rng, 60),
        *focused_scenes(),
    ]
    return {
        "schema_version": 1,
        "kind": "fully_synthetic_group_chat_workload",
        "seed": seed,
        "source_note": (
            "Timing, actor concentration, @ and media rates fitted to ignored "
            "on-host aggregate; no real IDs, messages or timestamps included."
        ),
        "semantic_note": (
            "All Chinese text and intent labels are developer-authored; intent "
            "prevalence is an explicit assumption, not estimated from server data."
        ),
        "scenes": scenes,
    }


def choice(dimension: str, selected: str) -> Choice:
    return Choice(
        choice=selected,
        probabilities={key: float(key == selected) for key in CRITERIA[dimension]},
    )


class FixtureObserver:
    def __init__(self, labels: dict[str, dict[str, object]]) -> None:
        self.labels = labels
        self.calls: list[dict[str, object]] = []

    async def evaluate(self, snapshot: object) -> Observation:
        focus = snapshot.focus  # type: ignore[attr-defined]
        label = self.labels[focus.ref.event_id]
        self.calls.append({"focus": focus.ref.event_id, "at": snapshot.issued_at})  # type: ignore[attr-defined]
        return Observation(
            observation_id=f"fixture:{focus.ref.event_id}:{snapshot.sequence}",  # type: ignore[attr-defined]
            snapshot=snapshot,
            provider="developer_fixture_labels",
            model_revision="no-model",
            rubric_revision=REVISION,
            received_at=snapshot.issued_at,  # type: ignore[attr-defined]
            answers={
                "interaction_mark": choice("interaction_mark", str(label["act"])),
                "information_state": choice("information_state", str(label["information"])),
                "floor_state": choice("floor_state", str(label["floor"])),
                "boundary_scope": choice("boundary_scope", str(label["boundary"])),
            },
        )


async def replay_scene(
    scene: dict[str, object],
    seed: int,
    *,
    intrinsic_allowed: bool = False,
    horizon: float | None = None,
    simulated_unanswered_send: bool = False,
    parameters: AutonomyParameters = DEFAULT_AUTONOMY_PARAMETERS,
) -> dict[str, object]:
    records = scene["events"]
    assert isinstance(records, list) and records
    labels = {str(row["id"]): row for row in records}
    scope = Scope(conversation_id=f"synthetic:{scene['id']}", generation=1)
    start = float(records[0]["at"])
    controller = Controller(scope, start - 1, parameters=parameters)
    observer = FixtureObserver(labels)
    session = ObservationSession(controller, observer)
    proposals: list[dict[str, object]] = []
    now = start - 1

    async def tick(at: float) -> None:
        await session.evaluate_due(at)
        proposal = controller.advance(
            at,
            controller_epoch=0,
            host_available=True,
            intrinsic_allowed=intrinsic_allowed,
        )
        if proposal is None:
            return
        sources = [source.event_id for source in proposal.sources]
        proposals.append(
            {
                "at": round(at, 2),
                "kind": proposal.kind.value,
                "thread": proposal.thread,
                "target": proposal.target_hint,
                "sources": sources,
                "no_at_sources": [
                    source for source in sources if not labels[source]["mentions_bot"]
                ],
                "gold_acts": [labels[source]["act"] for source in sources],
                "age_seconds": (
                    round(at - max(float(labels[source]["at"]) for source in sources), 2)
                    if sources
                    else None
                ),
            }
        )
        # Recording-only executor: one model request per Work. Optional send
        # is synthetic, unacknowledged, and never reaches a gateway.
        effects = [
            Effect(
                effect_id=f"fixture-model:{len(proposals)}",
                kind="compute",
                at=at,
            )
        ]
        if simulated_unanswered_send:
            effects.append(
                Effect(
                    effect_id=f"fixture-send:{len(proposals)}",
                    kind="message",
                    at=at,
                    actual_targets=("group",),
                )
            )
        accepted = controller.observe_run_feedback(
            Feedback(
                run_ref=f"fixture-run:{len(proposals)}",
                proposal_id=proposal.proposal_id,
                sequence=1,
                outcome="completed" if simulated_unanswered_send else "no_reply",
                at=at,
                considered_refs=proposal.sources,
                effects=tuple(effects),
            )
        )
        if not accepted:
            raise AssertionError("fixture feedback was rejected")

    async def until(target: float) -> None:
        nonlocal now
        while now + 2 < target:
            probe = now + 2
            live = bool(session.queue.pending) or any(
                candidate.support.valid_until > probe
                and controller.state.consumed.get(candidate.event.ref.event_id, 0)
                < candidate.event.ref.revision
                for candidate in controller.state.candidates.values()
            )
            if not live:
                if not intrinsic_allowed:
                    break
                probe = min(now + 60, target - 0.001)
            now = probe
            await tick(now)
        now = target

    for row in records:
        at = float(row["at"])
        await until(at)
        ref = SourceRef(event_id=str(row["id"]), revision=1)
        reply = row.get("reply_to")
        item = ScopedEvent(
            scope=scope,
            ref=ref,
            thread=str(row["thread"]),
            author=str(row["author"]),
            target=str(row["target"]),
            text=str(row["text"]),
            at=at,
            kind=str(row["kind"]),
            reply_to=SourceRef(event_id=str(reply), revision=1) if reply else None,
        )
        if row["kind"] == "self":
            controller.observe_committed_event(item)
            controller.observe_committed_effect(
                f"fixture-direct:{ref.event_id}",
                Effect(
                    effect_id=f"effect:{ref.event_id}",
                    kind="message",
                    at=at,
                    actual_targets=(item.target,),
                ),
            )
        elif row["direct"]:
            controller.observe_committed_event(item)
            controller.state.consumed[ref.event_id] = 1
        else:
            session.controller.observe_committed_event(item)
            session.request_observation(item.ref)
        await tick(at)
    await until(horizon if horizon is not None else float(records[-1]["at"]) + 100)
    await tick(now)
    human = [row for row in records if row["kind"] == "human"]
    observed_at = {str(call["focus"]): float(call["at"]) for call in observer.calls}
    for proposal in proposals:
        if not proposal["sources"]:
            proposal["unobserved_boundary_before_proposal"] = []
            continue
        latest_source_at = max(float(labels[source]["at"]) for source in proposal["sources"])
        proposal["unobserved_boundary_before_proposal"] = [
            row["id"]
            for row in human
            if row["act"] in {"close_topic", "ask_yuki_stop"}
            and row["thread"] == proposal["thread"]
            and row["target"] == proposal["target"]
            and latest_source_at < float(row["at"]) <= float(proposal["at"])
            and observed_at.get(str(row["id"]), math.inf) > float(proposal["at"])
        ]
    gaps = [float(b["at"]) - float(a["at"]) for a, b in zip(human, human[1:], strict=False)]
    return {
        "scene": scene["id"],
        "human_messages": len(human),
        "human_without_at": sum(not row["mentions_bot"] for row in human),
        "interarrival_le_60_fraction": round(sum(gap <= 60 for gap in gaps) / len(gaps), 4)
        if gaps
        else None,
        "observer_calls": len(observer.calls),
        "observer_trace": observer.calls
        if not str(scene["id"]).startswith("server_shape_")
        else [],
        "queue_expired": session.queue.expired,
        "queue_dropped": session.queue.dropped,
        "proposals": proposals,
        "proposal_count": len(proposals),
        "proposals_with_no_at_source": sum(bool(row["no_at_sources"]) for row in proposals),
        "proposals_after_unobserved_boundary": sum(
            bool(row["unobserved_boundary_before_proposal"]) for row in proposals
        ),
        "recorded_outcome": "no_reply_for_every_proposal",
        "sent_messages": 0,
    }


async def replay(fixture: dict[str, object]) -> dict[str, object]:
    scenes = fixture["scenes"]
    assert isinstance(scenes, list)
    rows = [
        await replay_scene(scene, int(fixture["seed"]) + index)
        for index, scene in enumerate(scenes)
    ]
    return {
        "kind": "offline_fixture_controller_replay",
        "provenance": provenance(fixture),
        "fixture_kind": fixture["kind"],
        "provider_requests": 0,
        "main_agent_requests": 0,
        "qq_messages_sent": 0,
        "total_human_messages": sum(int(row["human_messages"]) for row in rows),
        "total_observer_calls": sum(int(row["observer_calls"]) for row in rows),
        "total_proposals": sum(int(row["proposal_count"]) for row in rows),
        "total_proposals_with_no_at_source": sum(
            int(row["proposals_with_no_at_source"]) for row in rows
        ),
        "scenes": rows,
        "limitations": [
            "Fixture one-hot semantic labels are not Jev predictions or "
            "independent human annotation.",
            "Idealized thread/target IDs bypass Yuki's live canonical routing "
            "and ambiguity resolution.",
            "Every proposal receives a synthetic NO_REPLY; no Main Agent, "
            "tools, permissions, or QQ delivery are exercised.",
            "Virtual tick and opportunity timings are not HTTP or live response latency.",
        ],
    }


async def replay_many(fixture: dict[str, object], runs: int) -> dict[str, object]:
    if runs < 2:
        raise ValueError("multiple replay requires at least two runs")
    base_seed = int(fixture["seed"])
    trials = []
    for index in range(runs):
        sampled = {**fixture, "seed": base_seed + index * 1009}
        result = await replay(sampled)
        trials.append(
            {
                "seed": sampled["seed"],
                "proposals": result["total_proposals"],
                "no_at_proposals": result["total_proposals_with_no_at_source"],
                "pending_boundary_proposals": sum(
                    int(scene["proposals_after_unobserved_boundary"]) for scene in result["scenes"]
                ),
                "by_scene": {
                    str(scene["scene"]): int(scene["proposal_count"]) for scene in result["scenes"]
                },
            }
        )

    def summary(field: str) -> dict[str, float | int]:
        values = sorted(int(trial[field]) for trial in trials)
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        return {
            "min": values[0],
            "median": median,
            "max": values[-1],
            "mean": round(sum(values) / len(values), 3),
        }

    return {
        "kind": "offline_fixture_controller_multiseed",
        "provenance": provenance(fixture),
        "fixture_kind": fixture["kind"],
        "runs": runs,
        "provider_requests": 0,
        "main_agent_requests": 0,
        "qq_messages_sent": 0,
        "summary": {
            field: summary(field)
            for field in ("proposals", "no_at_proposals", "pending_boundary_proposals")
        },
        "trials": trials,
        "limitations": [
            "Only controller random seeds vary; workload and developer-authored labels are fixed.",
            "This is not Jev, Main Agent, host admission, or QQ behavioral validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument("--profile", type=Path, required=True)
    build_command.add_argument("--output", type=Path, required=True)
    build_command.add_argument("--seed", type=int, default=20260923)
    replay_command = commands.add_parser("replay")
    replay_command.add_argument("--fixture", type=Path, required=True)
    replay_command.add_argument("--output", type=Path, required=True)
    replay_command.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.profile, seed=args.seed)
    else:
        fixture = json.loads(args.fixture.read_text(encoding="utf-8-sig"))
        result = asyncio.run(replay(fixture) if args.runs == 1 else replay_many(fixture, args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "scenes": len(result["scenes"]) if "scenes" in result else None,
                "human_messages": result.get("total_human_messages"),
                "proposals": result.get("total_proposals"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
