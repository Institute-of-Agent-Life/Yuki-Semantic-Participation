"""Offline queue/controller replay. No provider, Host database, Main or QQ execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yuki_participation.autonomy_parameters import AutonomyParameters
from yuki_participation.controller import Controller
from yuki_participation.models import (
    Choice,
    HostUnitOption,
    Observation,
    Scope,
    ScopedEvent,
    SourceRef,
)
from yuki_participation.participation import ParticipationUnit, UnitBinding
from yuki_participation.rubric import CRITERIA, REVISION
from yuki_participation.scheduling import ObservationQueue
from yuki_participation.self_report import SelfDelta, SelfReport


def choice(dimension: str, selected: str) -> Choice:
    return Choice(
        choice=selected, probabilities={key: float(key == selected) for key in CRITERIA[dimension]}
    )


def replay(
    *, span: int, gap: int, latency: float, mode: str, active: bool = True, duration: int = 90
) -> dict[str, object]:
    scope = Scope(conversation_id="offline-review-span", generation=1)
    c = Controller(scope, 90, AutonomyParameters(continuation_review_span=span))
    first = ScopedEvent(
        scope=scope,
        ref=SourceRef(event_id="origin", revision=1),
        thread="topic",
        author="A",
        target="A",
        text="Synthetic established interaction",
        at=90,
        source_order=0,
    )
    c.observe_committed_event(first)
    original = UnitBinding(
        scope=scope,
        unit=ParticipationUnit(thread="topic", target="A"),
        actor="A",
        basis=(first.ref,),
    )
    c.observe_unit_input(original, first.ref)
    c.observe_unit_hint(
        original,
        SelfReport(
            run_ref="initial",
            sequence=1,
            response_id="initial",
            at=90,
            source=first.ref,
            delta=SelfDelta(engage="stay"),
        ),
    )
    queue = ObservationQueue(scope)
    fast = deferred = calls = ready = post_change_fast = 0
    first_deferred = None
    inflight = None
    completed = 0
    waits = []
    # One real tick every two virtual seconds, like Host _loop without its wall work.
    for offset in range(duration + 22):
        now = 100 + offset
        if offset < duration and offset % gap == 0:
            identity = offset // gap + 1
            focus = ScopedEvent(
                scope=scope,
                ref=SourceRef(event_id=f"opaque-{identity}", revision=1),
                thread=f"new-{identity}",
                author="A",
                target="A",
                text="Synthetic human input",
                at=now,
                source_order=identity,
                unit_ambiguous=True,
                unit_options=(
                    HostUnitOption(key="old", thread="topic", target="A"),
                    HostUnitOption(key="new", thread=f"new-{identity}", target="A"),
                ),
            )
            c.observe_committed_event(focus)
            matched = c.participation_view(focus, now).matched_unit
            if matched is not None:
                actual = original.model_copy(update={"basis": (*original.basis, focus.ref)})
                c.observe_unit_input(actual, focus.ref, continuation=True)
                fast += 1
                post_change_fast += int(mode == "topic_change" and offset >= 30)
            else:
                deferred += 1
                if first_deferred is None:
                    first_deferred = offset
            # All new real messages can be observed independently of fast admission.
            queue.offer(focus)
        if offset % 2:
            continue
        if inflight is not None and now >= inflight[0]:
            _, snapshot = inflight
            source = snapshot.focus
            negative = mode == "topic_change" and source.at >= 130
            unknown = mode == "unknown"
            act = "unknown" if unknown else "other_exchange" if negative else "extend_yuki"
            floor = "unknown" if unknown else "other" if negative else "yuki"
            selected = "unknown" if unknown else "old"
            answers = {
                "interaction_mark": choice("interaction_mark", act),
                "information_state": choice("information_state", "new"),
                "floor_state": choice("floor_state", floor),
                "boundary_scope": choice("boundary_scope", "unknown"),
                "unit_selection": Choice(
                    choice=selected,
                    probabilities={
                        "old": float(selected == "old"),
                        "new": 0.0,
                        "unknown": float(selected == "unknown"),
                    },
                ),
            }
            c.apply_semantic_observation(
                Observation(
                    observation_id=f"offline-{snapshot.sequence}",
                    snapshot=snapshot,
                    provider="offline-one-hot",
                    model_revision="synthetic",
                    rubric_revision=REVISION,
                    received_at=now,
                    answers=answers,
                    resolved_unit=source.unit_options[0] if not unknown else None,
                )
            )
            completed += 1
            ready += int(not negative and not unknown)
            waits.append(now - source.at)
            queue.finish(snapshot.sequence)
            inflight = None
        raw = queue.take(now, active=active, context=tuple(c.state.events.values()))
        if raw is not None:
            snapshot = c.freeze_observation(raw)
            queue.in_flight = snapshot
            inflight = (now + latency, snapshot)
            calls += 1
    return {
        "fixed_active_queue": active,
        "queue_seconds": [2, 8, 15] if active else [3, 12, 20],
        "span": span,
        "message_gap_seconds": gap,
        "observer_elapsed_seconds": latency,
        "mode": mode,
        "input_count": fast + deferred,
        "fast_admitted": fast,
        "needs_semantic_path": deferred,
        "first_semantic_path_at_seconds": first_deferred,
        "post_topic_change_fast_admitted": post_change_fast,
        "simulated_observations": calls,
        "completed_observations": completed,
        "positive_foci": ready,
        "queue_expired": queue.expired,
        "queue_dropped": queue.dropped,
        "observed_focus_delay_seconds": {"min": min(waits), "max": max(waits)},
        "fast_path_does_not_wait_for_observer": True,
        "proposals_created": len(c.state.proposals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", type=float, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    latencies = args.latency or [6.002977]
    rows = [
        replay(span=span, gap=gap, latency=latency, mode=mode, active=active)
        for active in (True, False)
        for latency in latencies
        for mode in ("positive", "unknown", "topic_change")
        for gap in (1, 3, 5, 10)
        for span in (2, 3, 4, 6)
    ]
    output = {
        "evidence": "synthetic offline controller/queue timing; not model accuracy or Main latency",
        "tick_seconds": 2,
        "queue_profile_boundary": "fixed active/inactive comparisons, not live Host mode switching",
        "input_duration_seconds": 90,
        "drain_seconds": 22,
        "observation_horizon_seconds": 112,
        "main_execution_and_semantic_promotion": "not simulated",
        "timeout_boundary": (
            "real ObservationSession timeout/backoff plus spent-span regression in "
            "tests/test_seed_retirement_fastpath.py; unknown is not timeout"
        ),
        "rows": rows,
    }
    encoded = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
