"""Replay actual archived/current sessions with anonymous sources and virtual time.

No provider, Main, production database or platform send is invoked. Each request
count is a real ObservationSession -> synthetic observer.evaluate invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ORIGINAL = "438f204f9ebe2b8c1b698b68c0f1b3ac8b4facef"
VARIANTS = ("original", "coalesced_queue_only", "sparse")
SCENARIOS = {
    "burst": (12, 1),
    "continuous": (90, 1),
    "sparse": (90, 30),
    "independent": (30, 3),
    "quiet": (12, 1),
    "failure": (90, 3),
    "topic_change": (90, 3),
}


async def replay(variant: str, *, scenario: str, latency: float) -> dict[str, object]:
    import httpx

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
    from yuki_participation.self_report import SelfDelta, SelfReport
    from yuki_participation.session import ObservationSession

    def choice(dimension: str, selected: str) -> Choice:
        return Choice(
            choice=selected,
            probabilities={key: float(key == selected) for key in CRITERIA[dimension]},
        )

    scope = Scope(conversation_id="synthetic-sparse-review", generation=1)
    controller = Controller(scope, 90)
    origin = ScopedEvent(
        scope=scope,
        ref=SourceRef(event_id="origin", revision=1),
        thread="topic",
        author="person:A",
        target="person:A",
        text="Synthetic established interaction",
        at=90,
        source_order=0,
    )
    controller.observe_committed_event(origin)
    binding = UnitBinding(
        scope=scope,
        unit=ParticipationUnit(thread="topic", target="person:A"),
        actor="person:A",
        basis=(origin.ref,),
    )
    controller.observe_unit_input(binding, origin.ref)
    controller.observe_unit_hint(
        binding,
        SelfReport(
            run_ref="original-main",
            sequence=1,
            response_id="stay",
            at=90,
            source=origin.ref,
            delta=SelfDelta(engage="stay"),
        ),
    )
    now = 100
    release = None
    task = None
    deadline = None
    calls = []
    waits = []
    failures = 0
    brake_applied_at = None

    class Observer:
        async def evaluate(self, snapshot):
            nonlocal release, deadline, failures, brake_applied_at
            calls.append({"at": now, "focus_at": snapshot.focus.at})
            release = asyncio.Event()
            deadline = now + latency
            await release.wait()
            if scenario == "failure" and len(calls) == 1:
                failures += 1
                raise httpx.ReadTimeout("Synthetic offline transport failure")
            negative = scenario == "topic_change" and snapshot.focus.at >= 130
            if negative and brake_applied_at is None:
                brake_applied_at = now
            waits.append(now - snapshot.focus.at)
            return Observation(
                observation_id=f"synthetic:{snapshot.sequence}",
                snapshot=snapshot,
                provider="offline-one-hot",
                model_revision="synthetic",
                rubric_revision=REVISION,
                received_at=now,
                answers={
                    "interaction_mark": choice(
                        "interaction_mark", "other_exchange" if negative else "extend_yuki"
                    ),
                    "information_state": choice("information_state", "new"),
                    "floor_state": choice("floor_state", "other" if negative else "yuki"),
                    "boundary_scope": choice("boundary_scope", "unknown"),
                    "unit_selection": Choice(
                        choice="old", probabilities={"old": 1, "new": 0, "unknown": 0}
                    ),
                },
                resolved_unit=snapshot.focus.unit_options[0],
            )

    session = ObservationSession(controller, Observer())
    duration, gap = SCENARIOS[scenario]
    fast_admitted = enqueued = replaced = max_pending = 0
    current = binding
    latest = origin
    for now in range(100, 100 + duration + 90):
        offset = now - 100
        if offset < duration and offset % gap == 0:
            actor = f"person:{offset}" if scenario == "independent" else "person:A"
            focus = ScopedEvent(
                scope=scope,
                ref=SourceRef(event_id=f"source:{offset}", revision=1),
                thread=f"new:{offset}",
                author=actor,
                target=actor,
                text="Synthetic ordinary input",
                at=now,
                source_order=offset + 1,
                unit_ambiguous=True,
                observation_priority=scenario == "independent",
                unit_options=(
                    HostUnitOption(key="old", thread="topic", target=actor),
                    HostUnitOption(key="new", thread=f"new:{offset}", target=actor),
                ),
            )
            controller.observe_committed_event(focus)
            matched = controller.participation_view(focus, now).matched_unit
            if matched is not None:
                current = binding.model_copy(update={"basis": (*binding.basis, focus.ref)})
                controller.observe_unit_input(current, focus.ref, continuation=True)
                fast_admitted += 1
                latest = focus
            before = set(session.queue.pending)
            dropped = session.queue.dropped
            enqueued += int(session.request_observation(focus.ref))
            replaced += max(
                0, len(before - session.queue.pending.keys()) - (session.queue.dropped - dropped)
            )
            max_pending = max(max_pending, len(session.queue.pending))
        if scenario == "quiet" and offset == duration:
            controller.observe_unit_hint(
                current,
                SelfReport(
                    run_ref="synthetic-main",
                    sequence=1,
                    response_id="quiet",
                    at=latest.at,
                    source=latest.ref,
                    delta=SelfDelta(engage="quiet"),
                ),
            )
        if offset % 2:
            continue
        if task is not None and deadline is not None and now >= deadline:
            release.set()
            await task
            task = release = deadline = None
        kwargs = {"active": True} if variant != "sparse" else {}
        if task is None:
            task = asyncio.create_task(session.evaluate_due(now, **kwargs))
            await asyncio.sleep(0)
            if task.done():
                await task
                task = None
        else:
            # Another Host tick may clean covered pending, but cannot replace HTTP.
            await session.evaluate_due(now, **kwargs)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return {
        "variant": variant,
        "scenario": scenario,
        "message_gap_seconds": gap,
        "source_count": len(range(0, duration, gap)),
        "observation_requests_enqueued": enqueued,
        "observer_evaluate_starts": len(calls),
        "fast_admitted": fast_admitted,
        "replaced_pending": replaced,
        "max_pending": max_pending,
        "expired": session.queue.expired,
        "dropped": session.queue.dropped,
        "remaining_pending": len(session.queue.pending),
        "transport_failures": failures,
        "request_timeline": calls,
        "focus_to_result_seconds": {"min": min(waits), "max": max(waits)} if waits else None,
        "topic_change_to_brake_seconds": brake_applied_at - 130 if brake_applied_at else None,
    }


def prepare_queue_only(root: Path, checkout: Path) -> None:
    scheduler = (root / "src/yuki_participation/scheduling.py").read_text(encoding="utf-8")
    assert scheduler.count("self.last_call + 12") == 1
    assert scheduler.count("pending.last_dirty + 3, pending.first_dirty + 20") == 1
    scheduler = scheduler.replace("self.last_call + 12", "self.last_call + 8").replace(
        "pending.last_dirty + 3, pending.first_dirty + 20",
        "pending.last_dirty + 2, pending.first_dirty + 15",
    )
    (checkout / "src/yuki_participation/scheduling.py").write_text(scheduler, encoding="utf-8")
    session_path = checkout / "src/yuki_participation/session.py"
    session = session_path.read_text(encoding="utf-8")
    assert session.count("            active=active,\n") == 1
    assert session.count("        self.queue.offer(event, kind)") == 1
    session = session.replace("            active=active,\n", "")
    session = session.replace(
        "self.queue.offer(snapshot.focus, snapshot.kind)", "self.queue.retry()"
    )
    session = session.replace(
        "        self.queue.offer(event, kind)",
        "        matched = self.controller.participation_view(\n"
        "            event, max(self.controller.state.now, event.at), check_budget=False\n"
        "        ).matched_unit\n"
        "        self.queue.offer(event, kind, unit=matched.unit if matched else None)",
    )
    session_path.write_text(session, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=VARIANTS)
    parser.add_argument("--latency", type=float, default=6.002977)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(
            json.dumps(
                [
                    asyncio.run(replay(args.worker, scenario=scenario, latency=args.latency))
                    for scenario in SCENARIOS
                ]
            )
        )
        return
    root = Path(__file__).resolve().parents[1]
    archive = subprocess.check_output(
        ["git", "archive", ORIGINAL, "src/yuki_participation"], cwd=root
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="sparse-review-") as temporary:
        for variant in VARIANTS:
            checkout = Path(temporary) / variant
            checkout.mkdir()
            if variant == "sparse":
                shutil.copytree(root / "src", checkout / "src")
            else:
                with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                    tar.extractall(checkout, filter="data")
                if variant == "coalesced_queue_only":
                    prepare_queue_only(root, checkout)
            result = subprocess.check_output(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    variant,
                    "--latency",
                    str(args.latency),
                ],
                env={**os.environ, "PYTHONPATH": str(checkout / "src")},
                cwd=checkout,
            )
            rows.extend(json.loads(result))
    output = {
        "evidence": (
            "Actual ObservationSession with synthetic observer, anonymous sources "
            "and virtual time; no paid API calls"
        ),
        "original_revision": ORIGINAL,
        "profiles": {
            "original": [2, 8, 15],
            "coalesced_queue_only": [2, 8, 15],
            "sparse": [3, 12, 20],
        },
        "queue_only_boundary": (
            "Original controller/session with only new queue, offer/retry integration "
            "and original active timing"
        ),
        "tick_seconds": 2,
        "observer_elapsed_seconds": args.latency,
        "drain_seconds": 90,
        "not_simulated": (
            "Main execution, Host state-change extra review requests, provider accuracy, "
            "real tokens/cache or online latency"
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
