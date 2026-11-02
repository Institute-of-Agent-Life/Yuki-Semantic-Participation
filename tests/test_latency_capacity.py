"""Source-capacity regressions at the actual Controller persistence boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from yuki_participation.controller import Controller
from yuki_participation.models import Effect, Scope, ScopedEvent, SourceRef
from yuki_participation.participation import ParticipationUnit, UnitBinding
from yuki_participation.self_report import SelfDelta, SelfReport
from yuki_participation.store import SnapshotStore


@pytest.fixture(scope="module")
def saved_live_relation_at_1024(tmp_path_factory):
    """Construct only through Controller's real event/admission/effect APIs.

    The initial direct admission and first effect are assumed Host facts here;
    this fixture does not claim to be Host or gateway E2E.
    """
    scope = Scope(conversation_id="synthetic-capacity-scope", generation=1)
    controller = Controller(scope, 100.0)
    binding = None
    for index in range(1, 1025):
        now = 100.0 + index / 10
        controller.advance(now, controller_epoch=0, host_available=False)
        ref = SourceRef(event_id=f"event:{index}", revision=1)
        event = ScopedEvent(
            scope=scope,
            ref=ref,
            at=now,
            thread="event:1",
            author="synthetic-person",
            target="synthetic-person",
            text="synthetic source",
            source_order=index,
        )
        assert controller.observe_committed_event(event)
        if index == 1:
            binding = UnitBinding(
                scope=scope,
                unit=ParticipationUnit(thread="event:1", target="synthetic-person"),
                actor="synthetic-person",
                basis=(ref,),
            )
        else:
            assert controller.participation_view(event, now).matched_unit is not None
        current = (
            binding if index == 1 else binding.model_copy(update={"basis": (*binding.basis, ref)})
        )
        assert controller.observe_unit_input(current, ref, continuation=index != 1)
        if index == 1:
            assert controller.observe_unit_expression(
                current,
                "synthetic-run-original",
                Effect(
                    effect_id="synthetic-original-effect",
                    kind="message",
                    at=now,
                    actual_targets=("group",),
                ),
            )
    assert len(controller.state.seen) == 1024
    store = SnapshotStore(tmp_path_factory.mktemp("live-capacity") / "state.sqlite")
    try:
        assert store.save(controller.state, expected_revision=0) == 1
        loaded = store.load(scope)
        assert loaded is not None and loaded[0] == 1
        return loaded[1], binding
    finally:
        store.close()


def _record(name: str, value: dict):
    root = os.environ.get("YUKI_LATENCY_ARTIFACT_DIR")
    if root:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_live_relation_crosses_1024_and_remains_after_restore(saved_live_relation_at_1024):
    state, binding = saved_live_relation_at_1024
    controller = Controller.restore(state.model_copy(deep=True), state.now)
    ref = SourceRef(event_id="event:1025", revision=1)
    event = ScopedEvent(
        scope=state.scope,
        ref=ref,
        at=state.now + 0.1,
        thread="event:1",
        author="synthetic-person",
        target="synthetic-person",
        text="synthetic next continuation",
        source_order=1025,
    )
    controller.advance(event.at, controller_epoch=0, host_available=False)
    observed = controller.observe_committed_event(event)
    view = controller.participation_view(event, event.at) if observed else None
    _record(
        "live_relation_cross_1024",
        {
            "boundary": (
                "synthetic library events/admissions/effect, real Controller and snapshot store"
            ),
            "seen_before": len(state.seen),
            "observed": observed,
            "matched": bool(view and view.matched_unit),
            "units_after": len(controller._participation_checkpoint().units),
        },
    )
    assert observed and view is not None and view.matched_unit is not None
    current = binding.model_copy(update={"basis": (*binding.basis, ref)})
    assert controller.observe_unit_input(current, ref, continuation=True)
    assert controller.observe_unit_hint(
        current,
        SelfReport(
            run_ref="synthetic-run-original",
            sequence=2,
            response_id="synthetic-late-quiet",
            at=event.at,
            source=ref,
            delta=SelfDelta(engage="quiet"),
        ),
    )
    restored = Controller.restore(controller.state.model_copy(deep=True), event.at)
    assert any(
        unit.hint and unit.hint.delta.engage == "quiet"
        for unit in restored._participation_checkpoint().units.values()
    )


def test_unseen_source_withdrawal_at_1024_does_not_revive(saved_live_relation_at_1024):
    state, _binding = saved_live_relation_at_1024
    controller = Controller.restore(state.model_copy(deep=True), state.now)
    ref = SourceRef(event_id="event:4096", revision=1)
    try:
        controller.observe_source_change(ref)
    except ValueError as error:
        _record("withdrawal_at_1024", {"error": str(error), "seen_before": len(state.seen)})
        raise
    event = ScopedEvent(
        scope=state.scope,
        ref=ref,
        at=state.now,
        thread="event:4096",
        author="synthetic-person",
        target="synthetic-person",
        text="withdrawn source",
        source_order=4096,
    )
    assert controller.state.invalidated.get(ref.event_id) == ref.revision
    assert not controller.observe_committed_event(event)
    _record("withdrawal_at_1024", {"error": None, "same_revision_rejected": True})
