"""Repeated old admissions avoid redundant work without losing effect invalidation."""

from __future__ import annotations

from yuki_participation.controller import Controller
from yuki_participation.models import Effect, Scope, ScopedEvent, SourceRef
from yuki_participation.participation import ParticipationUnit, UnitBinding


def _consumed_old_admission():
    scope = Scope(conversation_id="synthetic-replay", generation=1)
    controller = Controller(scope, 100.0)
    first = SourceRef(event_id="event:1", revision=1)
    second = SourceRef(event_id="event:2", revision=1)
    binding = UnitBinding(
        scope=scope,
        unit=ParticipationUnit(thread="event:1", target="person-a"),
        actor="person-a",
        basis=(first,),
    )
    for index, ref in enumerate((first, second), 1):
        now = 100.0 + index
        controller.advance(now, controller_epoch=0, host_available=False)
        event = ScopedEvent(
            scope=scope,
            ref=ref,
            at=now,
            thread="event:1",
            author="person-a",
            target="person-a",
            text="synthetic source",
            source_order=index,
        )
        assert controller.observe_committed_event(event)
        current = binding if index == 1 else binding.model_copy(update={"basis": (first, second)})
        assert controller.observe_unit_input(current, ref, continuation=index == 2)
        if index == 1:
            assert controller.observe_unit_expression(
                current,
                "synthetic-original-run",
                Effect(
                    effect_id="synthetic-original-effect",
                    kind="message",
                    at=now,
                    actual_targets=("group",),
                ),
            )
    # Establish the real post-advance steady state and its retirement projection.
    controller.advance(103.0, controller_epoch=0, host_available=False)
    controller.take_changes()
    return controller, binding, first


def test_consumed_old_admission_replay_has_no_redundant_prune(monkeypatch):
    controller, binding, old = _consumed_old_admission()
    before = controller.state.model_dump_json()
    calls = []
    original = Controller._prune_participation

    def trace(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Controller, "_prune_participation", trace)
    assert controller.observe_unit_input(binding, old, continuation=False)
    assert controller.state.model_dump_json() == before
    assert controller.take_changes() == (False, False)
    assert not calls, "unchanged consumed admission repeated full participation pruning"


def test_replay_removes_invalidated_effect_and_does_not_revive_association():
    controller, binding, old = _consumed_old_admission()
    effect_id = "synthetic-original-effect"
    effect = controller.state.effects[effect_id]
    assert effect_id in controller._participation_checkpoint().expressions
    controller.remove_state_entry("effects", effect_id)
    assert controller.observe_unit_input(binding, old, continuation=False)
    assert effect_id not in controller._participation_checkpoint().expressions
    controller.set_state_entry("effects", effect_id, effect)
    assert controller.observe_unit_input(binding, old, continuation=False)
    assert effect_id not in controller._participation_checkpoint().expressions
