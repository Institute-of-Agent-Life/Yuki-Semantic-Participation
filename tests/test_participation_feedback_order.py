"""Late feedback follows original input order, including equal source timestamps."""

import pytest

from tests.test_controller import event, observation
from tests.test_participation_continuation import admitted, report
from yuki_participation.controller import Controller, State
from yuki_participation.models import Effect
from yuki_participation.observer import JevObserver
from yuki_participation.participation import CHECKPOINT_KEY


def _report(binding, engage, *, at, run, response, sequence=1):
    return report(engage, at=at, run=run, response=response, sequence=sequence).model_copy(
        update={"source": binding.basis[-1]}
    )


def _pair(*, old_act="other_exchange", old_floor="other", ids=("event:9", "event:10")):
    controller, origin, unit = admitted()
    assert controller.observe_unit_hint(unit, report("stay", at=100))
    first = event(ids[0], at=101, source_order=9)
    first_binding = unit.model_copy(update={"basis": (origin.ref, first.ref)})
    assert controller.observe_committed_event(first)
    assert controller.observe_unit_input(first_binding, first.ref, continuation=True)
    old = observation(first, sequence=1, act=old_act, floor=old_floor)
    old = old.model_copy(update={"snapshot": controller.freeze_observation(old.snapshot)})
    second = event(ids[1], at=101, source_order=10)
    second_binding = unit.model_copy(update={"basis": (origin.ref, second.ref)})
    assert controller.observe_committed_event(second)
    assert controller.observe_unit_input(second_binding, second.ref, continuation=True)
    new = observation(second, sequence=2, act="extend_yuki")
    new = new.model_copy(update={"snapshot": controller.freeze_observation(new.snapshot)})
    return controller, origin, first_binding, second_binding, old, new


def _matched(controller):
    focus = event("event:11", at=102, source_order=11)
    controller.observe_committed_event(focus)
    return controller.participation_view(focus, 102).matched_unit


def test_late_older_activation_hint_cannot_undo_quiet_at_the_same_source_time():
    controller, _, first, second, _, _ = _pair()
    assert controller.observe_unit_hint(
        second, _report(second, "quiet", at=101, run="new-activation", response="new-response")
    )
    assert not controller.observe_unit_hint(
        first, _report(first, "stay", at=101, run="old-activation", response="old-response")
    )
    assert controller.participating_units(102)[0].engage == "quiet"
    assert _matched(controller) is None


def test_newer_activation_hint_can_update_intent_at_the_same_source_time():
    controller, _, first, second, _, _ = _pair()
    assert controller.observe_unit_hint(
        first, _report(first, "quiet", at=101, run="old-activation", response="old-response")
    )
    assert controller.observe_unit_hint(
        second, _report(second, "stay", at=101, run="new-activation", response="new-response")
    )
    assert controller.participating_units(102)[0].engage == "stay"
    assert _matched(controller) is not None


@pytest.mark.parametrize("ids", [("event:9", "event:10"), ("z-old", "a-new")])
def test_late_negative_review_cannot_replace_a_newer_confirmed_invitation(ids):
    controller, _, _, _, old, new = _pair(ids=ids)
    assert controller.apply_semantic_observation(new)
    assert controller.participating_units(102)[0].semantic_continuation
    assert controller.apply_semantic_observation(old)
    assert controller.participating_units(102)[0].semantic_continuation
    assert _matched(controller) is not None


def test_late_negative_review_still_retires_later_unconfirmed_inputs():
    controller, _, _, _, old, _ = _pair()
    assert controller.apply_semantic_observation(old)
    assert not controller.participating_units(102)[0].semantic_continuation
    assert _matched(controller) is None


def test_confirmation_preserves_original_source_and_later_admissions():
    controller, origin, _, second, _, new = _pair()
    third = event("event:11", at=102, source_order=11)
    assert controller.observe_committed_event(third)
    assert controller.observe_unit_input(
        second.model_copy(update={"basis": (origin.ref, third.ref)}),
        third.ref,
        continuation=True,
    )
    assert controller.apply_semantic_observation(new)
    unit = controller._participation_checkpoint().units[controller._unit_key("topic", "A")]
    assert unit.review.source == new.snapshot.focus.ref
    assert third.ref in unit.continuation_refs


def test_quiet_is_not_reopened_by_a_late_positive_review_or_actual_send():
    controller, _, _, second, old, _ = _pair(old_act="extend_yuki", old_floor="yuki")
    assert controller.observe_unit_hint(
        second, _report(second, "quiet", at=101, run="new-activation", response="new-response")
    )
    assert controller.apply_semantic_observation(old)
    assert controller.observe_unit_expression(
        second,
        "original-main",
        Effect(effect_id="actual-message", kind="message", at=102, actual_targets=("group",)),
    )
    assert controller.participating_units(102)[0].engage == "quiet"
    assert _matched(controller) is None


def test_cold_old_state_learns_source_order_without_reopening_quiet():
    controller, _, first, second, old, new = _pair()
    assert controller.observe_unit_hint(
        second, _report(second, "quiet", at=101, run="new-activation", response="new-response")
    )
    raw = controller.state.model_dump(mode="json")
    for source in raw["events"].values():
        source.pop("source_order", None)
    for source in raw["seen"].values():
        source.pop("source_order", None)
    for unit in raw["host_checkpoint"][CHECKPOINT_KEY]["units"].values():
        if unit["hint"] is not None:
            unit["hint"].pop("source", None)
    restored = Controller.restore(State.model_validate(raw), 102)
    original_events = {
        key: source.model_dump_json() for key, source in restored.state.events.items()
    }
    for source in (old.snapshot.focus, new.snapshot.focus):
        # The Host rereads the same internal event, now providing its original row order.
        assert not restored.observe_committed_event(source)
        assert restored.state.seen[source.ref.event_id].source_order == source.source_order
    assert {
        key: source.model_dump_json() for key, source in restored.state.events.items()
    } == original_events
    assert not restored.observe_unit_hint(
        first, _report(first, "stay", at=101, run="old-activation", response="old-response")
    )
    assert restored.participating_units(102)[0].engage == "quiet"
    # An actual newer response on C is still permitted; equal time is not a blanket rejection.
    assert restored.observe_unit_hint(
        second,
        _report(
            second,
            "stay",
            at=101,
            run="new-activation",
            response="newer-response",
            sequence=2,
        ),
    )
    assert restored.participating_units(102)[0].engage == "stay"


def test_learning_order_after_http_freeze_preserves_the_original_snapshot_and_apply():
    controller, origin, unit = admitted()
    assert controller.observe_unit_hint(unit, report("stay", at=100))
    source = event("opaque-source", at=101)
    assert controller.observe_committed_event(source)
    assert controller.observe_unit_input(
        unit.model_copy(update={"basis": (origin.ref, source.ref)}),
        source.ref,
        continuation=True,
    )
    actual = observation(source, act="extend_yuki")
    actual = actual.model_copy(update={"snapshot": controller.freeze_observation(actual.snapshot)})
    original_snapshot = actual.snapshot.model_dump_json()
    observer = JevObserver.__new__(JevObserver)
    observer.model = "fixture"
    observer._identity = lambda: {"display_name": "Yuki", "aliases": ()}
    encoded = observer._encode_request(actual.snapshot)
    assert not controller.observe_committed_event(source.model_copy(update={"source_order": 9}))
    assert controller.state.seen[source.ref.event_id].source_order == 9
    assert controller.state.events[source.ref.event_id] == source
    assert actual.snapshot.model_dump_json() == original_snapshot
    assert observer._encode_request(actual.snapshot) == encoded
    assert controller.apply_semantic_observation(actual)
