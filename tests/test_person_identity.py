"""Current same-human decisions over untouched original participation facts."""

from tests.test_controller import SCOPE, observation
from yuki_participation.controller import Controller, State
from yuki_participation.models import Effect, ScopedEvent, Snapshot, SourceRef
from yuki_participation.observer import JevObserver
from yuki_participation.participation import ParticipationUnit, UnitBinding
from yuki_participation.self_report import SelfDelta, SelfReport


def root(actor):
    return "C" if actor in {"A", "B", "C"} else actor


def source(c, number, actor, at, thread="topic"):
    value = ScopedEvent(
        scope=c.state.scope,
        ref=SourceRef(event_id=f"event:{number}", revision=1),
        thread=thread,
        author=actor,
        target=actor,
        text="synthetic source",
        at=at,
        source_order=number,
    )
    assert c.observe_committed_event(value)
    return value


def binding(event, *, basis=None):
    return UnitBinding(
        scope=event.scope,
        unit=ParticipationUnit(thread=event.thread, target=event.author),
        actor=event.author,
        basis=basis or (event.ref,),
    )


def start(c, number, actor, at, engage="stay", thread="topic"):
    event = source(c, number, actor, at, thread)
    unit = binding(event)
    assert c.observe_unit_input(unit, event.ref)
    assert hint(c, unit, event, engage)
    return event, unit


def hint(c, unit, event, engage, *, response=None):
    return c.observe_unit_hint(
        unit,
        SelfReport(
            run_ref=f"run:{event.ref.event_id}",
            sequence=1,
            response_id=response or event.ref.event_id,
            at=event.at,
            source=event.ref,
            delta=SelfDelta(engage=engage),
        ),
    )


def units(c):
    return c._read_participation_checkpoint().units


def test_merge_keeps_new_quiet_after_late_send_and_hint_but_allows_real_later_main():
    c = Controller(SCOPE, 100)
    a, ua = start(c, 1, "A", 100)
    _, _ub = start(c, 2, "B", 200, "quiet")
    assert c.observe_unit_expression(
        ua, "run-A", Effect(effect_id="late-send", kind="message", at=250, actual_targets=("A",))
    )
    assert not hint(c, ua, a, "stay", response="late-old-stay")
    focus = source(c, 3, "C", 251)
    original = c.state.model_dump_json()
    assert c.participation_view(focus, 251, actor_identity=root).matched_unit is None
    assert c.participating_units(251, actor_identity=root)[0].engage == "quiet"
    assert c.state.model_dump_json() == original
    later, uc = start(c, 4, "C", 300)
    assert c.participation_view(later, 300, actor_identity=root).matched_unit is not None
    assert c.participating_units(300, actor_identity=root)[0].engage == "stay"
    assert len(units(c)) == 3
    assert units(c)[c._unit_key("topic", "A")].binding == ua
    assert units(c)[c._unit_key("topic", "C")].binding == uc


def test_same_person_different_discussion_does_not_get_global_quiet():
    c = Controller(SCOPE, 100)
    start(c, 1, "A", 100, "quiet", "one")
    start(c, 2, "B", 101, "stay", "two")
    focus = source(c, 3, "C", 102, "two")
    view = c.participation_view(focus, 102, actor_identity=root)
    assert view.matched_unit.unit.thread == "two"
    assert len(view.candidates) == 2


def populated():
    c = Controller(SCOPE, 100)
    a, ua = start(c, 1, "A", 100)
    for i in range(2, 5):
        event = source(c, i, "A", 99 + i)
        assert c.observe_unit_input(ua, event.ref, continuation=True)
    b, ub = start(c, 10, "B", 110)
    for i in range(11, 13):
        event = source(c, i, "B", 100 + i)
        assert c.observe_unit_input(ub, event.ref, continuation=True)
    return c, a, ua, b, ub


def test_merge_counts_original_sources_once_and_preserves_unknown_progress():
    c, a, ua, _b, _ub = populated()
    focus = source(c, 20, "C", 120)
    assert c.participating_units(120, actor_identity=root)[0].unconfirmed_inputs == 5
    assert c.participation_view(focus, 120, actor_identity=root).matched_unit is None
    original = c.state.model_dump_json()
    for number in range(2, 5):
        assert c.observe_unit_input(ua, c.state.events[f"event:{number}"].ref, continuation=True)
    assert c.observe_unit_input(ua, a.ref)
    assert c.state.model_dump_json() == original
    checkpoint = c._read_participation_checkpoint()
    key = c._unit_key("topic", "A")
    checkpoint.units[key] = checkpoint.units[key].model_copy(update={"fast_inputs": None})
    c._save_participation(checkpoint)
    assert c.participating_units(120, actor_identity=root)[0].unconfirmed_inputs is None
    assert c.participation_view(focus, 120, actor_identity=root).matched_unit is None


def test_paid_review_keeps_each_original_watermark_and_not_later_inputs():
    c, _a, ua, _b, ub = populated()
    focus = source(c, 20, "C", 120)
    snapshot = c.freeze_observation(
        Snapshot(scope=SCOPE, focus=focus, context=(), sequence=1, issued_at=120),
        actor_identity=root,
    )
    assert {(item.actor, item.fast_sequence) for item in snapshot.unit_reviews} == {
        ("A", 3),
        ("B", 2),
    }
    before = snapshot.model_dump_json()
    for number, actor, unit in ((21, "A", ua), (22, "B", ub)):
        event = source(c, number, actor, 121)
        assert c.observe_unit_input(unit, event.ref, continuation=True)
    actual = observation(focus, act="extend_yuki").model_copy(
        update={"snapshot": snapshot, "received_at": 122}
    )
    assert c.apply_semantic_observation(actual, actor_identity=root)
    assert snapshot.model_dump_json() == before
    assert c.participating_units(122, actor_identity=root)[0].unconfirmed_inputs == 2
    assert {unit.review.covered_fast_sequence for unit in units(c).values()} == {2, 3}
    assert {unit.binding.actor for unit in units(c).values()} == {"A", "B"}
    assert not c.apply_semantic_observation(actual, actor_identity=root)


def test_merge_negative_review_closes_same_discussion_without_reobserving_old_focus():
    c, _a, ua, _b, ub = populated()
    focus = source(c, 20, "C", 120)
    snapshot = c.freeze_observation(
        Snapshot(scope=SCOPE, focus=focus, context=(), sequence=1, issued_at=120),
        actor_identity=root,
    )
    actual = observation(focus, act="other_exchange", floor="other").model_copy(
        update={"snapshot": snapshot}
    )
    assert c.apply_semantic_observation(actual, actor_identity=root)
    newer = source(c, 21, "C", 121)
    assert (
        c.participation_view(newer, 121, actor_identity=root, check_budget=False).matched_unit
        is None
    )
    assert set(c.state.observations) == {focus.ref.event_id}
    assert units(c)[c._unit_key("topic", "A")].binding == ua
    assert units(c)[c._unit_key("topic", "B")].binding == ub


def test_flattened_a_b_c_identity_cold_restore_keeps_checkpoint_and_admissions():
    c, _a, _ua, _b, _ub = populated()
    original = c.state.model_dump(mode="json")
    restored = Controller.restore(State.model_validate(original), 120)
    assert restored.state.host_checkpoint == c.state.host_checkpoint
    assert restored.state.events == c.state.events
    assert restored.state.consumed == c.state.consumed
    before = restored.state.model_dump_json()
    current = restored.participating_units(120, actor_identity=root)
    assert len(current) == 1 and current[0].unconfirmed_inputs == 5
    assert restored.state.model_dump_json() == before
    separate = Controller(SCOPE.model_copy(update={"generation": 2}), 120)
    assert not separate.participating_units(120, actor_identity=root)


def test_new_jev_projection_uses_current_person_without_rewriting_snapshot():
    c = Controller(SCOPE, 100)
    a, _ua = start(c, 1, "A", 100)
    b, _ub = start(c, 2, "B", 101, "quiet")
    snapshot = c.freeze_observation(
        Snapshot(scope=SCOPE, focus=b, context=(a,), sequence=1, issued_at=102), actor_identity=root
    )
    original = snapshot.model_dump_json()
    state = JevObserver._semantic_state(snapshot, actor_identity=root)
    assert state["focus"]["author"] == state["context"][0]["author"] == "C"
    assert {unit["target"] for unit in state["yuki_participation"]["units"]} == {"C"}
    assert snapshot.model_dump_json() == original
    assert snapshot.focus.author == "B" and snapshot.context[0].author == "A"
