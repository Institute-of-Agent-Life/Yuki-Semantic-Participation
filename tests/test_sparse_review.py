"""Sparse review preserves real inputs, paid snapshots and independent brakes."""

import asyncio

import httpx
import pytest

from tests.test_controller import SCOPE, answer, event, observation
from tests.test_participation_continuation import report
from tests.test_seed_retirement_fastpath import established, input_for, scored
from yuki_participation.autonomy_parameters import AutonomyParameters
from yuki_participation.controller import Controller, State
from yuki_participation.models import Snapshot
from yuki_participation.participation import CHECKPOINT_KEY, ParticipationUnit, UnitBinding
from yuki_participation.scheduling import ObservationQueue
from yuki_participation.session import ObservationSession


def test_matched_unit_replaces_only_ordinary_pending_and_preserves_first_dirty():
    queue = ObservationQueue(SCOPE)
    unit = ParticipationUnit(thread="topic", target="A")
    invite = event("explicit-invite", at=100, observation_priority=True)
    queue.offer(invite, unit=unit)
    for position in range(21):
        queue.offer(event(f"ordinary:{position}", at=100 + position), unit=unit)
    queue.offer(event("unmatched", at=120))
    assert set(queue.pending) == {"explicit-invite", "ordinary:20", "unmatched"}
    assert queue.pending["ordinary:20"].first_dirty == 100
    snapshot = queue.take(120, context=())
    assert snapshot.focus == invite
    queue.finish(snapshot.sequence)
    assert queue.take(131, context=()) is None
    assert queue.take(132, context=()).focus.ref.event_id == "ordinary:20"


def test_continuous_replacement_reaches_original_max_wait_without_initial_interval():
    queue = ObservationQueue(SCOPE)
    unit = ParticipationUnit(thread="topic", target="A")
    for position in range(21):
        queue.offer(event(f"source:{position}", at=position), unit=unit)
        snapshot = queue.take(position, context=())
        assert (snapshot is not None) == (position == 20)
    assert snapshot.focus.ref.event_id == "source:20"
    assert queue.sequence == 1


def test_newer_pending_survives_old_request_failure_and_cancel_retry():
    queue = ObservationQueue(SCOPE)
    unit = ParticipationUnit(thread="topic", target="A")
    queue.offer(event("paid", at=100), unit=unit)
    snapshot = queue.take(103, context=())
    frozen = snapshot.model_dump_json()
    queue.offer(event("newer", at=104), unit=unit)
    queue.retry()
    assert list(queue.pending) == ["newer"]
    assert queue.in_flight.model_dump_json() == frozen
    restored = ObservationQueue(SCOPE)
    restored.restore(
        queue.checkpoint(),
        now=105,
        events={item.ref.event_id: item for item in (snapshot.focus, event("newer", at=104))},
    )
    assert list(restored.pending) == ["newer"]
    assert restored.last_call == 103


def test_restore_coalesces_old_per_event_pending_without_changing_invitation():
    c, _, original = established()
    old, _ = input_for(c, original, "old", 1)
    new, _ = input_for(c, original, "new", 2)
    invite = event("independent", at=103, observation_priority=True)
    c.observe_committed_event(invite)
    session = ObservationSession(c, None)
    checkpoint = session.queue.checkpoint()
    checkpoint["pending"] = [
        {
            "ref": source.ref.model_dump(),
            "kind": "conversation",
            "first_dirty": source.at,
            "last_dirty": source.at,
        }
        for source in (old, new, invite)
    ]
    c.state.observer_checkpoint["queue"] = checkpoint
    restored = ObservationSession(Controller.restore(c.state, 104), None)
    assert set(restored.queue.pending) == {"new", "independent"}
    assert restored.queue.pending["new"].first_dirty == old.at
    assert restored.queue.last_call == float("-inf")


@pytest.mark.asyncio
async def test_quiet_clears_only_covered_admitted_review_and_preserves_new_invitation():
    c, _, original = established()
    old, _ = input_for(c, original, "old", 1)
    newer, current = input_for(c, original, "processed", 2)
    session = ObservationSession(c, None)
    session.request_observation(old.ref)
    session.request_observation(newer.ref)
    assert c.observe_unit_hint(current, report("quiet", at=102, response="done"))
    assert not await session.evaluate_due(105)
    assert not session.queue.pending
    # A real future invitation is still observed, without resetting last_call.
    invite = event("later-invitation", at=106, observation_priority=True)
    c.observe_committed_event(invite)
    session.request_observation(invite.ref)
    assert list(session.queue.pending) == [invite.ref.event_id]


@pytest.mark.asyncio
async def test_quiet_does_not_clear_unadmitted_focus_even_if_older():
    c, _, original = established()
    unadmitted, _ = input_for(c, original, "not-processed", 1, admit=False)
    _, current = input_for(c, original, "processed", 2)
    snapshots = []

    class Observer:
        async def evaluate(self, snapshot):
            snapshots.append(snapshot)
            return scored(snapshot)

    session = ObservationSession(c, Observer())
    session.request_observation(unadmitted.ref)
    assert c.observe_unit_hint(current, report("quiet", at=102, response="done"))
    assert await session.evaluate_due(105)
    assert snapshots[0].focus == unadmitted


@pytest.mark.asyncio
async def test_quiet_while_http_runs_never_cancels_paid_snapshot_or_requeues_old_failure():
    c, _, original = established()
    first, _ = input_for(c, original, "paid", 1)
    started, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    class DelayedFailure:
        async def evaluate(self, snapshot):
            snapshots.append(snapshot)
            started.set()
            await release.wait()
            raise httpx.ReadTimeout("offline synthetic")

    session = ObservationSession(c, DelayedFailure())
    session.request_observation(first.ref)
    task = asyncio.create_task(session.evaluate_due(104))
    await started.wait()
    paid = snapshots[0].model_dump_json()
    later, current = input_for(c, original, "later", 5)
    session.request_observation(later.ref)
    assert c.observe_unit_hint(current, report("quiet", at=105, response="done"))
    assert not await session.evaluate_due(106)
    assert not session.queue.pending
    assert snapshots[0].model_dump_json() == paid
    assert not task.done()
    release.set()
    assert not await task
    assert not await session.evaluate_due(107)
    assert not session.queue.pending
    assert len(snapshots) == 1


def test_checkpoint_upgrade_keeps_real_quiet_review_binding_and_original_sources():
    c, _, original = established()
    source, current = input_for(c, original, "continuation", 1)
    snap = c.freeze_observation(
        Snapshot(scope=SCOPE, focus=source, context=(), sequence=1, issued_at=103)
    )
    assert c.apply_semantic_observation(scored(snap, act="other_exchange", floor="other"))
    assert c.observe_unit_hint(current, report("quiet", at=101, response="done"))
    raw = c.state.model_dump(mode="json")
    key = c._unit_key("topic", "A")
    unit = raw["host_checkpoint"][CHECKPOINT_KEY]["units"][key]
    unit.pop("continuation_refs")
    unit["fast_sequence"] = 1
    unit["fast_inputs"] = [
        {"source": source.ref.model_dump(), "at": 101, "sequence": 1, "source_order": 1}
    ]
    unit["review"]["covered_fast_sequence"] = 1
    restored = Controller.restore(State.model_validate(raw), 104)
    migrated = restored._participation_checkpoint().units[key]
    assert migrated.binding == original
    assert migrated.continuation_refs == (source.ref,)
    assert migrated.hint.delta.engage == "quiet"
    assert migrated.review.source == source.ref
    assert not restored.participating_units(104)[0].semantic_continuation
    assert "fast_sequence" not in restored.state.model_dump_json()
    assert (
        AutonomyParameters.model_validate({"continuation_review_span": 4}).model_dump()
        == AutonomyParameters().model_dump()
    )


@pytest.mark.asyncio
async def test_current_identity_is_used_for_unique_matched_queue_without_storing_a_resolver():
    c, _, original = established()
    source = event("merged-person", at=101).model_copy(update={"author": "B", "target": "B"})
    c.observe_committed_event(source)
    session = ObservationSession(c, None)
    session.request_observation(
        source.ref, actor_identity=lambda actor: "A" if actor == "B" else actor
    )
    assert session.queue.pending[source.ref.event_id].unit == original.unit
    assert not hasattr(session, "actor_identity")


def test_context_uses_original_source_order_and_excludes_known_later_same_time_input():
    queue = ObservationQueue(SCOPE)
    before = event("event:9", at=100, source_order=9)
    focus = event("focus", at=100, source_order=11)
    middle = event("event:10", at=100, source_order=10)
    after = event("event:12", at=100, source_order=12)
    queue.offer(focus)
    snapshot = queue.take(103, context=(after, middle, before))
    assert [source.ref.event_id for source in snapshot.context] == ["event:9", "event:10"]


@pytest.mark.asyncio
async def test_quiet_covers_actual_observed_supplement_without_changing_original_hint_source():
    c, origin, original = established()
    source, _ = input_for(c, original, "observed-supplement", 1)
    session = ObservationSession(c, None)
    session.request_observation(source.ref)
    actual_observed = original.model_copy(update={"basis": (*original.basis, source.ref)})
    quiet = report("quiet", at=100, response="done").model_copy(update={"source": origin.ref})
    assert c.observe_unit_hint(actual_observed, quiet)
    assert not await session.evaluate_due(104)
    assert not session.queue.pending
    assert c._participation_checkpoint().units[c._unit_key("topic", "A")].hint.source == origin.ref


@pytest.mark.parametrize("same_time", [False, True])
def test_group_stop_releases_for_another_persons_target_invitation_in_original_order(same_time):
    c = Controller(SCOPE, 100)
    stop = event("event:9", at=100, source_order=9)
    c.observe_committed_event(stop)
    stopped = observation(stop, act="ask_yuki_stop")
    stopped = stopped.model_copy(
        update={
            "answers": {
                **stopped.answers,
                "boundary_scope": answer("boundary_scope", "group_thread"),
            }
        }
    )
    assert c.apply_semantic_observation(stopped)
    invite = event("event:10", at=100 if same_time else 101, source_order=10).model_copy(
        update={"author": "B", "target": "B"}
    )
    c.observe_committed_event(invite)
    assert c._closed(invite)
    assert c.apply_semantic_observation(observation(invite))
    assert not c._closed(invite)
    assert c.state.boundaries[stop.ref.event_id].released_by == invite.ref


@pytest.mark.parametrize("brake", ["semantic", "quiet"])
def test_new_brake_retires_older_pending_invitation_but_allows_new_real_invitation(brake):
    c, _, original = established()
    old, _ = input_for(c, original, "old-invitation", 1, admit=False)

    def evaluate(source, sequence, *, act="invite_yuki", floor="yuki"):
        snapshot = c.freeze_observation(
            Snapshot(scope=SCOPE, focus=source, context=(), sequence=sequence, issued_at=source.at)
        )
        assert c.apply_semantic_observation(scored(snapshot, act=act, floor=floor))

    evaluate(old, 1)
    assert c.participation_view(old, 102).addressed
    later, actual = input_for(c, original, "later-brake", 2)
    if brake == "semantic":
        evaluate(later, 2, act="other_exchange", floor="other")
    else:
        assert c.observe_unit_hint(actual, report("quiet", at=102, response="done"))
    assert not c.participation_view(old, 103).addressed
    fresh, _ = input_for(c, original, "fresh-invitation", 4, admit=False)
    evaluate(fresh, 3)
    assert c.participation_view(fresh, 105).addressed
    assert not c.participation_view(old, 105).addressed
    assert old.ref.event_id not in c.state.candidates


def test_same_person_unit_representative_change_keeps_one_review_key_and_original_authors():
    def identity(actor):
        return "C" if actor in {"A", "B"} else actor

    c, _, original = established()
    first = event("first-pending", at=101)
    c.observe_committed_event(first)
    session = ObservationSession(c, None)
    session.request_observation(first.ref, actor_identity=identity)
    other = event("other-original-person", at=102).model_copy(update={"author": "B", "target": "B"})
    c.observe_committed_event(other)
    other_binding = UnitBinding(
        scope=SCOPE,
        unit=ParticipationUnit(thread="topic", target="B"),
        actor="B",
        basis=(other.ref,),
    )
    assert c.observe_unit_input(other_binding, other.ref)
    assert c.observe_unit_hint(other_binding, report("stay", at=102, response="other-stay"))
    later = event("later-pending", at=103)
    c.observe_committed_event(later)
    assert (
        c.participation_view(later, 103, actor_identity=identity).matched_unit.unit != original.unit
    )
    session.request_observation(later.ref, actor_identity=identity)
    assert list(session.queue.pending) == [later.ref.event_id]
    assert session.queue.pending[later.ref.event_id].unit.target == "C"
    assert c.state.events[first.ref.event_id].author == "A"
    assert c.state.events[other.ref.event_id].author == "B"
