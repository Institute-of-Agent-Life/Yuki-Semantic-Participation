"""Offline retirement, original-source feedback and bounded optimistic continuation."""

import asyncio
import json

import httpx
import pytest
from jev_factory import jev_observer
from pydantic import ValidationError
from test_controller import SCOPE, event, observation
from test_participation_continuation import admitted, binding, report

from yuki_participation.autonomy_parameters import AutonomyParameters
from yuki_participation.controller import Controller, State
from yuki_participation.models import (
    CandidateKind,
    Choice,
    Effect,
    Feedback,
    HostUnitOption,
    Proposal,
    ScopedEvent,
    Snapshot,
    SourceRef,
)
from yuki_participation.participation import CHECKPOINT_KEY
from yuki_participation.rubric import REVISION, questions
from yuki_participation.self_report import SelfDelta, SelfReport, extract_tail
from yuki_participation.session import ObservationSession


def input_for(c, original, key, position, *, at=None, admit=True):
    source = event(key, at=at if at is not None else 100 + position).model_copy(
        update={
            "thread": f"new-{key}",
            "source_order": position,
            "unit_ambiguous": True,
            "unit_options": (
                HostUnitOption(key="old", thread=original.unit.thread, target=original.unit.target),
                HostUnitOption(key="new", thread=f"new-{key}", target=original.unit.target),
            ),
        }
    )
    assert c.observe_committed_event(source)
    current = binding(
        source,
        thread=original.unit.thread,
        target=original.unit.target,
        basis=(*original.basis, source.ref),
    )
    if admit:
        assert c.observe_unit_input(current, source.ref, continuation=True)
    return source, current


def freeze(c, source, sequence=1):
    return c.freeze_observation(
        Snapshot(
            scope=source.scope,
            focus=source,
            context=tuple(
                e for e in c.state.events.values() if e.ref != source.ref and e.at <= source.at
            ),
            sequence=sequence,
            issued_at=source.at + 0.25,
        )
    )


def scored(snapshot, *, act="extend_yuki", floor="yuki", selection="old"):
    source = snapshot.focus
    selected = next((o for o in source.unit_options if o.key == selection), None)
    original = observation(source, sequence=snapshot.sequence, act=act, floor=floor)
    answers = dict(original.answers)
    if source.unit_ambiguous:
        answers["unit_selection"] = Choice(
            choice=selection,
            probabilities={
                key: float(key == selection)
                for key in [*(o.key for o in source.unit_options), "unknown"]
            },
        )
    return original.model_copy(
        update={
            "snapshot": snapshot,
            "answers": answers,
            "resolved_unit": selected,
            "received_at": snapshot.issued_at + 0.25,
        }
    )


def established():
    c, first, original = admitted()
    assert c.observe_unit_hint(original, report("stay", at=100, run="initial"))
    return c, first, original


def test_continuation_has_no_review_quota_across_hydration_and_cold_restore():
    c, first, original = established()
    b, bb = input_for(c, original, "b", 1)
    d, bd = input_for(c, original, "d", 2)
    for _ in range(3):
        for source, actual in ((first, original), (b, bb), (d, bd)):
            assert c.observe_unit_input(actual, source.ref, continuation=source != first)
    assert c._participation_checkpoint().units[c._unit_key("topic", "A")].continuation_refs == (
        b.ref,
        d.ref,
    )
    assert (
        c.state.host_checkpoint[CHECKPOINT_KEY]["units"][c._unit_key("topic", "A")]["input_ref"]
        == d.ref.model_dump()
    )
    next_source, next_binding = input_for(c, original, "next", 3, admit=False)
    assert c.participation_view(next_source, 103).needs_observation
    assert c.continuation_current(next_source.ref, next_binding, 103)
    restored = Controller.restore(
        State.model_validate_json(c.state.model_dump_json()), 103, parameters=c.parameters
    )
    assert restored.continuation_current(next_source.ref, next_binding, 103)
    assert not c.state.boundaries


def test_positive_focus_keeps_original_review_source_and_later_admissions():
    c, _, original = established()
    b, _ = input_for(c, original, "b", 1)
    snap = freeze(c, b)
    later, _ = input_for(c, original, "later", 2)
    assert c.apply_semantic_observation(scored(snap))
    assert c.participating_units(103)[0].semantic_continuation
    unit = c._participation_checkpoint().units[c._unit_key("topic", "A")]
    assert unit.review.source == b.ref
    assert unit.continuation_refs == (b.ref, later.ref)
    assert later.ref != snap.focus.ref


@pytest.mark.parametrize(
    "selection,act,floor",
    [
        ("old", "other_exchange", "other"),
        ("new", "open_group", "open"),
        ("old", "ask_yuki_stop", "other"),
    ],
)
def test_late_negative_review_reaches_original_unit_after_new_unreviewed_input(
    selection, act, floor
):
    c, _, original = established()
    b, bb = input_for(c, original, "b", 1)
    snap = freeze(c, b)
    input_for(c, original, "later", 2)
    assert c.apply_semantic_observation(scored(snap, selection=selection, act=act, floor=floor))
    assert c.observe_unit_hint(bb, report("stay", at=101, response="late-stay"))
    actual = Effect(effect_id="late-actual-send", kind="message", at=103)
    assert c.observe_unit_expression(bb, "original", actual)
    fresh, current = input_for(c, original, "fresh", 4, admit=False)
    assert not c.continuation_current(fresh.ref, current, 104)
    assert c.state.effects[actual.effect_id].effect == actual
    assert c.participating_units(104)[0].engage == "stay"
    assert bool(c.state.boundaries) == (act == "ask_yuki_stop")


@pytest.mark.parametrize("quiet_first", [False, True])
def test_local_quiet_and_negative_review_are_independent_facts(quiet_first):
    c, _, original = established()
    source, actual = input_for(c, original, "b", 1)
    snap = freeze(c, source)
    if quiet_first:
        assert c.observe_unit_hint(actual, report("quiet", at=101, response="quiet"))
    assert c.apply_semantic_observation(scored(snap, act="other_exchange", floor="other"))
    if not quiet_first:
        assert c.observe_unit_hint(actual, report("quiet", at=101, response="quiet"))
    unit = c.participating_units(102)[0]
    assert unit.engage == "quiet" and not unit.semantic_continuation
    assert not c.state.boundaries
    c = Controller.restore(State.model_validate_json(c.state.model_dump_json()), 103)
    assert unit.engage == "quiet" and not unit.semantic_continuation


def test_unknown_keeps_existing_relationship_without_buying_confirmation():
    c, _, original = established()
    source, _ = input_for(c, original, "b", 1)
    assert c.apply_semantic_observation(
        scored(freeze(c, source), act="unknown", floor="unknown", selection="unknown")
    )
    assert c.participating_units(102)[0].semantic_continuation
    fresh, actual = input_for(c, original, "fresh", 2, admit=False)
    assert c.continuation_current(fresh.ref, actual, 102)
    assert not c.state.boundaries


def test_new_real_invitation_can_establish_its_source_without_late_old_review_revoking_it():
    c, _, original = established()
    older, old_binding = input_for(c, original, "older", 1)
    old_snapshot = freeze(c, older)
    current, actual = input_for(c, original, "new-invitation", 2, admit=False)
    direct = binding(current, thread="topic", basis=(current.ref,))
    assert c.observe_unit_input(direct, current.ref, continuation=False)
    assert c.observe_unit_hint(direct, report("join", at=102, response="new-join"))
    assert c.apply_semantic_observation(scored(old_snapshot, act="other_exchange", floor="other"))
    assert not c.observe_unit_hint(old_binding, report("stay", at=101, response="old-stay"))
    fresh, actual = input_for(c, direct, "fresh", 3, admit=False)
    assert c.continuation_current(fresh.ref, actual, 103)


def test_old_namespace_missing_progress_keeps_real_relationship_without_quota():
    c, _, original = established()
    raw = c.state.model_dump(mode="json")
    for unit in raw["host_checkpoint"][CHECKPOINT_KEY]["units"].values():
        for name in ("fast_sequence", "fast_inputs", "review"):
            unit.pop(name, None)
    c = Controller.restore(State.model_validate(raw), 101)
    source, actual = input_for(c, original, "b", 2, admit=False)
    assert c.continuation_current(source.ref, actual, 102)
    assert c.apply_semantic_observation(scored(freeze(c, source)))
    assert c.continuation_current(source.ref, actual, 103)


@pytest.mark.parametrize("engage", ["join", "stay", "quiet"])
def test_real_encoder_projects_frozen_related_intent_and_existing_thread_map(engage):
    c, _, original = established()
    assert c.observe_unit_hint(original, report(engage, at=101, response="intent"))
    source, _ = input_for(c, original, "focus", 2, admit=False)
    snapshot = freeze(c, source)
    encoded = jev_observer(
        "synthetic-not-a-credential", identity=lambda: {"display_name": "Yuki", "aliases": ()}
    )._encode_request(snapshot)
    payload = json.loads(encoded)
    state = payload["state"]
    local = state["yuki_participation"]["units"][0]
    old_option = next(o for o in state["unit_options"] if o["key"] == "old")
    assert local == {
        "thread": old_option["thread"],
        "target": "A",
        "engage": engage,
        "source_seconds_from_focus": -1.0,
    }
    assert state["projection_revision"] == "semantic-state-v3"
    assert "source_order" not in encoded.decode()
    assert "response" not in state["yuki_participation"]
    assert "seed_fit" not in payload["questions"]
    original_bytes = encoded
    assert c.observe_unit_hint(
        original,
        report("quiet" if engage != "quiet" else "join", at=103, response="new-state", sequence=2),
    )
    assert (
        jev_observer(
            "synthetic-not-a-credential", identity=lambda: {"display_name": "Yuki", "aliases": ()}
        )._encode_request(snapshot)
        == original_bytes
    )
    assert snapshot.projection_revision == "semantic-state-v3"
    assert snapshot.participation.units[0].engage == engage


def test_missing_hint_is_absent_scope_intent_remains_separate_and_ambiguous_units_not_chosen():
    c, first, original = admitted()
    other_source = event("other", at=101).model_copy(update={"thread": "other-topic"})
    c.observe_committed_event(other_source)
    other = binding(other_source)
    c.observe_unit_input(other, other_source.ref)
    assert c.observe_unit_hint(other, report("quiet", at=101, response="other-quiet"))
    source, _ = input_for(c, original, "focus", 2, admit=False)
    source = source.model_copy(
        update={
            "unit_options": (
                *source.unit_options,
                HostUnitOption(key="other", thread="other-topic", target="A"),
            )
        }
    )
    c.state.events[source.ref.event_id] = source
    snapshot = freeze(c, source)
    assert snapshot.participation.scope_intent is None
    assert [(u.thread, u.engage) for u in snapshot.participation.units] == [
        ("other-topic", "quiet")
    ]
    assert len(snapshot.focus.unit_options) == 3
    assert not c.state.engagement_report


@pytest.mark.asyncio
async def test_inflight_request_freezes_once_new_reports_do_not_requeue_and_real_revocation_wins():
    c, _, original = established()
    source, actual = input_for(c, original, "focus", 1, admit=False)
    started, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    class Delayed:
        async def evaluate(self, snapshot):
            snapshots.append(snapshot)
            started.set()
            await release.wait()
            return scored(snapshot)

    session = ObservationSession(c, Delayed())
    assert session.request_observation(source.ref)
    task = asyncio.create_task(session.evaluate_due(104))
    await started.wait()
    submitted = snapshots[0].model_dump_json()
    assert c.observe_unit_hint(original, report("quiet", at=102, response="while-http"))
    assert not session.queue.pending
    assert snapshots[0].model_dump_json() == submitted
    c.observe_source_change(source.ref)
    release.set()
    assert not await task
    assert source.ref.event_id not in c.state.observations
    assert c.participating_units(104)[0].engage == "quiet"
    assert not session.queue.pending


def test_seed_retirement_reader_preserves_pending_and_authorized_ordinary_basis_without_text():
    c, first, original = established()
    seed = first.model_dump(mode="json")
    seed.update(
        ref={"event_id": "memory:42", "revision": 7},
        kind="seed",
        author="SELF",
        text="retired-memory-body",
    )
    raw = c.state.model_dump(mode="json")
    raw["events"]["memory:42"] = seed
    raw["seen"]["memory:42"] = {"revision": 7, "at": 100}
    raw.update(
        content_keys={"memory:42": "fingerprint"},
        seed_claims={"fingerprint": 100},
        seed_attempts={},
    )
    unit = raw["host_checkpoint"][CHECKPOINT_KEY]["units"][c._unit_key("topic", "A")]
    unit["binding"]["basis"].append(seed["ref"])
    unit["hint_basis"].append(seed["ref"])
    unit["hint"]["delta"]["mood"] = "retired"
    proposal = Proposal(
        proposal_id="original-pending",
        scope=SCOPE,
        controller_epoch=0,
        kind=CandidateKind.CONTACT,
        thread="topic",
        target_hint="A",
        sources=(SourceRef(event_id="memory:42", revision=7),),
        created_at=90,
        expires_at=99,
    )
    raw["proposals"][proposal.proposal_id] = {
        **proposal.model_dump(mode="json"),
        "source_fingerprints": ["old"],
    }
    raw["pending"] = proposal.proposal_id
    state = State.model_validate(raw)
    assert "retired-memory-body" not in state.model_dump_json()
    assert "mood" not in state.model_dump_json()
    c = Controller.restore(state, 105)
    assert c.state.pending == proposal.proposal_id
    assert c.participating_units(105)[0].engage == "stay"
    session = ObservationSession(c, None)
    assert not session.request_observation(SourceRef(event_id="memory:42", revision=7))
    assert c.observe_run_feedback(
        Feedback(
            run_ref="old-run",
            proposal_id=proposal.proposal_id,
            sequence=1,
            outcome="accepted",
            at=105,
            considered_refs=proposal.sources,
        )
    )
    assert c.observe_run_feedback(
        Feedback(
            run_ref="old-run",
            proposal_id=proposal.proposal_id,
            sequence=2,
            outcome="completed",
            at=106,
            effects=(
                Effect(effect_id="compute", kind="compute", at=105),
                Effect(
                    effect_id="actual-message", kind="message", at=106, actual_targets=("group",)
                ),
            ),
        )
    )
    assert c.state.pending is None
    assert c.state.work_pulses["old-run"].public_at == 106
    assert c.state.proposal_runs[proposal.proposal_id] == "old-run"
    assert len(c.state.effects) == 2
    c.observe_source_change(SourceRef(event_id="memory:42", revision=7))
    assert not c.participating_units(107)


def test_old_queued_seed_is_dropped_new_human_restores_with_current_frozen_report():
    c, _, original = established()
    source, _ = input_for(c, original, "focus", 1, admit=False)
    session = ObservationSession(c, None)
    session.request_observation(source.ref)
    checkpoint = session.queue.checkpoint()
    checkpoint["pending"].append(
        {
            "ref": {"event_id": "old-seed", "revision": 1},
            "kind": "recall",
            "first_dirty": 100,
            "last_dirty": 100,
        }
    )
    checkpoint["in_flight"] = {
        "ref": {"event_id": "old-seed", "revision": 1},
        "kind": "contact",
        "sequence": 4,
    }
    restored = type(session.queue)(SCOPE)
    restored.restore(checkpoint, now=103, events=c.state.events)
    assert len(restored.pending) == 1 and restored.sequence == 4
    snapshot = restored.take(104, context=tuple(c.state.events.values()))
    snapshot = c.freeze_observation(snapshot)
    assert snapshot.focus == source and snapshot.sequence == 5
    assert snapshot.participation.units[0].engage == "stay"
    assert all(e.kind != "seed" for e in snapshot.context)


def test_retired_parameters_are_narrowly_migrated_and_mood_does_not_become_empty_feedback():
    profile = AutonomyParameters.model_validate(
        {
            "source_interval_seconds": 17,
            "recall_source_decay_seconds": 3600,
            "contact_source_decay_seconds": 1800,
            "independent_source_factor": 0.35,
            "continuation_review_span": 2,
        }
    )
    assert profile.source_interval_seconds == 17
    assert "continuation_review_span" not in profile.model_dump()
    assert "recall_source_decay_seconds" not in profile.model_dump()
    with pytest.raises(ValidationError):
        AutonomyParameters.model_validate({"typo_unknown": 1})
    with pytest.raises(ValidationError):
        ScopedEvent.model_validate({**event().model_dump(), "kind": "seed"})
    body, delta = extract_tail('body<yuki-state>{"mood":"retired"}</yuki-state>')
    assert body == "body" and delta is None
    old = SelfReport(
        run_ref="run",
        sequence=1,
        response_id="r",
        at=100,
        delta=SelfDelta.model_validate({"engage": "stay", "mood": "retired"}),
    )
    assert old.delta.engage == "stay" and "mood" not in old.model_dump_json()
    snapshot = Snapshot.model_validate(
        {
            "scope": SCOPE.model_dump(),
            "focus": event().model_dump(),
            "context": [],
            "sequence": 1,
            "issued_at": 100,
        }
    )
    assert snapshot.projection_revision is None and snapshot.participation is None
    assert "seed_fit" not in questions()
    assert REVISION == "v6-zh-6"


@pytest.mark.asyncio
async def test_real_session_timeout_and_cold_backoff_do_not_remove_real_continuation():
    c, _, original = established()
    source, _ = input_for(c, original, "source", 1)

    class Unavailable:
        async def evaluate(self, snapshot):
            raise httpx.ReadTimeout("synthetic offline timeout")

    session = ObservationSession(c, Unavailable())
    session.request_observation(source.ref)
    assert not await session.evaluate_due(104)
    assert session.health.failures == 1 and session.retry_after == 134
    fresh, actual = input_for(c, original, "later", 5, admit=False)
    assert c.continuation_current(fresh.ref, actual, 105)
    assert c._participation_checkpoint().units[c._unit_key("topic", "A")].review is None
    restored = Controller.restore(
        State.model_validate_json(c.state.model_dump_json()), 105, parameters=c.parameters
    )
    recovered = ObservationSession(restored, Unavailable())
    assert recovered.retry_after == 134
    assert not await recovered.evaluate_due(106)
    assert recovered.health.failures == 1
    assert restored.continuation_current(fresh.ref, actual, 106)
    assert not restored.state.boundaries


def test_scope_self_report_and_local_quiet_both_reach_one_frozen_request():
    c, _, original = admitted()
    assert c.observe_unit_hint(original, report("quiet", at=100))
    invite = event("independent-self-opportunity", at=101)
    c.observe_committed_event(invite)
    c.apply_semantic_observation(observation(invite))
    proposal = c.advance(102, controller_epoch=0, host_available=True)
    assert proposal is not None
    c.observe_run_feedback(
        Feedback(
            run_ref="self-real-run",
            proposal_id=proposal.proposal_id,
            sequence=1,
            outcome="accepted",
            at=102,
        )
    )
    own = SelfReport(
        run_ref="self-real-run",
        sequence=1,
        response_id="self-response",
        at=102,
        delta=SelfDelta(engage="join"),
    )
    assert c.observe_self_report(own)
    focus, _ = input_for(c, original, "focus", 3, admit=False)
    snapshot = freeze(c, focus)
    assert snapshot.participation.scope_intent.engage == "join"
    assert snapshot.participation.units[0].engage == "quiet"
    payload = jev_observer(
        "synthetic-not-a-credential", identity=lambda: {"display_name": "Yuki", "aliases": ()}
    )._semantic_state(snapshot)
    assert payload["yuki_participation"]["scope_intent"]["engage"] == "join"
    assert payload["yuki_participation"]["units"][0]["engage"] == "quiet"
    assert c.state.engagement_report == own


def test_report_order_uses_only_its_actual_bound_source():
    c, _, original = established()
    source, actual = input_for(c, original, "source", 1)
    foreign = event("other-unit-source", at=101).model_copy(
        update={"thread": "other", "author": "B", "target": "B", "source_order": 99}
    )
    c.observe_committed_event(foreign)
    unrelated = report("quiet", at=101, run="foreign", response="wrong-source").model_copy(
        update={"source": foreign.ref}
    )
    assert not c.observe_unit_hint(actual, unrelated)
    changed_time = report("quiet", at=102, run="source", response="wrong-time").model_copy(
        update={"source": source.ref}
    )
    assert not c.observe_unit_hint(actual, changed_time)
    assert c.participating_units(102)[0].engage == "stay"
