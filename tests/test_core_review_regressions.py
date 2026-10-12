"""Reproductions from the independent V6 review; all sources are synthetic."""

import math

import pytest

from yuki_participation import dynamics
from yuki_participation.controller import Controller
from yuki_participation.models import (
    Choice,
    Effect,
    Feedback,
    Observation,
    Scope,
    ScopedEvent,
    Snapshot,
    SourceRef,
)
from yuki_participation.rubric import CRITERIA, REVISION
from yuki_participation.session import ObservationSession
from yuki_participation.store import SnapshotStore

SCOPE = Scope(conversation_id="synthetic-review", generation=1)


def source(key="focus", *, at=100, revision=1, **changes):
    return ScopedEvent(
        scope=SCOPE,
        ref=SourceRef(event_id=key, revision=revision),
        thread="topic",
        author="A",
        target="A",
        text=f"Synthetic review source {key}",
        at=at,
    ).model_copy(update=changes)


def choice(dimension, selected):
    return Choice(
        choice=selected,
        probabilities={key: float(key == selected) for key in CRITERIA[dimension]},
    )


def observation(event, *, sequence=1, act="invite_yuki", info="refine", context=()):
    return Observation(
        observation_id=f"synthetic-{event.ref.event_id}-{event.ref.revision}-{sequence}",
        snapshot=Snapshot(
            scope=SCOPE,
            focus=event,
            context=context,
            sequence=sequence,
            issued_at=event.at,
        ),
        provider="synthetic_fixture",
        model_revision="fixture",
        rubric_revision=REVISION,
        received_at=event.at,
        answers={
            "interaction_mark": choice("interaction_mark", act),
            "information_state": choice("information_state", info),
            "floor_state": choice("floor_state", "yuki"),
            "boundary_scope": choice("boundary_scope", "target_thread"),
        },
    )


def controller():
    return Controller(SCOPE, 100)


def propose(c, event, *, at=105, sequence=1, epoch=0):
    assert c.observe_committed_event(event)
    assert c.apply_semantic_observation(observation(event, sequence=sequence))
    proposal = c.advance(at, controller_epoch=epoch, host_available=True)
    assert proposal is not None
    return proposal


def feedback(proposal, *, run="run", sequence=1, outcome="accepted", at=105, effects=()):
    return Feedback(
        run_ref=run,
        proposal_id=proposal.proposal_id,
        sequence=sequence,
        outcome=outcome,
        at=at,
        considered_refs=proposal.sources,
        actual_targets=("A",),
        effects=effects,
    )


@pytest.mark.parametrize("act", ["invite_yuki", "ask_yuki_stop"])
def test_revoked_context_removes_derived_observation_candidate_and_boundary(act):
    c = controller()
    anchor, event = source("anchor", at=99), source()
    assert c.observe_committed_event(anchor)
    assert c.observe_committed_event(event)
    assert c.apply_semantic_observation(observation(event, act=act, context=(anchor,)))
    if act == "invite_yuki":
        assert c.opportunity_scores(105)[event.ref.event_id] > 0
    else:
        assert c.state.boundaries

    c.observe_source_change(anchor.ref)

    assert event.ref.event_id not in c.state.observations
    assert event.ref.event_id not in c.state.candidates
    assert event.ref.event_id not in c.state.boundaries
    assert not c.opportunity_scores(105)
    assert c.belief("topic", "A", 105) == dynamics.IDLE


def test_late_old_revision_retraction_cannot_delete_new_source_or_its_interpretation():
    c = controller()
    old, current = source(), source(revision=2)
    assert c.observe_committed_event(old)
    assert c.observe_committed_event(current)
    score = observation(current, sequence=2)
    assert c.apply_semantic_observation(score)
    stored = c.state.observations[current.ref.event_id]

    c.observe_source_change(old.ref)

    assert c.state.events[current.ref.event_id] == current
    assert c.state.observations[current.ref.event_id] == stored
    assert stored.observation_id == score.observation_id
    assert c.opportunity_scores(105)[current.ref.event_id] > 0


@pytest.mark.parametrize("retracted", ["focus", "anchor"])
def test_expired_raw_stop_source_can_still_be_explicitly_retracted(retracted):
    c = controller()
    event = source()
    anchor = source("anchor", at=99)
    assert c.observe_committed_event(anchor)
    assert c.observe_committed_event(event)
    assert c.apply_semantic_observation(observation(event, act="ask_yuki_stop", context=(anchor,)))
    c.advance(1000, controller_epoch=0, host_available=False)
    assert event.ref.event_id not in c.state.observations
    assert event.ref.event_id in c.state.boundaries

    c.observe_source_change(event.ref if retracted == "focus" else anchor.ref)

    assert event.ref.event_id not in c.state.boundaries


def test_reinterpretation_recomputes_attention_without_reusing_old_stimulus():
    c = controller()
    event = source()
    c.observe_committed_event(event)
    c.apply_semantic_observation(observation(event))
    assert c.opportunity_scores(105)[event.ref.event_id] > 0
    assert c.state.attentions[event.ref.event_id] > 0.5

    assert c.apply_semantic_observation(
        observation(event, sequence=2, act="off_focus", info="repeat")
    )

    assert sum(c.opportunity_scores(105).values()) == 0
    assert event.ref.event_id not in c.state.attentions
    assert c.belief("topic", "A", 105) == dynamics.IDLE


@pytest.mark.parametrize("kind", ["compute", "tool"])
def test_non_message_effect_cannot_create_mutual_engagement(kind):
    c = controller()
    proposal = propose(c, source())
    before = c.belief("topic", "A", 105)
    assert c.observe_run_feedback(
        feedback(
            proposal,
            outcome="completed",
            effects=(Effect(effect_id="non-message", kind=kind, at=105, actual_targets=("A",)),),
        )
    )
    assert c.belief("topic", "A", 105) == before
    assert c.belief("topic", "A", 105)[3] == 0


def test_replayed_expression_receipt_does_not_count_as_a_second_expression():
    c = controller()
    proposal = propose(c, source())
    effect = Effect(effect_id="expression", kind="message", at=105, actual_targets=("A",))
    assert c.observe_run_feedback(feedback(proposal, effects=(effect,)))
    before = c.belief("topic", "A", 106)
    exposure = c._exposure(106)

    assert c.observe_run_feedback(
        feedback(
            proposal,
            sequence=2,
            outcome="completed",
            at=106,
            effects=(effect,),
        )
    )

    assert len(c.state.effects) == 1
    assert c.belief("topic", "A", 106) == before
    assert c._exposure(106) == exposure


def test_one_proposal_cannot_be_bound_to_another_run():
    c = controller()
    proposal = propose(c, source())
    assert c.observe_run_feedback(feedback(proposal, run="first"))
    assert not c.observe_run_feedback(feedback(proposal, run="second", outcome="completed"))
    assert set(c.state.feedback) == {"first"}


@pytest.mark.parametrize("outcome", ["completed", "interrupted", "no_reply", "rejected"])
def test_terminal_feedback_cannot_reopen_a_run_with_higher_sequence(outcome):
    c = controller()
    proposal = propose(c, source())
    assert c.observe_run_feedback(feedback(proposal, outcome=outcome))
    assert not c.observe_run_feedback(feedback(proposal, sequence=2, outcome="accepted", at=106))
    assert c.state.feedback["run"].outcome == outcome


def test_late_real_effect_after_interruption_is_recorded_without_reopening_run():
    c = controller()
    proposal = propose(c, source())
    assert c.observe_run_feedback(feedback(proposal))
    assert c.observe_run_feedback(feedback(proposal, sequence=2, outcome="interrupted", at=106))
    effect = Effect(
        effect_id="late-confirmed-message",
        kind="message",
        at=105.5,
        actual_targets=("A",),
    )

    assert c.observe_run_feedback(
        feedback(
            proposal,
            sequence=3,
            outcome="interrupted",
            at=107,
            effects=(effect,),
        )
    )

    assert c.state.feedback["run"].outcome == "interrupted"
    assert c.state.effects[effect.effect_id].effect == effect
    assert c.state.pending is None
    assert not c.observe_run_feedback(feedback(proposal, sequence=4, outcome="accepted", at=108))


def test_late_older_proposal_receipt_cannot_lower_consumption_or_acceptance_time():
    c = controller()
    old = propose(c, source())
    c.advance(106, controller_epoch=1, host_available=False)
    current = propose(c, source(revision=2, at=106), at=111, sequence=2, epoch=1)
    assert c.observe_run_feedback(feedback(current, run="new", at=111))
    assert c.state.consumed["focus"] == 2

    assert c.observe_run_feedback(feedback(old, run="old", at=105))
    assert c.state.consumed["focus"] == 2
    assert c.state.last_accepted == 111
    assert not c.opportunity_scores(112)


@pytest.mark.parametrize("outcome", ["completed", "no_reply", "interrupted"])
def test_terminal_first_receipt_consumes_original_proposal_without_considered_refs(outcome):
    c = controller()
    event = source()
    proposal = propose(c, event)
    result = Feedback(
        run_ref="recovered-run",
        proposal_id=proposal.proposal_id,
        sequence=3,
        outcome=outcome,
        at=106,
    )

    assert c.observe_run_feedback(result)

    assert c.state.consumed[event.ref.event_id] == event.ref.revision
    assert c.state.last_accepted == 106
    assert c.advance(107, controller_epoch=0, host_available=True) is None
    assert c.advance(120, controller_epoch=0, host_available=True) is None


@pytest.mark.parametrize("outcome", ["busy", "rejected"])
def test_non_admission_receipt_does_not_consume_an_unexecuted_proposal(outcome):
    c = controller()
    proposal = propose(c, source())
    assert c.observe_run_feedback(feedback(proposal, outcome=outcome, at=106))
    assert not c.state.consumed
    assert c.state.last_accepted < 0


def test_pending_observation_survives_real_snapshot_roundtrip(tmp_path):
    session = ObservationSession(controller(), None)
    event = source()
    queue_observation(session, event)
    session.queue.sequence = 7
    session.health.failure(105)
    session.retry_after = 135
    session.checkpoint()
    store = SnapshotStore(tmp_path / "controller.sqlite3")
    try:
        store.save(session.controller.state, expected_revision=0)
        _, state = store.load(SCOPE)
    finally:
        store.close()

    restored = ObservationSession(Controller.restore(state, 110), None)
    queue_observation(restored, event)  # Replay must neither lose nor duplicate the queued source.

    assert len(restored.queue.pending) == 1
    assert next(iter(restored.queue.pending.values())).event.ref == event.ref
    assert restored.queue.sequence == 7
    assert restored.health.failures == 1
    assert restored.retry_after == 135


def test_legal_bounded_source_volume_remains_checkpointable(tmp_path):
    c = controller()
    for index in range(30):
        event = source(str(index), at=100 + index, text=f"{index}:".ljust(8000, "x"))
        context = tuple(c.state.events.values())[-6:]
        assert c.observe_committed_event(event)
        assert c.apply_semantic_observation(observation(event, sequence=index + 1, context=context))
    assert sum(len(e.text.encode()) for e in c.state.events.values()) <= 256 * 1024
    assert c.state.candidates
    store = SnapshotStore(tmp_path / "controller.sqlite3")
    try:
        store.save(c.state, expected_revision=0)
        _, restored = store.load(SCOPE)
        assert restored.scope == SCOPE
        assert restored.seen == c.state.seen
    finally:
        store.close()


@pytest.mark.parametrize("outcome", ["accepted", "completed"])
def test_time_window_compaction_preserves_decay_and_does_not_replay_old_effects(outcome):
    c = controller()
    proposal = propose(c, source())
    effect = Effect(effect_id="expression", kind="message", at=105, actual_targets=("A",))
    assert c.observe_run_feedback(feedback(proposal, outcome=outcome, effects=(effect,)))
    initial = c.belief("topic", "A", 105)

    for now in (706, 900, 1400):
        c.advance(now, controller_epoch=0, host_available=False)
        assert c.belief("topic", "A", now) == pytest.approx(dynamics.decay(initial, now - 105))
        assert c.belief("topic", "A", now)[4] == 0
        assert not c.state.observations
    if outcome == "accepted":
        assert effect.effect_id in c.state.effects
        assert c.state.feedback["run"].outcome == "accepted"


def test_capacity_compaction_preserves_belief_with_one_monotonic_time_boundary():
    c = controller()
    event = source(text="x" * 12000)
    assert c.observe_committed_event(event)
    score = observation(event)
    assert c.apply_semantic_observation(score)
    initial = c.belief("topic", "A", 100)
    for index in range(1, 24):
        assert c.observe_committed_event(
            source(str(index), at=100 + index, text=f"{index}:".ljust(12000, "x"))
        )

    assert c.state.replay_after >= 100
    assert all(e.at > c.state.replay_after for e in c.state.events.values())
    assert c.belief("topic", "A", 130) == pytest.approx(dynamics.decay(initial, 30))
    assert not c.apply_semantic_observation(score.model_copy(update={"observation_id": "late"}))
    assert not c.observe_committed_event(source("previously-unseen-but-too-old", at=100))


def test_capacity_compaction_keeps_equal_timestamp_actions_on_the_same_side():
    c = controller()
    event = source(text="x" * 12000)
    assert c.observe_committed_event(event)
    assert c.apply_semantic_observation(observation(event))
    initial = c.belief("topic", "A", 100)
    for index in range(1, 22):
        assert c.observe_committed_event(source(str(index), text=f"{index}:".ljust(12000, "x")))

    assert c.state.replay_after == 100
    assert not c.state.events
    assert c.belief("topic", "A", 110) == pytest.approx(dynamics.decay(initial, 10))


def test_withdrawing_retired_evidence_clears_unreconstructable_derived_baseline():
    c = controller()
    event, anchor = source(), source("anchor", at=99)
    c.observe_committed_event(anchor)
    c.observe_committed_event(event)
    c.apply_semantic_observation(observation(event, context=(anchor,)))
    c.advance(706, controller_epoch=0, host_available=False)
    assert c.belief("topic", "A", 706)[1] > 0

    c.observe_source_change(anchor.ref)

    assert c.belief("topic", "A", 706) == dynamics.IDLE
    assert c.state.baseline_invalidations == 1


def test_baselines_have_an_explicit_sixty_four_unit_capacity():
    c = controller()
    for index in range(65):
        event = source(str(index), at=100 + index, target=f"person-{index}")
        assert c.observe_committed_event(event)
        assert c.apply_semantic_observation(observation(event, sequence=index + 1))

    c.advance(1000, controller_epoch=0, host_available=False)

    assert len(c.state.belief_baselines) == 64
    assert c.state.baseline_evictions == 1
    assert c.belief("topic", "person-64", 1000)[1] > 0
    assert c.belief("topic", "person-0", 1000) == dynamics.IDLE
    c.advance(1001, controller_epoch=0, host_available=False)
    assert c.state.baseline_evictions == 1


def test_restore_does_not_mutate_the_supplied_snapshot_during_compaction():
    c = controller()
    event = source()
    c.observe_committed_event(event)
    c.apply_semantic_observation(observation(event))
    snapshot = c.state
    original = snapshot.model_dump_json()

    restored = Controller.restore(snapshot, 706)

    assert snapshot.model_dump_json() == original
    assert event.ref.event_id in snapshot.events
    assert not restored.state.events
    assert restored.belief("topic", "A", 706)[1] > 0


def test_new_observation_after_compaction_starts_from_the_decayed_baseline():
    c = controller()
    event = source()
    c.observe_committed_event(event)
    c.apply_semantic_observation(observation(event))
    initial = c.belief("topic", "A", 100)
    c.advance(706, controller_epoch=0, host_available=False)
    fresh = source("new-invitation", at=706)
    c.observe_committed_event(fresh)
    c.apply_semantic_observation(observation(fresh, sequence=2))

    expected = dynamics.observe(
        dynamics.decay(initial, 606),
        choice("interaction_mark", "invite_yuki"),
        matched_self=False,
    )
    assert c.belief("topic", "A", 706) == pytest.approx(expected)


def test_retiring_original_self_anchor_does_not_undo_still_retained_reciprocity():
    c = controller()
    anchor = source("self-anchor", at=100, kind="self")
    human = source("human-reply", at=110, reply_to=anchor.ref)
    c.observe_committed_event(anchor)
    c.observe_committed_event(human)
    c.apply_semantic_observation(observation(human, act="extend_yuki", context=(anchor,)))
    initial = c.belief("topic", "A", 110)
    assert initial[3] > 0

    c.advance(706, controller_epoch=0, host_available=False)

    assert anchor.ref.event_id not in c.state.events
    assert human.ref.event_id in c.state.events
    assert c.belief("topic", "A", 706) == pytest.approx(dynamics.decay(initial, 596))
    c.observe_source_change(anchor.ref)
    assert c.belief("topic", "A", 706) == dynamics.IDLE


def mixed_choice(dimension, probabilities):
    full = {key: probabilities.get(key, 0.0) for key in CRITERIA[dimension]}
    return Choice(choice=max(full, key=full.get), probabilities=full)


@pytest.mark.parametrize(
    "dimension, probabilities",
    [
        ("information_state", {"unknown": 0.42, "new": 0.30, "refine": 0.28}),
        ("information_state", {"unknown": 0.4, "new": 0.4, "refine": 0.2}),
        ("floor_state", {"unknown": 0.42, "yuki": 0.35, "open": 0.23}),
    ],
)
def test_unknown_dominant_or_tied_dimension_does_not_create_a_determinate_estimate(
    dimension,
    probabilities,
):
    c = controller()
    event = source()
    c.observe_committed_event(event)
    score = observation(event)
    answers = {**score.answers, dimension: mixed_choice(dimension, probabilities)}
    assert c.apply_semantic_observation(score.model_copy(update={"answers": answers}))
    assert c.state.observations[event.ref.event_id].answers[dimension] == answers[dimension]
    if dimension == "information_state":
        # Topic novelty is irrelevant to a clear invitation addressed to Yuki.
        assert event.ref.event_id in c.state.candidates
    else:
        assert not c.state.candidates


@pytest.mark.parametrize(
    "dimension, probabilities",
    [
        ("interaction_mark", {"unknown": 0.42, "close_topic": 0.30, "ask_yuki_stop": 0.28}),
        ("boundary_scope", {"unknown": 0.42, "target_thread": 0.30, "group_thread": 0.28}),
    ],
)
def test_unknown_winning_act_or_scope_cannot_create_a_hard_stop(dimension, probabilities):
    c = controller()
    event = source()
    c.observe_committed_event(event)
    score = observation(event, act="ask_yuki_stop")
    answers = {**score.answers, dimension: mixed_choice(dimension, probabilities)}
    assert c.apply_semantic_observation(score.model_copy(update={"answers": answers}))
    assert not c.state.boundaries
    if dimension == "interaction_mark":
        assert c.belief("topic", "A", 100)[4] == pytest.approx(0.8 * 0.58)


def test_known_winning_dimension_preserves_full_distribution_without_renormalization():
    c = controller()
    event = source()
    c.observe_committed_event(event)
    score = observation(event)
    info = mixed_choice("information_state", {"new": 0.55, "refine": 0.25, "unknown": 0.20})
    assert c.apply_semantic_observation(
        score.model_copy(
            update={
                "answers": {**score.answers, "information_state": info},
            }
        )
    )
    assert c.state.candidates[event.ref.event_id].value.value == pytest.approx(0.875)


def test_conflicting_effect_ids_within_one_receipt_are_rejected_atomically():
    c = controller()
    proposal = propose(c, source())
    message = Effect(effect_id="same", kind="message", at=105, actual_targets=("A",))
    tool = message.model_copy(update={"kind": "tool"})
    before = c.state.model_dump_json()

    assert not c.observe_run_feedback(
        feedback(proposal, outcome="completed", effects=(message, tool)),
    )

    assert c.state.model_dump_json() == before


def test_identical_effect_ids_within_one_receipt_count_once():
    c = controller()
    proposal = propose(c, source())
    message = Effect(effect_id="same", kind="message", at=105, actual_targets=("A",))
    assert c.observe_run_feedback(
        feedback(proposal, outcome="completed", effects=(message, message)),
    )
    assert len(c.state.effects) == 1
    assert len(c.state.work_pulses) == 1


def test_activity_survives_source_capacity_compaction_and_restart():
    c = controller()
    for index in range(21):
        assert c.observe_committed_event(source(str(index), text=f"{index}:".ljust(12000, "x")))
    before = c._activity(100)
    assert c.observe_committed_event(source("capacity-overflow", text="x" * 12000))

    expected = 1 - 0.9**22
    assert c._activity(100) == pytest.approx(expected)
    assert c._activity(100) > before
    assert not c.state.events
    restored = Controller.restore(c.state, 110)
    assert restored._activity(110) == pytest.approx(expected * math.exp(-10 / 60))
    restored.advance(1000, controller_epoch=0, host_available=False)
    assert restored._activity(1000) == pytest.approx(expected * math.exp(-900 / 60))


def test_compute_trace_survives_receipt_compaction_and_restart():
    c = controller()
    proposal = propose(c, source())
    kind, tau = "compute", 180
    effect = Effect(effect_id="cost", kind=kind, at=105, actual_targets=("A",))
    assert c.observe_run_feedback(feedback(proposal, outcome="completed", effects=(effect,)))
    assert c._trace(kind, 105, tau) == 0.25

    c.advance(706, controller_epoch=0, host_available=False)

    assert not c.state.effects
    assert c._trace(kind, 706, tau) == pytest.approx(0.25 * math.exp(-601 / tau))
    restored = Controller.restore(c.state, 1200)
    assert restored._trace(kind, 1200, tau) == pytest.approx(0.25 * math.exp(-1095 / tau))


def test_direct_invitation_is_not_delayed_by_an_unscored_same_unit_fragment():
    c = controller()
    invitation = source()
    assert c.observe_committed_event(invitation)
    assert c.apply_semantic_observation(observation(invitation))
    assert c.opportunity_scores(105)[invitation.ref.event_id] > 0
    for kind in ("self", "seed"):
        assert c.observe_committed_event(source(kind, at=105, kind=kind))
        assert c.opportunity_scores(105)[invitation.ref.event_id] > 0
    assert c.observe_committed_event(source("other-thread", at=105, thread="different"))
    assert c.observe_committed_event(source("other-target", at=105, target="B"))
    assert not c.observe_committed_event(
        source(
            "foreign-scope",
            at=105,
            scope=Scope(conversation_id="other-group", generation=1),
        )
    )
    assert c.opportunity_scores(105)[invitation.ref.event_id] > 0

    continuation = source("same-unit-fragment", at=105, reply_to=invitation.ref)
    assert c.observe_committed_event(continuation)

    assert c.opportunity_scores(105)[invitation.ref.event_id] > 0
    assert c.opportunity_scores(106)[invitation.ref.event_id] > 0
    c.observe_source_change(continuation.ref)
    assert c.opportunity_scores(105)[invitation.ref.event_id] > 0


def queue_observation(session, source):
    session.controller.observe_committed_event(source)
    session.request_observation(source.ref)
