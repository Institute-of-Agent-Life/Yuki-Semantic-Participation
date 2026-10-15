"""Synthetic invariants for intrinsic opportunities and implicit SELF continuation."""

import math

import pytest
from test_core_review_regressions import controller, feedback, observation, source

from yuki_participation.controller import Reception, WorkPulse
from yuki_participation.models import CandidateKind, Choice, Effect, HostUnitOption
from yuki_participation.scheduling import ObservationQueue


def _prime_group(c, *, count=35):
    for index in range(count):
        assert c.observe_committed_event(source(f"group-{index}", at=101 + index))
    c.advance(100 + count, controller_epoch=0, host_available=False)


def _next_intrinsic(c):
    sample = c._sample
    c._sample = lambda _sequence, _stream: 0.0
    try:
        for at in range(140, 4002, 2):
            proposal = c.advance(
                at, controller_epoch=0, host_available=True, intrinsic_allowed=True
            )
            if proposal is not None:
                return at, proposal
    finally:
        c._sample = sample
    raise AssertionError("expected an intrinsic opportunity after real human activity")


def test_intrinsic_opportunity_needs_no_semantic_source_and_does_not_repeat_immediately():
    c = controller()
    assert not c.opportunity_scores(2000)
    _prime_group(c)
    at, proposal = _next_intrinsic(c)
    assert proposal.kind is CandidateKind.INTRINSIC
    assert proposal.sources == () and proposal.support is None
    before = c.intrinsic_opportunity(at + 1)
    assert c.observe_run_feedback(
        feedback(
            proposal,
            outcome="no_reply",
            at=at + 1,
            effects=(Effect(effect_id="model-call", kind="compute", at=at + 1),),
        )
    )
    assert c._no_reply(at + 1) > 0
    assert 0 < c.intrinsic_opportunity(at + 2) < before
    assert c._work_density(at + 2, 1800) > 0
    assert (
        c.advance(at + 2, controller_epoch=0, host_available=True, intrinsic_allowed=True) is None
    )


def test_active_group_lull_has_a_tail_while_dormant_group_fades_without_a_floor():
    c = controller()
    assert c.intrinsic_opportunity(2000) == 0
    _prime_group(c)
    assert c._human_activity(500) > c._human_activity(2000) > c._human_activity(86400)
    assert c.intrinsic_opportunity(2000) > 0
    assert c.intrinsic_opportunity(7200) > 0
    assert 0 < c.intrinsic_opportunity(86400) < c.intrinsic_opportunity(7200)
    assert c.intrinsic_opportunity(10 * 86400) < c.intrinsic_opportunity(86400) / 1000


def test_more_real_human_history_increases_the_same_lull_rate():
    quiet, active = controller(), controller()
    assert quiet.observe_committed_event(source("one", at=101))
    _prime_group(active)
    assert active.intrinsic_opportunity(7200) > quiet.intrinsic_opportunity(7200)


def test_human_activity_seed_survives_compaction_without_double_counting():
    c = controller()
    c.initialize_human_activity(1000, ((100.0, 1), (400.0, 1)), 400.0)
    assert c.state.human_activity_initialized
    assert c.state.last_human_at == 400
    expected = sum(math.exp(-(1000 - at) / 172800) for at in (100, 400))
    assert c._human_activity(1000) == pytest.approx(expected)
    assert c.observe_committed_event(source("new", at=1001))
    c.advance(1800, controller_epoch=0, host_available=False)
    restored = type(c).restore(type(c.state).model_validate_json(c.state.model_dump_json()), 1800)
    assert restored._human_activity(1800) == pytest.approx(c._human_activity(1800))
    assert restored.state.last_human_at == 1001


def test_real_self_speech_does_not_count_as_another_autonomous_work():
    c = controller()
    _prime_group(c)
    before = c.intrinsic_opportunity(1900)
    assert before > 0
    c.advance(1900, controller_epoch=0, host_available=False)
    own = source("self-speech", at=1900, kind="self", target="group")
    assert c.observe_committed_event(own)
    assert c.intrinsic_opportunity(1900) == before
    c.advance(2600, controller_epoch=0, host_available=False)
    assert own.ref.event_id not in c.state.events
    assert c._work_density(2600, 1800) == 0


def test_no_reply_feedback_trace_survives_compaction_and_restore():
    c = controller()
    _prime_group(c)
    at, proposal = _next_intrinsic(c)
    assert c.observe_run_feedback(feedback(proposal, outcome="no_reply", at=at))
    c.advance(at + 601, controller_epoch=0, host_available=False)
    assert not c.state.feedback
    retained = c._no_reply(at + 601)
    assert retained > 0
    restored = type(c).restore(c.state, at + 900)
    assert restored._no_reply(at + 900) < retained


def test_one_autonomous_work_has_the_same_future_rate_with_one_or_many_messages():
    c = controller()
    _prime_group(c)
    at, proposal = _next_intrinsic(c)
    single = type(c).restore(c.state, at)
    many = type(c).restore(c.state, at)
    compute = Effect(effect_id="one-model-call", kind="compute", at=at + 1)
    first = Effect(effect_id="first", kind="message", at=at + 1, actual_targets=("group",))
    assert single.observe_run_feedback(
        feedback(proposal, outcome="completed", at=at + 1, effects=(compute, first))
    )
    assert many.observe_run_feedback(
        feedback(
            proposal,
            outcome="completed",
            at=at + 1,
            effects=(
                compute,
                first,
                Effect(effect_id="second", kind="message", at=at + 2, actual_targets=("group",)),
                Effect(effect_id="third", kind="message", at=at + 3, actual_targets=("group",)),
            ),
        )
    )
    assert single._work_density(at + 100, 1800) == many._work_density(at + 100, 1800)
    assert single._exposure(at + 100) == many._exposure(at + 100)
    assert single.intrinsic_opportunity(at + 100) == many.intrinsic_opportunity(at + 100)


def test_real_anchored_reply_reduces_unanswered_exposure():
    c = controller()
    _prime_group(c)
    at, proposal = _next_intrinsic(c)
    assert c.observe_run_feedback(
        feedback(
            proposal,
            outcome="completed",
            at=at + 1,
            effects=(
                Effect(effect_id="model", kind="compute", at=at + 1),
                Effect(effect_id="out", kind="message", at=at + 2, actual_targets=("group",)),
            ),
        )
    )
    outbound = source(
        "outbound-work", at=at + 2, kind="self", thread=proposal.thread, target="group"
    )
    assert c.observe_committed_event(outbound)
    assert c.observe_public_anchor("run", outbound.ref.event_id, at + 2)
    assert outbound.ref.event_id in c.state.outbound_anchors
    exposed = c._exposure(at + 100)
    reply = source("reply-to-work", at=at + 20, thread=proposal.thread, reply_to=outbound.ref)
    assert c.observe_committed_event(reply)
    assert c.apply_semantic_observation(observation(reply, act="extend_yuki", context=(outbound,)))
    assert c.state.observations[reply.ref.event_id].matching_self_anchor == outbound.ref
    assert outbound.ref.event_id in c.state.outbound_anchors
    assert c.state.receptions[reply.ref.event_id].score > 0
    assert c._exposure(at + 100) < exposed
    assert c._reception(at + 100) > 0


def test_unrelated_group_talk_is_only_weak_negative_reception_evidence():
    c = controller()
    _prime_group(c)
    at, proposal = _next_intrinsic(c)
    assert c.observe_run_feedback(
        feedback(
            proposal,
            outcome="completed",
            at=at + 1,
            effects=(Effect(effect_id="sent", kind="message", at=at + 1),),
        )
    )
    unrelated = source("other-people", at=at + 30)
    assert c.observe_committed_event(unrelated)
    assert c.apply_semantic_observation(observation(unrelated, act="other_exchange"))
    assert -0.03 < c._reception(at + 31) < 0
    assert c._exposure(at + 31) > 0


def test_reception_compaction_preserves_rate_without_unbounded_snapshot_growth():
    c = controller()
    c.state.work_pulses["run"] = WorkPulse(started_at=9000, public_at=9000)
    for index in range(3000):
        c.state.receptions[f"reply-{index}"] = Reception(
            run_ref="run", at=9000 + index / 3, score=0.5 if index == 0 else -0.001
        )
    before_r, before_e = c._reception(10000), c._exposure(10000)
    c.advance(10000, controller_epoch=0, host_available=False)
    assert len(c.state.receptions) <= 2048
    assert c._reception(10000) == pytest.approx(before_r)
    assert c._exposure(10000) == pytest.approx(before_e)
    reloaded = type(c.state).model_validate_json(c.state.model_dump_json())
    restored = type(c).restore(reloaded, 10000)
    assert restored._reception(10000) == pytest.approx(before_r)
    assert restored._exposure(10000) == pytest.approx(before_e)


def test_unavailable_time_and_restart_do_not_accumulate_unspent_opportunities():
    c = controller()
    c.advance(10000, controller_epoch=0, host_available=False)
    restored = type(c).restore(c.state, 100000)
    assert restored.state.last_sample_at == 100000
    restored._sample = lambda _sequence, _stream: 0.01
    assert (
        restored.advance(100000.1, controller_epoch=0, host_available=True, intrinsic_allowed=True)
        is None
    )


def test_real_direct_invitation_bypasses_autonomous_work_density():
    c = controller()
    for index in range(20):
        c.state.work_pulses[f"previous-{index}"] = WorkPulse(started_at=100)
    invitation = source("direct-invitation", at=101)
    assert c.observe_committed_event(invitation)
    assert c.apply_semantic_observation(observation(invitation, act="invite_yuki"))
    proposal = c.advance(105, controller_epoch=0, host_available=True, intrinsic_allowed=True)
    assert proposal is not None and proposal.sources == (invitation.ref,)


def test_semantic_selection_of_real_self_anchor_recognizes_unquoted_continuation():
    c = controller()
    anchor = source("outbound", at=100, kind="self", target="group")
    selected = HostUnitOption(key="self", thread="topic", target="A", self_anchor=anchor.ref)
    focus = source(
        "human",
        at=110,
        unit_ambiguous=True,
        unit_options=(HostUnitOption(key="new", thread="new", target="A"), selected),
    )
    assert c.observe_committed_event(anchor)
    assert c.observe_committed_event(focus)
    score = observation(focus, act="extend_yuki", context=(anchor,))
    score = score.model_copy(
        update={
            "answers": {
                **score.answers,
                "unit_selection": Choice(
                    choice="self", probabilities={"new": 0.0, "self": 1.0, "unknown": 0.0}
                ),
            },
            "resolved_unit": selected,
        }
    )
    assert c.apply_semantic_observation(score)
    assert c.state.observations[focus.ref.event_id].matching_self_anchor == anchor.ref
    assert c.belief("topic", "A", 110)[3] > 0


def test_self_option_alone_does_not_claim_a_continuation():
    c = controller()
    anchor = source("outbound", at=100, kind="self", target="group")
    selected = HostUnitOption(key="self", thread="topic", target="A", self_anchor=anchor.ref)
    focus = source(
        "human",
        at=110,
        unit_ambiguous=True,
        unit_options=(HostUnitOption(key="new", thread="new", target="A"), selected),
    )
    assert c.observe_committed_event(anchor)
    assert c.observe_committed_event(focus)
    score = observation(focus, act="other_exchange", context=(anchor,))
    score = score.model_copy(
        update={
            "answers": {
                **score.answers,
                "unit_selection": Choice(
                    choice="self", probabilities={"new": 0.0, "self": 1.0, "unknown": 0.0}
                ),
            },
            "resolved_unit": selected,
        }
    )
    assert c.apply_semantic_observation(score)
    assert c.state.observations[focus.ref.event_id].matching_self_anchor is None


def test_observation_queue_keeps_implicit_self_anchor_in_context():
    c = controller()
    anchor = source("outbound", at=100, kind="self", target="group")
    option = HostUnitOption(key="self", thread="topic", target="A", self_anchor=anchor.ref)
    focus = source(
        "human",
        at=110,
        unit_ambiguous=True,
        unit_options=(HostUnitOption(key="new", thread="new", target="A"), option),
    )
    fillers = tuple(source(f"filler-{index}", at=101 + index) for index in range(7))
    queue = ObservationQueue(c.state.scope)
    queue.offer(focus)
    snapshot = queue.take(140, active=False, context=(anchor, *fillers))
    assert snapshot is not None
    assert anchor in snapshot.context
