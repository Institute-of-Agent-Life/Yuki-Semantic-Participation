"""Synthetic V6 invariants, not real Jev or QQ acceptance."""

import hashlib
import math
import random

import pytest

from yuki_participation import dynamics
from yuki_participation.controller import Controller, State
from yuki_participation.models import (
    Choice,
    Effect,
    Feedback,
    HostUnitOption,
    Observation,
    Scope,
    ScopedEvent,
    Snapshot,
    SourceRef,
)
from yuki_participation.rubric import CRITERIA, REVISION
from yuki_participation.store import SnapshotConflict, SnapshotStore

SCOPE = Scope(conversation_id="group-test", generation=1)


def event(key="e1", at=100, **kwargs):
    return ScopedEvent(
        scope=SCOPE,
        ref=SourceRef(event_id=key, revision=1),
        at=at,
        thread="topic",
        author="A",
        target="A",
        text=f"合成材料 {hashlib.sha256(key.encode()).hexdigest()[:8]}",
        **kwargs,
    )


def answer(dimension, option):
    return Choice(choice=option, probabilities={k: float(k == option) for k in CRITERIA[dimension]})


def observation(source, *, sequence=1, act="invite_yuki", info="refine", floor="yuki", context=()):
    return Observation(
        observation_id=f"observation-{source.ref.event_id}-{sequence}",
        snapshot=Snapshot(
            scope=source.scope,
            focus=source,
            context=context,
            sequence=sequence,
            issued_at=source.at,
        ),
        provider="synthetic_fixture",
        model_revision="fixture",
        rubric_revision=REVISION,
        received_at=source.at,
        answers={
            "interaction_mark": answer("interaction_mark", act),
            "information_state": answer("information_state", info),
            "floor_state": answer("floor_state", floor),
            "boundary_scope": answer("boundary_scope", "target_thread"),
        },
    )


def controller():
    return Controller(SCOPE, 100)


def test_time_and_self_output_never_manufacture_closure_or_reciprocity():
    b = dynamics.self_expression(dynamics.IDLE)
    for _ in range(20):
        b = dynamics.self_expression(dynamics.decay(b, 10))
    assert b[3] == b[4] == 0
    assert sum(b) == pytest.approx(1)
    assert dynamics.decay((0, 0, 0, 1, 0), 500)[4] == 0


@pytest.mark.parametrize("act", ["unknown", "acknowledge", "other_exchange", "off_focus"])
def test_ack_unknown_and_unrelated_do_not_close(act):
    assert (
        dynamics.observe((0, 0, 0, 1, 0), answer("interaction_mark", act), matched_self=False)[4]
        == 0
    )


def test_duplicate_rescore_and_late_scores_are_source_replacements():
    c = controller()
    e = event()
    c.observe_committed_event(e)
    first = observation(e)
    assert c.apply_semantic_observation(first)
    before = c.belief("topic", "A", 100)
    assert not c.apply_semantic_observation(first)
    assert c.belief("topic", "A", 100) == before
    assert c.apply_semantic_observation(observation(e, sequence=2, act="close_topic"))
    assert not c.apply_semantic_observation(first)
    assert c.belief("topic", "A", 100)[4] == pytest.approx(0.8)
    assert c.belief("topic", "A", 100)[1] == 0


def test_scope_revision_context_and_generation_fences():
    c = controller()
    e = event()
    anchor = event("anchor", at=99)
    c.observe_committed_event(e)
    c.observe_committed_event(anchor)
    c.observe_source_change(anchor.ref)
    assert not c.apply_semantic_observation(observation(e, context=(anchor,)))
    foreign = e.model_copy(update={"scope": Scope(conversation_id="other", generation=1)})
    assert not c.observe_committed_event(foreign)
    assert not c.apply_semantic_observation(observation(foreign))
    c.observe_source_change(e.ref)
    assert not c.apply_semantic_observation(observation(e))


def test_unknown_does_not_create_high_default_floor():
    c = controller()
    e = event()
    c.observe_committed_event(e)
    c.apply_semantic_observation(observation(e, floor="unknown"))
    assert not c.state.candidates
    assert not c.opportunity_scores(110)


def test_addressed_invitation_can_use_host_new_unit_without_topic_selection():
    from yuki_participation.controller import State

    option = HostUnitOption(key="new", thread="topic", target="A")
    e = event(unit_ambiguous=True, unit_options=(option,))
    c = controller()
    assert c.observe_committed_event(e)
    observed = observation(e, info="unknown").model_copy(
        update={
            "answers": {
                **observation(e, info="unknown").answers,
                "unit_selection": Choice(
                    choice="unknown", probabilities={"new": 0.45, "unknown": 0.55}
                ),
            },
        }
    )
    assert c.apply_semantic_observation(observed)
    assert c.state.observations[e.ref.event_id].unit_resolution == "semantic_new"
    proposal = c.advance(100, controller_epoch=0, host_available=True)
    assert proposal is not None and proposal.sources == (e.ref,)
    assert proposal.thread == "topic" and proposal.target_hint == "A"

    # Old durable snapshots remain loadable without reviving the removed gate.
    saved = c.state.model_dump(mode="json")
    saved.update(
        threshold=0.6,
        hazard=0.3,
        intrinsic_base_at=80,
        last_intrinsic_at=99,
        self_reference_at=80,
        last_intrinsic_accepted_at=90,
    )
    saved["trace_baselines"] = {
        "ratio_message": {"at": 100, "value": 0.5},
        "social_context": {"at": 100, "value": 0.2},
    }
    restored = State.model_validate(saved)
    assert "threshold" not in restored.model_fields_set
    assert "hazard" not in restored.model_fields_set
    assert "self_reference_at" not in restored.model_fields_set
    assert "last_intrinsic_accepted_at" not in restored.model_fields_set
    assert "ratio_message" not in restored.trace_baselines
    assert restored.trace_baselines["social_context"].value == 0.2


def test_name_priority_cannot_turn_non_invitation_into_a_proposal():
    e = event(observation_priority=True)
    c = controller()
    c.observe_committed_event(e)
    assert c.apply_semantic_observation(observation(e, act="other_exchange"))
    assert c.advance(100, controller_epoch=0, host_available=True) is None


def test_old_snapshot_reconstructs_only_proven_recent_autonomous_work():
    c = controller()
    source = event()
    assert c.observe_committed_event(source)
    assert c.apply_semantic_observation(observation(source))
    proposal = c.advance(105, controller_epoch=0, host_available=True)
    assert proposal is not None
    assert c.observe_run_feedback(
        Feedback(
            run_ref="real-run",
            proposal_id=proposal.proposal_id,
            sequence=1,
            outcome="completed",
            at=106,
            effects=(
                Effect(effect_id="model", kind="compute", at=105),
                Effect(effect_id="send", kind="message", at=106),
            ),
        )
    )
    legacy = c.state.model_dump(mode="json")
    legacy.pop("work_pulses")
    restored = State.model_validate(legacy)
    assert restored.work_pulses["real-run"].started_at == 105
    assert restored.work_pulses["real-run"].public_at == 106


def test_observation_gets_fresh_window_but_cannot_resurrect_stale_invitation():
    e = event()
    c = controller()
    c.observe_committed_event(e)
    fresh = observation(e).model_copy(update={"received_at": 120})
    assert c.apply_semantic_observation(fresh)
    assert c.state.candidates[e.ref.event_id].support.valid_until == 195
    assert c.advance(120, controller_epoch=0, host_available=True) is not None

    old = controller()
    old.observe_committed_event(e)
    stale = observation(e).model_copy(update={"received_at": 191})
    assert old.apply_semantic_observation(stale)
    assert not old.state.candidates


def test_closed_boundary_survives_window_and_silence():
    c = controller()
    e = event()
    c.observe_committed_event(e)
    c.apply_semantic_observation(observation(e, act="ask_yuki_stop"))
    c.advance(1000, controller_epoch=0, host_available=False)
    fresh = event("new", at=1000)
    c.observe_committed_event(fresh)
    c.apply_semantic_observation(observation(fresh, act="open_group"))
    assert not c.opportunity_scores(1001)
    assert c.state.boundaries


def test_released_boundaries_do_not_block_a_new_invitation():
    c = controller()
    c.advance(101, controller_epoch=0, host_available=False)
    for index in range(257):
        stop = event(f"stop-{index}", at=100 + index / 1000)
        assert c.observe_committed_event(stop)
        assert c.apply_semantic_observation(observation(stop, act="ask_yuki_stop"))

    reopening = event("reopening", at=100.5)
    assert c.observe_committed_event(reopening)
    assert c.apply_semantic_observation(observation(reopening))
    assert len(c.state.boundaries) == 257
    assert all(boundary.released_by == reopening.ref for boundary in c.state.boundaries.values())
    assert reopening.ref.event_id in c.state.candidates

    c.advance(102, controller_epoch=0, host_available=False)
    legacy = c.state.model_dump(mode="json")
    legacy["capacity_blocked"] = True
    restored = Controller.restore(State.model_validate(legacy), 102)
    proposal = c.advance(103, controller_epoch=0, host_available=True)
    assert proposal is not None and proposal.sources == (reopening.ref,)
    resumed = restored.advance(103, controller_epoch=0, host_available=True)
    assert resumed is not None and resumed.sources == (reopening.ref,)


def test_support_expires_without_self_renewal():
    c = controller()
    e = event()
    c.observe_committed_event(e)
    c.apply_semantic_observation(observation(e))
    c.advance(195, controller_epoch=0, host_available=True)
    assert not c.opportunity_scores(195)


def test_proposal_consumption_survives_feedback_replay_and_restart(tmp_path):
    c = controller()
    e = event()
    c.observe_committed_event(e)
    c.apply_semantic_observation(observation(e))
    p = c.advance(105, controller_epoch=0, host_available=True)
    assert p is not None
    feedback = Feedback(
        run_ref="run1",
        proposal_id=p.proposal_id,
        sequence=1,
        outcome="accepted",
        at=105,
        considered_refs=(e.ref,),
    )
    assert c.observe_run_feedback(feedback)
    assert not c.observe_run_feedback(feedback)
    assert not c.opportunity_scores(106)
    store = SnapshotStore(tmp_path / "controller.sqlite3")
    revision = store.save(c.state, expected_revision=0)
    with pytest.raises(SnapshotConflict):
        store.save(c.state, expected_revision=0)
    _, saved = store.load(SCOPE)
    resumed = Controller.restore(saved, 110)
    assert resumed.state.pending is None
    assert "threshold" not in resumed.state.model_fields_set
    assert not resumed.advance(111, controller_epoch=0, host_available=True)
    assert store.save(resumed.state, expected_revision=revision) == 2
    store.close()


def test_accepted_proposal_releases_pending_but_retains_real_run_after_restart():
    c = controller()
    first = event()
    c.observe_committed_event(first)
    c.apply_semantic_observation(observation(first))
    proposal = c.advance(105, controller_epoch=0, host_available=True)
    assert proposal is not None
    accepted = Feedback(
        run_ref="retained-run",
        proposal_id=proposal.proposal_id,
        sequence=1,
        outcome="accepted",
        at=105,
        considered_refs=(first.ref,),
    )
    assert c.observe_run_feedback(accepted)
    assert c.state.pending is None
    old = c.state.model_copy(update={"pending": proposal.proposal_id}, deep=True)
    restored = Controller.restore(old, 106)
    assert restored.state.pending is None
    assert restored.state.proposal_runs[proposal.proposal_id] == "retained-run"
    second = event("second", at=107)
    restored.observe_committed_event(second)
    restored.apply_semantic_observation(observation(second))
    next_proposal = restored.advance(108, controller_epoch=0, host_available=True)
    assert next_proposal is not None
    assert next_proposal.proposal_id != proposal.proposal_id
    assert restored.observe_run_feedback(
        accepted.model_copy(update={"sequence": 2, "outcome": "completed", "at": 109})
    )
    assert restored.state.feedback["retained-run"].outcome == "completed"
    assert restored.state.pending == next_proposal.proposal_id


def test_disabled_time_and_epoch_change_cannot_create_stale_proposal():
    c = controller()
    e = event()
    c.observe_committed_event(e)
    c.apply_semantic_observation(observation(e))
    c.advance(105, controller_epoch=0, host_available=False)
    assert not c.state.pending
    c.advance(106, controller_epoch=1, host_available=True)
    assert not c.state.pending
    c.advance(1000, controller_epoch=1, host_available=True)
    assert c.state.skipped_seconds > 0
    assert not c.state.pending


def test_bounded_equations_on_random_simplex():
    rng = random.Random(20260920)
    for _ in range(100):
        raw = [rng.random() for _ in range(5)]
        b = tuple(x / sum(raw) for x in raw)
        for option in CRITERIA["interaction_mark"]:
            updated = dynamics.observe(
                dynamics.decay(b, rng.random() * 300),
                answer("interaction_mark", option),
                matched_self=True,
            )
            assert sum(updated) == pytest.approx(1)
            assert all(0 <= x <= 1 for x in updated)
        assert 0 <= dynamics.attention(rng.random(), rng.random(), 10) <= 1
        assert math.isfinite(sum(b))


def test_snapshot_accepts_large_state(tmp_path):
    c = controller()
    c.set_state_entry("host_checkpoint", "large", {"text": "x" * (1024 * 1024 + 1)})
    store = SnapshotStore(tmp_path / "large.sqlite3")
    assert store.save(c.state, expected_revision=0) == 1
    assert store.load(c.state.scope)[1].model_dump_json() == c.state.model_dump_json()
    store.close()
