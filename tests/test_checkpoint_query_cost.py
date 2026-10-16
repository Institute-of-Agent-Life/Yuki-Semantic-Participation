"""Synthetic operation counts, not wall-clock thresholds or production acceptance."""

from unittest.mock import patch

import pytest

from tests.test_controller import SCOPE, event, observation
from tests.test_participation_continuation import admitted, binding, report
from yuki_participation import dynamics
from yuki_participation.controller import Controller, RecordedEffect
from yuki_participation.models import Effect, Proposal, Snapshot
from yuki_participation.participation import CHECKPOINT_KEY, ParticipationCheckpoint, UnitState


class CountedFacts(dict):
    """Count visited facts, including scans hidden behind ordinary dict views."""

    def __init__(self, source):
        super().__init__(source)
        self.visits = 0

    def values(self):
        self.visits += len(self)
        return super().values()

    def items(self):
        self.visits += len(self)
        return super().items()


def population(size):
    c = Controller(SCOPE, 100)
    originals = []
    for index in range(size):
        source = event(f"origin-{index}").model_copy(
            update={
                "thread": f"topic-{index}",
                "author": f"person-{index}",
                "target": f"person-{index}",
            }
        )
        assert c.observe_committed_event(source)
        assert c.observe_unit_input(binding(source), source.ref)
        assert c.observe_unit_expression(
            binding(source),
            f"run-{index}",
            Effect(effect_id=f"sent-{index}", kind="message", at=101, actual_targets=("group",)),
        )
        originals.append(source)
    c.advance(704, controller_epoch=0, host_available=False)
    assert not c.state.events
    assert len(c.participating_units(704)) == size
    for index, origin in enumerate(originals):
        focus = origin.model_copy(
            update={"ref": origin.ref.model_copy(update={"event_id": f"focus-{index}"}), "at": 705}
        )
        assert c.observe_committed_event(focus)
        assert c.apply_semantic_observation(observation(focus, act="open_group", floor="open"))
    return c, originals


@pytest.mark.parametrize("size", [8, 32])
def test_actor_identity_sources_survive_retirement_without_deriving_participation(size):
    c, _ = population(size)
    c.advance(1310, controller_epoch=0, host_available=False)
    assert not c.state.events
    before = c.state.model_dump_json()
    with (
        patch.object(c, "participating_units", side_effect=AssertionError("derived participation")),
        patch.object(c, "_current_units", side_effect=AssertionError("current identity view")),
        patch.object(c, "_belief_actions", side_effect=AssertionError("global action replay")),
        patch.object(UnitState, "model_validate", side_effect=AssertionError("unit reparse")),
        patch.object(
            ParticipationCheckpoint, "model_validate", wraps=ParticipationCheckpoint.model_validate
        ) as decode,
    ):
        assert c.participation_actor_ids() == frozenset(f"person-{index}" for index in range(size))
        assert decode.call_count == 1
    assert c.state.model_dump_json() == before


@pytest.mark.parametrize("size", [8, 32])
def test_participation_and_freeze_decode_once_without_computing_belief(size):
    c, _ = population(size)
    focus = c.state.events["focus-0"]
    before = c.state.model_dump_json()
    with (
        patch.object(c, "belief", side_effect=AssertionError("unused belief calculation")),
        patch.object(c, "_belief_actions", side_effect=AssertionError("unused action replay")),
        patch.object(UnitState, "model_validate", side_effect=AssertionError("unit reparse")),
        patch.object(
            ParticipationCheckpoint, "model_validate", wraps=ParticipationCheckpoint.model_validate
        ) as decode,
    ):
        units = c.participating_units(706)
        assert len(units) == size and all(u.expressed for u in units)
        assert "belief" not in units[0].model_dump()
        assert decode.call_count == 1
        decode.reset_mock()
        frozen = c.freeze_observation(
            Snapshot(scope=SCOPE, focus=focus, context=(), sequence=1, issued_at=706)
        )
        assert frozen.unit_reviews
        assert decode.call_count == 1
    assert c.state.model_dump_json() == before


@pytest.mark.parametrize("size", [8, 32])
def test_nonempty_autonomy_scores_and_boundary_fold_scan_facts_once(size):
    c, _ = population(size)
    expected = {key: c.belief(e.thread, e.target, 706) for key, e in c.state.events.items()}
    facts = {
        name: CountedFacts(getattr(c.state, name)) for name in ("events", "observations", "effects")
    }
    c.state = c.state.model_copy(update=facts)
    with (
        patch.object(c, "_belief_actions", wraps=c._belief_actions) as actions,
        patch.object(UnitState, "model_validate", side_effect=AssertionError("unit reparse")),
        patch.object(
            ParticipationCheckpoint, "model_validate", wraps=ParticipationCheckpoint.model_validate
        ) as decode,
    ):
        scores = c.opportunity_scores(706)
        assert len(scores) == size and all(value > 0 for value in scores.values())
        assert actions.call_count == decode.call_count == 1
        assert facts["observations"].visits == size
        assert facts["effects"].visits <= 2 * size
        assert facts["events"].visits <= 3 * size
        for value in facts.values():
            value.visits = 0
        decode.reset_mock()
        actions.reset_mock()
        c._advance_replay_boundary(706)
        assert actions.call_count == decode.call_count == 1
        assert facts["observations"].visits == 2 * size
        assert facts["effects"].visits <= 3 * size
        for key, source in c.state.events.items():
            assert c.belief(source.thread, source.target, 706) == pytest.approx(expected[key])


def test_query_indices_do_not_outlive_quiet_or_source_withdrawal():
    c, originals = population(2)
    focus = c.state.events["focus-0"]
    assert c.participation_view(focus, 706).matched_unit is not None
    assert c.observe_unit_hint(binding(originals[0]), report("quiet", at=706))
    assert c.participation_view(focus, 706).matched_unit is None
    assert c.participating_units(706)[0].engage == "quiet"
    c.observe_source_change(originals[0].ref)
    assert all(u.unit.thread != originals[0].thread for u in c.participating_units(707))
    assert c.belief(originals[0].thread, originals[0].target, 707) == dynamics.IDLE


def test_admission_and_hint_validate_the_checkpoint_once_including_pruning():
    c, originals = population(2)
    focus = c.state.events["focus-0"]
    with patch.object(
        ParticipationCheckpoint, "model_validate", wraps=ParticipationCheckpoint.model_validate
    ) as decode:
        assert c.observe_unit_input(binding(originals[0]), focus.ref, continuation=True)
        assert decode.call_count == 1
        decode.reset_mock()
        assert c.observe_unit_hint(binding(originals[0]), report("quiet", at=706))
        assert decode.call_count == 1


def test_extended_namespace_keeps_existing_single_unit_read_and_refuses_writes():
    c, origin, unit = admitted()
    c._set(now=800)
    c._prune()
    speech = event("recent-self", at=800, kind="self").model_copy(update={"author": "SELF"})
    assert c.observe_committed_event(speech)
    assert c.observe_public_anchor("real-run", speech.ref.event_id, 800)
    c.state.proposals["accepted-proposal"] = Proposal(
        proposal_id="accepted-proposal",
        scope=SCOPE,
        controller_epoch=0,
        kind="conversation",
        thread="topic",
        target_hint="A",
        sources=(speech.ref,),
        created_at=800,
        expires_at=850,
    )
    effect = Effect(effect_id="actual-send", kind="message", at=800, actual_targets=("A",))
    c.state.effects[effect.effect_id] = RecordedEffect(
        proposal_id="accepted-proposal", run_ref="real-run", effect=effect
    )
    focus = event("focus", at=801)
    assert c.observe_committed_event(focus)
    c.state.host_checkpoint[CHECKPOINT_KEY]["future_extension"] = True
    before = c.state.model_dump_json()
    assert c.participation_view(focus, 801).matched_unit is not None
    assert c.continuation_current(focus.ref, unit, 801)
    assert not c.observe_unit_input(unit, focus.ref)
    assert not c.observe_unit_hint(unit, report("stay", at=801))
    assert not c.observe_unit_expression(unit, "real-run", effect)
    c._prune_participation()
    assert c.state.model_dump_json() == before
    # The original identity/revision proof is still required by that single read.
    c.state.invalidated[origin.ref.event_id] = origin.ref.revision
    assert not c.continuation_current(focus.ref, unit, 801)
