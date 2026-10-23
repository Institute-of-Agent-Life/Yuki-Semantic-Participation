"""Expired controller inputs are disposable; real execution receipts stay in the Host."""

from unittest.mock import patch

import pytest

from tests.test_checkpoint_query_cost import population
from tests.test_controller import SCOPE
from tests.test_participation_continuation import report
from yuki_participation.controller import Controller
from yuki_participation.models import Effect, Feedback, Proposal
from yuki_participation.participation import CHECKPOINT_KEY, ParticipationCheckpoint


def retired_population(size):
    controller, _ = population(size)
    controller.advance(1310, controller_epoch=0, host_available=False)
    checkpoint = ParticipationCheckpoint.model_validate(
        controller.state.host_checkpoint[CHECKPOINT_KEY]
    )
    assert len(checkpoint.units) == size
    assert all(unit.retired_refs for unit in checkpoint.units.values())
    assert not controller.state.events
    return controller


@pytest.mark.parametrize("size", [8, 32, 64])
@pytest.mark.parametrize("receipt_count", [3, 64])
def test_expired_compute_replay_does_not_rebuild_or_prune_checkpoint(size, receipt_count):
    controller = retired_population(size)
    before = controller.state.model_dump()
    effects = tuple(
        Effect(effect_id=f"expired-compute-{index}", kind="compute", at=100)
        for index in range(receipt_count)
    )
    with (
        patch.object(controller, "_prune", wraps=controller._prune) as prune,
        patch.object(
            controller,
            "_read_participation_checkpoint",
            wraps=controller._read_participation_checkpoint,
        ) as read,
        patch.object(
            controller, "_save_participation", wraps=controller._save_participation
        ) as save,
    ):
        for _ in range(3):
            for effect in effects:
                assert not controller.observe_committed_effect("original-run", effect)
        assert prune.call_count == read.call_count == save.call_count == 0
    assert controller.state.model_dump() == before


def test_expired_message_preserves_last_self_fact_without_restoring_short_term_effect():
    controller = retired_population(8)
    effect = Effect(effect_id="late-expired-send", kind="message", at=600)
    assert effect.at < controller.state.replay_after
    with patch.object(controller, "_prune", wraps=controller._prune) as prune:
        assert not controller.observe_committed_effect("old-run", effect)
        assert controller.state.last_self_message_at == 600
        assert not controller.observe_committed_effect("old-run", effect)
        assert prune.call_count == 0
    assert effect.effect_id not in controller.state.effects
    assert not controller.state.feedback and not controller.state.proposals
    assert not controller.observe_self_report(report("quiet", run="old-run", at=601))


@pytest.mark.parametrize("kind", ["compute", "message"])
def test_recent_late_effect_for_old_run_is_recorded_once_with_original_identity(kind):
    controller = retired_population(8)
    # Arrival belongs to an old run, but the confirmed execution itself is recent.
    effect = Effect(effect_id="late-recent-effect", kind=kind, at=1300)
    with (
        patch.object(controller, "_prune", wraps=controller._prune) as prune,
        patch.object(
            controller,
            "_read_participation_checkpoint",
            wraps=controller._read_participation_checkpoint,
        ) as read,
        patch.object(
            controller, "_save_participation", wraps=controller._save_participation
        ) as save,
    ):
        assert controller.observe_committed_effect("old-run", effect)
        assert prune.call_count == read.call_count == save.call_count == 0
        controller.advance(1311, controller_epoch=0, host_available=False)
        assert prune.call_count == read.call_count == 1
        assert save.call_count == 0
    assert controller.state.effects[effect.effect_id].effect == effect
    assert controller.state.effects[effect.effect_id].run_ref == "old-run"
    before = controller.state.model_dump()
    with patch.object(controller, "_prune", wraps=controller._prune) as prune:
        assert controller.observe_committed_effect("old-run", effect)
        assert not controller.observe_committed_effect("other-run", effect)
        assert not controller.observe_committed_effect(
            "old-run", effect.model_copy(update={"at": 1301})
        )
        assert prune.call_count == 0
    assert controller.state.model_dump() == before


def test_retained_expression_identity_precedes_expiration_filter():
    controller = retired_population(8)
    # An original old expression still supports a current participation unit.
    original = controller.state.effects["sent-0"]
    assert original.effect.at < controller.state.replay_after
    before = controller.state.model_dump()
    assert controller.observe_committed_effect(original.run_ref, original.effect)
    assert not controller.observe_committed_effect("other-run", original.effect)
    assert not controller.observe_committed_effect(
        original.run_ref, original.effect.model_copy(update={"at": 102})
    )
    assert controller.state.model_dump() == before


def test_recent_self_report_remains_eligible_without_historical_brake_replay():
    controller = retired_population(8)
    assert controller.observe_committed_effect(
        "original-run", Effect(effect_id="current-compute", kind="compute", at=1300)
    )
    self_report = report("quiet", run="original-run", at=1301)
    assert controller.observe_self_report(self_report)
    assert controller.state.engagement_report == self_report


def test_accepted_feedback_without_final_checkpoint_retires_and_releases_capacity():
    controller = Controller(SCOPE, 100)
    for index in range(257):
        proposal = Proposal(
            proposal_id=f"accepted-{index}",
            scope=SCOPE,
            controller_epoch=0,
            kind="intrinsic",
            thread="topic",
            target_hint="group",
            sources=(),
            created_at=100,
            expires_at=110,
        )
        controller.state.proposals[proposal.proposal_id] = proposal
        assert controller.observe_run_feedback(
            Feedback(
                run_ref=f"original-run-{index}",
                proposal_id=proposal.proposal_id,
                sequence=1,
                outcome="accepted",
                at=100,
                effects=(Effect(effect_id=f"compute-{index}", kind="compute", at=100),),
            )
        )
    controller.advance(101, controller_epoch=0, host_available=False)
    assert controller.state.capacity_blocked
    # A historical snapshot can still have the already-accepted proposal pending.
    controller._set(pending="accepted-0")
    controller.advance(1000, controller_epoch=0, host_available=False)
    assert not controller.state.feedback
    assert not controller.state.proposals
    assert not controller.state.proposal_runs
    assert not controller.state.effects
    assert controller.state.pending is None
    assert not controller.state.capacity_blocked
