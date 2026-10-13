"""Kinds are filtered before grouping/rates; disabled time never accumulates."""

import pytest
from test_core_review_regressions import controller, observation, source
from test_v6_completion import score_seed

from yuki_participation.models import CandidateKind


def candidates():
    c = controller()
    event = source("conversation", at=100)
    c.observe_committed_event(event)
    assert c.apply_semantic_observation(observation(event))
    score_seed(c, "recall", 101, kind=CandidateKind.RECALL)
    score_seed(c, "contact", 102, kind=CandidateKind.CONTACT)
    return c


@pytest.mark.parametrize(
    "allowed",
    [
        frozenset(),
        frozenset({CandidateKind.CONVERSATION}),
        frozenset({CandidateKind.RECALL, CandidateKind.CONTACT}),
    ],
)
def test_scores_and_groups_exclude_disabled_sources_before_sampling(allowed):
    c = candidates()
    all_scores = c.opportunity_scores(105)
    assert all_scores
    groups = c._eligible_groups(105, allowed_kinds=allowed)
    scores = c.opportunity_scores(105, allowed_kinds=allowed)
    assert set(groups) == set(scores)
    assert all(c.state.candidates[key].kind in allowed for key in groups)
    assert scores == {
        key: value for key, value in all_scores.items() if c.state.candidates[key].kind in allowed
    }


def test_disabled_advance_keeps_candidates_and_moves_the_original_clock():
    c = candidates()
    original = set(c.state.candidates)
    assert (
        c.advance(
            105,
            controller_epoch=0,
            host_available=True,
            allowed_kinds=frozenset(),
            intrinsic_allowed=False,
        )
        is None
    )
    assert c.state.last_sample_at == 105 and set(c.state.candidates) == original
    assert c.state.epoch == 0
    assert c.opportunity_scores(106, allowed_kinds=None) == c.opportunity_scores(106)
    c.advance(
        106,
        controller_epoch=0,
        host_available=True,
        allowed_kinds=frozenset({CandidateKind.RECALL}),
        intrinsic_allowed=False,
    )
    if c.state.pending:
        assert c.state.proposals[c.state.pending].kind is CandidateKind.RECALL
