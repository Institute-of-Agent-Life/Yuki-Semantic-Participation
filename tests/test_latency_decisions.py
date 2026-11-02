"""Synthetic scoring, nonempty proposal, original receipt and SQLite recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tests.test_controller import SCOPE, controller, event, observation
from yuki_participation.controller import Controller
from yuki_participation.models import Effect, Feedback
from yuki_participation.store import SnapshotStore


def test_scores_and_nonempty_proposal_replay_after_reopen(tmp_path):
    c = controller()
    first = event("event:101", at=100)
    second = event("event:102", at=101)
    invited = event("event:103", at=102).model_copy(update={"thread": "other-topic"})
    for source in (first, second):
        assert c.observe_committed_event(source)
        assert c.apply_semantic_observation(observation(source, act="open_group", floor="open"))
    assert c.observe_committed_event(invited)
    assert c.apply_semantic_observation(observation(invited))
    c.advance(102, controller_epoch=0, host_available=False)
    groups = c._eligible_groups(102)
    group_key = next(key for key, values in groups.items() if len(values) == 2)
    normal = c._opportunity_scores(102, groups, c._autonomy_pressure(102))[group_key]
    members = list(groups[group_key])
    # Deliberately synthetic library-only mixed shape, not a reachable Host claim.
    synthetic_self = members[1].event.model_copy(update={"kind": "self", "author": "SELF"})
    mixed_member = members[1].model_copy(update={"event": synthetic_self})
    mixed_groups = dict(groups)
    mixed_groups[group_key] = [members[0], mixed_member]
    mixed = c._opportunity_scores(102, mixed_groups, c._autonomy_pressure(102))[group_key]
    proposal = c.advance(103, controller_epoch=0, host_available=True)
    assert proposal is not None and proposal.sources == (invited.ref,)
    proposal_facts = proposal.model_dump(mode="json", exclude={"proposal_id"})
    receipt = Feedback(
        run_ref="synthetic-original-run",
        proposal_id=proposal.proposal_id,
        sequence=1,
        outcome="completed",
        at=104,
        considered_refs=(invited.ref,),
        effects=(
            Effect(
                effect_id="synthetic-original-effect",
                kind="message",
                at=104,
                actual_targets=("group",),
            ),
        ),
    )
    assert c.observe_run_feedback(receipt)
    assert not c.observe_run_feedback(receipt)
    store = SnapshotStore(tmp_path / "decisions.sqlite3")
    try:
        assert store.save(c.state, expected_revision=0) == 1
        _, saved = store.load(SCOPE)
    finally:
        store.close()
    restored = Controller.restore(saved, 105)
    assert restored.state.feedback[receipt.run_ref].outcome == "completed"
    assert restored.state.effects["synthetic-original-effect"].effect == receipt.effects[0]
    assert restored.state.proposal_runs[proposal.proposal_id] == receipt.run_ref
    directory = Path(os.environ.get("YUKI_LATENCY_ARTIFACT_DIR", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "boundary": (
            "real library Controller/SQLite; synthetic observations/effects; "
            "mixed self shape directly constructed, not Host"
        ),
        "two_human_score": normal,
        "human_plus_self_score": mixed,
        "proposal_facts_excluding_new_uuid": proposal_facts,
        "original_receipt_replay_idempotent": True,
        "original_run_and_effect_recovered": True,
    }
    (directory / "decision-replay.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )
