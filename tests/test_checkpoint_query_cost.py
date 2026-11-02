"""Synthetic operation counts, not wall-clock thresholds or production acceptance."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import ClassVar
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tests.test_controller import SCOPE, event, observation
from tests.test_participation_continuation import admitted, binding, report
from yuki_participation import dynamics
from yuki_participation.controller import Boundary, Controller, RecordedEffect, SeenSource, State
from yuki_participation.models import Choice, Effect, HostUnitOption, Proposal, Snapshot, SourceRef
from yuki_participation.participation import (
    CHECKPOINT_KEY,
    ParticipationCheckpoint,
    ParticipationUnit,
    UnitBinding,
    UnitState,
)
from yuki_participation.store import SnapshotStore


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


def test_current_invitation_checks_only_the_target_candidate(monkeypatch):
    controller = Controller(SCOPE, 100)
    source = event("invited")
    assert controller.observe_committed_event(source)
    assert controller.apply_semantic_observation(observation(source))
    assert controller.participation_view(source, 101).addressed
    monkeypatch.setattr(
        controller, "_current_units", lambda *_a, **_k: pytest.fail("built unrelated units")
    )
    monkeypatch.setattr(
        controller, "_eligible_groups", lambda *_a, **_k: pytest.fail("scanned all candidates")
    )
    assert controller.invitation_current(source, 101)
    assert not controller.invitation_current(source, 200)
    controller.observe_source_change(source.ref)
    assert not controller.invitation_current(source, 101)


def test_family_invitation_checks_related_boundary_and_quiet_without_global_scan(monkeypatch):
    controller, origin, original = admitted()
    source = event("family-invitation", at=101)
    assert controller.observe_committed_event(source)
    assert controller.apply_semantic_observation(observation(source))
    family = frozenset({"A"})
    assert controller.invitation_current(source, 101)
    assert controller.invitation_current_for_family(source, 101, family=family)
    monkeypatch.setattr(
        controller, "_closed", lambda *_a, **_k: pytest.fail("scanned all boundaries")
    )
    controller.set_state_entry(
        "boundaries",
        "family-stop",
        Boundary(source=source.ref, thread="topic", target="A", group_wide=True),
    )
    assert not controller.invitation_current_for_family(source, 101, family=family)
    controller.remove_state_entry("boundaries", "family-stop")
    assert controller.invitation_current_for_family(source, 101, family=family)
    assert controller.observe_unit_hint(original, report("quiet", at=102))
    assert not controller.invitation_current_for_family(source, 102, family=family)
    assert controller._valid(origin.ref)


def test_family_invitation_does_not_reclassify_a_generic_other_target_invitation():
    controller = Controller(SCOPE, 100)
    option = HostUnitOption(key="other", thread="other-topic", target="B")
    source = event("other-target", unit_ambiguous=True, unit_options=(option,))
    assert controller.observe_committed_event(source)
    original = observation(source)
    observed = original.model_copy(
        update={
            "resolved_unit": option,
            "answers": {
                **original.answers,
                "unit_selection": Choice(
                    choice="other", probabilities={"other": 1.0, "unknown": 0.0}
                ),
            },
        }
    )
    assert controller.apply_semantic_observation(observed)
    assert controller.state.candidates[source.ref.event_id].event.target == "B"
    assert controller.invitation_current(source, 101)
    assert not controller.invitation_current_for_family(source, 101, family=frozenset({"A"}))


@pytest.mark.parametrize("explicit_none", [False, True])
def test_first_real_observation_keeps_original_empty_checkpoint_wire(explicit_none):
    controller = Controller(SCOPE, 100)
    if explicit_none:
        controller.set_state_entry("host_checkpoint", CHECKPOINT_KEY, None)
    source = event("first-observation")
    assert controller.observe_committed_event(source)
    assert controller.state.host_checkpoint.get(CHECKPOINT_KEY) is None
    observed = observation(source, act="other_exchange", floor="open")
    assert controller.apply_semantic_observation(observed)
    assert controller.state.model_dump(mode="json")["host_checkpoint"][CHECKPOINT_KEY] == {
        "version": 1,
        "units": {},
        "expressions": {},
    }
    controller.take_changes()
    assert not controller.apply_semantic_observation(observed)
    assert controller.take_changes() == (False, False)


def test_steady_retirement_reuses_derived_refs_until_an_actual_change():
    controller, origin, original = admitted()
    controller.advance(701, controller_epoch=0, host_available=False)
    controller.advance(702, controller_epoch=0, host_available=False)
    before = controller._participation_checkpoint().model_dump_json()
    with (
        patch.object(
            controller, "_retire_participation", wraps=controller._retire_participation
        ) as retire,
        patch.object(
            controller, "_prune_participation", wraps=controller._prune_participation
        ) as prune,
    ):
        controller.advance(703, controller_epoch=0, host_available=False)
        assert retire.call_count == 0
        assert prune.call_count == 0
    assert controller._participation_checkpoint().model_dump_json() == before
    expected_sources = frozenset(
        controller._participation_refs(controller._participation_checkpoint())
    )
    with patch.object(
        controller, "_participation_refs", side_effect=AssertionError("rebuilt refs")
    ):
        assert controller.participation_source_refs() == expected_sources
    following = event("new-retirement", at=704)
    assert controller.observe_committed_event(following)
    actual = original.model_copy(update={"basis": (*original.basis, following.ref)})
    assert controller.observe_unit_input(actual, following.ref, continuation=True)
    with patch.object(
        controller, "_retire_participation", wraps=controller._retire_participation
    ) as retire:
        controller.advance(705, controller_epoch=0, host_available=False)
        assert retire.call_count > 0
    controller.observe_source_change(origin.ref)
    assert not controller.participating_units(705)


def test_unchanged_state_update_preserves_owner_and_dirty_flags():
    controller = Controller(SCOPE, 100)
    controller.take_changes()
    original = controller.state
    controller._set(now=controller.state.now, host_checkpoint=controller.state.host_checkpoint)
    assert controller.state is original
    assert controller.take_changes() == (False, False)


def test_known_checkpoint_is_loaded_once_and_reused_across_reads_and_feedback(tmp_path):
    controller, origin, original = admitted()
    wire = controller.state.model_dump(mode="json")
    with patch.object(
        ParticipationCheckpoint, "model_validate", wraps=ParticipationCheckpoint.model_validate
    ) as decode:
        restored = Controller.restore(State.model_validate(wire), 100)
        decoded_at_load = decode.call_count
        assert isinstance(restored.state.host_checkpoint[CHECKPOINT_KEY], ParticipationCheckpoint)
        assert restored.participation_actor_ids() == frozenset({"A"})
        assert restored.participating_units(100)
        fresh = event("typed-new", at=101)
        assert restored.observe_committed_event(fresh)
        actual = original.model_copy(update={"basis": (*original.basis, fresh.ref)})
        restored.take_changes()
        assert restored.observe_unit_input(actual, fresh.ref, continuation=True)
        assert restored.body_changed
        assert restored.participation_actor_ids() == frozenset({"A"})
        assert decode.call_count == decoded_at_load
        restored.take_changes()
        assert restored.observe_unit_input(actual, fresh.ref, continuation=True)
        assert restored.take_changes() == (False, False)
    store = SnapshotStore(tmp_path / "state.sqlite3")
    revision = store.save(restored.state, expected_revision=0)
    store.close()
    store = SnapshotStore(tmp_path / "state.sqlite3")
    loaded_revision, loaded = store.load(SCOPE)
    store.close()
    assert loaded_revision == revision
    reloaded = Controller.restore(loaded, 101)
    checkpoint = reloaded._participation_checkpoint()
    unit = checkpoint.units[reloaded._unit_key(original.unit.thread, original.unit.target)]
    assert unit.binding.basis == (origin.ref,)
    assert unit.input_ref == fresh.ref
    assert unit.continuation_refs == (fresh.ref,)


def test_host_participation_reads_distinguish_binding_refs_and_all_retained_sources():
    controller, originals = population(2)
    checkpoint = controller._participation_checkpoint()
    bindings = frozenset(ref for unit in checkpoint.units.values() for ref in unit.binding.basis)
    sources = frozenset(controller._participation_refs(checkpoint))
    assert controller.participation_binding_refs() == bindings
    assert controller.participation_source_refs() == sources
    assert controller.participation_identity_ids() == frozenset(
        actor
        for unit in checkpoint.units.values()
        for actor in (unit.binding.actor, unit.binding.unit.target)
    )
    assert {original.ref for original in originals} <= bindings


def test_host_participation_reports_and_continuations_are_isolated_views():
    controller, origin, original = admitted()
    assert controller.observe_unit_hint(original, report("stay", at=104))
    following = event("following", at=105)
    assert controller.observe_committed_event(following)
    actual = original.model_copy(update={"basis": (*original.basis, following.ref)})
    assert controller.observe_unit_input(actual, following.ref, continuation=True)
    before = controller.state.model_dump_json()
    reports = controller.participation_context_reports(original.unit.thread, original.unit.target)
    assert len(reports) == 1
    assert reports[0].actor == "A"
    assert reports[0].hint is not None
    assert reports[0].hint.engage == "stay"
    assert reports[0].hint.source is None
    assert reports[0].hint.basis == (origin.ref,)
    assert reports[0].hint.at == 104
    assert reports[0].hint.sequence == 1
    assert controller.participation_continuation_sources(
        original.unit.thread, original.actor, original.unit.target, (following.ref.event_id,)
    ) == {following.ref.event_id: following.ref}
    with pytest.raises(FrozenInstanceError):
        reports[0].actor = "altered"
    assert controller.state.model_dump_json() == before


def test_continuation_index_append_reset_and_restore_preserve_original_positions():
    controller, origin, original = admitted()
    key = controller._unit_key("topic", "A")
    checkpoint = controller._participation_checkpoint()
    unit = checkpoint.units[key]
    first = SourceRef(event_id="indexed", revision=1)
    revised = SourceRef(event_id="indexed", revision=2)
    other = SourceRef(event_id="other", revision=1)
    controller._put_participation_unit(
        checkpoint,
        key,
        unit.model_copy(update={"continuation_refs": (first, other, first, revised)}),
    )
    controller._save_participation(checkpoint)
    assert controller.participation_continuation_sources("topic", "A", "A", ("indexed",)) == {
        "indexed": revised
    }
    unit = checkpoint.units[key]
    assert controller._continuation_position(key, unit, first) == 2
    assert controller._continuation_position(key, unit, revised) == 3
    fresh = event("indexed-new", at=101)
    assert controller.observe_committed_event(fresh)
    # Prime after any real retirement needed to accept the source.
    controller.participation_continuation_sources("topic", "A", "A", ("indexed",))
    cached = controller._family_index.continuations[key]
    actual = original.model_copy(update={"basis": (origin.ref, fresh.ref)})
    assert controller.observe_unit_input(actual, fresh.ref, continuation=True)
    assert controller._family_index.continuations[key] is cached
    unit = controller._participation_checkpoint().units[key]
    assert controller._continuation_position(key, unit, fresh.ref) == 4
    restored = Controller.restore(
        State.model_validate(controller.state.model_dump(mode="json")), 101
    )
    assert restored.participation_continuation_sources("topic", "A", "A", ("indexed",)) == {
        "indexed": revised
    }
    independent = event("indexed-independent", at=102)
    assert restored.observe_committed_event(independent)
    assert restored.observe_unit_input(binding(independent), independent.ref)
    assert not restored.participation_continuation_sources("topic", "A", "A", ("indexed",))


@pytest.mark.parametrize("raw", [None, {"version": 2}, {"version": 1, "future": True}])
def test_host_required_participation_reads_reject_opaque_or_null_namespace(raw):
    controller = Controller(SCOPE, 100)
    controller.state.host_checkpoint[CHECKPOINT_KEY] = raw
    with pytest.raises(ValidationError):
        controller.participation_binding_refs()
    with pytest.raises(ValidationError):
        controller.participation_context_reports("topic", "A")
    with pytest.raises(ValidationError):
        controller.participation_continuation_sources("topic", "A", "A", ())
    assert not controller.participation_source_refs()
    assert not controller.participation_identity_ids()


def test_retired_source_membership_is_shared_per_unit_and_copy_cannot_reuse_stale_index():
    class CountingRef(SourceRef):
        hashes: ClassVar[int] = 0
        comparisons: ClassVar[int] = 0

        def __hash__(self):
            type(self).hashes += 1
            return super().__hash__()

        def __eq__(self, other):
            type(self).comparisons += 1
            return super().__eq__(other)

    refs = tuple(CountingRef(event_id=f"retired-{index}", revision=1) for index in range(32))
    controller = Controller(SCOPE, 1000)
    controller.state.seen.update({ref.event_id: SeenSource(revision=1, at=100) for ref in refs})
    unit = UnitState(
        binding=UnitBinding(
            scope=SCOPE,
            unit=ParticipationUnit(thread="topic", target="A"),
            actor="A",
            basis=(refs[0],),
        ),
        last_at=100,
        retired_refs=refs,
    )
    CountingRef.hashes = 0
    assert all(controller._unit_ref_valid(ref, unit) for ref in refs)
    assert CountingRef.hashes <= len(refs) * 3
    CountingRef.comparisons = 0
    assert all(ref in unit.retired_refs for ref in refs)
    assert CountingRef.comparisons <= len(refs) * 3
    copied = unit.model_copy(update={"retired_refs": refs[1:]})
    assert not controller._unit_ref_valid(refs[0], copied)
    assert controller._unit_ref_valid(refs[0], unit)


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


def test_retirement_index_matches_full_replay_across_changes_and_boundary():
    class FullRetirement(Controller):
        def _prune(self, *, normalize_checkpoint=False):
            self._retirement_index = None
            return super()._prune(normalize_checkpoint=normalize_checkpoint)

    seed, originals = population(3)
    wire = seed.state.model_dump(mode="json")
    fast = Controller.restore(State.model_validate(wire), 705)
    full = FullRetirement.restore(State.model_validate(wire), 705)

    def same():
        assert fast.state.model_dump_json() == full.state.model_dump_json()
        assert fast.take_changes() == full.take_changes()

    same()
    for now in (706, 707, 708):
        fast.advance(now, controller_epoch=0, host_available=False)
        full.advance(now, controller_epoch=0, host_available=False)
        same()
    for controller in (fast, full):
        controller.observe_source_change(originals[0].ref)
    same()
    fresh = originals[1].model_copy(
        update={"ref": SourceRef(event_id="index-new", revision=1), "at": 709}
    )
    updated = binding(originals[1]).model_copy(update={"basis": (originals[1].ref, fresh.ref)})
    for controller in (fast, full):
        assert controller.observe_committed_event(fresh)
        assert controller.observe_unit_input(updated, fresh.ref, continuation=True)
    same()
    for now in (710, 1305, 1310, 1311):
        fast.advance(now, controller_epoch=0, host_available=False)
        full.advance(now, controller_epoch=0, host_available=False)
        same()
    for controller in (fast, full):
        controller.remove_state_entry("effects", "sent-1")
        controller.advance(1312, controller_epoch=0, host_available=False)
    same()


def test_continuation_family_lookup_matches_full_view_without_scanning_other_people(monkeypatch):
    controller, originals = population(3)
    focus = controller.state.events["focus-0"]
    actual = binding(originals[0]).model_copy(update={"basis": (originals[0].ref, focus.ref)})
    assert controller.continuation_current(focus.ref, actual, 706)
    expected = controller.continuation_current(focus.ref, actual, 706)
    monkeypatch.setattr(
        controller, "_current_units", lambda *_a, **_k: pytest.fail("built whole current view")
    )
    monkeypatch.setattr(
        controller, "_participating_units", lambda *_a, **_k: pytest.fail("scanned all units")
    )
    assert (
        controller.continuation_current_for_family(
            focus.ref, actual, 706, family=frozenset({"person-0"})
        )
        is expected
    )
    controller.observe_source_change(originals[0].ref)
    assert not controller.continuation_current_for_family(
        focus.ref, actual, 706, family=frozenset({"person-0"})
    )


def test_continuation_family_matches_full_view_for_merge_anchor_self_and_boundary():
    controller, originals = population(3)
    family = frozenset({"person-0", "person-1"})

    def identity(actor):
        return "root" if actor in family else actor

    focus = event("family-focus", at=706).model_copy(
        update={"thread": "topic-0", "author": "person-0", "target": "person-0"}
    )
    assert controller.observe_committed_event(focus)

    def check(source, thread):
        candidate = UnitBinding(
            scope=SCOPE,
            unit=ParticipationUnit(thread=thread, target="person-0"),
            actor="person-0",
            basis=(source.ref,),
        )
        full = controller.continuation_current(source.ref, candidate, 707, actor_identity=identity)
        local = controller.continuation_current_for_family(
            source.ref, candidate, 707, family=family, actor_identity=identity
        )
        assert local == full
        return local

    assert not check(focus, "topic-0")  # Two related threads are ambiguous.
    anchor = event("family-anchor", at=706).model_copy(
        update={"thread": "topic-1", "author": "SELF", "target": "person-1", "kind": "self"}
    )
    assert controller.observe_committed_event(anchor)
    effect = controller.state.effects["sent-1"].effect
    assert controller.observe_unit_expression(
        binding(originals[1]), "run-1", effect, anchor=anchor.ref
    )
    replied = focus.model_copy(
        update={"ref": SourceRef(event_id="reply-family", revision=1), "reply_to": anchor.ref}
    )
    assert controller.observe_committed_event(replied)
    assert check(replied, "topic-1")  # Explicit anchor wins over another thread.
    controller.set_state_entry(
        "boundaries",
        "stop-family",
        Boundary(source=replied.ref, thread="topic-1", target="person-1", group_wide=True),
    )
    assert not check(replied, "topic-1")
    controller.remove_state_entry("boundaries", "stop-family")
    assert check(replied, "topic-1")

    proposal = Proposal(
        proposal_id="self-family-proposal",
        scope=SCOPE,
        controller_epoch=0,
        kind="conversation",
        thread="self-topic",
        target_hint="person-1",
        sources=(replied.ref,),
        created_at=706,
        expires_at=800,
    )
    controller.set_state_entry("proposals", proposal.proposal_id, proposal)
    self_effect = Effect(effect_id="self-family-effect", kind="message", at=706)
    controller.set_state_entry(
        "effects",
        self_effect.effect_id,
        RecordedEffect(
            proposal_id=proposal.proposal_id, run_ref="self-family-run", effect=self_effect
        ),
    )
    assert check(replied, "topic-1")  # The explicit anchor still outranks synthetic SELF.
    no_anchor = focus.model_copy(update={"ref": SourceRef(event_id="no-anchor", revision=1)})
    assert controller.observe_committed_event(no_anchor)
    assert not check(no_anchor, "self-topic")
    wire = controller.state.model_dump(mode="json")
    retained = wire["host_checkpoint"][CHECKPOINT_KEY]["units"]
    self_unit = deepcopy(retained[controller._unit_key("topic-1", "person-1")])
    self_unit["binding"]["actor"] = "SELF"
    self_unit["binding"]["unit"]["thread"] = "restored-self"
    self_unit["hint"] = report("stay", at=706).model_dump(mode="json")
    self_unit["hint_basis"] = self_unit["binding"]["basis"]
    retained[controller._unit_key("restored-self", "person-1")] = self_unit
    restored = Controller.restore(State.model_validate(wire), 707)
    assert any(unit.binding.actor == "SELF" for unit in restored.participating_units(707))
    actual = UnitBinding(
        scope=SCOPE,
        unit=ParticipationUnit(thread="topic-1", target="person-0"),
        actor="person-0",
        basis=(replied.ref,),
    )
    assert restored.continuation_current_for_family(
        replied.ref, actual, 707, family=family, actor_identity=identity
    ) == restored.continuation_current(replied.ref, actual, 707, actor_identity=identity)


def test_family_facts_detect_quiet_and_competitor_without_changing_sources():
    controller, originals = population(2)
    focus = controller.state.events["focus-0"]
    family = frozenset({"person-0"})
    before_sources = controller.continuation_family_sources(focus.ref, 706, family=family)
    before = controller.continuation_family_facts(focus.ref, 706, family=family)
    assert controller.observe_unit_hint(binding(originals[0]), report("quiet", at=706))
    assert controller.continuation_family_sources(focus.ref, 706, family=family) == before_sources
    quiet = controller.continuation_family_facts(focus.ref, 706, family=family)
    assert quiet != before
    checkpoint = controller._participation_checkpoint()
    existing = checkpoint.units[controller._unit_key("topic-0", "person-0")]
    other_binding = existing.binding.model_copy(
        update={"unit": ParticipationUnit(thread="other-thread", target="person-0")}
    )
    other = existing.model_copy(update={"binding": other_binding})
    controller._put_participation_unit(
        checkpoint, controller._unit_key("other-thread", "person-0"), other
    )
    controller._save_participation(checkpoint)
    assert controller.continuation_family_sources(focus.ref, 706, family=family) == before_sources
    assert controller.continuation_family_facts(focus.ref, 706, family=family) != quiet


def test_family_facts_distinguish_seen_from_real_unit_retirement_proof():
    controller, origin, original = admitted()
    assert controller.observe_unit_hint(original, report("quiet", at=100))
    checkpoint = controller._participation_checkpoint()
    key = controller._unit_key("topic", "A")
    ghost = SourceRef(event_id="unproved-retired", revision=1)
    controller.set_state_entry("seen", ghost.event_id, SeenSource(revision=1, at=100))
    unit = checkpoint.units[key].model_copy(update={"hint_basis": (ghost,)})
    controller._put_participation_unit(checkpoint, key, unit)
    controller._save_participation(checkpoint)

    def state_for(ref):
        facts = controller.continuation_family_facts(
            origin.ref, 101, family=frozenset({"A"}), binding=original
        )
        return next(item for item in facts.sources if item[0] == ref)

    assert state_for(ghost)[-1] is False  # Matching seen alone cannot revive a relation.
    unit = unit.model_copy(update={"retired_refs": (ghost,)})
    controller._put_participation_unit(checkpoint, key, unit)
    controller._save_participation(checkpoint)
    assert state_for(ghost)[-1] is True
    controller.set_state_entry("invalidated", ghost.event_id, ghost.revision)
    assert state_for(ghost)[-1] is False
    controller.remove_state_entry("seen", origin.ref.event_id)
    assert state_for(origin.ref)[-1] is True  # A current event does not require seen.


def test_family_facts_detect_unassociated_effect_receipt_change():
    controller, _ = population(1)
    focus = controller.state.events["focus-0"]
    family = frozenset({"person-0"})
    before = controller.continuation_family_facts(focus.ref, 706, family=family)
    before_sources = controller.continuation_family_sources(focus.ref, 706, family=family)
    assert controller.state.effects["sent-0"].proposal_id == "host:run-0"
    controller.remove_state_entry("effects", "sent-0")
    assert controller.continuation_family_sources(focus.ref, 706, family=family) == before_sources
    assert controller.continuation_family_facts(focus.ref, 706, family=family) != before


def test_family_lookup_keeps_original_effect_order_after_host_receipt_association():
    controller = Controller(SCOPE, 100)
    source = event("self-origin", at=100)
    assert controller.observe_committed_event(source)
    anchor = event("self-anchor", at=101).model_copy(
        update={"kind": "self", "author": "SELF", "target": "A", "thread": "self-topic"}
    )
    assert controller.observe_committed_event(anchor)
    assert controller.observe_public_anchor("first-run", anchor.ref.event_id, 101)
    proposals = [
        Proposal(
            proposal_id=f"self-p{number}",
            scope=SCOPE,
            controller_epoch=0,
            kind="conversation",
            thread="self-topic",
            target_hint="A",
            sources=(source.ref,),
            created_at=100,
            expires_at=200,
        )
        for number in (1, 2)
    ]
    for proposal in proposals:
        controller.set_state_entry("proposals", proposal.proposal_id, proposal)
    first = Effect(effect_id="self-e1", kind="message", at=101)
    second = Effect(effect_id="self-e2", kind="message", at=102)
    assert controller.observe_committed_effect("first-run", first)
    controller.set_state_entry(
        "effects",
        second.effect_id,
        RecordedEffect(proposal_id="self-p2", run_ref="second-run", effect=second),
    )
    controller.set_state_entry(
        "effects",
        first.effect_id,
        RecordedEffect(proposal_id="self-p1", run_ref="first-run", effect=first),
    )
    focus = event("self-focus", at=103, reply_to=anchor.ref)
    assert controller.observe_committed_event(focus)
    actual = UnitBinding(
        scope=SCOPE,
        unit=ParticipationUnit(thread="self-topic", target="A"),
        actor="A",
        basis=(focus.ref,),
    )
    assert controller.continuation_current(focus.ref, actual, 104)
    assert controller.continuation_current_for_family(
        focus.ref, actual, 104, family=frozenset({"A"})
    )


def test_retirement_reuses_latest_expression_ids_within_one_prune():
    controller, _ = population(3)
    controller._retirement_index = None
    with patch.object(
        controller, "_unit_expression_ids", wraps=controller._unit_expression_ids
    ) as expressions:
        controller._prune()
    # One pre-fold pass and one pass after invalid units/associations can change.
    assert expressions.call_count == 2


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
        assert decode.call_count == 0
    assert c.state.model_dump_json() == before


@pytest.mark.parametrize("size", [8, 32])
def test_participation_and_freeze_reuse_loaded_checkpoint_without_computing_belief(size):
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
        assert decode.call_count == 0
        decode.reset_mock()
        frozen = c.freeze_observation(
            Snapshot(scope=SCOPE, focus=focus, context=(), sequence=1, issued_at=706)
        )
        assert frozen.unit_reviews
        assert decode.call_count == 0
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
        assert actions.call_count == 1
        assert decode.call_count == 0
        assert facts["observations"].visits == size
        assert facts["effects"].visits <= 2 * size
        assert facts["events"].visits <= 3 * size
        for value in facts.values():
            value.visits = 0
        decode.reset_mock()
        actions.reset_mock()
        c._advance_replay_boundary(706)
        assert actions.call_count == 1
        assert decode.call_count == 0
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


def test_admission_and_hint_reuse_loaded_checkpoint_including_pruning():
    c, originals = population(2)
    focus = c.state.events["focus-0"]
    with patch.object(
        ParticipationCheckpoint, "model_validate", wraps=ParticipationCheckpoint.model_validate
    ) as decode:
        assert c.observe_unit_input(binding(originals[0]), focus.ref, continuation=True)
        assert decode.call_count == 0
        decode.reset_mock()
        assert c.observe_unit_hint(binding(originals[0]), report("quiet", at=706))
        assert decode.call_count == 0


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
    raw = c.state.model_dump(mode="json")["host_checkpoint"][CHECKPOINT_KEY]
    raw["future_extension"] = True
    c.state.host_checkpoint[CHECKPOINT_KEY] = raw
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
