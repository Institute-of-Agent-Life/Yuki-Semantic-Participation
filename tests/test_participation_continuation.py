"""Ordinary unit feedback, pure queries and real old-reader snapshot compatibility."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_controller import SCOPE, event, observation
from yuki_participation.controller import Controller
from yuki_participation.models import Choice, Effect, Feedback, HostUnitOption, Scope, SourceRef
from yuki_participation.participation import CHECKPOINT_KEY, ParticipationUnit, UnitBinding
from yuki_participation.self_report import SelfDelta, SelfReport
from yuki_participation.session import ObservationSession
from yuki_participation.store import SnapshotStore


def binding(source, *, thread=None, target=None, basis=None):
    return UnitBinding(
        scope=source.scope,
        unit=ParticipationUnit(thread=thread or source.thread, target=target or source.target),
        actor=source.author,
        basis=basis or (source.ref,),
    )


def save_historical_wire(store, state):
    """A frozen old producer's wire shape, not forward support in its strict reader."""
    raw = state.model_dump(mode="json")
    for original_event in raw["events"].values():
        original_event.pop("source_order", None)
        original_event.pop("mentions", None)
    for seen in raw["seen"].values():
        seen.pop("source_order", None)
    for report in (*raw["self_reports"].values(), raw["engagement_report"]):
        if report is not None:
            report.pop("source", None)
    store.connection.execute(
        "INSERT INTO snapshots VALUES(?,?,?,?)",
        (state.scope.conversation_id, state.scope.generation, 1, json.dumps(raw).encode()),
    )


def admitted():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    b = binding(source)
    assert c.observe_unit_input(b, source.ref)
    return c, source, b


def report(engage, *, sequence=1, at=104, run="ordinary", response="response"):
    return SelfReport(
        run_ref=run,
        sequence=sequence,
        response_id=response,
        at=at,
        delta=SelfDelta(engage=engage),
    )


def test_context_recording_does_not_automatically_queue_or_predict():
    c = Controller(SCOPE, 100)
    session = ObservationSession(c, None)
    source = event()
    session.observe(source)
    assert not session.queue.pending
    assert not c.state.candidates
    assert not hasattr(c, "predict_continuation")
    assert session.request_observation(source.ref)
    session.request_observation(source.ref)
    assert len(session.queue.pending) == 1
    assert not session.request_observation(source.ref.model_copy(update={"revision": 2}))


def test_pure_matching_does_not_resolve_a_new_focus_or_consume_it():
    c, _, b = admitted()
    c.observe_unit_hint(b, report("stay"))
    next_event = event(
        "next",
        at=105,
        unit_ambiguous=True,
        unit_options=(HostUnitOption(key="new", thread="new", target="A"),),
    )
    c.observe_committed_event(next_event)
    before = c.state.model_dump_json()
    for _ in range(3):
        view = c.participation_view(next_event, 105)
        assert view.source_valid and not view.current_resolved
        assert view.matched_unit.unit == ParticipationUnit(thread="topic", target="A")
        assert view.needs_observation
    assert c.state.model_dump_json() == before
    assert next_event.ref.event_id not in c.state.observations


def test_nonambiguous_host_fallback_is_not_a_semantic_observation():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    assert c.resolved_unit(source) is not None  # Existing Host unit API is not a semantic verdict.
    view = c.participation_view(source, 101)
    assert not view.current_resolved and view.current_unit is None
    assert view.needs_observation


def test_admitted_input_is_not_mutual_engagement_until_real_expression():
    c, source, b = admitted()
    assert c.belief("topic", "A", 100) == (0, 1, 0, 0, 0)
    effect = Effect(effect_id="social:one", kind="message", at=101, actual_targets=("group",))
    assert c.observe_unit_expression(b, "ordinary", effect)
    assert c.belief("topic", "A", 101)[3] > 0
    assert c.state.effects[effect.effect_id].effect.actual_targets == ("group",)
    assert not c.state.proposals
    assert c.state.consumed[source.ref.event_id] == 1


def test_logical_expression_replay_adds_each_real_anchor_without_double_count():
    c, _, b = admitted()
    effect = Effect(effect_id="social:one", kind="message", at=101, actual_targets=("group",))
    before = None
    for key in ("outbound-one", "outbound-two"):
        anchor = event(key, at=101, kind="self").model_copy(update={"author": "SELF"})
        c.observe_committed_event(anchor)
        assert c.observe_unit_expression(b, "ordinary", effect, anchor=anchor.ref)
        now = c.belief("topic", "A", 102)
        if before is not None:
            assert now == before
        before = now
    assert len(c.state.effects) == 1
    assert len(c.participating_units(102)[0].anchors) == 2
    assert not c.observe_unit_expression(b, "other-run", effect)
    assert not c.observe_unit_expression(b, "ordinary", effect.model_copy(update={"at": 103}))


@pytest.mark.parametrize("kind", ["compute", "tool"])
def test_nonexpression_effect_cannot_strengthen_an_ordinary_unit(kind):
    c, _, b = admitted()
    before = c.state.model_dump_json()
    assert not c.observe_unit_expression(b, "ordinary", Effect(effect_id="x", kind=kind, at=101))
    assert c.state.model_dump_json() == before


def test_quiet_is_local_optional_intent_without_user_stop_or_global_change():
    c, _, b = admitted()
    other = event("other", at=101).model_copy(update={"thread": "other-topic"})
    c.observe_committed_event(other)
    c.observe_unit_input(binding(other), other.ref)
    c.observe_unit_hint(binding(other), report("join", at=102, response="other-response"))
    assert c.observe_unit_hint(b, report("quiet"))
    assert not c.observe_unit_hint(b, report("quiet"))
    next_event = event("next", at=105)
    c.observe_committed_event(next_event)
    view = c.participation_view(next_event, 105)
    assert view.matched_unit.unit.thread == "other-topic"
    assert not c.state.boundaries
    assert c.state.engagement_report is None
    assert c.observe_unit_hint(b, report("stay", sequence=2, at=106, response="response-2"))
    assert c.participation_view(next_event, 106).ambiguous


def test_join_cannot_release_actual_stop_and_consumed_stop_can_still_be_observed():
    c, source, b = admitted()
    assert c.apply_semantic_observation(observation(source, act="ask_yuki_stop"))
    assert c.state.boundaries
    assert c.observe_unit_hint(b, report("join"))
    next_event = event("next", at=105)
    c.observe_committed_event(next_event)
    view = c.participation_view(next_event, 105)
    assert view.matched_unit is None
    assert c.participating_units(105)[0].closed


@pytest.mark.parametrize("change", ["edit", "withdraw", "generation"])
def test_source_change_or_new_generation_cannot_reuse_old_intent(change):
    c, source, b = admitted()
    c.observe_unit_hint(b, report("stay"))
    if change == "generation":
        other_scope = Scope(conversation_id=SCOPE.conversation_id, generation=2)
        assert not c.observe_unit_hint(b.model_copy(update={"scope": other_scope}), report("quiet"))
        assert not c.participation_view(
            source.model_copy(update={"scope": other_scope}), 105
        ).source_valid
    else:
        c.observe_source_change(source.ref)
        if change == "edit":
            c.observe_committed_event(
                source.model_copy(update={"ref": SourceRef(event_id="e1", revision=2)})
            )
        assert not c.participating_units(105)
        assert not c.observe_unit_hint(b, report("quiet"))


def test_self_group_anchor_is_a_candidate_not_all_members_engagement():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    c.apply_semantic_observation(observation(source))
    proposal = c.advance(101, controller_epoch=0, host_available=True)
    effect = Effect(effect_id="self-send", kind="message", at=102, actual_targets=("group",))
    c.observe_run_feedback(
        Feedback(
            run_ref="self-run",
            proposal_id=proposal.proposal_id,
            sequence=1,
            outcome="completed",
            at=102,
            effects=(effect,),
        )
    )
    anchor = event("self-out", at=102, kind="self").model_copy(
        update={"author": "SELF", "target": "group"}
    )
    c.observe_committed_event(anchor)
    c.observe_public_anchor("self-run", anchor.ref.event_id, 102)
    new_person = event("other-person", at=103, reply_to=anchor.ref).model_copy(
        update={"author": "B", "target": "B"}
    )
    c.observe_committed_event(new_person)
    view = c.participation_view(new_person, 103)
    assert view.candidates and view.candidates[0].anchors == (anchor.ref,)
    assert view.matched_unit is None and view.needs_observation


def test_addressed_false_filters_proposals_without_losing_original_focus():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    c.apply_semantic_observation(observation(source))
    assert c.participation_view(source, 101).addressed
    assert c.advance(101, controller_epoch=0, host_available=True, include_addressed=False) is None
    assert source.ref.event_id in c.state.candidates
    assert source.ref.event_id not in c.state.consumed
    assert c.advance(102, controller_epoch=0, host_available=True) is not None


def test_future_namespace_is_retained_but_not_used_or_overwritten():
    c, source, b = admitted()
    c.state.host_checkpoint[CHECKPOINT_KEY] = {"version": 2, "unknown": "retained"}
    before = c.state.model_dump_json()
    assert not c.participating_units(101)
    assert not c.observe_unit_hint(b, report("quiet"))
    assert c.state.model_dump_json() == before


def test_unaccepted_pending_retires_without_consuming_invitation_or_fake_feedback():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    c.apply_semantic_observation(observation(source))
    proposal = c.advance(101, controller_epoch=0, host_available=True)
    restored = Controller.restore(c.state, 102)
    assert restored.discard_unaccepted_proposal(proposal.proposal_id)
    assert restored.state.pending is None and not restored.state.feedback
    assert not restored.state.consumed and source.ref.event_id in restored.state.candidates
    assert restored.participation_view(source, 102).addressed
    assert not restored.discard_unaccepted_proposal(proposal.proposal_id)
    assert (
        restored.advance(103, controller_epoch=0, host_available=True, include_addressed=False)
        is None
    )


def test_accepted_pending_cannot_be_discarded_or_reidentified():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    c.apply_semantic_observation(observation(source))
    proposal = c.advance(101, controller_epoch=0, host_available=True)
    assert c.observe_run_feedback(
        Feedback(
            run_ref="original-run",
            proposal_id=proposal.proposal_id,
            sequence=1,
            outcome="accepted",
            at=102,
        )
    )
    before = c.state.model_dump_json()
    assert not c.discard_unaccepted_proposal(proposal.proposal_id)
    assert c.state.model_dump_json() == before


def test_old_predicted_candidate_restores_but_cannot_create_a_new_proposal():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    c.apply_semantic_observation(observation(source))
    candidate = c.state.candidates[source.ref.event_id]
    c.state.candidates[source.ref.event_id] = candidate.model_copy(
        update={"support": candidate.support.model_copy(update={"kind": "predicted"})}
    )
    restored = Controller.restore(c.state, 101)
    assert source.ref.event_id in restored.state.candidates
    assert not restored.participation_view(source, 101).addressed
    assert restored.advance(102, controller_epoch=0, host_available=True) is None


def test_bound_observation_dependencies_invalidate_unit_on_context_edit():
    c, source, original = admitted()
    context = event("context", at=99)
    c.observe_committed_event(context)
    b = original.model_copy(update={"basis": (source.ref, context.ref)})
    assert c.observe_unit_hint(b, report("stay"))
    c.observe_source_change(context.ref)
    # Only this hint used the context; the original unit admission did not.
    assert c.participating_units(104)[0].engage is None
    assert c.state.host_checkpoint[CHECKPOINT_KEY]["units"]
    focus = event("next", at=105)
    c.observe_committed_event(focus)
    assert c.participation_view(focus, 105).matched_unit is None


@pytest.mark.parametrize("hint", [None, "join", "stay"])
def test_no_reply_without_intent_does_not_self_prove_continuation(hint):
    c, _, b = admitted()
    if hint:
        c.observe_unit_hint(b, report(hint))
    next_event = event("next", at=105)
    c.observe_committed_event(next_event)
    view = c.participation_view(next_event, 105)
    assert not c.participating_units(105)[0].expressed
    assert (view.matched_unit is not None) == bool(hint)
    assert view.needs_observation


def test_confirmed_expression_without_hint_can_match_and_survives_baseline_compaction():
    c, _, b = admitted()
    c.observe_unit_expression(
        b,
        "ordinary",
        Effect(effect_id="confirmed", kind="message", at=101, actual_targets=("group",)),
    )
    next_event = event("next", at=105)
    c.observe_committed_event(next_event)
    assert c.participation_view(next_event, 105).matched_unit.expressed
    before = c.belief("topic", "A", 111)
    c._advance_replay_boundary(105)
    assert c.belief("topic", "A", 111) == before


def test_old_observed_invitation_does_not_advertise_expired_addressed_admission():
    c = Controller(SCOPE, 100)
    source = event()
    c.observe_committed_event(source)
    c.apply_semantic_observation(observation(source))
    assert c.participation_view(source, 101).addressed
    assert not c.participation_view(source, 300).addressed


@pytest.mark.parametrize("corrected", ["different", "unknown"])
def test_semantic_correction_of_admitted_focus_cannot_match_old_unit_on_next_focus(corrected):
    c = Controller(SCOPE, 100)
    options = (
        HostUnitOption(key="old", thread="topic", target="A"),
        HostUnitOption(key="new", thread="different", target="A"),
    )
    source = event(unit_ambiguous=True, unit_options=options)
    c.observe_committed_event(source)
    b = binding(source)
    c.observe_unit_input(b, source.ref)
    effect = Effect(effect_id="confirmed", kind="message", at=101, actual_targets=("group",))
    c.observe_unit_expression(b, "ordinary", effect)
    before = c.state.effects[effect.effect_id]
    choice = options[1] if corrected == "different" else None
    obs = observation(source).model_copy(update={"resolved_unit": choice})
    obs = obs.model_copy(
        update={
            "answers": {
                **obs.answers,
                "unit_selection": Choice(
                    choice="new" if choice else "unknown",
                    probabilities={
                        "old": 0.0,
                        "new": 1.0 if choice else 0.0,
                        "unknown": 0.0 if choice else 1.0,
                    },
                ),
            }
        }
    )
    assert c.apply_semantic_observation(obs)
    next_event = event("next", at=105)
    c.observe_committed_event(next_event)
    before_query = c.state.model_dump_json()
    view = c.participation_view(next_event, 105)
    assert view.needs_observation
    assert (view.matched_unit is None) == (corrected == "different")
    assert c.participating_units(105)[0].semantic_continuation == (corrected == "unknown")
    assert c.state.model_dump_json() == before_query
    assert c.state.effects[effect.effect_id] == before
    assert c.state.host_checkpoint[CHECKPOINT_KEY]["expressions"]


@pytest.mark.parametrize("namespace_version", [1, 2])
@pytest.mark.parametrize("retired", [False, True])
def test_historical_wire_old_reader_load_save_keeps_namespace_and_other_host_keys(
    tmp_path, namespace_version, retired
):
    revision = "b9c7cc71f8dbaf9bc4b45419e0270a08bf48fe9a"
    available = subprocess.run(["git", "cat-file", "-e", revision], capture_output=True)
    if available.returncode:
        pytest.skip("Frozen old reader git object is unavailable in this checkout")
    c, _, b = admitted()
    c.observe_unit_hint(b, report("stay"))
    if retired:
        c.advance(1800, controller_epoch=0, host_available=False)
        assert c.participating_units(1800)[0].engage == "stay"
    if namespace_version == 2:
        c.state.host_checkpoint[CHECKPOINT_KEY] = {"version": 2, "unknown": ["retained"]}
    c.state.host_checkpoint["outbound_threads"] = {"event:old": "old-thread"}
    namespace = c.state.host_checkpoint[CHECKPOINT_KEY]
    database = tmp_path / "participation.sqlite3"
    store = SnapshotStore(database)
    save_historical_wire(store, c.state)
    store.close()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/yuki_participation"], text=True
    ).splitlines()
    for path in paths:
        if not path.endswith(".py"):
            continue
        target = tmp_path / Path(path).relative_to("src")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(["git", "show", f"{revision}:{path}"]))
    code = """
from pathlib import Path
from yuki_participation.autonomy_parameters import AutonomyParameters
from yuki_participation.controller import Controller
from yuki_participation.models import Scope
from yuki_participation.store import SnapshotStore
s=SnapshotStore(Path('participation.sqlite3'))
revision,state=s.load(Scope(conversation_id='group-test',generation=1))
c=Controller.restore(state,max(110,state.now))
s.save(c.state,expected_revision=revision)
s.close()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=True,
    )
    store = SnapshotStore(database)
    revision, state = store.load(SCOPE)
    store.close()
    assert revision == 2
    assert state.host_checkpoint[CHECKPOINT_KEY] == namespace
    assert state.host_checkpoint["outbound_threads"] == {"event:old": "old-thread"}
    restored = Controller.restore(state, max(110, state.now))
    if namespace_version == 1 and not retired:
        assert restored.participating_units(110)[0].engage == "stay"
    else:
        assert not restored.participating_units(110)
    assert json.loads(state.model_dump_json())["host_checkpoint"][CHECKPOINT_KEY] == namespace


def test_new_admitted_input_and_hydration_replay_do_not_reset_local_quiet():
    c, source, b = admitted()
    assert c.observe_unit_hint(b, report("quiet"))
    assert c.observe_unit_input(b, source.ref)
    next_event = event("next", at=105)
    c.observe_committed_event(next_event)
    assert c.observe_unit_input(binding(next_event), next_event.ref)
    restored = Controller.restore(c.state, 106)
    assert not restored.observe_committed_event(next_event)
    assert restored.observe_unit_input(binding(next_event), next_event.ref)
    assert restored.participating_units(106)[0].engage == "quiet"
    later = event("later", at=107)
    restored.observe_committed_event(later)
    assert restored.participation_view(later, 107).matched_unit is None


def test_fifty_continuations_keep_establishment_binding_and_each_actual_effect_basis(tmp_path):
    c, original, origin = admitted()
    assert c.observe_unit_hint(origin, report("stay"))
    bindings = {}
    for index in range(1, 51):
        focus = event(f"next-{index}", at=105 + index)
        c.observe_committed_event(focus)
        matched = c.participation_view(focus, focus.at).matched_unit
        assert matched is not None and matched.binding == origin
        actual = matched.binding.model_copy(update={"basis": (*matched.binding.basis, focus.ref)})
        assert len(actual.basis) == 2
        assert c.observe_unit_input(actual, focus.ref, continuation=True)
        assert c.observe_unit_input(actual, focus.ref, continuation=True)
        assert c.observe_unit_hint(
            actual,
            report("stay", sequence=index + 1, at=focus.at, response=f"response-{index}"),
        )
        effect = Effect(
            effect_id=f"send-{index}", kind="message", at=focus.at, actual_targets=("group",)
        )
        assert c.observe_unit_expression(actual, f"run-{index}", effect)
        bindings[effect.effect_id] = actual.model_dump(mode="json")
        namespace = c.state.host_checkpoint[CHECKPOINT_KEY]
        unit = next(iter(namespace["units"].values()))
        assert unit["binding"] == origin.model_dump(mode="json")
        assert unit["input_ref"] == focus.ref.model_dump(mode="json")
        assert unit["hint_basis"] == [original.ref.model_dump(), focus.ref.model_dump()]
        if index == 25:
            store = SnapshotStore(tmp_path / "units.sqlite3")
            store.save(c.state, expected_revision=0)
            _, restored = store.load(SCOPE)
            store.close()
            c = Controller.restore(restored, focus.at, parameters=c.parameters)
    namespace = c.state.host_checkpoint[CHECKPOINT_KEY]
    assert {k: v["binding"] for k, v in namespace["expressions"].items()} == bindings
    assert len(c.state.effects) == 50
    assert all(record.effect.actual_targets == ("group",) for record in c.state.effects.values())
    # An intermediate input was not used by the current admission/hint or the origin.
    c.observe_source_change(SourceRef(event_id="next-10", revision=1))
    assert c.participating_units(156)[0].engage == "stay"
    assert "send-10" not in c.state.host_checkpoint[CHECKPOINT_KEY]["expressions"]
    assert "send-10" in c.state.effects  # The actual transport receipt remains historical.


@pytest.mark.parametrize("changed", ["origin", "latest"])
def test_fixed_unit_binding_still_rejects_changed_establishment_or_latest_input(changed):
    c, source, origin = admitted()
    focus = event("latest", at=105)
    c.observe_committed_event(focus)
    actual = origin.model_copy(update={"basis": (source.ref, focus.ref)})
    assert c.observe_unit_input(actual, focus.ref, continuation=True)
    assert c.observe_unit_hint(actual, report("stay", at=105))
    c.observe_source_change(source.ref if changed == "origin" else focus.ref)
    next_event = event("unobserved", at=106)
    c.observe_committed_event(next_event)
    view = c.participation_view(next_event, 106)
    assert view.matched_unit is None and view.needs_observation


def test_different_actor_cannot_reuse_previous_actors_unit_establishment():
    c, _, origin = admitted()
    focus = event("other-actor", at=105).model_copy(update={"author": "B"})
    c.observe_committed_event(focus)
    actual = origin.model_copy(update={"actor": "B", "basis": (focus.ref,)})
    assert c.observe_unit_input(actual, focus.ref)
    assert c.participating_units(105)[0].binding == actual
    assert not c.observe_unit_input(origin, focus.ref)
