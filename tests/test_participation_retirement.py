"""Finite raw replay retirement does not invent an engagement expiry."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_controller import SCOPE, event, observation
from tests.test_participation_continuation import admitted, binding, report, save_historical_wire
from yuki_participation.autonomy_parameters import AutonomyParameters
from yuki_participation.controller import Controller, SeenSource
from yuki_participation.models import Choice, Effect, HostUnitOption, SourceRef
from yuki_participation.participation import CHECKPOINT_KEY
from yuki_participation.store import SnapshotStore


def tick(controller, now):
    controller.advance(now, controller_epoch=controller.state.epoch, host_available=False)


@pytest.mark.parametrize("intent", [None, "stay", "quiet"])
@pytest.mark.parametrize("flow", ["continuous", "paused"])
def test_real_unit_survives_raw_retirement_and_store_reopen(tmp_path, intent, flow):
    c, origin, established = admitted()
    c.set_parameters(AutonomyParameters(continuation_review_span=16))
    effect = Effect(effect_id="accepted", kind="message", at=101, actual_targets=("group",))
    assert c.observe_unit_expression(established, "original-run", effect)
    if intent:
        assert c.observe_unit_hint(established, report(intent))
    for index in range(1, 9) if flow == "continuous" else ():
        now = 100 + index * 120
        tick(c, now)
        focus = event(f"input-{index}", at=now)
        assert c.observe_committed_event(focus)
        view = c.participation_view(focus, now)
        assert (view.matched_unit is not None) == (intent != "quiet")
        actual = established.model_copy(update={"basis": (origin.ref, focus.ref)})
        assert c.observe_unit_input(actual, focus.ref, continuation=True)
        if intent:
            assert c.observe_unit_hint(
                actual, report(intent, at=now, sequence=index + 1, response=f"response-{index}")
            )
    tick(c, 1800)
    assert not c.state.events and not c.state.observations
    assert not c._valid(origin.ref)
    assert c.state.seen[origin.ref.event_id].revision == origin.ref.revision
    assert c.state.effects[effect.effect_id].effect == effect
    store = SnapshotStore(tmp_path / "state.sqlite3")
    store.save(c.state, expected_revision=0)
    store.close()
    store = SnapshotStore(tmp_path / "state.sqlite3")
    _, state = store.load(c.state.scope)
    store.close()
    c = Controller.restore(state, 1800, parameters=c.parameters)
    focus = event("fresh-unaddressed", at=1801)
    assert c.observe_committed_event(focus)
    unit = c.participating_units(1801)[0]
    assert unit.binding == established and unit.expressed
    assert unit.engage == intent
    assert (c.participation_view(focus, 1801).matched_unit is not None) == (intent != "quiet")
    assert not c.state.observations and not c.state.candidates
    assert not c.observe_committed_event(origin)


def test_resource_retirement_preserves_unit_but_not_raw_or_every_old_receipt():
    c, origin, established = admitted()
    for index in range(1, 271):
        focus = event(f"source-{index}", at=100 + index / 10)
        assert c.observe_committed_event(focus)
        actual = established.model_copy(update={"basis": (origin.ref, focus.ref)})
        assert c.observe_unit_input(actual, focus.ref)
        assert c.observe_unit_expression(
            actual, f"run-{index}", Effect(effect_id=f"effect-{index}", kind="message", at=focus.at)
        )
    assert len(c.state.events) <= 256
    assert origin.ref.event_id not in c.state.events
    assert c.participation_view(focus, focus.at).matched_unit is not None
    tick(c, 1800)
    assert not c.state.events
    assert set(c.state.effects) == {"effect-270"}
    assert set(c.state.host_checkpoint[CHECKPOINT_KEY]["expressions"]) == {"effect-270"}
    assert len(c.state.seen) == 2  # Origin and the last actual admission/expression source.


@pytest.mark.parametrize("changed", ["origin", "latest", "hint", "expression"])
def test_retired_source_withdrawal_cannot_resurrect_intent_or_expression(changed):
    c, origin, established = admitted()
    latest = event("latest", at=105)
    context = event("context", at=106)
    expression_context = event("expression-context", at=107)
    for source in (latest, context, expression_context):
        assert c.observe_committed_event(source)
    assert c.observe_unit_input(
        established.model_copy(update={"basis": (origin.ref, latest.ref)}),
        latest.ref,
        continuation=True,
    )
    assert c.observe_unit_hint(
        established.model_copy(update={"basis": (origin.ref, context.ref)}), report("stay", at=108)
    )
    assert c.observe_unit_expression(
        established.model_copy(update={"basis": (origin.ref, expression_context.ref)}),
        "run",
        Effect(effect_id="message", kind="message", at=109),
    )
    tick(c, 1800)
    sources = {
        "origin": origin,
        "latest": latest,
        "hint": context,
        "expression": expression_context,
    }
    c.observe_source_change(sources[changed].ref)
    tick(c, 1801)
    c = Controller.restore(c.state, 1802)
    fresh = event("new", at=1802)
    assert c.observe_committed_event(fresh)
    view = c.participation_view(fresh, 1802)
    if changed in {"origin", "latest"}:
        assert view.matched_unit is None
    elif changed == "hint":
        assert view.matched_unit is not None and view.matched_unit.engage is None
    else:
        assert view.matched_unit is not None and not view.matched_unit.expressed
    assert not c._valid(sources[changed].ref)


@pytest.mark.parametrize("correction", ["unknown", "different"])
def test_retirement_keeps_conclusive_correction_distinct_from_unknown(correction):
    c, source, established = admitted()
    assert c.observe_unit_hint(established, report("stay"))
    other = HostUnitOption(key="other", thread="other", target="A")
    source = source.model_copy(update={"unit_ambiguous": True, "unit_options": (other,)})
    c.state.events[source.ref.event_id] = source
    corrected = observation(source, sequence=2, act="unknown", floor="unknown")
    corrected = corrected.model_copy(
        update={
            "resolved_unit": other if correction == "different" else None,
            "answers": {
                **corrected.answers,
                "unit_selection": Choice(
                    choice="other" if correction == "different" else "unknown",
                    probabilities={
                        "other": float(correction == "different"),
                        "unknown": float(correction == "unknown"),
                    },
                ),
            },
        }
    )
    assert c.apply_semantic_observation(corrected)
    assert c.participating_units(110)[0].semantic_continuation == (correction == "unknown")
    tick(c, 1800)
    fresh = event("new", at=1801)
    assert c.observe_committed_event(fresh)
    assert (c.participation_view(fresh, 1801).matched_unit is None) == (correction == "different")


def test_arbitrary_seen_fingerprint_is_not_a_live_source_or_unit_establishment():
    c, _, established = admitted()
    fake = SourceRef(event_id="never-established", revision=1)
    tick(c, 1800)
    c.state.seen[fake.event_id] = SeenSource(revision=1, at=100)
    assert not c._valid(fake)
    assert not c.observe_unit_hint(
        established.model_copy(update={"basis": (fake,)}), report("stay")
    )
    assert not c.state.candidates


def test_fingerprint_resource_pressure_retires_derived_units_before_rejecting_new_input():
    c, _, _ = admitted()
    for index in range(64):
        now = 1000 + index * 700
        tick(c, now)
        sources = [event(f"basis-{index}-{i}", at=now + i) for i in range(16)]
        for source in sources:
            assert c.observe_committed_event(source)
        actual = binding(sources[-1], thread=f"topic-{index}", basis=tuple(s.ref for s in sources))
        assert c.observe_unit_input(actual, sources[-1].ref)
    tick(c, now + 700)
    assert len(c.state.seen) == 1024
    fresh = event("new-actual-source", at=now + 701)
    assert c.observe_committed_event(fresh)
    assert len(c.state.seen) <= 1024
    assert len(c.participating_units(fresh.at)) == 63
    assert not c.state.capacity_blocked
    oldest = SourceRef(event_id="basis-0-0", revision=1)
    assert not c._valid(oldest)
    assert oldest.event_id not in c.state.seen


def different_unit_review(c, source):
    other = HostUnitOption(key="other", thread="other", target="A")
    raw = source.model_copy(update={"unit_ambiguous": True, "unit_options": (other,)})
    c.state.events[source.ref.event_id] = raw
    observed = observation(raw, act="unknown", floor="unknown")
    observed = observed.model_copy(
        update={
            "resolved_unit": other,
            "answers": {
                **observed.answers,
                "unit_selection": Choice(
                    choice="other", probabilities={"other": 1.0, "unknown": 0.0}
                ),
            },
        }
    )
    assert c.apply_semantic_observation(observed)
    assert not c.participating_units(110)[0].semantic_continuation
    return raw


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_old_reader_fold_and_unrelated_stop_seen_do_not_resurrect_unknown_unit(
    tmp_path, legacy_checkpoint
):
    revision = "b9c7cc71f8dbaf9bc4b45419e0270a08bf48fe9a"
    if subprocess.run(["git", "cat-file", "-e", revision], capture_output=True).returncode:
        pytest.skip("Frozen old reader git object is unavailable in this checkout")
    c, source, established = admitted()
    assert c.observe_unit_hint(established, report("stay"))
    legacy_units = c.state.host_checkpoint[CHECKPOINT_KEY]["units"]
    raw = different_unit_review(c, source)
    stop = event("other-stop", at=110).model_copy(
        update={"thread": "other-thread", "author": "B", "target": "B"}
    )
    assert c.observe_committed_event(stop)
    assert c.apply_semantic_observation(observation(stop, act="ask_yuki_stop", context=(raw,)))
    if legacy_checkpoint:
        # Actual pre-fix namespaces kept invalid units and had no retirement proof.
        for value in legacy_units.values():
            value.pop("retired_refs", None)
        c.state.host_checkpoint[CHECKPOINT_KEY]["units"] = legacy_units
    store = SnapshotStore(tmp_path / "state.sqlite3")
    save_historical_wire(store, c.state)
    store.close()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision, "src/yuki_participation"], text=True
    ).splitlines()
    for path in paths:
        if path.endswith(".py"):
            target = tmp_path / Path(path).relative_to("src")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.check_output(["git", "show", f"{revision}:{path}"]))
    code = """
from pathlib import Path
from yuki_participation.controller import Controller
from yuki_participation.models import Scope
from yuki_participation.store import SnapshotStore
s=SnapshotStore(Path('state.sqlite3'))
revision,state=s.load(Scope(conversation_id='group-test',generation=1))
c=Controller.restore(state,1800)
s.save(c.state,expected_revision=revision)
s.close()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=True,
        capture_output=True,
    )
    store = SnapshotStore(tmp_path / "state.sqlite3")
    _, state = store.load(SCOPE)
    store.close()
    assert source.ref.event_id not in state.events
    assert source.ref.event_id in state.seen  # Pinned by an unrelated actual stop.
    c = Controller.restore(state, 1801)
    fresh = event("new", at=1801)
    assert c.observe_committed_event(fresh)
    assert c.participation_view(fresh, 1801).matched_unit is None


@pytest.mark.parametrize("feedback", ["hint", "expression"])
def test_late_real_feedback_cannot_reestablish_reinterpreted_unit(feedback):
    c, source, established = admitted()
    assert c.observe_unit_hint(established, report("stay"))
    different_unit_review(c, source)
    if feedback == "hint":
        assert c.observe_unit_hint(
            established, report("stay", sequence=2, at=111, response="late-response")
        )
    else:
        receipt = Effect(effect_id="actually-delivered", kind="message", at=111)
        assert c.observe_unit_expression(established, "original-run", receipt)
        assert c.state.effects[receipt.effect_id].effect == receipt
        assert receipt.effect_id in c.state.host_checkpoint[CHECKPOINT_KEY]["expressions"]
        assert c.observe_unit_expression(established, "original-run", receipt)
        assert len(c.state.effects) == 1
    assert not c.participating_units(112)[0].semantic_continuation
    tick(c, 1800)
    fresh = event("new", at=1801)
    assert c.observe_committed_event(fresh)
    assert c.participation_view(fresh, 1801).matched_unit is None
    # A genuine new admission establishes its own unit before a new Main hint.
    current = binding(fresh)
    assert c.observe_unit_input(current, fresh.ref)
    assert c.observe_unit_hint(
        current, report("stay", at=1802, response="new-response", run="new-activation")
    )
    assert c.participation_view(fresh, 1802).matched_unit is not None
