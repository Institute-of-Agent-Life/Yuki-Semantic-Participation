"""Narrow Host reads/writes retain the old wire and full-view semantics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from yuki_participation.controller import Controller, State
from yuki_participation.models import Scope, ScopedEvent, SourceRef
from yuki_participation.store import SnapshotStore


def _wire_controller(case: str) -> Controller:
    scope = Scope(conversation_id="synthetic-legacy-key-scope", generation=1)
    controller = Controller(scope, 100.0)
    for index, author in ((1, "person-a"), (2, "person-b"), (3, "SELF")):
        assert controller.observe_committed_event(
            ScopedEvent(
                scope=scope,
                ref=SourceRef(event_id=f"event:{index}", revision=1),
                at=100.0 + index,
                thread="thread-old",
                author=author,
                target="person-a",
                text="synthetic source",
                kind="self" if author == "SELF" else "human",
                source_order=index,
            )
        )
    payload = controller.state.model_dump(mode="json")

    def unit(actor, ref, last_at):
        return {
            "binding": {
                "scope": scope.model_dump(mode="json"),
                "unit": {"thread": "thread-old", "target": "person-a"},
                "actor": actor,
                "basis": [{"event_id": f"event:{ref}", "revision": 1}],
            },
            "last_at": last_at,
        }

    canonical = controller._unit_key("thread-old", "person-a")
    if case == "single_noncanonical":
        units = {"legacy-key": unit("person-a", 1, 108.0)}
    elif case == "self_only":
        units = {"self-legacy-key": unit("SELF", 3, 108.0)}
    elif case == "future_only":
        units = {"legacy-key": unit("person-a", 1, 111.0)}
    else:
        units = {
            canonical: unit("person-a", 1, 109.0 if case == "owned_first" else 107.0),
            "duplicate-legacy-key": unit("person-b", 2, 108.0),
            "self-legacy-key": unit("SELF", 3, 109.5),
        }
    payload["host_checkpoint"]["participation_v1"] = {
        "version": 1,
        "units": units,
        "expressions": {},
    }
    return Controller.restore(State.model_validate(payload), 110.0)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("single_noncanonical", ("event:1",)),
        ("self_only", ()),
        ("future_only", ()),
        ("owned_first", ("event:1",)),
        ("other_first", ()),
    ],
)
def test_own_basis_matches_full_view_with_legacy_keys(case, expected):
    controller = _wire_controller(case)
    old_oracle = next(
        (
            unit.binding.basis
            for unit in controller.participating_units(110.0)
            if unit.unit.thread == "thread-old"
            and unit.unit.target == "person-a"
            and unit.binding.actor == "person-a"
        ),
        (),
    )
    assert tuple(ref.event_id for ref in old_oracle) == expected
    directory = os.environ.get("YUKI_LATENCY_ARTIFACT_DIR")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"own-basis-{case}.json").write_text(
            json.dumps({"case": case, "old_oracle": list(expected)}, indent=2) + "\n",
            encoding="utf-8",
        )
    assert controller.participation_own_basis("thread-old", "person-a", 110.0) == old_oracle


def test_host_mapping_entry_dirty_and_wire_survive_restore(tmp_path):
    scope = Scope(conversation_id="synthetic-host-mapping", generation=1)
    controller = Controller(scope, 100.0)
    controller.set_state_entry("host_checkpoint", "opaque_future", {"version": 2, "x": [1, None]})
    controller.take_changes()
    controller.set_host_mapping_entry("source_versions", "event:1", [1, "digest-a"])
    assert controller.take_changes()[1]
    controller.set_host_mapping_entry("source_versions", "event:1", [1, "digest-a"])
    assert not controller.take_changes()[1]
    controller.set_host_mapping_entry("source_versions", "event:1", [2, "digest-b"])
    controller.set_host_mapping_entry("source_versions", "event:2", [1, "digest-c"])
    assert controller.take_changes()[1]
    store = SnapshotStore(tmp_path / "mapping.sqlite")
    try:
        assert store.save(controller.state, expected_revision=0) == 1
        loaded = store.load(scope)
        assert loaded is not None
        reopened = Controller.restore(loaded[1], 100.0)
        assert reopened.state.model_dump(mode="json")["host_checkpoint"] == {
            "opaque_future": {"version": 2, "x": [1, None]},
            "source_versions": {"event:1": [2, "digest-b"], "event:2": [1, "digest-c"]},
        }
    finally:
        store.close()
