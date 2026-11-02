"""End-to-end Controller/SQLite recovery for retired-source ownership.

All sources and effects in this file are synthetic library fixtures, not Host or
gateway evidence. No fixed function-call or wall-clock budget is asserted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tests.test_controller import event
from tests.test_participation_continuation import admitted
from yuki_participation.controller import Controller
from yuki_participation.models import Effect
from yuki_participation.snapshot import freeze
from yuki_participation.store import SnapshotStore


def test_retired_source_copy_and_mutation_survive_snapshot_reopen(tmp_path: Path):
    evidence = {
        "scenario": "retired-source deep copy, actual mutation, duplicate and invalidation",
        "boundary": (
            "real library Controller and local SQLite; "
            "synthetic sources/effect; no Host, QQ or provider"
        ),
    }
    artifact_dir = Path(os.environ.get("YUKI_LATENCY_ARTIFACT_DIR", str(tmp_path)))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    store = None
    try:
        c, original, established = admitted()
        second = event("synthetic-second", at=101)
        assert c.observe_committed_event(second)
        actual = established.model_copy(update={"basis": (original.ref, second.ref)})
        assert c.observe_unit_input(actual, second.ref, continuation=True)
        effect = Effect(
            effect_id="synthetic-confirmed", kind="message", at=102, actual_targets=("group",)
        )
        assert c.observe_unit_expression(actual, "synthetic-original-run", effect)
        c.advance(800, controller_epoch=c.state.epoch, host_available=False)
        assert not c.state.events
        checkpoint = c._read_participation_checkpoint()
        assert checkpoint is not None and len(checkpoint.units) == 1
        unit = next(iter(checkpoint.units.values()))
        assert original.ref in unit.retired_refs and second.ref in unit.retired_refs

        copied = unit.model_copy(deep=True)
        narrowed = copied.model_copy(update={"retired_refs": (original.ref,)})
        assert original.ref in copied.retired_refs and second.ref in copied.retired_refs
        assert original.ref in narrowed.retired_refs and second.ref not in narrowed.retired_refs
        assert second.ref in unit.retired_refs

        store = SnapshotStore(tmp_path / "participation.sqlite3")
        frozen = freeze(c.state)
        revision = store.save_frozen(frozen, expected_revision=0)
        c.take_changes()
        original_json = c.state.model_dump_json()
        assert c.participating_units(800) and c.participating_units(800)
        assert c.state.model_dump_json() == original_json
        assert c.take_changes() == (False, False)

        new_focus = event("synthetic-third", at=801)
        assert c.observe_committed_event(new_focus)
        c.take_changes()  # Isolate the real admission's own dirty propagation.
        prior_owner = c._read_participation_checkpoint()
        actual = established.model_copy(update={"basis": (original.ref, new_focus.ref)})
        assert c.observe_unit_input(actual, new_focus.ref, continuation=True)
        same_owner_after_input = prior_owner is c._read_participation_checkpoint()
        _, body_changed = c.take_changes()
        assert body_changed
        updated = freeze(c.state, body_changed=body_changed, saved_layout=frozen.layout)
        assert updated.body is not None
        revision = store.save_frozen(updated, expected_revision=revision)
        loaded_revision, saved = store.load(c.state.scope)
        assert loaded_revision == revision
        restored = Controller.restore(saved, 801)
        units = restored.participating_units(801)
        assert len(units) == 1 and units[0].binding == established and units[0].expressed
        assert restored.state.effects[effect.effect_id].effect == effect
        checkpoint_after = restored._read_participation_checkpoint()
        assert checkpoint_after is not None
        restored_unit = next(iter(checkpoint_after.units.values()))
        assert restored_unit.input_ref == new_focus.ref
        assert restored._unit_ref_valid(original.ref, restored_unit)

        restored.take_changes()
        assert restored.observe_unit_input(actual, new_focus.ref, continuation=True)
        assert restored.take_changes() == (False, False)
        restored.observe_source_change(original.ref)
        restored.advance(802, controller_epoch=restored.state.epoch, host_available=False)
        assert not restored.participating_units(802)
        _, body_changed = restored.take_changes()
        assert body_changed
        invalidated = freeze(restored.state, body_changed=body_changed, saved_layout=updated.layout)
        revision = store.save_frozen(invalidated, expected_revision=revision)
        store.close()
        store = SnapshotStore(tmp_path / "participation.sqlite3")
        loaded_revision, loaded = store.load(c.state.scope)
        final = Controller.restore(loaded, 802)
        assert loaded_revision == revision and not final.participating_units(802)
        evidence.update(
            deep_copy_and_retired_membership=True,
            in_place_owner_reused=same_owner_after_input,
            changed_input_persisted=True,
            duplicate_did_not_dirty=True,
            withdrawal_not_resurrected=True,
            synthetic_effect_preserved_before_invalidation=True,
            effect_still_in_transient_controller_after_invalidation=effect.effect_id
            in final.state.effects,
        )
    finally:
        if store is not None:
            store.close()
        (artifact_dir / "retired-source-persistence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
