"""Real SQLite progress/body commits retain the old mathematical State and capacity."""

import json
import sqlite3

import pytest

from yuki_participation.controller import BeliefBaseline, Controller, State
from yuki_participation.models import Scope
from yuki_participation.snapshot import HOT_FIELDS, freeze, thaw
from yuki_participation.store import SnapshotConflict, SnapshotStore

SCOPE = Scope(conversation_id="progress-scope", generation=4)


def controller():
    c = Controller(SCOPE, 1000)
    c.advance(1000, controller_epoch=0, host_available=False)
    return c


def test_all_38_fields_and_hot_capacity_tokens(tmp_path, monkeypatch):
    c = controller()
    c._set(host_checkpoint={"extension": {"padding": "资料" * 20000}})
    store = SnapshotStore(tmp_path / "state.db")
    first = freeze(c.state)
    assert len(State.model_fields) == 38
    body = json.loads(first.body)
    assert set(body["state"]) | HOT_FIELDS == set(State.model_fields)
    assert thaw(SCOPE, first.progress, first.body).model_dump_json() == c.state.model_dump_json()
    revision = store.save_frozen(first, expected_revision=0)
    original_body = store.connection.execute("SELECT body FROM snapshot_body").fetchone()[0]
    c.take_changes()[1]
    for number, value in enumerate((0, -0.0, 1e-6, 1e20), start=1):
        c.advance(1000 + number * 2, controller_epoch=0, host_available=False)
        c._set(skipped_seconds=value, sample_sequence=2**90 + number)
        length = len(c.state.model_dump_json().encode())
        with monkeypatch.context() as m:
            m.setattr(
                State, "model_dump_json", lambda *_a, **_k: pytest.fail("hot save encoded body")
            )
            current = freeze(c.state, body_changed=c.take_changes()[1], saved_layout=first.layout)
        assert current.body is None
        revision = store.save_frozen(current, expected_revision=revision)
        assert (
            store.connection.execute("SELECT logical_bytes FROM snapshot_head").fetchone()[0]
            == length
        )
        assert store.load(SCOPE)[1].model_dump_json() == c.state.model_dump_json()
        assert (
            store.connection.execute("SELECT body FROM snapshot_body").fetchone()[0]
            == original_body
        )
    store.close()


def test_layout_order_and_optional_shape_require_full_save():
    c = controller()
    a = BeliefBaseline(thread="a", target="p", at=400, value=(1, 0, 0, 0, 0), last_action=200)
    b = a.model_copy(update={"thread": "b"})
    c._set(belief_baselines={"a": a, "b": b})
    previous = freeze(c.state)
    c._set(belief_baselines={"b": b, "a": a})
    result = freeze(c.state, body_changed=False, saved_layout=previous.layout)
    assert result.body is not None
    assert list(thaw(SCOPE, result.progress, result.body).belief_baselines) == ["b", "a"]


@pytest.mark.parametrize("side", ["snapshot_head", "snapshot_body"])
def test_missing_half_is_not_an_empty_controller(tmp_path, side):
    store = SnapshotStore(tmp_path / "state.db")
    assert store.load_raw(SCOPE) is None
    store.save(controller().state, expected_revision=0)
    store.connection.execute(f"DELETE FROM {side}")
    with pytest.raises(ValueError, match="snapshot_storage_pair"):
        store.load_raw(SCOPE)
    store.close()


def test_cas_and_global_capacity_include_other_generation(tmp_path):
    c = controller()
    length = len(c.state.model_dump_json().encode())
    store = SnapshotStore(tmp_path / "state.db", total_bytes=length * 2 - 1)
    assert store.save(c.state, expected_revision=0) == 1
    with pytest.raises(SnapshotConflict):
        store.save(c.state, expected_revision=0)
    other = c.state.model_copy(update={"scope": SCOPE.model_copy(update={"generation": 5})})
    with pytest.raises(ValueError, match="global_snapshot_capacity"):
        store.save(other, expected_revision=0)
    assert store.load(SCOPE)[0] == 1 and store.load(other.scope) is None
    store.close()


def legacy_file(path, state):
    payload = json.dumps(json.loads(state.model_dump_json()), ensure_ascii=False, indent=3)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE snapshots(scope TEXT,generation INTEGER,revision INTEGER,"
            "payload BLOB,PRIMARY KEY(scope,generation))"
        )
        db.execute(
            "INSERT INTO snapshots VALUES(?,?,?,?)",
            (SCOPE.conversation_id, SCOPE.generation, 17, payload),
        )
    return len(payload.encode())


def test_migration_is_format_only_preserves_revision_and_original_size(tmp_path, monkeypatch):
    c = controller()
    path = tmp_path / "old.db"
    old_size = legacy_file(path, c.state)
    for method in ("restore", "advance", "_prune"):
        monkeypatch.setattr(
            Controller, method, lambda *_a, **_k: pytest.fail("migration ran controller")
        )
    store = SnapshotStore(path)
    revision, state = store.load(SCOPE)
    assert revision == 17 and state.model_dump_json() == c.state.model_dump_json()
    assert (
        store.connection.execute("SELECT logical_bytes FROM snapshot_head").fetchone()[0]
        == old_size
    )
    assert not store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name='snapshots'"
    ).fetchall()
    assert store.save(state, expected_revision=17) == 18
    assert store.connection.execute("SELECT logical_bytes FROM snapshot_head").fetchone()[0] == len(
        state.model_dump_json().encode()
    )
    store.close()


def test_migration_ddl_failure_leaves_whole_old_format(tmp_path):
    c = controller()
    path = tmp_path / "old.db"
    legacy_file(path, c.state)

    class InterruptedStore(SnapshotStore):
        def _create(self):
            super()._create()
            raise RuntimeError("interrupted upgrade")

    with pytest.raises(RuntimeError, match="interrupted"):
        InterruptedStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT revision FROM snapshots").fetchone() == (17,)
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='snapshot_head'").fetchall()
    store = SnapshotStore(path)
    assert store.load(SCOPE)[0] == 17
    store.close()
