"""Real SQLite progress/body commits retain the mathematical State and CAS."""

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


def test_all_38_fields_and_hot_saves(tmp_path, monkeypatch):
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
        with monkeypatch.context() as m:
            m.setattr(
                State, "model_dump_json", lambda *_a, **_k: pytest.fail("hot save encoded body")
            )
            current = freeze(c.state, body_changed=c.take_changes()[1], saved_layout=first.layout)
        assert current.body is None
        revision = store.save_frozen(current, expected_revision=revision)
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


def test_cold_freeze_does_not_encode_a_second_complete_state(monkeypatch):
    c = controller()
    monkeypatch.setattr(
        State, "model_dump_json", lambda *_a, **_k: pytest.fail("encoded full state twice")
    )
    value = freeze(c.state)
    assert value.body is not None
    assert thaw(SCOPE, value.progress, value.body) == c.state


@pytest.mark.parametrize("side", ["snapshot_head", "snapshot_body"])
def test_missing_half_is_not_an_empty_controller(tmp_path, side):
    store = SnapshotStore(tmp_path / "state.db")
    assert store.load_raw(SCOPE) is None
    store.save(controller().state, expected_revision=0)
    store.connection.execute(f"DELETE FROM {side}")
    with pytest.raises(ValueError, match="snapshot_storage_pair"):
        store.load_raw(SCOPE)
    store.close()


def test_cas_and_other_generation_are_independent(tmp_path):
    c = controller()
    store = SnapshotStore(tmp_path / "state.db")
    assert store.save(c.state, expected_revision=0) == 1
    with pytest.raises(SnapshotConflict):
        store.save(c.state, expected_revision=0)
    other = c.state.model_copy(update={"scope": SCOPE.model_copy(update={"generation": 5})})
    assert store.save(other, expected_revision=0) == 1
    assert store.load(SCOPE)[0] == 1 and store.load(other.scope)[0] == 1
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


def legacy_pair_file(path, state, *, missing=None):
    value = freeze(state)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE snapshot_head(scope TEXT NOT NULL,generation INTEGER NOT NULL,"
            "revision INTEGER NOT NULL,progress BLOB NOT NULL,"
            "logical_bytes INTEGER NOT NULL,PRIMARY KEY(scope,generation))"
        )
        db.execute(
            "CREATE TABLE snapshot_body(scope TEXT NOT NULL,generation INTEGER NOT NULL,"
            "body BLOB NOT NULL,logical_size_delta INTEGER NOT NULL,"
            "PRIMARY KEY(scope,generation))"
        )
        if missing != "head":
            db.execute(
                "INSERT INTO snapshot_head VALUES(?,?,?,?,?)",
                (SCOPE.conversation_id, SCOPE.generation, 17, value.progress, 42),
            )
        if missing != "body":
            db.execute(
                "INSERT INTO snapshot_body VALUES(?,?,?,?)",
                (SCOPE.conversation_id, SCOPE.generation, value.body, -11),
            )
    return value


def test_legacy_pair_migrates_atomically_without_rewriting_payload(tmp_path):
    c = controller()
    path = tmp_path / "old_pair.db"
    before = legacy_pair_file(path, c.state)
    store = SnapshotStore(path)
    assert store.load_raw(SCOPE) == (17, before.progress, before.body)
    assert store.load(SCOPE)[1] == c.state
    assert [row[1] for row in store.connection.execute("PRAGMA table_info(snapshot_head)")] == [
        "scope",
        "generation",
        "revision",
        "progress",
    ]
    assert [row[1] for row in store.connection.execute("PRAGMA table_info(snapshot_body)")] == [
        "scope",
        "generation",
        "body",
    ]
    assert store.save(c.state, expected_revision=17) == 18
    store.close()


@pytest.mark.parametrize("missing", ["head", "body"])
def test_legacy_pair_migration_preserves_local_missing_side(tmp_path, missing):
    path = tmp_path / "broken_pair.db"
    legacy_pair_file(path, controller().state, missing=missing)
    store = SnapshotStore(path)
    with pytest.raises(ValueError, match="snapshot_storage_pair"):
        store.load(SCOPE)
    other_scope = SCOPE.model_copy(update={"generation": 5})
    other = controller().state.model_copy(update={"scope": other_scope})
    assert store.save(other, expected_revision=0) == 1
    assert store.load(other_scope)[1] == other
    assert store.connection.execute("SELECT count(*) FROM snapshot_" + missing).fetchone()[0] == 1
    store.close()


def test_legacy_pair_ddl_failure_rolls_back_both_tables(tmp_path):
    path = tmp_path / "old_pair.db"
    before = legacy_pair_file(path, controller().state)

    class InterruptedStore(SnapshotStore):
        def _create(self):
            super()._create()
            raise RuntimeError("interrupted upgrade")

    with pytest.raises(RuntimeError, match="interrupted"):
        InterruptedStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT revision,progress FROM snapshot_head").fetchone() == (
            17,
            before.progress,
        )
        assert db.execute("SELECT body FROM snapshot_body").fetchone() == (before.body,)
        assert "logical_bytes" in [r[1] for r in db.execute("PRAGMA table_info(snapshot_head)")]


def test_migration_is_format_only_preserves_revision(tmp_path, monkeypatch):
    c = controller()
    path = tmp_path / "old.db"
    legacy_file(path, c.state)
    for method in ("restore", "advance", "_prune"):
        monkeypatch.setattr(
            Controller, method, lambda *_a, **_k: pytest.fail("migration ran controller")
        )
    store = SnapshotStore(path)
    revision, state = store.load(SCOPE)
    assert revision == 17 and state.model_dump_json() == c.state.model_dump_json()
    assert not store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name='snapshots'"
    ).fetchall()
    assert store.save(state, expected_revision=17) == 18
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


def test_dirty_captures_signed_zero_and_nested_unknown_namespace_order():
    c = Controller(SCOPE, 0)
    c.set_state_entry("host_checkpoint", "extension", {"nested": {"a": 1, "b": 2}})
    first = freeze(c.state)
    c.take_changes()
    c._set(now=-0.0)
    assert c.take_changes() == (True, False)
    hot = freeze(c.state, body_changed=False, saved_layout=first.layout)
    assert hot.body is None
    assert thaw(SCOPE, hot.progress, first.body).model_dump_json() == c.state.model_dump_json()
    c.set_state_entry("host_checkpoint", "extension", {"nested": {"b": 2, "a": 1}})
    assert c.take_changes() == (False, True)
    reordered = freeze(c.state, body_changed=True, saved_layout=first.layout)
    assert (
        thaw(SCOPE, reordered.progress, reordered.body).model_dump_json()
        == c.state.model_dump_json()
    )
