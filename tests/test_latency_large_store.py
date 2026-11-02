"""Synthetic multi-generation SQLite recovery and legacy scope isolation."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from yuki_participation.controller import Controller
from yuki_participation.models import Scope
from yuki_participation.snapshot import freeze
from yuki_participation.store import SnapshotStore


@pytest.fixture
def artifact(tmp_path, request):
    directory = Path(os.environ.get("YUKI_LATENCY_ARTIFACT_DIR", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    evidence = {
        "scenario": request.node.name,
        "boundary": "real library Controller and local SQLite; synthetic data; no Host/network",
    }
    try:
        yield evidence
    finally:
        name = request.node.name.replace("[", "-").replace("]", "")
        (directory / f"{name}.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )


def test_more_than_16mib_across_generations_reopens_without_capping_scope(tmp_path, artifact):
    store = SnapshotStore(tmp_path / "large.sqlite3")
    padding = "x" * (1024 * 1024 + 4096)
    completed = []
    artifact.update(requested_generations=17, payload_bytes_per_generation=len(padding), saved=0)
    try:
        for number in range(1, 18):
            scope = Scope(conversation_id="synthetic-multigeneration", generation=number)
            controller = Controller(scope, 100)
            controller.set_state_entry(
                "host_checkpoint", "opaque_synthetic", {"padding": padding, "index": number}
            )
            value = freeze(controller.state)
            assert value.body is not None and len(value.body) > 1024 * 1024
            assert store.save_frozen(value, expected_revision=0) == 1
            completed.append((scope, len(value.body)))
            artifact["saved"] = len(completed)
        store.close()
        store = SnapshotStore(tmp_path / "large.sqlite3")
        for scope, length in completed:
            revision, loaded = store.load(scope)
            unknown = loaded.host_checkpoint["opaque_synthetic"]
            assert revision == 1 and unknown["index"] == scope.generation
            assert unknown["padding"] == padding and length > 1024 * 1024
        artifact.update(
            reopened=len(completed),
            body_bytes_total=sum(n for _, n in completed),
            all_payloads_preserved=True,
        )
        assert sum(length for _, length in completed) > 16 * 1024 * 1024
    finally:
        store.close()


@pytest.mark.parametrize("missing", ["head", "body"])
def test_legacy_orphan_scope_does_not_block_other_scope(tmp_path, artifact, missing):
    path = tmp_path / "legacy.sqlite3"
    good_scope = Scope(conversation_id="synthetic-good", generation=1)
    bad_scope = Scope(conversation_id="synthetic-broken", generation=1)
    good = freeze(Controller(good_scope, 100).state)
    bad = freeze(Controller(bad_scope, 100).state)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE snapshot_head(scope TEXT NOT NULL,generation INTEGER NOT NULL,"
            "revision INTEGER NOT NULL,progress BLOB NOT NULL,logical_bytes INTEGER NOT NULL,"
            "PRIMARY KEY(scope,generation))"
        )
        db.execute(
            "CREATE TABLE snapshot_body(scope TEXT NOT NULL,generation INTEGER NOT NULL,"
            "body BLOB NOT NULL,logical_size_delta INTEGER NOT NULL,PRIMARY KEY(scope,generation))"
        )
        db.execute(
            "INSERT INTO snapshot_head VALUES(?,?,?,?,?)",
            (good_scope.conversation_id, 1, 7, good.progress, 7),
        )
        db.execute(
            "INSERT INTO snapshot_body VALUES(?,?,?,?)",
            (good_scope.conversation_id, 1, good.body, 1),
        )
        if missing != "head":
            db.execute(
                "INSERT INTO snapshot_head VALUES(?,?,?,?,?)",
                (bad_scope.conversation_id, 1, 9, bad.progress, 9),
            )
        if missing != "body":
            db.execute(
                "INSERT INTO snapshot_body VALUES(?,?,?,?)",
                (bad_scope.conversation_id, 1, bad.body, 1),
            )
    store = SnapshotStore(path)
    try:
        assert store.load_raw(good_scope) == (7, good.progress, good.body)
        with pytest.raises(ValueError, match="snapshot_storage_pair"):
            store.load(bad_scope)
        with pytest.raises(ValueError, match="snapshot_storage_pair"):
            store.save_frozen(bad, expected_revision=9 if missing == "body" else 0)
        assert store.save_frozen(good, expected_revision=7) == 8
        assert store.load_raw(good_scope) == (8, good.progress, good.body)
        artifact.update(
            missing=missing,
            normal_scope_revision=8,
            original_good_bytes_retained=True,
            bad_scope_load_and_save_rejected=True,
        )
    finally:
        store.close()
