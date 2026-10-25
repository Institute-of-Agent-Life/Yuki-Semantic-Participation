"""A single CAS snapshot, with numerical progress isolated from its cold body."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

from .controller import State
from .models import Scope
from .snapshot import SnapshotWrite, freeze, thaw


class SnapshotConflict(RuntimeError):
    pass


class SnapshotStore:
    def __init__(
        self, path: Path, *, per_scope_bytes: int = 1024 * 1024, total_bytes: int = 16 * 1024 * 1024
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.per_scope_bytes = per_scope_bytes
        self.total_bytes = total_bytes
        try:
            self._initialize()
        except BaseException:
            self.connection.close()
            raise

    def _initialize(self) -> None:
        tables = {
            r[0]
            for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        old = "snapshots" in tables
        head, body = "snapshot_head" in tables, "snapshot_body" in tables
        if head != body or (old and head):
            raise ValueError("snapshot_storage_format")
        if old:
            self._migrate()
        elif not head:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self._create()
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def _create(self) -> None:
        self.connection.execute("""CREATE TABLE snapshot_head(
            scope TEXT NOT NULL,generation INTEGER NOT NULL,revision INTEGER NOT NULL,
            progress BLOB NOT NULL,logical_bytes INTEGER NOT NULL,
            PRIMARY KEY(scope,generation))""")
        self.connection.execute("""CREATE TABLE snapshot_body(
            scope TEXT NOT NULL,generation INTEGER NOT NULL,body BLOB NOT NULL,
            logical_size_delta INTEGER NOT NULL,PRIMARY KEY(scope,generation))""")

    def _migrate(self) -> None:
        from .snapshot_v1 import LegacyState

        rows = self.connection.execute(
            "SELECT scope,generation,revision,payload FROM snapshots"
        ).fetchall()
        prepared = []
        for scope, generation, revision, payload in rows:
            legacy = LegacyState.model_validate_json(payload)
            key = Scope(conversation_id=scope, generation=generation)
            if legacy.scope != key:
                raise ValueError("snapshot_scope_mismatch")
            frozen = freeze(cast(State, legacy))
            length = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
            prepared.append((scope, generation, revision, frozen, length))
        expected = {(scope, generation, revision) for scope, generation, revision, _ in rows}
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            current = set(
                self.connection.execute("SELECT scope,generation,revision FROM snapshots")
            )
            if current != expected:
                raise SnapshotConflict("snapshot_changed")
            self._create()
            for scope, generation, revision, value, length in prepared:
                self.connection.execute(
                    "INSERT INTO snapshot_head VALUES(?,?,?,?,?)",
                    (scope, generation, revision, value.progress, length),
                )
                self.connection.execute(
                    "INSERT INTO snapshot_body VALUES(?,?,?,?)",
                    (scope, generation, value.body, value.logical_size_delta),
                )
            self.connection.execute("DROP TABLE snapshots")
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def load_raw(self, scope: Scope) -> tuple[int, bytes, bytes] | None:
        row = self.connection.execute(
            """WITH wanted(scope,generation) AS (VALUES(?,?))
            SELECT h.revision,h.progress,b.body FROM wanted w
            LEFT JOIN snapshot_head h USING(scope,generation)
            LEFT JOIN snapshot_body b USING(scope,generation)""",
            (scope.conversation_id, scope.generation),
        ).fetchone()
        if row == (None, None, None):
            return None
        if row[0] is None or row[1] is None or row[2] is None:
            raise ValueError("snapshot_storage_pair")
        return row[0], row[1], row[2]

    def load(self, scope: Scope) -> tuple[int, State] | None:
        row = self.load_raw(scope)
        return (row[0], thaw(scope, row[1], row[2])) if row is not None else None

    def save(self, state: State, *, expected_revision: int) -> int:
        return self.save_frozen(freeze(state), expected_revision=expected_revision)

    def save_frozen(self, value: SnapshotWrite, *, expected_revision: int) -> int:
        key = (value.scope.conversation_id, value.scope.generation)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """WITH wanted(scope,generation) AS (VALUES(?,?))
                SELECT h.revision,h.logical_bytes,b.logical_size_delta FROM wanted w
                LEFT JOIN snapshot_head h USING(scope,generation)
                LEFT JOIN snapshot_body b USING(scope,generation)""",
                key,
            ).fetchone()
            if (row[0] if row[0] is not None else 0) != expected_revision:
                raise SnapshotConflict("snapshot_changed")
            if (row[0] is None) != (row[2] is None):
                raise ValueError("snapshot_storage_pair")
            delta = value.logical_size_delta if value.body is not None else row[2]
            if delta is None:
                raise ValueError("snapshot_body_missing")
            length = delta + len(value.progress)
            if length > self.per_scope_bytes:
                raise ValueError("scope_snapshot_capacity")
            used = self.connection.execute(
                "SELECT coalesce(sum(logical_bytes),0) FROM snapshot_head"
            ).fetchone()[0]
            if used - (row[1] or 0) + length > self.total_bytes:
                raise ValueError("global_snapshot_capacity")
            revision = expected_revision + 1
            if value.body is not None:
                self.connection.execute(
                    """INSERT INTO snapshot_body VALUES(?,?,?,?)
                    ON CONFLICT(scope,generation) DO UPDATE SET body=excluded.body,
                    logical_size_delta=excluded.logical_size_delta""",
                    (*key, value.body, delta),
                )
            self.connection.execute(
                """INSERT INTO snapshot_head VALUES(?,?,?,?,?)
                ON CONFLICT(scope,generation) DO UPDATE SET revision=excluded.revision,
                progress=excluded.progress,logical_bytes=excluded.logical_bytes""",
                (*key, revision, value.progress, length),
            )
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        return revision

    def delete_scope(self, scope: Scope, *, expected_revision: int) -> None:
        """Only the host retires real scopes after reconciling their original receipts."""
        key = (scope.conversation_id, scope.generation)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """WITH wanted(scope,generation) AS (VALUES(?,?))
                SELECT h.revision,b.logical_size_delta FROM wanted w
                LEFT JOIN snapshot_head h USING(scope,generation)
                LEFT JOIN snapshot_body b USING(scope,generation)""",
                key,
            ).fetchone()
            if (row[0] is None) != (row[1] is None):
                raise ValueError("snapshot_storage_pair")
            if (row[0] if row[0] is not None else 0) != expected_revision:
                raise SnapshotConflict("snapshot_changed")
            self.connection.execute("DELETE FROM snapshot_head WHERE scope=? AND generation=?", key)
            self.connection.execute("DELETE FROM snapshot_body WHERE scope=? AND generation=?", key)
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.connection.close()
