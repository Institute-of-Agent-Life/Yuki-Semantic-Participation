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
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
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
        elif head:
            head_columns = {
                r[1] for r in self.connection.execute("PRAGMA table_info(snapshot_head)")
            }
            body_columns = {
                r[1] for r in self.connection.execute("PRAGMA table_info(snapshot_body)")
            }
            head_size = "logical_bytes" in head_columns
            body_size = "logical_size_delta" in body_columns
            if head_size != body_size:
                raise ValueError("snapshot_storage_format")
            if head_size:
                self._migrate_pair()
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
            progress BLOB NOT NULL,
            PRIMARY KEY(scope,generation))""")
        self.connection.execute("""CREATE TABLE snapshot_body(
            scope TEXT NOT NULL,generation INTEGER NOT NULL,body BLOB NOT NULL,
            PRIMARY KEY(scope,generation))""")

    def _migrate_pair(self) -> None:
        """Remove size bookkeeping, retaining each existing pair and its exact bytes."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute("ALTER TABLE snapshot_head RENAME TO snapshot_head_old")
            self.connection.execute("ALTER TABLE snapshot_body RENAME TO snapshot_body_old")
            self._create()
            self.connection.execute("""INSERT INTO snapshot_head(scope,generation,revision,progress)
                SELECT scope,generation,revision,progress FROM snapshot_head_old""")
            self.connection.execute("""INSERT INTO snapshot_body(scope,generation,body)
                SELECT scope,generation,body FROM snapshot_body_old""")
            self.connection.execute("DROP TABLE snapshot_head_old")
            self.connection.execute("DROP TABLE snapshot_body_old")
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

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
            prepared.append((scope, generation, revision, frozen))
        expected = {(scope, generation, revision) for scope, generation, revision, _ in rows}
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            current = set(
                self.connection.execute("SELECT scope,generation,revision FROM snapshots")
            )
            if current != expected:
                raise SnapshotConflict("snapshot_changed")
            self._create()
            for scope, generation, revision, value in prepared:
                self.connection.execute(
                    "INSERT INTO snapshot_head VALUES(?,?,?,?)",
                    (scope, generation, revision, value.progress),
                )
                self.connection.execute(
                    "INSERT INTO snapshot_body VALUES(?,?,?)",
                    (scope, generation, value.body),
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
                SELECT h.revision,b.scope FROM wanted w
                LEFT JOIN snapshot_head h USING(scope,generation)
                LEFT JOIN snapshot_body b USING(scope,generation)""",
                key,
            ).fetchone()
            if (row[0] if row[0] is not None else 0) != expected_revision:
                raise SnapshotConflict("snapshot_changed")
            if (row[0] is None) != (row[1] is None):
                raise ValueError("snapshot_storage_pair")
            if value.body is None and row[1] is None:
                raise ValueError("snapshot_body_missing")
            revision = expected_revision + 1
            if value.body is not None:
                self.connection.execute(
                    """INSERT INTO snapshot_body VALUES(?,?,?)
                    ON CONFLICT(scope,generation) DO UPDATE SET body=excluded.body""",
                    (*key, value.body),
                )
            self.connection.execute(
                """INSERT INTO snapshot_head VALUES(?,?,?,?)
                ON CONFLICT(scope,generation) DO UPDATE SET revision=excluded.revision,
                progress=excluded.progress""",
                (*key, revision, value.progress),
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
                SELECT h.revision,b.scope FROM wanted w
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
