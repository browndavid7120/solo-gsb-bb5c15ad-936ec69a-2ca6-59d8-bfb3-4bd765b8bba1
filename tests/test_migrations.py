"""Ordered, versioned schema migrations (app/db.py ``init_db``)."""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager

import pytest

from app import db


@contextmanager
def _temp_db():
    """Point ADDRESS_DB_PATH at a fresh throwaway file for the block."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # the app creates the file itself
    old = os.environ.get("ADDRESS_DB_PATH")
    os.environ["ADDRESS_DB_PATH"] = path
    try:
        yield path
    finally:
        if old is None:
            os.environ.pop("ADDRESS_DB_PATH", None)
        else:
            os.environ["ADDRESS_DB_PATH"] = old
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(path + suffix)
            except OSError:
                pass


def _version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {r[0] for r in rows}


# The match_logs table as it existed before consensus_json and the durable
# review-choice snapshot columns were introduced.
_LEGACY_MATCH_LOGS = """
CREATE TABLE match_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    query            TEXT NOT NULL,
    status           TEXT NOT NULL,
    score            REAL,
    selected_id      INTEGER REFERENCES addresses(id) ON DELETE SET NULL,
    candidate_id     INTEGER REFERENCES addresses(id) ON DELETE SET NULL,
    token_score      REAL,
    house_score      REAL,
    matched_tokens   TEXT,
    conflicts        TEXT,
    candidates_json  TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    reviewed_at      TEXT,
    review_action    TEXT
);
"""

_LEGACY_ADDRESSES = """
CREATE TABLE addresses (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    full_text            TEXT NOT NULL,
    tokens_json          TEXT NOT NULL,
    street_tokens_json   TEXT NOT NULL DEFAULT '[]',
    locality_tokens_json TEXT NOT NULL DEFAULT '[]',
    house                TEXT,
    line1                TEXT,
    city                 TEXT,
    state                TEXT,
    postal_code          TEXT,
    country              TEXT,
    external_id          TEXT,
    raw_json             TEXT NOT NULL,
    created_at           TEXT NOT NULL
);
"""


def _build_legacy_db(conn: sqlite3.Connection) -> int:
    """Create a pre-versioning database (user_version 0, old columns) with one
    address and one reviewed log missing its snapshot; return the log id."""
    conn.executescript(_LEGACY_ADDRESSES)
    conn.executescript(_LEGACY_MATCH_LOGS)
    cur = conn.execute(
        """
        INSERT INTO addresses
            (full_text, tokens_json, raw_json, created_at)
        VALUES ('100 Legacy Lane, Oldtown', '[]', '{}', '2020-01-01T00:00:00+00:00')
        """
    )
    addr_id = int(cur.lastrowid)
    cur = conn.execute(
        """
        INSERT INTO match_logs
            (query, status, score, selected_id, candidates_json,
             created_at, reviewed_at, review_action)
        VALUES ('legacy query', 'reviewed', 0.72, ?, '[]',
                '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00',
                'confirmed')
        """,
        (addr_id,),
    )
    conn.commit()
    assert _version(conn) == 0  # no versioning back then
    return int(cur.lastrowid)


def test_fresh_database_gets_current_version_and_full_schema():
    with _temp_db():
        conn = db.connect()
        try:
            db.init_db(conn)
            assert _version(conn) == db.SCHEMA_VERSION
            assert {
                "addresses", "match_logs", "upload_previews",
                "ab_experiments", "review_claims", "mapping_batches",
            } <= _tables(conn)
            cols = _columns(conn, "match_logs")
            assert {"consensus_json", "selected_address_id",
                    "selected_address_text"} <= cols
            # The connection is left clean (no open transaction).
            assert not conn.in_transaction
        finally:
            conn.close()


def test_legacy_unversioned_database_is_migrated_and_backfilled():
    with _temp_db():
        conn = db.connect()
        try:
            log_id = _build_legacy_db(conn)
            db.init_db(conn)

            assert _version(conn) == db.SCHEMA_VERSION
            # Missing columns were added to the legacy table ...
            cols = _columns(conn, "match_logs")
            assert {"consensus_json", "selected_address_id",
                    "selected_address_text"} <= cols
            # ... tables introduced later were created ...
            assert {"upload_previews", "ab_experiments",
                    "review_claims", "mapping_batches"} <= _tables(conn)
            # ... and the review snapshot was backfilled from the live row.
            row = conn.execute(
                "SELECT selected_address_id, selected_address_text"
                " FROM match_logs WHERE id = ?",
                (log_id,),
            ).fetchone()
            addr = conn.execute("SELECT id, full_text FROM addresses").fetchone()
            assert row["selected_address_id"] == addr["id"]
            assert row["selected_address_text"] == addr["full_text"]
        finally:
            conn.close()


def test_completed_migrations_are_not_re_executed():
    with _temp_db():
        conn = db.connect()
        try:
            db.init_db(conn)
            # A "legacy-style" row written after the database reached the
            # current version must stay untouched: the backfill migration
            # already ran and is never repeated.
            addr_id = db.insert_address(
                conn, full_text="1 Later Ave", tokens=[], house="1",
                fields={}, raw={},
            )
            cur = conn.execute(
                """
                INSERT INTO match_logs
                    (query, status, score, selected_id, candidates_json,
                     created_at, reviewed_at, review_action)
                VALUES ('later query', 'reviewed', 0.5, ?, '[]',
                        '2021-01-01T00:00:00+00:00', '2021-01-01T00:00:00+00:00',
                        'confirmed')
                """,
                (addr_id,),
            )
            log_id = int(cur.lastrowid)
            conn.commit()

            db.init_db(conn)  # second run on the same connection
            other = db.connect()  # and a fresh connection on the same file
            try:
                db.init_db(other)
            finally:
                other.close()

            assert _version(conn) == db.SCHEMA_VERSION
            row = conn.execute(
                "SELECT selected_address_id, selected_address_text"
                " FROM match_logs WHERE id = ?",
                (log_id,),
            ).fetchone()
            assert row["selected_address_id"] is None
            assert row["selected_address_text"] is None
        finally:
            conn.close()


def test_newer_database_version_is_refused_without_touching_the_file():
    with _temp_db():
        conn = db.connect()
        try:
            conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
            conn.commit()
            with pytest.raises(db.SchemaVersionError):
                db.init_db(conn)
            # Nothing was written: version unchanged, no tables created.
            assert _version(conn) == db.SCHEMA_VERSION + 1
            assert _tables(conn) == set()
            assert not conn.in_transaction
        finally:
            conn.close()


def test_failed_migration_rolls_back_and_startup_can_rerun(monkeypatch):
    with _temp_db():
        conn = db.connect()
        try:
            def _boom(c: sqlite3.Connection) -> None:
                c.execute("CREATE TABLE migration_marker (id INTEGER)")
                raise RuntimeError("boom")

            patched = (db.MIGRATIONS[0], _boom, db.MIGRATIONS[2])
            monkeypatch.setattr(db, "MIGRATIONS", patched)

            with pytest.raises(RuntimeError, match="boom"):
                db.init_db(conn)
            # The failed step rolled back completely: its table is gone and
            # only the first (committed) migration's version is recorded.
            assert _version(conn) == 1
            assert "migration_marker" not in _tables(conn)
            assert "addresses" in _tables(conn)
            assert not conn.in_transaction

            monkeypatch.undo()
            # A later startup resumes from the recorded version and finishes.
            db.init_db(conn)
            assert _version(conn) == db.SCHEMA_VERSION
            assert {"consensus_json", "selected_address_id",
                    "selected_address_text"} <= _columns(conn, "match_logs")
        finally:
            conn.close()


def test_concurrent_init_runs_each_migration_exactly_once(monkeypatch):
    with _temp_db():
        counts = [0] * db.SCHEMA_VERSION
        count_lock = threading.Lock()
        original = db.MIGRATIONS

        def _counted(index, fn):
            def run(conn: sqlite3.Connection) -> None:
                with count_lock:
                    counts[index] += 1
                fn(conn)
            return run

        monkeypatch.setattr(
            db, "MIGRATIONS",
            tuple(_counted(i, fn) for i, fn in enumerate(original)),
        )

        # Connections are created up front (connect() itself is not the racing
        # part); the four init_db calls are released simultaneously.
        conns = [db.connect() for _ in range(4)]
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def worker(conn: sqlite3.Connection) -> None:
            try:
                barrier.wait(timeout=10)
                db.init_db(conn)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(c,)) for c in conns]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        for c in conns:
            c.close()

        assert not errors
        # Every migration ran exactly once across all racing connections.
        assert counts == [1] * db.SCHEMA_VERSION

        conn = db.connect()
        try:
            assert _version(conn) == db.SCHEMA_VERSION
        finally:
            conn.close()
