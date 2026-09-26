"""Ordered, versioned schema migrations (``app/db.py`` ``init_db``)."""
from __future__ import annotations

import threading

import pytest

from app import db


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    """Point the app at a fresh throwaway database file."""
    path = str(tmp_path / "migration_test.db")
    monkeypatch.setenv("ADDRESS_DB_PATH", path)
    return path


def _applied_versions(conn) -> list[int]:
    rows = conn.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    return [r[0] for r in rows]


def _match_log_columns(conn) -> set[str]:
    return {r[1] for r in conn.execute("PRAGMA table_info(match_logs)")}


def _table_names(conn) -> set[str]:
    return {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


# The pre-migration schema: match_logs without the consensus/snapshot columns
# and no schema_migrations version table at all.
LEGACY_SCHEMA = """
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


def test_fresh_database_migrates_to_current_version(db_path):
    conn = db.connect()
    try:
        db.init_db(conn)
        assert _applied_versions(conn) == [1, 2, 3]
        assert db.SCHEMA_VERSION == 3
        assert {
            "addresses",
            "match_logs",
            "upload_previews",
            "ab_experiments",
            "review_claims",
            "mapping_batches",
            "schema_migrations",
        } <= _table_names(conn)
        assert {
            "consensus_json",
            "selected_address_id",
            "selected_address_text",
        } <= _match_log_columns(conn)
    finally:
        conn.close()


def test_init_db_is_idempotent(db_path):
    conn = db.connect()
    try:
        db.init_db(conn)
        first = conn.execute(
            "SELECT version, applied_at FROM schema_migrations ORDER BY version"
        ).fetchall()
        # A second initialisation must not re-apply (or re-register) anything.
        db.init_db(conn)
        second = conn.execute(
            "SELECT version, applied_at FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [tuple(r) for r in first] == [tuple(r) for r in second]
    finally:
        conn.close()


def test_legacy_database_without_version_is_migrated(db_path):
    conn = db.connect()
    try:
        conn.executescript(LEGACY_SCHEMA)
        cur = conn.execute(
            "INSERT INTO addresses (full_text, tokens_json, raw_json, created_at)"
            " VALUES ('1 Legacy Lane', '[]', '{}', '2020-01-01T00:00:00+00:00')"
        )
        addr_id = cur.lastrowid
        conn.execute(
            "INSERT INTO match_logs"
            " (query, status, score, selected_id, candidates_json,"
            "  created_at, reviewed_at, review_action)"
            " VALUES ('legacy query', 'reviewed', 0.72, ?, '[]',"
            " '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00', 'confirmed')",
            (addr_id,),
        )
        conn.commit()
    finally:
        conn.close()

    conn = db.connect()
    try:
        db.init_db(conn)
        # Version registered, missing columns added, missing tables created.
        assert _applied_versions(conn) == [1, 2, 3]
        assert {
            "consensus_json",
            "selected_address_id",
            "selected_address_text",
        } <= _match_log_columns(conn)
        assert {
            "upload_previews",
            "ab_experiments",
            "review_claims",
            "mapping_batches",
        } <= _table_names(conn)
        # The review-snapshot backfill ran for the legacy reviewed row.
        row = conn.execute(
            "SELECT selected_address_id, selected_address_text FROM match_logs"
        ).fetchone()
        assert row["selected_address_id"] == addr_id
        assert row["selected_address_text"] == "1 Legacy Lane"
    finally:
        conn.close()


def test_concurrent_init_runs_each_migration_exactly_once(db_path):
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            conn = db.connect()
            try:
                barrier.wait(timeout=10)
                db.init_db(conn)
            finally:
                conn.close()
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors

    conn = db.connect()
    try:
        # Each migration was applied exactly once (the PRIMARY KEY would have
        # rejected a duplicate registration) and both connections agree on
        # the final version.
        assert _applied_versions(conn) == [1, 2, 3]
    finally:
        conn.close()


def test_newer_database_version_is_rejected_untouched(db_path):
    conn = db.connect()
    try:
        db.init_db(conn)
        conn.execute(
            "INSERT INTO addresses (full_text, tokens_json, raw_json, created_at)"
            " VALUES ('1 Future Road', '[]', '{}', '2026-01-01T00:00:00+00:00')"
        )
        # Simulate a database migrated by a newer program version.
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at)"
            " VALUES (?, '2026-01-01T00:00:00+00:00')",
            (db.SCHEMA_VERSION + 1,),
        )
        conn.commit()
        tables_before = _table_names(conn)
        columns_before = _match_log_columns(conn)

        with pytest.raises(db.SchemaVersionError):
            db.init_db(conn)

        # The refusal left the database completely untouched.
        assert _applied_versions(conn) == [1, 2, 3, db.SCHEMA_VERSION + 1]
        assert _table_names(conn) == tables_before
        assert _match_log_columns(conn) == columns_before
        assert conn.execute("SELECT COUNT(*) FROM addresses").fetchone()[0] == 1
        # The connection is still usable (the failed init rolled back cleanly).
        assert not conn.in_transaction
    finally:
        conn.close()


def test_failed_migration_rolls_back_and_next_start_resumes(db_path):
    original = db.MIGRATIONS[2]

    def broken_step(conn) -> None:
        conn.execute("ALTER TABLE match_logs ADD COLUMN boom TEXT")
        raise RuntimeError("boom")

    db.MIGRATIONS[2] = broken_step
    try:
        conn = db.connect()
        try:
            with pytest.raises(RuntimeError, match="boom"):
                db.init_db(conn)
            # Step 1 committed; the failed step rolled back entirely: its
            # partial schema change is gone and its version was never
            # registered.
            assert _applied_versions(conn) == [1]
            assert "boom" not in _match_log_columns(conn)
        finally:
            conn.close()
    finally:
        db.MIGRATIONS[2] = original

    # A later start (fresh connection, fixed program) resumes from the
    # version the database was left at and completes the remaining steps.
    conn = db.connect()
    try:
        db.init_db(conn)
        assert _applied_versions(conn) == [1, 2, 3]
        assert {
            "consensus_json",
            "selected_address_id",
            "selected_address_text",
        } <= _match_log_columns(conn)
        assert "boom" not in _match_log_columns(conn)
    finally:
        conn.close()
