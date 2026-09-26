"""Concurrency regression tests for the CSV preview/confirm dedup gap.

The preview used to classify rows (valid/duplicate/invalid) *outside* any
transaction and only afterwards open the write transaction that saves the
table snapshot + voucher.  Another connection committing the same full_text
or external_id in that window produced a voucher whose "valid" rows were no
longer unique against the snapshotted table, so the confirm wrote duplicate
addresses.

The preview now runs on a dedicated per-request connection and takes the
SQLite write lock *first*: the existing-address reads, the classification,
the table snapshot and the voucher save all happen inside that one
``BEGIN IMMEDIATE`` transaction.  A concurrent import therefore either
commits before the lock is taken (its rows are seen and counted as
duplicates) or after the lock is released (the confirm rejects the voucher's
now-stale snapshot with 409).
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import db, ingest  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.main import app, get_conn  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _preview(client, csv_bytes: bytes, name: str = "p.csv") -> dict:
    r = client.post("/api/addresses/preview",
                    files={"file": (name, csv_bytes, "text/csv")})
    assert r.status_code == 200
    return r.json()


def _confirm(client, csv_bytes: bytes, token: str, name: str = "p.csv"):
    return client.post(
        "/api/addresses/confirm",
        files={"file": (name, csv_bytes, "text/csv")},
        data={"preview_token": token},
    )


def _concurrent_import(rows: list[dict]) -> dict:
    """A concurrent import committing on its own connection (what another
    worker's direct upload would do)."""
    other = db.connect()
    try:
        return ingest.add_many(other, rows)
    finally:
        other.close()


def _table_rows(full_text: str) -> list[sqlite3.Row]:
    """All address rows with this exact full_text (direct read, no paging)."""
    other = db.connect()
    try:
        return other.execute(
            "SELECT * FROM addresses WHERE full_text = ?", (full_text,)
        ).fetchall()
    finally:
        other.close()


def test_preview_and_confirm_use_dedicated_connections(client):
    """An open transaction on the shared connection must not interfere with
    preview/confirm: they pin their own per-request connections.  If they
    still used the shared connection, ``BEGIN IMMEDIATE`` would fail with
    "cannot start a transaction within a transaction" and the endpoints
    would return 500."""
    csv_bytes = b"external_id,address\nDED-1,60 Dedicated Conn Court\n"
    shared = get_conn()
    shared.execute("BEGIN")
    shared.execute("SELECT 1").fetchone()
    try:
        assert shared.in_transaction
        preview = _preview(client, csv_bytes, "d.csv")
        assert preview["valid"] == 1
        r = _confirm(client, csv_bytes, preview["preview_token"], "d.csv")
        assert r.status_code == 200
        assert r.json()["inserted"] == 1
    finally:
        shared.rollback()
    assert len(_table_rows("60 Dedicated Conn Court")) == 1


def test_classification_runs_inside_preview_write_transaction(client, monkeypatch):
    """The row classification must happen after the preview's write lock is
    taken, so no other connection can commit between the classification and
    the table snapshot (the window the old code left open)."""
    csv_bytes = b"external_id,address\nLOCK-1,70 Lock Window Road\n"
    in_transaction = []
    interloper_blocked = []
    real_plan_rows = ingest.plan_rows

    def plan_rows_spy(conn, rows, skip_duplicates=True):
        in_transaction.append(conn.in_transaction)
        # Another connection tries to commit the very same full_text right in
        # the middle of the classification.  The preview already holds the
        # SQLite write lock, so the interloper cannot commit here.
        other = sqlite3.connect(get_settings().db_path, timeout=0.05)
        try:
            other.execute(
                "INSERT INTO addresses"
                " (full_text, tokens_json, raw_json, created_at)"
                " VALUES ('70 Lock Window Road', '[]', '{}',"
                " '2000-01-01T00:00:00+00:00')"
            )
            other.commit()
        except sqlite3.OperationalError:
            interloper_blocked.append(True)
            other.rollback()
        finally:
            other.close()
        return real_plan_rows(conn, rows, skip_duplicates=skip_duplicates)

    monkeypatch.setattr(ingest, "plan_rows", plan_rows_spy)
    preview = _preview(client, csv_bytes)
    assert in_transaction == [True]
    assert interloper_blocked == [True]
    assert preview["valid"] == 1

    # The confirm writes exactly the previewed valid row -- the interloper
    # never landed -- and the write summary matches the preview counts.
    r = _confirm(client, csv_bytes, preview["preview_token"])
    assert r.status_code == 200
    assert r.json()["inserted"] == preview["valid"] == 1
    assert r.json()["duplicates"] == preview["duplicates"] == 0
    assert len(_table_rows("70 Lock Window Road")) == 1


def test_import_committed_before_preview_lock_counts_as_duplicate(client):
    """A concurrent import that commits *before* the preview takes its write
    lock is seen by the classification: rows matching its full_text or
    external_id are counted as duplicates, never written twice."""
    assert _concurrent_import([
        {"address": "40 Prelock Lane, Lockville", "external_id": "PRE-LOCK-1"},
        {"address": "41 Prelock Lane, Lockville", "external_id": "PRE-LOCK-2"},
    ])["inserted"] == 2

    csv_bytes = (
        b"external_id,address\n"
        # same external_id AND same full_text as the committed import
        b'PRE-LOCK-1,"40 Prelock Lane, Lockville"\n'
        # same external_id, different text -> still a duplicate
        b"PRE-LOCK-2,99 Other Prelock Road\n"
        # fresh row -> the only valid one
        b'PRE-LOCK-3,"42 Prelock Lane, Lockville"\n'
    )
    preview = _preview(client, csv_bytes)
    assert preview["valid"] == 1
    assert preview["duplicates"] == 2

    r = _confirm(client, csv_bytes, preview["preview_token"])
    assert r.status_code == 200
    result = r.json()
    # The write summary matches exactly what the preview showed.
    assert result["inserted"] == preview["valid"] == 1
    assert result["duplicates"] == preview["duplicates"] == 2
    # The pre-existing rows were not duplicated.
    assert len(_table_rows("40 Prelock Lane, Lockville")) == 1
    assert len(_table_rows("99 Other Prelock Road")) == 0
    assert len(_table_rows("42 Prelock Lane, Lockville")) == 1


def test_confirm_rejects_import_committed_after_preview(client):
    """The exact race from the bug report, now closed: the interloper commits
    the SAME full_text/external_id after the preview's lock is released but
    before the confirm.  The confirm must reject the stale snapshot (409)
    instead of writing a duplicate."""
    csv_bytes = b'external_id,address\nPOST-LOCK-1,"50 Postlock Way, Lockville"\n'
    preview = _preview(client, csv_bytes)
    assert preview["valid"] == 1

    # The concurrent import lands in the preview <-> confirm window.
    assert _concurrent_import([
        {"address": "50 Postlock Way, Lockville", "external_id": "POST-LOCK-1"},
    ])["inserted"] == 1

    r = _confirm(client, csv_bytes, preview["preview_token"])
    assert r.status_code == 409
    # The failed confirm wrote nothing: still exactly one such row.
    assert len(_table_rows("50 Postlock Way, Lockville")) == 1

    # Re-previewing against the new table state counts the row as a
    # duplicate; confirming that voucher writes nothing and stays consistent.
    preview2 = _preview(client, csv_bytes)
    assert preview2["valid"] == 0
    assert preview2["duplicates"] == 1
    r = _confirm(client, csv_bytes, preview2["preview_token"])
    assert r.status_code == 200
    assert r.json()["inserted"] == preview2["valid"] == 0
    assert r.json()["duplicates"] == preview2["duplicates"] == 1
    assert len(_table_rows("50 Postlock Way, Lockville")) == 1
