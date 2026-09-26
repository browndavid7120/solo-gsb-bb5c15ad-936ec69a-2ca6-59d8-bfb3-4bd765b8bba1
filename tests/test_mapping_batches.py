"""Tests for the external-address mapping batch feature."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import ingest  # noqa: E402
from app.main import app, get_conn, get_settings  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _isolated_tables(client):
    """Reset the shared test DB so other modules' uploads/logs don't leak in."""
    conn = get_conn()
    conn.execute("DELETE FROM mapping_batches")
    conn.execute("DELETE FROM match_logs")
    conn.execute("DELETE FROM addresses")
    conn.commit()
    sample = Path(get_settings().sample_csv)
    ingest.add_many(conn, ingest.parse_csv(sample.read_bytes()))
    yield
    conn.execute("DELETE FROM mapping_batches")
    conn.commit()


def _csv(text: str) -> bytes:
    return text.encode("utf-8")


def _std_id(external_id: str) -> int:
    """Current id of a seeded sample address (autoincrement does not reset
    after the per-module DELETE/reload, so ids must be resolved dynamically)."""
    row = get_conn().execute(
        "SELECT id FROM addresses WHERE external_id = ?", (external_id,)
    ).fetchone()
    assert row is not None, external_id
    return int(row[0])


def _upload(client, content: bytes, name: str = "batch.csv"):
    return client.post(
        "/api/mapping-batches",
        files={"file": (name, content, "text/csv")},
    )


# --------------------------------------------------------------- happy path


def test_batch_scores_rows_on_one_snapshot_and_persists(client):
    content = _csv(
        "biz_no,address\n"
        # Commas inside the address require CSV quoting; the raw text (including the
        # commas) must be stored verbatim.
        'A-1,"123 Main Street, New York, NY 10001"\n'  # auto
        'A-2,"1010 Lake Dr, Austin TX"\n'                # needs review
        'A-3,"1 Milky Way, Mars Base"\n'          # no match
    )
    r = _upload(client, content)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["id"].startswith("MAP-")
    assert data["address_count"] >= 15
    assert data["table_snapshot"]["count"] == data["address_count"]
    assert data["summary"] == {
        "total": 3, "auto_matched": 1, "needs_review": 1,
        "no_match": 1, "id_conflict": 0, "unresolved": 2,
    }
    rows = {row["biz_no"]: row for row in data["rows"]}
    main_street = _std_id("STD-001")
    a1 = rows["A-1"]
    assert a1["status"] == "auto_matched"
    assert a1["target_id"] == main_street
    assert a1["target_full_text"] == "123 Main Street"
    a2 = rows["A-2"]
    assert a2["status"] == "needs_review"
    assert a2["target_id"] is None
    assert a2["best_id"] is not None and a2["candidates"]
    assert a2["reason"].startswith("未自动命中（待确认）")
    a3 = rows["A-3"]
    assert a3["status"] == "no_match"
    assert a3["target_id"] is None
    # A best candidate is still scored (here below the reject threshold), but it
    # is never accepted as the mapping target.
    assert a3["best_score"] is not None and a3["best_score"] < 0.35
    assert a3["reason"].startswith("未自动命中（无匹配）")

    # The batch is persisted with its snapshot digest and can be reviewed by id.
    got = client.get(f"/api/mapping-batches/{data['id']}")
    assert got.status_code == 200
    body = got.json()
    assert body["id"] == data["id"]
    assert body["table_snapshot"]["digest"] == data["table_snapshot"]["digest"]
    assert {r["raw_address"] for r in body["rows"]} == {
        "123 Main Street, New York, NY 10001",
        "1010 Lake Dr, Austin TX",
        "1 Milky Way, Mars Base",
    }

    listing = client.get("/api/mapping-batches")
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert items[0]["id"] == data["id"]
    assert items[0]["summary"]["total"] == 3
    assert "rows" not in items[0]


def test_batch_does_not_write_match_logs(client):
    before = client.get("/api/health").json()["needs_review"]
    r = _upload(client, _csv(
        "biz_no,address\n"
        "A-2,1010 Lake Dr, Austin TX\n"
        "A-3,1 Milky Way, Mars Base\n"
    ))
    assert r.status_code == 200
    # Needs-review rows from a batch must not appear in the ordinary review queue.
    after = client.get("/api/health").json()["needs_review"]
    assert after == before
    assert get_conn().execute("SELECT COUNT(*) FROM match_logs").fetchone()[0] == 0


def test_non_csv_extension_rejected(client):
    r = client.post(
        "/api/mapping-batches",
        files={"file": ("x.txt", b"biz_no,address\n", "text/plain")},
    )
    assert r.status_code == 400


# ----------------------------------------------------------- id conflicts


def test_same_biz_no_different_addresses_is_conflict_never_picked(client):
    r = _upload(client, _csv(
        "biz_no,address\n"
        "B-1,50 Hillcrest Road, Dallas TX 75201\n"
        "B-1,52 Hillcrest Road, Dallas TX 75202\n"
        "C-1,742 Evergreen Tce\n"
        "C-1,742 Evergreen Tce\n"           # same biz + same address: not a conflict
    ))
    assert r.status_code == 200, r.text
    hill_8 = _std_id("STD-008")
    hill_9 = _std_id("STD-009")
    evergreen = _std_id("STD-015")
    s = r.json()["summary"]
    assert s["id_conflict"] == 2
    assert s["auto_matched"] == 2
    assert s["unresolved"] == 2

    for row in r.json()["rows"]:
        if row["biz_no"] == "B-1":
            assert row["status"] == "id_conflict"
            # Never auto-assign a target, even though each row would auto-hit.
            assert row["target_id"] is None
            assert row["target_full_text"] is None
            # The underlying scoring and candidates are still visible on review.
            assert row["match_status"] == "auto_matched"
            assert row["best_id"] in (hill_8, hill_9)
            assert row["candidates"]
            assert "B-1" in row["reason"] and "编号冲突" in row["reason"]
            assert "不任选一行" in row["reason"]
        else:
            assert row["biz_no"] == "C-1"
            assert row["status"] == "auto_matched"
            assert row["target_id"] == evergreen


def test_three_way_conflict_lists_every_member_line(client):
    r = _upload(client, _csv(
        "biz_no,address\n"
        "D-1,50 Hillcrest Road, Dallas TX 75201\n"
        "D-1,52 Hillcrest Road, Dallas TX 75202\n"
        "D-1,1010 Lake Dr, Austin TX\n"
    ))
    assert r.status_code == 200
    assert r.json()["summary"]["id_conflict"] == 3
    reason = r.json()["rows"][0]["reason"]
    assert "第 2 行" in reason and "第 3 行" in reason and "第 4 行" in reason


# --------------------------------------------------------- CSV validation


@pytest.mark.parametrize("header", [
    "id,foo\n",                 # no address column
    "biz_no,foo\n",             # no address column
    "reference,city\n",        # only a locality column, no line1/address
])
def test_missing_required_column_rejected(client, header):
    r = _upload(client, _csv(header + "X,New York\n"))
    assert r.status_code == 400
    assert "列" in r.json()["detail"]


def test_empty_biz_no_or_address_rejected(client):
    r1 = _upload(client, _csv("biz_no,address\n,742 Evergreen Tce\n"))
    assert r1.status_code == 400 and "业务编号为空" in r1.json()["detail"]
    r2 = _upload(client, _csv("biz_no,address\nX,\n"))
    assert r2.status_code == 400 and "原始地址为空" in r2.json()["detail"]


def test_no_data_rows_rejected(client):
    r = _upload(client, _csv("biz_no,address\n"))
    assert r.status_code == 400 and "数据行" in r.json()["detail"]


def test_more_than_100_rows_rejected(client):
    body = "biz_no,address\n" + "".join(
        f"X{i},742 Evergreen Tce\n" for i in range(101))
    r = _upload(client, _csv(body))
    assert r.status_code == 400 and "100" in r.json()["detail"]


def test_exactly_100_rows_accepted(client):
    body = "biz_no,address\n" + "".join(
        f"X{i},742 Evergreen Tce\n" for i in range(100))
    r = _upload(client, _csv(body))
    assert r.status_code == 200
    assert r.json()["summary"]["total"] == 100


def test_blank_lines_are_not_counted(client):
    # A blank physical line is skipped like in the standard-address importer.
    r = _upload(client, _csv(
        "biz_no,address\n"
        "A-1,742 Evergreen Tce\n"
        "\n"
        "A-2,742 Evergreen Tce\n"
    ))
    assert r.status_code == 200
    assert r.json()["summary"]["total"] == 2


def test_header_aliases_and_structured_fallback(client):
    r = _upload(client, _csv(
        "ref,line1,city,state\n"
        "R-9,123 Main Street,New York,NY\n"
    ))
    assert r.status_code == 200, r.text
    row = r.json()["rows"][0]
    assert row["biz_no"] == "R-9"
    assert row["raw_address"] == "123 Main Street, New York, NY"
    assert row["target_id"] == _std_id("STD-001")


# ------------------------------------------------------------------- export


def test_export_csv_while_snapshot_unchanged(client):
    r = _upload(client, _csv(
        "biz_no,address\n"
        'A-1,"123 Main Street, New York, NY 10001"\n'
        "A-2,1010 Lake Dr, Austin TX\n"
        "B-1,50 Hillcrest Road, Dallas TX 75201\n"
        "B-1,52 Hillcrest Road, Dallas TX 75202\n"
    ))
    batch_id = r.json()["id"]
    er = client.get(f"/api/mapping-batches/{batch_id}/export")
    assert er.status_code == 200
    assert er.headers["content-type"].startswith("text/csv")
    assert "attachment" in er.headers["content-disposition"]
    # BOM first, then the header row
    raw = er.content
    assert raw[:3] == b"\xef\xbb\xbf"
    decoded_text = raw.decode("utf-8-sig")
    table = list(csv.reader(io.StringIO(decoded_text)))
    assert table[0] == [
        "业务编号", "原始地址", "匹配状态", "标准地址ID",
        "标准地址全文", "首选候选得分", "原因",
    ]
    by_biz = {row[0]: row for row in table[1:]}
    # Auto hit: target filled in.
    assert by_biz["A-1"][2] == "自动命中"
    assert by_biz["A-1"][3] == str(_std_id("STD-001"))
    assert by_biz["A-1"][4] == "123 Main Street"
    # Not auto-hit: original biz kept, empty target id, explicit reason.
    assert by_biz["A-2"][3] == ""
    assert by_biz["A-2"][5]  # best-candidate score is still recorded
    assert by_biz["A-2"][6].startswith("未自动命中（待确认）")
    # Id conflict: empty target + conflict reason for every member row.
    conflict_rows = [row for row in table[1:] if row[0] == "B-1"]
    assert len(conflict_rows) == 2
    for row in conflict_rows:
        assert row[2] == "编号冲突"
        assert row[3] == ""
        assert "编号冲突" in row[6]


def test_export_refused_after_table_change(client):
    r = _upload(client, _csv(
        "biz_no,address\nA-1,742 Evergreen Tce\n"))
    batch_id = r.json()["id"]
    # Mutate the standard-address table after the batch was built.
    conn = get_conn()
    ingest.add_many(conn, [{"external_id": "NEW-1",
                          "full_text": "999 Brand New Road, Nowhere NH 03000"}])
    er = client.get(f"/api/mapping-batches/{batch_id}/export")
    assert er.status_code == 409
    assert "快照" in er.json()["detail"] and "重新" in er.json()["detail"]

    # The batch can still be reviewed (its frozen results are untouched).
    assert client.get(f"/api/mapping-batches/{batch_id}").status_code == 200


def test_export_unknown_batch_404(client):
    r = client.get("/api/mapping-batches/MAP-does-not-exist/export")
    assert r.status_code == 404


def test_get_unknown_batch_404(client):
    assert client.get("/api/mapping-batches/MAP-nope").status_code == 404
