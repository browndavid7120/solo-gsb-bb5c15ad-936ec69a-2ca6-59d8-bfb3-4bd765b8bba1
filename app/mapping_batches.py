"""外部地址映射批次 (external-address mapping batches).

A batch is a web-uploaded CSV of at most 100 rows, each carrying a **business
id** (``biz_no``) and a **raw address** (``address``).  Every row is scored
with the **existing matcher** under the **default rules**, against one single
frozen snapshot of the standard-address table; the rows are read once inside a
``BEGIN IMMEDIATE`` transaction and the snapshot digest is persisted alongside
the per-row results, exactly like the A/B experiments.

Batch runs are pure scoring: the review-consensus memory is **not** consulted
and **nothing is written to ``match_logs``**.

Rows sharing the same business id but carrying different addresses are an
**id conflict** (``id_conflict``): every such row is marked, its target left
empty and a conflict reason naming the other raw addresses is recorded -- the
server never silently picks one row.

A batch can be reviewed later (raw text, match status, candidates and the
conflict reason are all persisted).  Downloading the complete mapping CSV
re-checks the **current** address-table digest against the batch snapshot:
when the table changed since the batch ran, the export is refused (409) and
the user must build a new batch.  Rows that were not auto-hit (or that carry
an id conflict) keep their original business id with an empty target id and
an explicit reason column.
"""
from __future__ import annotations

import codecs
import csv
import io
import json
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Any

from . import db, ingest
from .config import Settings
from .matcher import match_address, result_to_dict

# A batch CSV may contain at most this many data rows (the header is free).
MAX_ROWS = 100

# Header aliases recognised for the two required input columns.  Earlier
# entries win when a CSV contains more than one matching header, so specific
# names (``biz_no``) take precedence over generic ones (``id``).
BIZ_NO_ALIASES = (
    "biz_no", "biz_id", "business_id", "business_no", "business_code",
    "external_id", "ref", "ref_no", "reference", "order_no", "order_id",
    "业务编号", "业务id", "客户编号", "外部编号", "id",
)
ADDRESS_ALIASES = (
    "address", "addr", "raw_address", "original_address",
    "full_address", "formatted_address", "full_text", "standard_address",
    "原始地址", "地址",
)
# Accepted as the address source when no direct address column exists: the
# raw address is then composed from line1 + city/state/postal_code/country.
ADDRESS_PART_ALIASES = ("line1", "street", "address_line1")

STATUS_AUTO = "auto_matched"
STATUS_REVIEW = "needs_review"
STATUS_NO_MATCH = "no_match"
STATUS_ID_CONFLICT = "id_conflict"

STATUS_LABELS = {
    STATUS_AUTO: "自动命中",
    STATUS_REVIEW: "待确认",
    STATUS_NO_MATCH: "无匹配",
    STATUS_ID_CONFLICT: "编号冲突",
}


class MappingBatchError(Exception):
    """The uploaded batch CSV cannot be processed (HTTP 400)."""


class MappingBatchNotFound(Exception):
    """The requested batch id does not exist (HTTP 404)."""


class MappingBatchTableChanged(Exception):
    """The address table changed since the batch ran; export is refused (409)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ CSV ingestion --


def _pick_column(fields: list[str], aliases: tuple[str, ...]) -> int | None:
    """Index of the highest-priority matching header.

    Alias priority (earlier in ``aliases`` wins) beats header order, so a
    specific name such as ``biz_no`` is preferred over a generic ``id`` when
    a CSV happens to contain both.  Matching is case/space insensitive.
    """
    priority = {alias.lower(): j for j, alias in enumerate(aliases)}
    best_idx: int | None = None
    best_priority = len(aliases)
    for i, field in enumerate(fields):
        rank = priority.get((field or "").strip().lower())
        if rank is not None and rank < best_priority:
            best_priority = rank
            best_idx = i
    return best_idx


def parse_batch_csv(content: bytes) -> tuple[list[str], list[tuple[int, list[str]]]]:
    """Decode a batch CSV into ``(headers, [(physical_line_no, values)])``.

    Blank physical lines are skipped (they are not counted toward the 100-row
    limit, mirroring the standard-address importer).
    """
    try:
        # utf-8-sig transparently strips a possible BOM from Excel exports.
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MappingBatchError(f"CSV 解析失败: {exc}") from exc
    reader = csv.reader(io.StringIO(text))
    try:
        headers = next(reader)
    except StopIteration:
        raise MappingBatchError("CSV 文件为空，须包含表头与数据行")
    headers = [h.strip() for h in headers]
    rows: list[tuple[int, list[str]]] = []
    for raw in reader:
        if not raw:  # csv.reader yields [] only for an empty physical line
            continue
        values = list(raw)
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))
        rows.append((reader.line_num, values))
    return headers, rows


def validate_and_extract(content: bytes) -> list[dict[str, Any]]:
    """Parse + validate the batch upload; return the extracted entries.

    Each entry is ``{"line_no", "biz_no", "address"}`` with stripped,
    non-empty values.  Any structural problem (missing column, too many rows,
    blank cell) raises :class:`MappingBatchError` and no batch is created.
    """
    headers, raw_rows = parse_batch_csv(content)
    biz_idx = _pick_column(headers, BIZ_NO_ALIASES)
    if biz_idx is None:
        raise MappingBatchError(
            "CSV 缺少业务编号列（可识别列名："
            + " / ".join(BIZ_NO_ALIASES[:6]) + " 等）")
    addr_idx = _pick_column(headers, ADDRESS_ALIASES)
    part_idx = _pick_column(headers, ADDRESS_PART_ALIASES)
    if addr_idx is None and part_idx is None:
        raise MappingBatchError(
            "CSV 缺少原始地址列（可识别列名："
            + " / ".join(ADDRESS_ALIASES[:6])
            + "；或用 line1/street 等结构化列拼接）")

    if len(raw_rows) == 0:
        raise MappingBatchError("CSV 没有数据行，至少需要 1 行（最多 100 行）")
    if len(raw_rows) > MAX_ROWS:
        raise MappingBatchError(
            f"CSV 数据行 {len(raw_rows)} 行，超出单次最多 {MAX_ROWS} 行的限制，"
            "请拆分后分批上传")

    entries: list[dict[str, Any]] = []
    for line_no, values in raw_rows:
        biz_no = values[biz_idx].strip() if biz_idx < len(values) else ""
        if not biz_no:
            raise MappingBatchError(f"CSV 第 {line_no} 行业务编号为空，无法建立映射")
        address = ""
        if addr_idx is not None and addr_idx < len(values) and values[addr_idx].strip():
            address = values[addr_idx].strip()
        elif part_idx is not None:
            # No direct address value: compose it from the structured
            # component columns (line1 + city/state/postal_code/country),
            # reusing the standard-address composition rules.
            address = _build_address(headers, values)
        if not address:
            raise MappingBatchError(f"CSV 第 {line_no} 行原始地址为空，无法匹配")
        entries.append({"line_no": line_no, "biz_no": biz_no, "address": address})
    return entries


def _build_address(headers: list[str], values: list[str]) -> str:
    """Compose the raw address from structured component columns
    (``line1``/street + city/state/postal_code/country), reusing the
    standard-address header aliases and composition rules."""
    row = {headers[i]: values[i] for i in range(min(len(headers), len(values)))}
    return ingest.normalize_row(row).get("full_text", "")


# ------------------------------------------------------------------- running --


def _id_conflict_groups(entries: list[dict[str, str]]
                        ) -> dict[str, list[int]]:
    """Business ids whose rows carry more than one distinct raw address.

    Returns ``{biz_no: [entry_index, ...]}`` for the conflicting groups only;
    the same business id repeated with the *same* address is allowed.
    """
    groups: dict[str, list[int]] = {}
    addresses: dict[str, set[str]] = {}
    for i, entry in enumerate(entries):
        groups.setdefault(entry["biz_no"], []).append(i)
        addresses.setdefault(entry["biz_no"], set()).add(entry["address"])
    return {biz: idxs for biz, idxs in groups.items()
            if len(addresses[biz]) > 1}


def _conflict_reason(biz_no: str, entries: list[dict[str, str]],
                     group_indices: list[int]) -> str:
    distinct = list(dict.fromkeys(entries[i]["address"] for i in group_indices))
    lines = "、".join(f"第 {entries[i]['line_no']} 行" for i in group_indices)
    shown = "；".join(f"「{addr}」" for addr in distinct[:5])
    more = f" 等 {len(distinct)} 个不同地址" if len(distinct) > 5 else ""
    return (
        f"业务编号“{biz_no}”在 {lines} 出现 {len(group_indices)} 次，"
        f"但原始地址不一致（{shown}{more}）；编号冲突，系统不任选一行，目标 ID 留空"
    )


def _unmatched_reason(status: str, match_reason: str) -> str:
    label = STATUS_LABELS.get(status, status)
    return f"未自动命中（{label}）：{match_reason}"


def create_batch(conn: sqlite3.Connection, content: bytes, filename: str,
                 settings: Settings) -> dict[str, Any]:
    """Validate the upload, score every row on one table snapshot, persist.

    Pure scoring with the default settings: the review-consensus memory is
    not consulted and no ``match_logs`` row is written.  All reads and the
    batch insert happen in one ``BEGIN IMMEDIATE`` transaction, so the saved
    snapshot digest is exactly the snapshot the scores were computed against.
    """
    entries = validate_and_extract(content)
    conflicts = _id_conflict_groups(entries)

    counters = {
        STATUS_AUTO: 0, STATUS_REVIEW: 0,
        STATUS_NO_MATCH: 0, STATUS_ID_CONFLICT: 0,
    }
    rows_out: list[dict[str, Any]] = []

    now = datetime.now(timezone.utc)
    batch_id = f"MAP-{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}"
    created_at = now.isoformat(timespec="seconds")

    with db.immediate_transaction(conn):
        table_rows = db.all_addresses(conn)
        snapshot_json = db.address_table_snapshot(conn)

        # entry index -> group conflict reason (precomputed for every member)
        conflict_reason_by_index: dict[int, str] = {}
        for biz_no, idxs in conflicts.items():
            reason = _conflict_reason(biz_no, entries, idxs)
            for i in idxs:
                conflict_reason_by_index[i] = reason

        for i, entry in enumerate(entries):
            result = match_address(entry["address"], table_rows, settings)
            data = result_to_dict(result)
            data.pop("consensus", None)  # batches are pure scoring
            is_conflict = i in conflict_reason_by_index
            if is_conflict:
                status = STATUS_ID_CONFLICT
                reason = conflict_reason_by_index[i]
            else:
                status = result.status
                if status == STATUS_AUTO:
                    reason = (
                        f"自动命中标准地址（{result.reason}），目标 ID 已填入"
                    )
                else:
                    reason = _unmatched_reason(status, result.reason)
            counters[status] += 1
            best = result.best
            rows_out.append({
                "line_no": entry["line_no"],
                "biz_no": entry["biz_no"],
                "raw_address": entry["address"],
                "status": status,
                "status_label": STATUS_LABELS[status],
                "reason": reason,
                # Target is only populated for a clean automatic hit; conflict
                # and non-auto rows keep an empty target id on export.
                "target_id": best.address["id"] if (
                    best is not None and status == STATUS_AUTO
                ) else None,
                "target_full_text": best.address["full_text"] if (
                    best is not None and status == STATUS_AUTO
                ) else None,
                "best_id": data["best_id"],
                "best_score": data["best_score"],
                "match_status": result.status,
                "match_reason": result.reason,
                "normalized": data["normalized"],
                "candidates": data["candidates"],
            })

        summary = {
            "total": len(rows_out),
            "auto_matched": counters[STATUS_AUTO],
            "needs_review": counters[STATUS_REVIEW],
            "no_match": counters[STATUS_NO_MATCH],
            "id_conflict": counters[STATUS_ID_CONFLICT],
            "unresolved": (
                counters[STATUS_REVIEW]
                + counters[STATUS_NO_MATCH]
                + counters[STATUS_ID_CONFLICT]
            ),
        }
        db.insert_mapping_batch(
            conn,
            batch_id=batch_id,
            filename=filename,
            table_snapshot=snapshot_json,
            address_count=len(table_rows),
            rows=rows_out,
            summary=summary,
            created_at=created_at,
        )

    return {
        "id": batch_id,
        "created_at": created_at,
        "filename": filename,
        "address_count": len(table_rows),
        "table_snapshot": json.loads(snapshot_json),
        "summary": summary,
        "rows": rows_out,
    }


# ------------------------------------------------------------- serialization --


def serialize_batch(row: sqlite3.Row) -> dict[str, Any]:
    """Reconstruct the full batch document from a persisted row."""
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "address_count": row["address_count"],
        "table_snapshot": json.loads(row["table_snapshot"]),
        "summary": json.loads(row["summary_json"]),
        "rows": json.loads(row["rows_json"]),
    }


def serialize_batch_summary(row: sqlite3.Row) -> dict[str, Any]:
    """Lightweight shape for the batch history list."""
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "address_count": row["address_count"],
        "table_snapshot": json.loads(row["table_snapshot"]),
        "summary": json.loads(row["summary_json"]),
    }


# ------------------------------------------------------------------ exporting --


EXPORT_HEADER = [
    "业务编号", "原始地址", "匹配状态", "标准地址ID",
    "标准地址全文", "首选候选得分", "原因",
]


def build_export_csv(conn: sqlite3.Connection, batch_id: str
                     ) -> tuple[str, str, bytes]:
    """Build the complete mapping CSV for a batch.

    The **current** address-table digest is re-checked against the batch
    snapshot first: if the table changed since the batch ran the export is
    refused with :class:`MappingBatchTableChanged` (HTTP 409) and the caller
    must create a fresh batch.  Non-auto and id-conflict rows keep their
    original business id, ship an empty target id and carry an explicit
    reason.  Returns ``(filename, created_at, csv_bytes)``.
    """
    batch = db.get_mapping_batch(conn, batch_id)
    if batch is None:
        raise MappingBatchNotFound(f"映射批次 {batch_id} 不存在")
    current_snapshot = db.address_table_snapshot(conn)
    if current_snapshot != batch["table_snapshot"]:
        raise MappingBatchTableChanged(
            "标准地址表在批次生成后已发生变化（快照摘要不一致），"
            "该批次映射结果可能已失效，已拒绝导出；请基于当前地址表重新建立批次"
        )

    rows = json.loads(batch["rows_json"])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(EXPORT_HEADER)
    for row in rows:
        writer.writerow([
            row["biz_no"],
            row["raw_address"],
            row.get("status_label") or STATUS_LABELS.get(
                row["status"], row["status"]),
            row["target_id"] if row["target_id"] is not None else "",
            row["target_full_text"] or "",
            row["best_score"] if row["best_score"] is not None else "",
            row["reason"],
        ])
    # BOM so Excel opens the UTF-8 file with the right encoding.
    return (
        f"mapping_{batch_id}.csv",
        batch["created_at"],
        codecs.BOM_UTF8 + buffer.getvalue().encode("utf-8"),
    )