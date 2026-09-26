"""Similarity scoring and candidate selection.

The address is scored along two independent axes:

* **tokens**  - street name / city / suffix words (after normalisation),
  scored with a set-based precision/recall (F1). Pairs that only match
  fuzzily use a Levenshtein ratio above 0.72;
* **house number** - scored separately so that ``100 Main St`` and
  ``1000 Main St`` are never confused despite near-identical street text.
  Compound numbers (ranges ``12-14``, slash numbers ``12/14``) are compared
  segment by segment including their separator structure, so ``12-14``
  never scores full marks against ``12-16``.

Final score = ``TOKEN_WEIGHT * token_f1 + (1-TOKEN_WEIGHT) * house_score``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .config import Settings
from .normalization import Normalized, normalize

# ----------------------------------------------------------------- Levenshtein


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalised edit-distance similarity in [0, 1] (standard DP)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    dist = prev[-1]
    return 1.0 - dist / max(len(a), len(b))


_CJK_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_WORD_RUN = re.compile(r"[a-z]")


def _script(token: str) -> str:
    if _CJK_RUN.search(token):
        return "cjk"
    if _WORD_RUN.search(token):
        return "word"
    return "digit"


def _fuzzy_equal(a: str, b: str) -> tuple[bool, float]:
    """Whether two tokens of the same script count as a fuzzy match."""
    if a == b:
        return True, 1.0
    # Scripts are never compared across each other.
    if _script(a) != _script(b):
        return False, 0.0
    # Short tokens (including CJK unigrams and abbreviated suffixes such as
    # "st"/"ter") must match exactly: containment on them would produce
    # meaningless hits like "ter" inside "water".
    if len(a) < 3 or len(b) < 3:
        return False, 0.0
    # Exact containment handles CJK bigrams / abbreviated names sensibly.
    if a in b or b in a:
        shorter = min(len(a), len(b))
        longer = max(len(a), len(b))
        ratio = 0.75 + 0.25 * (shorter / longer)
        return True, ratio
    ratio = levenshtein_ratio(a, b)
    return ratio >= 0.72, ratio


# ------------------------------------------------------------------- scoring


# Country names (English + common CJK forms) are treated as optional metadata;
# normalisation already drops them on the candidate side.
_COUNTRY_TOKENS = {
    "usa", "us", "u", "america", "unitedstates", "united", "states",
    "china", "prc", "中国", "中", "国",
    "uk", "britain", "england", "英国", "kingdom", "great",
    "germany", "德国",
}


def _is_metadata(token: str) -> bool:
    # Standalone 5/6-digit tokens are postcodes, not street words.
    return bool(re.fullmatch(r"\d{5,6}", token)) or token in _COUNTRY_TOKENS


@dataclass
class TokenScore:
    precision: float
    recall: float
    f1: float
    matched: list[dict[str, Any]] = field(default_factory=list)
    query_only: list[str] = field(default_factory=list)
    candidate_only: list[str] = field(default_factory=list)


# Tolerance for comparing similarity sums: sums closer than this are ties,
# resolved by the stable canonical order rather than by float noise.
_SIM_EPS = 1e-9


def _assign_component(a_toks: list[str], b_toks: list[str],
                      adj: dict[str, list[tuple[str, float]]],
                      flipped: bool) -> list[tuple[str, str]]:
    """Optimal one-to-one assignment inside one connected component.

    ``a_toks`` is the side being walked (in canonical token-string order)
    while ``b_toks`` is the DP bitmask side. Among all assignments that
    maximise first the pair count and then the summed similarity, the walk
    takes the lexicographically smallest pair list in walked-side order --
    a stable tie-break that depends only on the tokens, never on input order.
    """
    b_index = {t: j for j, t in enumerate(b_toks)}
    edges: list[list[tuple[int, float]]] = [
        [(b_index[b], ratio) for b, ratio in adj[a]] for a in a_toks
    ]

    @lru_cache(maxsize=None)
    def best_from(i: int, mask: int) -> tuple[int, float]:
        """Best (pair_count, similarity_sum) for a_toks[i:] given ``mask``."""
        if i == len(a_toks):
            return (0, 0.0)
        best_count, best_sim = best_from(i + 1, mask)  # a_toks[i] unmatched
        for j, ratio in edges[i]:
            bit = 1 << j
            if mask & bit:
                continue
            count, sim = best_from(i + 1, mask | bit)
            count += 1
            sim += ratio
            if count > best_count or (
                count == best_count and sim > best_sim + _SIM_EPS
            ):
                best_count, best_sim = count, sim
        return (best_count, best_sim)

    pairs: list[tuple[str, str]] = []
    mask = 0
    for i, a in enumerate(a_toks):
        target_count, target_sim = best_from(i, mask)
        for j, ratio in edges[i]:
            bit = 1 << j
            if mask & bit:
                continue
            count, sim = best_from(i + 1, mask | bit)
            if (count + 1 == target_count
                    and sim + ratio >= target_sim - _SIM_EPS):
                pairs.append((b_toks[j], a) if flipped else (a, b_toks[j]))
                mask |= bit
                break
    return pairs


def _pair_tokens(q_tokens: list[str], c_scored: list[str]):
    """One-to-one optimal assignment between query and candidate tokens.

    A greedy query-order walk can strand legal fuzzy pairs (query
    ``aaa, aaba`` vs candidate ``aaa, aaab`` would match only one pair,
    although two legal pairs exist). Instead the assignment is solved
    globally within the component: maximise the number of pairs first, then
    the summed similarity. Ties are broken by a stable canonical order that
    depends only on the token strings, so permuting the tokens on either
    side changes neither the score nor the match details.
    """
    if not q_tokens or not c_scored:
        return [], set()

    # Legal fuzzy edges; the relation is symmetric, so one adjacency pair
    # serves both orientations.
    q_adj: dict[str, list[tuple[str, float]]] = {t: [] for t in q_tokens}
    c_adj: dict[str, list[tuple[str, float]]] = {t: [] for t in c_scored}
    ratios: dict[tuple[str, str], float] = {}
    for qt in q_tokens:
        for ct in c_scored:
            ok, ratio = _fuzzy_equal(qt, ct)
            if ok:
                q_adj[qt].append((ct, ratio))
                c_adj[ct].append((qt, ratio))
                ratios[(qt, ct)] = ratio
    for neighbours in list(q_adj.values()) + list(c_adj.values()):
        neighbours.sort()

    # The assignment decomposes over the connected components of the edge
    # graph, so the exact search stays tiny even for long token lists (CJK
    # tokens, for instance, only ever form exact-match edges).
    pairs: list[tuple[str, str]] = []
    seen_q: set[str] = set()
    seen_c: set[str] = set()
    for seed in sorted(q_adj):
        if seed in seen_q or not q_adj[seed]:
            continue
        comp_q: list[str] = []
        comp_c: list[str] = []
        seen_q.add(seed)
        stack = [("q", seed)]
        while stack:
            side, token = stack.pop()
            if side == "q":
                comp_q.append(token)
                for nbr, _ in q_adj[token]:
                    if nbr not in seen_c:
                        seen_c.add(nbr)
                        stack.append(("c", nbr))
            else:
                comp_c.append(token)
                for nbr, _ in c_adj[token]:
                    if nbr not in seen_q:
                        seen_q.add(nbr)
                        stack.append(("q", nbr))
        comp_q.sort()
        comp_c.sort()
        # Walk the larger side, bitmask over the smaller one.
        if len(comp_q) >= len(comp_c):
            pairs.extend(_assign_component(comp_q, comp_c, q_adj, flipped=False))
        else:
            pairs.extend(_assign_component(comp_c, comp_q, c_adj, flipped=True))

    pairs.sort()
    matched = [
        {"query": q, "candidate": c, "similarity": round(ratios[(q, c)], 3)}
        for q, c in pairs
    ]
    return matched, {c for _, c in pairs}


def _score_component(query: list[str], candidate: list[str]) -> TokenScore:
    q_tokens = list(dict.fromkeys(query))
    c_tokens = list(dict.fromkeys(candidate))

    # Candidate-only metadata (postcode, country) must not penalise recall
    # when the query omits it; metadata present in the query is still scored.
    q_set = set(q_tokens)
    c_scored = [t for t in c_tokens if not (_is_metadata(t) and t not in q_set)]

    matched, used = _pair_tokens(q_tokens, c_scored)
    c_available = set(c_scored) - used

    matched_q = {m["query"] for m in matched}
    query_only = sorted(t for t in q_tokens if t not in matched_q)
    candidate_only = sorted(c_available)

    precision = len(matched) / len(q_tokens) if q_tokens else 0.0
    recall = len(used) / len(c_scored) if c_scored else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return TokenScore(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        matched=matched,
        query_only=query_only,
        candidate_only=candidate_only,
    )


def score_tokens(query: list[str], candidate: list[str]) -> TokenScore:
    """Backward-compatible single-component token scoring (used in tests)."""
    return _score_component(query, candidate)


# Street is the dominant signal; locality is corroborating evidence.
STREET_WEIGHT = 0.75
LOCALITY_WEIGHT = 0.25


@dataclass
class AddressTokenScore:
    street: TokenScore
    locality: TokenScore
    f1: float


def score_address_tokens(q_street: list[str], q_locality: list[str],
                         c_street: list[str], c_locality: list[str]
                         ) -> AddressTokenScore:
    street = _score_component(q_street, c_street)
    # When the query names no locality at all, locality must not influence
    # the result (a street-only query is still a perfectly specified input).
    if not q_locality:
        locality = _score_component([], c_locality)
        total_f1 = street.f1
    else:
        locality = _score_component(q_locality, c_locality)
        total_f1 = STREET_WEIGHT * street.f1 + LOCALITY_WEIGHT * locality.f1
    return AddressTokenScore(street=street, locality=locality,
                             f1=round(total_f1, 3))


# Canonical house numbers may be compound: ranges (``12-14``) or slash
# numbers (``12/14``). They are split into numeric (or letter) segments
# plus the separators between them, and both take part in scoring.
_HOUSE_SEPARATORS = re.compile(r"([-/])")
_HOUSE_NUMERIC_PART = re.compile(r"^(\d+)([a-z]?)$")
_HOUSE_LETTER_PART = re.compile(r"^[a-z]+$")


def _split_house(value: str) -> tuple[list[tuple[int | None, str]], list[str]]:
    """Split a canonical house number into segments and separators.

    ``12b``   -> ``([(12, "b")], [])``
    ``12-14`` -> ``([(12, ""), (14, "")], ["-"])``
    ``12/14`` -> ``([(12, ""), (14, "")], ["/"])``
    """
    segments: list[tuple[int | None, str]] = []
    separators: list[str] = []
    for part in _HOUSE_SEPARATORS.split(value):
        if not part:
            continue
        if part in ("-", "/"):
            separators.append(part)
            continue
        m = _HOUSE_NUMERIC_PART.match(part)
        if m:
            segments.append((int(m.group(1)), m.group(2)))
        elif _HOUSE_LETTER_PART.match(part):
            segments.append((None, part))
        else:
            segments.append((None, ""))
    return segments, separators


def _house_segment_score(q_num: int | None, q_letter: str,
                         c_num: int | None, c_letter: str) -> float:
    """Similarity of one pair of house-number segments (plain-number rules)."""
    if q_num is None or c_num is None:
        return 1.0 if (q_num, q_letter) == (c_num, c_letter) else 0.0
    if q_num == c_num:
        return 0.7 if q_letter != c_letter else 1.0
    bigger = max(q_num, c_num)
    closeness = 1 - abs(q_num - c_num) / bigger
    # A wrong number usually means a different property, so non-exact values
    # are capped low (~0.3) even when numerically adjacent; perfect street
    # tokens plus a wrong house number therefore stay below the auto
    # threshold and go to human review.
    return min(0.3, closeness ** 4 * 0.3)


def score_house(query: str | None, candidate: str | None) -> float:
    """House-number similarity with a harsh wrong-number penalty.

    * both missing  -> 0.5 (neutral: no information either way);
    * one missing   -> 0.15 (weak evidence, not a hard veto);
    * identical     -> 1.0;
    * same digits with letter differences (``12b`` vs ``12``) -> 0.7;
    * numbers close in value get a small partial score;
    * completely different numbers -> 0.0.

    Compound numbers (ranges ``12-14``, slash numbers ``12/14``) must agree
    on **every** segment and on the separator structure: a different later
    segment (``12-14`` vs ``12-16``) or a different structure (``12-14`` vs
    ``12/14`` or vs ``12``) is a wrong house number and scores accordingly
    low, so it cannot auto-match under the default thresholds.
    """
    if not query and not candidate:
        return 0.5
    if not query or not candidate:
        return 0.15
    if query == candidate:
        return 1.0

    q_segments, q_separators = _split_house(query)
    c_segments, c_separators = _split_house(candidate)

    if q_separators or c_separators:
        # At least one side is compound. A different separator structure
        # (range vs slash vs plain number) is a different property.
        if (q_separators != c_separators
                or len(q_segments) != len(c_segments)):
            return 0.0
        # Same structure: every segment must agree and the worst segment
        # decides, so ``12-14`` vs ``12-16`` scores like the wrong number
        # ``14`` vs ``16`` instead of full marks on the shared ``12``.
        return round(min(
            _house_segment_score(q_num, q_letter, c_num, c_letter)
            for (q_num, q_letter), (c_num, c_letter)
            in zip(q_segments, c_segments)
        ), 3)

    q_num, q_letter = q_segments[0] if q_segments else (None, "")
    c_num, c_letter = c_segments[0] if c_segments else (None, "")
    if q_num is None or c_num is None:
        return 0.0
    return round(_house_segment_score(q_num, q_letter, c_num, c_letter), 3)


# ----------------------------------------------------------------- matching


@dataclass
class CandidateResult:
    address: dict[str, Any]
    score: float
    token_score: AddressTokenScore
    house_score: float
    conflicts: list[dict[str, Any]]


@dataclass
class MatchResult:
    query: str
    normalized_query: Normalized
    status: str                  # auto_matched | needs_review | no_match
    reason: str
    best: CandidateResult | None
    runner_up: CandidateResult | None
    candidates: list[CandidateResult]
    thresholds: dict[str, float]
    # Populated by app/consensus.py from the reviewed-record memory; scoring
    # itself never touches this.
    consensus: dict[str, Any] | None = None


def _candidate_address(row: Any) -> dict[str, Any]:
    import json

    return {
        "id": row["id"],
        "full_text": row["full_text"],
        "house": row["house"],
        "line1": row["line1"],
        "city": row["city"],
        "state": row["state"],
        "postal_code": row["postal_code"],
        "country": row["country"],
        "external_id": row["external_id"],
        "raw": json.loads(row["raw_json"]),
    }


def _detect_conflicts(ts: AddressTokenScore, query_house: str | None,
                      cand_house: str | None, house_score: float) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    if query_house and cand_house and house_score < 0.99:
        conflicts.append({
            "field": "house_number",
            "query": query_house,
            "candidate": cand_house,
            "similarity": house_score,
            "message": "门牌号不一致",
        })
    # Locality mismatches (city/state/postcode present on both sides but not
    # matching) are surfaced as explicit conflict fields for reviewers.
    for component_name, component in (
        ("locality", ts.locality),
        ("street", ts.street),
    ):
        for m in component.matched:
            if m["query"] != m["candidate"] and m["similarity"] < 0.999:
                conflicts.append({
                    "field": f"{component_name}_token",
                    "query": m["query"],
                    "candidate": m["candidate"],
                    "similarity": m["similarity"],
                    "message": (
                        "地区词元近似但不完全相同"
                        if component_name == "locality"
                        else "道路词元近似但不完全相同"
                    ),
                })
    for missing in ts.locality.query_only:
        conflicts.append({
            "field": "locality",
            "query": missing,
            "candidate": None,
            "similarity": 0.0,
            "message": "查询中的地区字段在候选中缺失",
        })
    return conflicts


def match_address(query_text: str, rows: list[Any], settings: Settings,
                  top_n: int | None = 5) -> MatchResult:
    """Score ``query_text`` against every row and derive the match verdict.

    ``top_n`` caps how many scored candidates are kept in the result (the
    verdict itself only ever looks at the best two). ``None`` keeps the full
    scored list -- used by the self-match audit, which must see every address
    scoring close to the record's own score, beyond the usual top-5 cut.
    """
    query_norm = normalize(query_text)
    results: list[CandidateResult] = []

    for row in rows:
        import json

        c_street = json.loads(row["street_tokens_json"] or "[]")
        c_locality = json.loads(row["locality_tokens_json"] or "[]")
        if not c_street and not c_locality:  # tolerate legacy rows
            c_street = json.loads(row["tokens_json"])
        ts = score_address_tokens(
            query_norm.street_tokens, query_norm.locality_tokens,
            c_street, c_locality,
        )
        hs = score_house(query_norm.house, row["house"])
        score = settings.token_weight * ts.f1 + (1 - settings.token_weight) * hs

        conflicts = _detect_conflicts(ts, query_norm.house, row["house"], hs)
        results.append(CandidateResult(
            address=_candidate_address(row),
            score=round(score, 4),
            token_score=ts,
            house_score=hs,
            conflicts=conflicts,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[:top_n] if top_n is not None else list(results)
    best = top[0] if top else None
    runner_up = top[1] if len(top) > 1 else None

    if best is None or best.score < settings.reject_threshold:
        status, reason = "no_match", "所有候选得分均低于拒绝阈值"
    elif runner_up is not None and (best.score - runner_up.score) < settings.margin_threshold:
        status, reason = "needs_review", "最高分与次高分差距过小，结果存在歧义"
    elif best.score < settings.auto_threshold:
        status, reason = "needs_review", "最高得分低于自动确认阈值（可能门牌号或地区字段冲突）"
    else:
        status, reason = "auto_matched", "得分达到自动确认阈值且领先优势明显"

    return MatchResult(
        query=query_text,
        normalized_query=query_norm,
        status=status,
        reason=reason,
        best=best,
        runner_up=runner_up,
        candidates=top,
        thresholds={
            "auto_threshold": settings.auto_threshold,
            "margin_threshold": settings.margin_threshold,
            "reject_threshold": settings.reject_threshold,
            "token_weight": settings.token_weight,
        },
    )


# ------------------------------------------------------------- serialization

def candidate_to_dict(c: CandidateResult) -> dict[str, Any]:
    matched = c.token_score.street.matched + c.token_score.locality.matched
    query_only = (
        c.token_score.street.query_only + c.token_score.locality.query_only
    )
    candidate_only = (
        c.token_score.street.candidate_only + c.token_score.locality.candidate_only
    )
    return {
        "address": c.address,
        "score": c.score,
        "token_f1": c.token_score.f1,
        "street_f1": c.token_score.street.f1,
        "locality_f1": c.token_score.locality.f1,
        "token_precision": round(
            (c.token_score.street.precision + c.token_score.locality.precision) / 2,
            3,
        ),
        "token_recall": round(
            (c.token_score.street.recall + c.token_score.locality.recall) / 2,
            3,
        ),
        "house_score": c.house_score,
        "matched_tokens": matched,
        "query_only_tokens": query_only,
        "candidate_only_tokens": candidate_only,
        "street_tokens": {
            "matched": c.token_score.street.matched,
            "query_only": c.token_score.street.query_only,
            "candidate_only": c.token_score.street.candidate_only,
        },
        "locality_tokens": {
            "matched": c.token_score.locality.matched,
            "query_only": c.token_score.locality.query_only,
            "candidate_only": c.token_score.locality.candidate_only,
        },
        "conflicts": c.conflicts,
    }


def result_to_dict(result: MatchResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "normalized": {
            "text": result.normalized_query.text,
            "tokens": result.normalized_query.tokens,
            "house": result.normalized_query.house,
        },
        "status": result.status,
        "reason": result.reason,
        "best_id": result.best.address["id"] if result.best else None,
        "best_score": result.best.score if result.best else None,
        "runner_up_id": result.runner_up.address["id"] if result.runner_up else None,
        "runner_up_score": result.runner_up.score if result.runner_up else None,
        "thresholds": result.thresholds,
        "candidates": [candidate_to_dict(c) for c in result.candidates],
        "consensus": result.consensus,
    }
