"""Address text normalisation.

The normaliser turns messy real-world address text into a comparable form:

* Unicode is unified (NFKC, diacritics folded, full-width digits/letters
  converted, traditional Chinese characters mapped to simplified forms);
* everything is lower-cased and stripped of redundant punctuation;
* common road suffixes are canonicalised in both English
  (``street/st.`` -> ``st``) and Chinese (``北京東路`` -> ``北京东路``);
* house numbers are extracted with their own normalisation rules
  (``#12-B`` / ``No. 12b`` / ``12号`` -> ``12b``) and handled separately
  from the word tokens; ranges and slash numbers stay whole
  (``12-14`` / ``12–14`` / ``12/14``);
* unit/sub-premise references (``Apt 5``, ``Unit 7B``, ``3楼`` ...) are
  removed before house extraction wherever they sit -- leading, trailing,
  across a comma or separated by spaces only -- so the unit number can
  never act as the house number or as a street/locality token. In
  ``Apt 5, 123 Main St`` the house number is therefore ``123``;

Tokens are additionally split into two components so scoring can treat them
differently:

* ``street_tokens``   - street line: name, suffix, directional, unit/building;
* ``locality_tokens`` - city, state/province, postcode (and country, which is
  treated as optional metadata).

A query that omits the city must therefore not be penalised against a
candidate that has one.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ----------------------------------------------------------------- mappings --

# English road suffixes: many spellings -> one canonical short form.
_ENGLISH_ABBREVIATIONS: dict[str, str] = {
    "street": "st", "str": "st", "st.": "st",
    "road": "rd", "rd.": "rd",
    "avenue": "ave", "av": "ave", "ave.": "ave", "avenida": "ave",
    "boulevard": "blvd", "boul": "blvd", "blvd.": "blvd",
    "drive": "dr", "dr.": "dr",
    "lane": "ln", "ln.": "ln",
    "court": "ct", "ct.": "ct",
    "terrace": "ter", "tce": "ter",
    "place": "pl", "pl.": "pl",
    "square": "sq", "sq.": "sq",
    "apartment": "apt", "apt.": "apt",
    "suite": "ste", "ste.": "ste",
    "floor": "fl", "fl.": "fl",
    "north": "n", "south": "s", "east": "e", "west": "w",
}

# Chinese traditional -> simplified unification (small but useful set).
_CJK_CHAR_MAP: dict[str, str] = {
    # directional / common
    "東": "东", "西": "西", "南": "南", "北": "北",
    "號": "号", "樓": "楼", "棟": "栋", "單": "单", "元": "元",
    "區": "区", "縣": "县", "市": "市", "鎮": "镇", "鄉": "乡",
    "門": "门", "衖": "弄", "巷": "巷",
    "臺": "台", "灣": "湾", "廣": "广", "國": "国",
    # road suffix characters stay as themselves
    "街": "街", "路": "路", "道": "道",
}

_CJK = r"㐀-䶿一-鿿豈-﫿"

# Country words are optional: ignored on the candidate side when the query
# does not mention them.
_COUNTRY_WORDS = {
    "usa", "us", "america", "united", "states", "unitedstates",
    "china", "prc", "uk", "britain", "england", "great", "kingdom",
    "germany", "deutschland", "france", "japan", "canada", "australia",
    "中国", "英国", "德国", "法国", "日本", "加拿大", "澳大利亚",
}

# US/CA-style state codes (uppercased in source; compared lower-cased).
_STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc", "on", "qc", "bc", "ab",
}

# Markers that may introduce a house number.
_HOUSE_PREFIX = re.compile(
    r"(?:#+\s*|no\.?\s*|nr\.?\s*|nummer\s*|n[oº]\.?\s*)",
    re.IGNORECASE,
)
_BUILDING_WORDS = re.compile(
    r"(?:apt|ste|suite|unit|fl(?:oor)?|rm|room|bldg|building|栋|幢|号楼|单元|室|樓|楼)",
    re.IGNORECASE,
)
# A unit/sub-premise reference: "Apt 5", "Unit 7B", "Suite 300", "3楼",
# "2栋", "Apt #5". Such numbers identify a unit, never the house number --
# even when the reference is written *before* the street line
# ("Apt 5, 123 Main St") -- so the whole reference is stripped before house
# extraction. This runs before abbreviation normalisation, so both the long
# ("suite", "apartment") and canonical ("ste") forms are listed; bare "fl"
# is deliberately excluded because it is the Florida state abbreviation.
# CJK markers may sit on either side of the number.
_UNIT_NUMBER = r"\d+[a-z0-9-]*"
_UNIT_REFERENCE = re.compile(
    r"(?<![a-z0-9])(?:apt|ste|unit|rm|bldg)"
    r"[.\s]+#?\s*" + _UNIT_NUMBER + r"\b"
    r"|(?<![a-z0-9])fl\.[.\s]*#?\s*" + _UNIT_NUMBER + r"\b"
    r"|(?<![a-z0-9])(?:floor|room|building|suite|apartment)"
    r"[.\s]+#?\s*" + _UNIT_NUMBER + r"\b"
    r"|(?<![a-z0-9])" + _UNIT_NUMBER + r"\s*(?:单元|号楼|栋|幢|室|楼|樓)"
    r"|(?:单元|号楼|栋|幢|室|楼|樓)\s*#?\s*" + _UNIT_NUMBER + r"(?![0-9a-z])",
    re.IGNORECASE,
)
_TRAILING_CJK_NO = re.compile(r"([0-9]+[a-z]?)\s*(?:号|號)")
# Compound numbers stay whole: ranges ``12-14`` / ``12–14`` and slash
# numbers ``12/14`` are extracted as one house number.
_NUMBER = re.compile(r"\d+[a-z]?(?:[-–/](?:\d+[a-z]?|[a-z]))*")
_POSTCODE = re.compile(r"^\d{5,6}(?:-\d{4})?$")

# ------------------------------------------------------------------ result --


@dataclass(frozen=True)
class Normalized:
    original: str
    text: str                                # all components, joined
    street_text: str = ""
    house: str | None = None
    street_tokens: list[str] = field(default_factory=list)
    locality_tokens: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> list[str]:
        """All comparable tokens (street first, then locality)."""
        return self.street_tokens + self.locality_tokens


# ----------------------------------------------------------------- helpers --


def _fold_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    folded = "".join(ch for ch in decomposed if unicodedata.combining(ch) == 0)
    return folded.replace("ß", "ss").replace("æ", "ae").replace("ø", "o")


def _unify_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _fold_diacritics(text)
    return "".join(_CJK_CHAR_MAP.get(ch, ch) for ch in text)


def _canonical_house(raw: str) -> str:
    """``12-B`` / ``12 b`` -> ``12b``; keep ranges like ``12-14`` as ``12-14``."""
    raw = raw.strip().lower().replace(" ", "")
    parts = [p for p in re.split(r"([-–/])", raw) if p]
    out: list[str] = []
    for part in parts:
        if part in {"-", "–", "/"}:
            out.append("-" if part == "–" else part)
        else:
            m = re.match(r"^(\d+)([a-z]?)$", part)
            out.append(m.group(1) + m.group(2) if m else part)
    return "".join(out)


def extract_house(text: str) -> tuple[str | None, str]:
    """Return ``(house_number, text_without_house)``.

    Unit/sub-premise references (``Apt 5``, ``Unit 7B``, ``3楼`` ...) are
    removed first, wherever they appear -- leading ("Apt 5, 123 Main St"),
    trailing ("123 Main St Apt 5") or mid-line -- so a unit number can
    neither be mistaken for the house number nor leak into the locality.
    """
    text = _UNIT_REFERENCE.sub(" ", text)

    # 1a. prefix markers: #12, No. 12
    m = _HOUSE_PREFIX.search(text)
    if m:
        after = text[m.end():]
        nm = _NUMBER.match(after.lstrip())
        if nm:
            value = _canonical_house(nm.group(0))
            span_end = m.end() + len(after) - len(after.lstrip()) + nm.end()
            return value, text[:m.start()] + " " + text[span_end:]

    # 1b. trailing CJK 号 marker: 北京东路12号
    m = _TRAILING_CJK_NO.search(text)
    if m:
        return _canonical_house(m.group(1)), text[:m.start()] + " " + text[m.end():]

    # 2/3. scan number candidates left to right, skip postcodes/building refs.
    for nm in _NUMBER.finditer(text):
        end = nm.span()[1]
        tail = text[end:end + 12].lstrip()
        if _BUILDING_WORDS.match(tail):
            continue
        rest = text[end:]
        if re.fullmatch(r"\d{5}", nm.group(0)) and re.fullmatch(
            r"[^0-9a-z" + _CJK + r"]*", rest, re.IGNORECASE
        ):
            continue
        return _canonical_house(nm.group(0)), text[:nm.start()] + " " + text[end:]

    return None, text


def _replace_abbreviations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        word = match.group(0).lower().strip(".")
        canonical = _ENGLISH_ABBREVIATIONS.get(word)
        return canonical if canonical is not None else match.group(0)

    return re.sub(r"[A-Za-z]+\.?", repl, text)


def tokenize(text: str) -> list[str]:
    """Split normalised text into comparable tokens.

    * ASCII words/digits become single tokens (``main``, ``st``, ``10001``);
    * runs of CJK characters yield unigrams plus overlapping bigrams
      (``北京东路`` -> ``北``, ``京``, ``东``, ``路``, ``北京``, ``京东``, ``东路``),
      which tolerates minor variations in Chinese addresses.
    """
    tokens: list[str] = []
    for match in re.finditer(rf"[a-z0-9]+|[{_CJK}]+", text):
        chunk = match.group(0)
        if re.fullmatch(rf"[{_CJK}]+", chunk):
            tokens.extend(chunk)
            tokens.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
        else:
            tokens.append(chunk)
    return tokens


_PUNCT_RE = re.compile(r"[^\w\s" + _CJK + r"]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = _replace_abbreviations(text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize(value: str) -> Normalized:
    """Normalise a free-text address."""
    if value is None:
        value = ""
    text = _unify_unicode(value).lower()

    # Preserve comma-separated segments so street vs locality can be split.
    # Splitting *before* house extraction matters for leading unit references
    # ("Apt 5, 123 Main St"): the unit-only first segment must not claim the
    # role (or the number) of the real street line that follows it.
    raw_segments = [seg.strip() for seg in re.split(r"[,，;；]", text) if seg.strip()]
    if not raw_segments:
        raw_segments = [text]

    house: str | None = None
    cleaned_segments: list[tuple[str, str]] = []  # (original, cleaned)
    for raw_seg in raw_segments:
        seg_house, seg_rest = extract_house(raw_seg)
        cleaned = _clean(seg_rest)
        if house is None and seg_house is not None:
            house = seg_house
        if cleaned:
            cleaned_segments.append((raw_seg, cleaned))

    # Decide each segment's role from its *original* text -- classification
    # markers (CJK road characters, unit words) are stripped out of the
    # cleaned form but still indicate that the segment is a street line.
    street_parts: list[str] = []
    locality_parts: list[str] = []
    for idx, (raw_seg, seg) in enumerate(cleaned_segments):
        words = seg.split()
        if idx == 0:
            # The first surviving segment is the street line: leading
            # unit-only segments ("Apt 5,") disappear during stripping, so
            # the street line is the segment that actually remains.
            street_parts.append(seg)
            continue
        is_cjk = bool(re.search(rf"[{_CJK}]", raw_seg))
        non_country_words = [w for w in words if w not in _COUNTRY_WORDS]
        if is_cjk:
            # Chinese city names: short CJK segment (上海, 北京市) -> locality,
            # but a segment containing a road character belongs to street.
            if any(ch in raw_seg for ch in "路街道巷弄"):
                street_parts.append(seg)
            else:
                locality_parts.extend(non_country_words)
        elif all(
            _POSTCODE.fullmatch(w)
            or (len(w) == 2 and w in _STATE_CODES)
            or w in _COUNTRY_WORDS
            for w in words
        ):
            locality_parts.extend(non_country_words)
        elif len(non_country_words) <= 3 and not _BUILDING_WORDS.search(raw_seg):
            # "New York", "Los Angeles", "Springfield" -> locality
            locality_parts.extend(non_country_words)
        else:
            street_parts.append(seg)

    street_text = " ".join(street_parts)
    locality_text = " ".join(locality_parts)
    all_text = _WS_RE.sub(" ", f"{street_text} {locality_text}").strip()

    return Normalized(
        original=value,
        text=all_text,
        street_text=street_text,
        house=house,
        street_tokens=list(dict.fromkeys(tokenize(street_text))),
        locality_tokens=list(dict.fromkeys(tokenize(locality_text))),
    )


def normalize_parts(*, line1: str | None = None, city: str | None = None,
                    state: str | None = None, postal_code: str | None = None,
                    country: str | None = None, full_text: str | None = None
                    ) -> Normalized:
    """Normalise structured address fields (used by the CSV/JSON importer)."""
    if full_text:
        base = normalize(full_text)
    else:
        base = Normalized(original="", text="")

    extra_street_parts: list[str] = []
    extra_locality_parts: list[str] = []

    if line1 and not full_text:
        n = normalize(line1)
        extra_street_parts.append(n.street_text)
        house = n.house
    else:
        house = base.house

    if city:
        extra_locality_parts.append(_clean(_unify_unicode(city).lower()))
    if state:
        extra_locality_parts.append(_clean(_unify_unicode(state).lower()))
    if postal_code:
        extra_locality_parts.append(
            _clean(_unify_unicode(postal_code).lower())
        )

    street_text = _WS_RE.sub(
        " ", (base.street_text + " " + " ".join(extra_street_parts)).strip()
    ).strip()
    locality_tokens = list(dict.fromkeys(
        base.locality_tokens + tokenize(" ".join(extra_locality_parts))
    ))
    locality_text = " ".join(locality_tokens)
    # Country is intentionally dropped: optional metadata.

    street_tokens = list(dict.fromkeys(
        base.street_tokens + tokenize(" ".join(extra_street_parts))
    ))
    locality_tokens = list(dict.fromkeys(
        base.locality_tokens + tokenize(" ".join(extra_locality_parts))
    ))
    all_text = _WS_RE.sub(" ", f"{street_text} {locality_text}").strip()
    original = full_text or ", ".join(
        p for p in (line1, city, state, postal_code, country) if p
    )
    return Normalized(
        original=original,
        text=all_text,
        street_text=street_text,
        house=house,
        street_tokens=street_tokens,
        locality_tokens=locality_tokens,
    )
