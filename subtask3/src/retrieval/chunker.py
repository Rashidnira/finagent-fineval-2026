"""System D chunkers: multilingual news + row-aware financial statements.

Losslessness contract
---------------------
Both chunkers emit chunks whose `text` fields are verbatim, contiguous
slices of the parsed source span, in order, with no gaps:

    "".join(c.text for c in news_chunks)   == news_section_body
    "\\n".join(c.text for c in fs_chunks)  == statement_block_text

Overlap for long news sections is stored separately (`overlap_prefix`), so
it never perturbs the reconstruction invariant. The original text is never
translated, re-encoded, or normalized in place; `search_text` is a derived,
deterministic lexical view kept alongside the original.
"""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, asdict, field

import regex as re

# ---------------------------------------------------------------- constants

NEWS_TARGET_CHARS = 1100     # start looking for a boundary once a chunk holds this much
NEWS_HARD_MAX = 1600         # never exceed without a forced sub-split
NEWS_MIN_TAIL = 300          # merge a smaller trailing chunk into its predecessor
NEWS_OVERLAP_CHARS = 180     # overlap prefix for multi-chunk sections

FS_MAX_DATA_ROWS = 12        # cap of value-bearing rows per statement chunk

_CJK_TERMINATORS = "。！？…"
_ASCII_TERMINATORS = ".!?"
_CLOSERS = "」』”’\"'）)】〉》"

# ------------------------------------------------------- lexical preprocessing

_WS_RUN = re.compile(r"\s+")
_TOKEN_RE = re.compile(
    r"\p{Han}"                                            # CJK ideograph: 1 char = 1 token
    r"|[\p{Hiragana}\p{Katakana}ー]+"                      # kana runs
    r"|[\p{Latin}\p{Greek}\p{Cyrillic}][\p{Latin}\p{Greek}\p{Cyrillic}'’\-]*"
    r"|\d[\d.,]*%?"                                       # numbers incl. 1,234.5 / 14.6%
)


def normalize_for_search(text: str) -> str:
    """Deterministic lexical normalization for retrieval indexing only.

    NFKC (folds full-width forms and non-breaking spaces), casefold,
    whitespace collapse. The original text is stored separately and never
    altered.
    """
    return _WS_RUN.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def lexical_tokens(text: str) -> list[str]:
    """Language-agnostic token stream over the normalized text (Han chars
    as single tokens, kana/Latin/Greek runs as words, numbers kept whole)."""
    return _TOKEN_RE.findall(normalize_for_search(text))


# ---------------------------------------------------------------- chunk model

@dataclass
class Chunk:
    chunk_id: str
    task_id: str
    dataset: str
    source_type: str            # "news" | "financial_statement"
    language: str               # ISO-ish code; statements are English ("en")
    chunk_index: int            # index within (task_id, collection)
    text: str                   # verbatim original slice (lossless core)
    search_text: str            # normalized lexical view
    char_start: int             # span into the original query string
    char_end: int
    sha1: str
    # news-only
    language_name: str | None = None
    overlap_prefix: str = ""    # verbatim tail of the previous chunk (context only)
    # financial-statement-only
    statement_type: str | None = None    # income_statement | balance_sheet | cash_flow | unknown
    statement_index: int | None = None   # table position within the filing context
    section_label: str | None = None     # nearest row-group heading, e.g. "Current assets:"
    periods: list[str] = field(default_factory=list)
    table_header: str | None = None      # column-header/period lines for prompt context

    def to_dict(self) -> dict:
        return asdict(self)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------- news chunking

def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous partition of `text` at multilingual sentence boundaries.

    Boundary after CJK terminators (。！？…) always, after ASCII .!? only
    when followed by whitespace (protects decimals like 760.98 and
    intra-word dots). Trailing closers/whitespace attach to the finished
    sentence so concatenation reproduces the input exactly.
    """
    n = len(text)
    spans: list[tuple[int, int]] = []
    start = i = 0
    while i < n:
        ch = text[i]
        end = None
        if ch in _CJK_TERMINATORS:
            j = i + 1
            while j < n and text[j] in _CJK_TERMINATORS + _ASCII_TERMINATORS:
                j += 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            end = j
        elif ch in _ASCII_TERMINATORS:
            j = i + 1
            while j < n and text[j] in _ASCII_TERMINATORS:
                j += 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if j >= n or text[j] in " \t\n":
                end = j
        if end is not None:
            while end < n and text[end] in " \t\n":
                end += 1
            spans.append((start, end))
            start = i = end
        else:
            i += 1
    if start < n:
        spans.append((start, n))
    return spans


def _subsplit(text: str, s: int, e: int, hard_max: int) -> list[tuple[int, int]]:
    """Split an oversized sentence at clause commas, else hard-cut."""
    if e - s <= hard_max:
        return [(s, e)]
    commas = [i for i in range(s, e - 1) if text[i] in "，、,；;"]
    if commas:
        # cut at the comma nearest the midpoint, recurse on both halves
        mid = s + (e - s) // 2
        cut = min(commas, key=lambda i: abs(i - mid)) + 1
        if s < cut < e:
            return _subsplit(text, s, cut, hard_max) + _subsplit(text, cut, e, hard_max)
    return [(s, min(s + hard_max, e))] + _subsplit(text, min(s + hard_max, e), e, hard_max)


def pack_spans(text: str,
               target: int = NEWS_TARGET_CHARS,
               hard_max: int = NEWS_HARD_MAX,
               min_tail: int = NEWS_MIN_TAIL) -> list[tuple[int, int]]:
    """Group sentence spans into chunk spans (a contiguous partition).

    A section that already fits within `hard_max` stays whole — short
    paragraphs are never split arbitrarily.
    """
    if len(text) <= hard_max:
        return [(0, len(text))] if text else []
    spans: list[tuple[int, int]] = []
    for s, e in sentence_spans(text):
        spans.extend(_subsplit(text, s, e, hard_max))

    chunks: list[tuple[int, int]] = []
    cur_start = cur_end = spans[0][0]
    for s, e in spans:
        if cur_end > cur_start and (e - cur_start) > hard_max:
            chunks.append((cur_start, cur_end))
            cur_start = s
        cur_end = e
        if (cur_end - cur_start) >= target:
            chunks.append((cur_start, cur_end))
            cur_start = cur_end
    if cur_end > cur_start:
        chunks.append((cur_start, cur_end))

    if (len(chunks) >= 2 and chunks[-1][1] - chunks[-1][0] < min_tail
            and chunks[-1][1] - chunks[-2][0] <= int(hard_max * 1.25)):
        chunks[-2:] = [(chunks[-2][0], chunks[-1][1])]
    return chunks


def _overlap_prefix(text: str, chunk_start: int, limit: int = NEWS_OVERLAP_CHARS) -> str:
    """Verbatim tail of the preceding text, trimmed to a whitespace boundary
    so Latin words are not cut mid-word (CJK needs no such boundary)."""
    if chunk_start == 0:
        return ""
    raw = text[max(0, chunk_start - limit):chunk_start]
    ws = raw.find(" ")
    if 0 < ws < len(raw) - 1 and chunk_start - limit > 0:
        raw = raw[ws + 1:]
    return raw


def chunk_news_section(task_id: str, dataset: str, language_name: str,
                       language: str, body_text: str, body_start: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, (s, e) in enumerate(pack_spans(body_text)):
        core = body_text[s:e]
        chunks.append(Chunk(
            chunk_id=f"{dataset}:{task_id}:news:{language}:{idx:03d}",
            task_id=task_id,
            dataset=dataset,
            source_type="news",
            language=language,
            language_name=language_name,
            chunk_index=idx,
            text=core,
            search_text=normalize_for_search(core),
            char_start=body_start + s,
            char_end=body_start + e,
            sha1=_sha1(core),
            overlap_prefix=_overlap_prefix(body_text, s),
        ))
    return chunks


# ------------------------------------------------- financial statement chunking

_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
          r"October|November|December)")
_DATE_RE = re.compile(_MONTH + r"[\s ]*\d{1,2},[\s ]*\d{4}")
_ENDED_RE = re.compile(
    r"(?:Three|Six|Nine|Twelve)[\s ]*Months[\s ]*Ended[\s ]*"
    + f"(?:{_MONTH}" + r"[\s ]*\d{1,2},?[\s ]*(?:\d{4})?)?")
_YEAR_RE = re.compile(r"\b(?:2019|2020|2021)\b")

_STATEMENT_KEYWORDS = {
    "cash_flow": [
        ("statements of cash flows", 5), ("statement of cash flows", 5),
        ("operating activities", 2), ("investing activities", 2),
        ("financing activities", 2), ("net cash from operations", 3),
        ("net cash used in financing", 3), ("net cash from (used in) investing", 3),
        ("net cash provided by", 2), ("net cash used", 1),
        ("cash and cash equivalents, beginning", 2),
    ],
    "balance_sheet": [
        ("balance sheet", 5), ("total assets", 3), ("total liabilities", 3),
        ("current assets", 2), ("current liabilities", 2),
        ("stockholders' equity", 2), ("stockholders’ equity", 2),
        ("shareholders", 1), ("treasury stock", 1), ("retained earnings", 1),
        ("goodwill", 1), ("accounts payable", 1),
    ],
    "income_statement": [
        ("statements of operations", 5), ("statement of operations", 5),
        ("statements of earnings", 5), ("income statements", 5),
        ("statements of income", 5), ("condensed consolidated statement of income", 5),
        ("net sales", 2), ("cost of revenue", 2), ("cost of products sold", 2),
        ("cost of goods sold", 2), ("gross margin", 2), ("gross profit", 2),
        ("operating income", 2), ("earnings per share", 2), ("diluted", 1),
        ("research and development", 1), ("total revenue", 2),
    ],
}


def classify_statement(block_text: str) -> str:
    t = normalize_for_search(block_text)
    scores = {
        name: sum(w for kw, w in kws if kw in t)
        for name, kws in _STATEMENT_KEYWORDS.items()
    }
    best = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    if ranked[0] == 0 or ranked[0] == ranked[1]:
        return "unknown"
    return best


def _cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    return [c.strip() for c in stripped.strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return all(set(c) <= set(":- ") for c in cells) and any("-" in c for c in cells)


def _is_data_row(cells: list[str]) -> bool:
    return any(c for c in cells[1:])


def _is_heading_row(cells: list[str]) -> bool:
    first = cells[0]
    return bool(first) and not any(c for c in cells[1:]) \
        and not _is_separator_row(cells) and first.lower() != "column 1"


def extract_periods(block_text: str, max_lines: int = 15) -> list[str]:
    head = "\n".join(block_text.splitlines()[:max_lines])
    found: list[str] = []
    for rx in (_ENDED_RE, _DATE_RE):
        for m in rx.finditer(head):
            frag = _WS_RUN.sub(" ", m.group(0).replace(" ", " ")).strip()
            if frag and frag not in found:
                found.append(frag)
    if not found:
        for m in _YEAR_RE.finditer(head):
            if m.group(0) not in found:
                found.append(m.group(0))
    return found


def chunk_statement_block(task_id: str, dataset: str, statement_index: int,
                          block_text: str, block_start: int) -> list[Chunk]:
    """Row-aware chunking: contiguous line spans that break at row-group
    headings (never mid-row, never merging across headings) with a cap on
    value-bearing rows per chunk."""
    lines = block_text.split("\n")
    statement_type = classify_statement(block_text)
    periods = extract_periods(block_text)

    # line offsets into block_text for span bookkeeping
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1  # +1 for the joining "\n"

    # table header = everything up to (and incl.) the markdown separator row
    header_end = 0
    for i, ln in enumerate(lines[:4]):
        cells = _cells(ln)
        if cells and _is_separator_row(cells):
            header_end = i + 1
            break
    table_header = "\n".join(lines[:header_end]) if header_end else None

    chunk_line_ranges: list[tuple[int, int]] = []
    cur_start, data_rows = 0, 0
    current_heading: str | None = None
    headings_by_chunk: list[str | None] = []

    for i in range(header_end, len(lines)):
        cells = _cells(lines[i])
        if cells is None:
            continue
        if _is_heading_row(cells):
            if data_rows > 0:
                chunk_line_ranges.append((cur_start, i))
                headings_by_chunk.append(current_heading)
                cur_start, data_rows = i, 0
            current_heading = cells[0]
        elif _is_data_row(cells):
            data_rows += 1
            if data_rows > FS_MAX_DATA_ROWS:
                chunk_line_ranges.append((cur_start, i))
                headings_by_chunk.append(current_heading)
                cur_start, data_rows = i, 1
    chunk_line_ranges.append((cur_start, len(lines)))
    headings_by_chunk.append(current_heading)

    chunks: list[Chunk] = []
    for idx, ((ls, le), heading) in enumerate(zip(chunk_line_ranges, headings_by_chunk)):
        core = "\n".join(lines[ls:le])
        span_start = block_start + offsets[ls]
        search_rows = []
        for ln in lines[ls:le]:
            cells = _cells(ln)
            if cells and _is_data_row(cells) or (cells and _is_heading_row(cells)):
                compact = " | ".join(c for c in cells if c)
                if compact:
                    search_rows.append(compact)
        chunks.append(Chunk(
            chunk_id=f"{dataset}:{task_id}:fs:t{statement_index}:{idx:03d}",
            task_id=task_id,
            dataset=dataset,
            source_type="financial_statement",
            language="en",
            chunk_index=idx,
            text=core,
            search_text=normalize_for_search(" ; ".join(search_rows)),
            char_start=span_start,
            char_end=span_start + len(core),
            sha1=_sha1(core),
            statement_type=statement_type,
            statement_index=statement_index,
            section_label=heading,
            periods=periods,
            table_header=table_header,
        ))
    return chunks
