"""Offset-based parser for PolyFiQA query contexts (System D input layer).

Every extracted piece is returned as an exact character span into the
original query string, so losslessness is verifiable: `q[start:end]` must
equal the stored text, and the leftover characters between spans must be
only structural markup (headers, `---` separators, whitespace).

Observed context grammar (verified over all 19 task_ids in both supplied
files, see reports/DATA_AUDIT.md):

    <instruction>
    Context:
    Financial Statements:
    <markdown table>            (3 tables; UVV filings have 4)
    ---
    <markdown table>
    ---
    ...
    ---
    English News:
    <single-line body>
    Chinese News:
    <single-line body>
    Japanese News:
    <single-line body>
    Spanish News:
    <single-line body>
    Greek News:
    <single-line body>
    Question:
    <question text>
    Answer:
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FS_MARKER = "Financial Statements:"
CTX_MARKER = "Context:"

NEWS_HEADER_RE = re.compile(r"(?m)^(English|Chinese|Japanese|Spanish|Greek) News:[ \t]*\n")
TABLE_SEP_RE = re.compile(r"(?m)^---[ \t]*$")
QUESTION_RE = re.compile(r"(?m)^Question:")

LANG_CODE = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "Spanish": "es",
    "Greek": "el",
}


@dataclass
class Span:
    """A verbatim slice of the original query string."""
    start: int
    end: int
    text: str

    @classmethod
    def of(cls, q: str, start: int, end: int) -> "Span":
        return cls(start, end, q[start:end])


@dataclass
class NewsSection:
    language_name: str          # section header name, e.g. "Chinese"
    language: str               # ISO-ish code, e.g. "zh"
    body: Span                  # verbatim body (leading/trailing ws trimmed via span)


@dataclass
class ParsedContext:
    task_id: str
    query_sha1: str
    instruction: Span
    statement_blocks: list[Span] = field(default_factory=list)
    news_sections: list[NewsSection] = field(default_factory=list)
    question: Span | None = None


def _trimmed_span(q: str, start: int, end: int) -> Span:
    """Shrink [start, end) to exclude leading/trailing whitespace, keeping
    the interior verbatim."""
    while start < end and q[start] in " \t\r\n":
        start += 1
    while end > start and q[end - 1] in " \t\r\n":
        end -= 1
    return Span.of(q, start, end)


def parse_context(task_id: str, q: str) -> ParsedContext:
    import hashlib

    ctx_pos = q.index(CTX_MARKER)
    fs_pos = q.index(FS_MARKER, ctx_pos)
    fs_body_start = fs_pos + len(FS_MARKER)

    news_headers = list(NEWS_HEADER_RE.finditer(q, fs_body_start))
    if not news_headers:
        raise ValueError(f"{task_id}: no news sections found")

    q_matches = list(QUESTION_RE.finditer(q, news_headers[-1].end()))
    if not q_matches:
        raise ValueError(f"{task_id}: no trailing Question: block")
    question_pos = q_matches[0].start()

    parsed = ParsedContext(
        task_id=task_id,
        query_sha1=hashlib.sha1(q.encode("utf-8")).hexdigest(),
        instruction=_trimmed_span(q, 0, ctx_pos),
    )

    # ---- financial statement tables: split on standalone `---` lines ----
    fs_region_end = news_headers[0].start()
    cursor = fs_body_start
    boundaries = [m for m in TABLE_SEP_RE.finditer(q, fs_body_start, fs_region_end)]
    for sep in boundaries:
        blk = _trimmed_span(q, cursor, sep.start())
        if blk.text:
            parsed.statement_blocks.append(blk)
        cursor = sep.end()
    tail = _trimmed_span(q, cursor, fs_region_end)
    if tail.text:
        parsed.statement_blocks.append(tail)

    # ---- news sections: header → next header (or Question:) ----
    ends = [m.start() for m in news_headers[1:]] + [question_pos]
    for m, end in zip(news_headers, ends):
        name = m.group(1)
        parsed.news_sections.append(NewsSection(
            language_name=name,
            language=LANG_CODE[name],
            body=_trimmed_span(q, m.end(), end),
        ))

    parsed.question = _trimmed_span(q, question_pos, len(q))
    verify_coverage(parsed, q)
    return parsed


def verify_coverage(parsed: ParsedContext, q: str) -> None:
    """Assert nothing but structural markup lies between extracted spans.

    Rebuilds the evidence region (Financial Statements ... Question:) by
    blanking every extracted span plus known markers; whatever survives must
    be whitespace or `---` separators, otherwise source text would be lost.
    """
    buf = list(q)

    def blank(start: int, end: int) -> None:
        for i in range(start, end):
            buf[i] = " "

    blank(parsed.instruction.start, parsed.instruction.end)
    ctx_pos = q.index(CTX_MARKER)
    blank(ctx_pos, ctx_pos + len(CTX_MARKER))
    fs_pos = q.index(FS_MARKER, ctx_pos)
    blank(fs_pos, fs_pos + len(FS_MARKER))
    for blk in parsed.statement_blocks:
        blank(blk.start, blk.end)
    for m in NEWS_HEADER_RE.finditer(q):
        blank(m.start(), m.end())
    for sec in parsed.news_sections:
        blank(sec.body.start, sec.body.end)
    blank(parsed.question.start, parsed.question.end)

    residue = "".join(buf)
    residue = TABLE_SEP_RE.sub("", residue)
    if residue.strip():
        leftover = residue.strip()[:200]
        raise AssertionError(
            f"{parsed.task_id}: unextracted context text would be lost: {leftover!r}")
