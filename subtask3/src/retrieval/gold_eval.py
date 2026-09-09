"""Deterministic Easy-gold proxy metrics for retrieval evaluation.

Gold answers are used ONLY here, as evaluation targets — never as retrieval
queries, never to build indexes, never for Expert rows (the participant
test file has no answer column at all).

Measures
--------
A. Gold-number coverage: financial numerics extracted from the gold answer
   (signs, %, currencies, parenthesized negatives, B/M magnitude suffixes
   recognized); a number counts as covered when its comma-stripped core
   appears among the numeric tokens of the retrieved statement chunks, or —
   for B/billion-suffixed values — when a chunk value matches within 0.5%
   after millions scaling (statements report in $M).
   Excluded as non-financial: bare years 2018-2021 and unflagged single
   digits.
B. News-evidence coverage: normalized-token set recall of the gold
   `News Evidence:` body against retrieved news chunks, plus a near-exact
   quote-hit rate (quoted spans, whitespace/punct-insensitive substring
   match). `None` evidence rows are excluded, tracked separately.
C. Overall lexical support: token-set recall of the whole gold answer
   against the union of retrieved news+statement chunks (diagnostic only).
"""
from __future__ import annotations

from dataclasses import dataclass

import regex as re

from src.retrieval.chunker import lexical_tokens, normalize_for_search

EVIDENCE_LABEL = "News Evidence:"

_NUM_RE = re.compile(
    r"\(?[-+]?[$€¥£]\s?\d[\d,]*(?:\.\d+)?\s?(?:%|B\b|M\b|bn\b|billion|million)?\)?"
    r"|\(?[-+]?\d[\d,]*(?:\.\d+)?\s?(?:%|B\b|M\b|bn\b|billion|million)\)?"
    r"|\(?[-+]?\d[\d,]*(?:\.\d+)?\)?")
_CHUNK_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_QUOTE_RE = re.compile(r"“([^”]{10,})”|\"([^\"]{10,})\"|「([^」]{5,})」|«([^»]{10,})»")
_YEARS = {"2018", "2019", "2020", "2021"}


@dataclass
class GoldNumber:
    raw: str
    core: str          # comma-stripped digits(+decimal), no sign/currency
    negative: bool
    percent: bool
    currency: bool
    magnitude: str     # "", "B", "M"


def parse_gold_answer(ans: str) -> tuple[str, str, bool]:
    """→ (answer_section, evidence_body, evidence_is_none)."""
    i = ans.find(EVIDENCE_LABEL)
    if i == -1:
        return ans.strip(), "", True
    answer_sec = ans[:i].strip()
    body = ans[i + len(EVIDENCE_LABEL):].strip()
    is_none = normalize_for_search(body).strip('."”“\' ') in {"none", ""}
    return answer_sec, body, is_none


def extract_gold_numbers(text: str) -> list[GoldNumber]:
    out, seen = [], set()
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        core_m = _CHUNK_NUM_RE.search(raw)
        if not core_m:
            continue
        core = core_m.group(0).replace(",", "").rstrip(".")
        flags = dict(
            negative="(" in raw or raw.lstrip().startswith("-"),
            percent="%" in raw,
            currency=any(c in raw for c in "$€¥£"),
            magnitude=("B" if re.search(r"B\b|bn\b|billion", raw) else
                       "M" if re.search(r"M\b|million", raw) else ""),
        )
        # non-financial exclusions
        if core in _YEARS and not any([flags["percent"], flags["currency"],
                                       flags["magnitude"]]):
            continue
        if len(core) == 1 and "." not in core and not any(
                [flags["percent"], flags["currency"], flags["magnitude"]]):
            continue
        key = (core, flags["percent"], flags["magnitude"])
        if key in seen:
            continue
        seen.add(key)
        out.append(GoldNumber(raw=raw.strip(), core=core, **flags))
    return out


def chunk_number_cores(texts: list[str]) -> set[str]:
    cores = set()
    for t in texts:
        for m in _CHUNK_NUM_RE.finditer(t):
            cores.add(m.group(0).replace(",", "").rstrip("."))
    return cores


def number_covered(g: GoldNumber, cores: set[str]) -> bool:
    if g.core in cores:
        return True
    # magnitude-normalized match: statements report in millions
    scale = {"B": 1000.0, "M": 1.0}.get(g.magnitude)
    if scale is None:
        return False
    try:
        target = float(g.core) * scale
    except ValueError:
        return False
    if target == 0:
        return False
    for c in cores:
        try:
            v = float(c)
        except ValueError:
            continue
        if v > 0 and abs(v - target) / target <= 0.005:
            return True
    return False


def extract_quotes(evidence_body: str) -> list[str]:
    quotes = []
    for m in _QUOTE_RE.finditer(evidence_body):
        q = next(g for g in m.groups() if g)
        quotes.append(q)
    return quotes


def _squash(text: str) -> str:
    """Normalized, whitespace- and punctuation-insensitive comparison form."""
    return re.sub(r"[\s\p{P}]+", "", normalize_for_search(text))


def quote_hit(quote: str, news_texts: list[str]) -> bool:
    needle = _squash(quote)
    if len(needle) < 8:
        return False
    hay = _squash(" ".join(news_texts))
    return needle in hay


def token_set_recall(gold_text: str, retrieved_texts: list[str]) -> float | None:
    gold = set(lexical_tokens(gold_text))
    if not gold:
        return None
    have = set()
    for t in retrieved_texts:
        have.update(lexical_tokens(t))
    return len(gold & have) / len(gold)
