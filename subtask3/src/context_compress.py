"""Lossless-content whitespace compression.

The supplied query contexts pad markdown financial tables with long space
runs (30-45% of all characters). Collapsing horizontal whitespace runs and
excess blank lines removes NO content — every figure, label, and news
sentence is preserved verbatim — but cuts input tokens enough to fit the
Gemma free-tier 16,000-input-token/minute admission bucket (measured
2026-08-19: HON_20201030 grouped 18,518 → 16,648 tokens).

This is deterministic preprocessing, not truncation: nothing is removed
except repeated whitespace. Quote-grounding validation is unaffected (the
validator compares under aggressive normalization).
"""
import re

_HSPACE = re.compile(r"[ \t]{2,}")
_BLANKS = re.compile(r"\n{3,}")


def compress_ws(text: str) -> str:
    return _BLANKS.sub("\n\n", _HSPACE.sub(" ", text))
