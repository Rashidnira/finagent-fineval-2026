"""Deterministic answer validation: format, numeric grounding, quote grounding.

Used in Phase 5 (Easy benchmark evaluation) and Phase 8 (Expert validation).
All checks are against the supplied context only — never external references.
"""
import unicodedata

import regex as re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_QUOTE = re.compile(r'[“"«「]([^”"»」]{6,600})[”"»」]')
_FENCE = re.compile(r"```")
_ELLIPSIS = re.compile(r"\.{3}|…")


def _norm_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def _norm_agg(s: str) -> str:
    """NFKC + lowercase + letters/digits only — robust to punctuation and
    quote-mark variants (calibrated on gold: removes 13/22 false mismatches)."""
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[^\p{L}\p{Nd}]+", "", s)


def quote_grounded(quote: str, ctx_agg: str) -> bool:
    """A quote is grounded if every ellipsis-separated fragment (annotators
    splice quotes with '...') appears in the aggressively-normalized context."""
    fragments = [f for f in _ELLIPSIS.split(quote) if len(_norm_agg(f)) >= 6]
    if not fragments:
        return True
    return all(_norm_agg(f) in ctx_agg for f in fragments)


def numbers_in(text: str) -> set:
    return {_norm_num(m) for m in _NUM.findall(text)}


def validate(prediction: str, context: str, evidence_label: str) -> dict:
    """Return per-row validation flags. evidence_label is
    'News Evidence:' (Easy) or 'Financial Statements Evidence:' (Expert)."""
    flags = []
    pred = prediction.strip()

    # --- format ---
    if not pred:
        flags.append("empty")
    if "Answer:" not in pred:
        flags.append("missing_answer_label")
    if evidence_label not in pred:
        flags.append("missing_evidence_label")
    words = len(pred.split())
    if words > 110:
        flags.append("over_length")
    if _FENCE.search(pred):
        flags.append("code_fence")

    # --- numeric grounding ---
    ctx_nums = numbers_in(context)
    pred_nums = numbers_in(pred)
    untraceable = sorted(n for n in pred_nums if n not in ctx_nums)
    # numbers plausibly derived (ratios/percentages) rather than copied:
    derived = [n for n in untraceable if "." in n or len(n) <= 3]
    hard_untraceable = [n for n in untraceable if n not in derived]
    if hard_untraceable:
        flags.append("untraceable_numbers")

    # --- quote grounding ---
    ctx_agg = _norm_agg(context)
    quotes = _QUOTE.findall(pred)
    unverbatim = [q[:60] for q in quotes if not quote_grounded(q, ctx_agg)]
    if unverbatim:
        flags.append("unverbatim_quotes")

    return {
        "flags": flags,
        "ok": not flags,
        "word_count": words,
        "n_pred_numbers": len(pred_nums),
        "n_untraceable_numbers": len(hard_untraceable),
        "untraceable_numbers": hard_untraceable[:8],
        "n_derived_numbers": len(derived),
        "n_quotes": len(quotes),
        "n_unverbatim_quotes": len(unverbatim),
        "unverbatim_quotes": unverbatim[:4],
    }
