"""Tokenizers for the two candidate ROUGE-1 scorers.

Scorer A — "multifinben_default":
    Replicates google-research `rouge_score.tokenize.tokenize(text, stemmer=None)`,
    which is what MultiFinBen's official eval uses via `evaluate.load("rouge")`
    (repo xueqingpeng/MultiFinBen @ a3fc54082d0b, tasks/multilingual/ml_utils.py).
    Behavior: lowercase, replace every non-[a-z0-9] char with a space, split.
    Consequence: CJK/kana/Greek text produces NO tokens (metric-invisible).

Scorer B — "finmmeval_multiscript":
    Reconstruction of the FinMMEval 2026 Task 2 scorer description
    (arXiv:2607.19867 §evaluation): Unicode NFKC normalization, lowercasing,
    then a regex that tokenizes words/numbers as runs and CJK characters
    individually; kana, Arabic-script and Greek/Latin ranges are covered.
    The exact regex is not published; this reconstruction treats Han,
    Hiragana and Katakana as single-character tokens and Latin/Greek/
    Cyrillic/Arabic alphanumeric runs as word tokens.
"""
import unicodedata

import regex as re

# --- Scorer A ---------------------------------------------------------------
_LATIN_STRIP = re.compile(r"[^a-z0-9]+")


def tokenize_latin_default(text: str) -> list:
    """rouge_score default tokenizer (no stemming)."""
    return [t for t in _LATIN_STRIP.split(text.lower()) if t]


# --- Scorer B ---------------------------------------------------------------
# Single-character scripts (character-level tokens per the paper: "CJK
# characters" and Japanese kana).
# Word scripts: Latin, Greek, Arabic letters, digits — tokenized as runs.
_MULTISCRIPT = re.compile(
    r"[\p{Han}\p{Hiragana}\p{Katakana}]"          # one token per CJK/kana char
    r"|[\p{Latin}\p{Greek}\p{Cyrillic}\p{Arabic}\p{Nd}]+"  # word/number runs
)


def tokenize_multiscript(text: str) -> list:
    text = unicodedata.normalize("NFKC", text).lower()
    return _MULTISCRIPT.findall(text)


TOKENIZERS = {
    "multifinben_default": tokenize_latin_default,
    "finmmeval_multiscript": tokenize_multiscript,
}
