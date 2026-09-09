"""Phase 2 scorer tests: identity, analytical hand-computed cases, fidelity
against the reference google rouge_score implementation, and the CJK-quote
policy case (user-mandated: bounds the quoting-policy consequence of the
scorer ambiguity)."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scorer import rouge1_prf, score_pairs
from src.tokenizer import tokenize_latin_default, tokenize_multiscript

GOLD = ("Answer: Microsoft's total revenue increased to $35.0B in Q3 FY2020, "
        "a 14.6% YoY growth from $30.6B.\n"
        "News Evidence: Chinese news: \"微软第三财季营收达到350亿美元\"")


def test_identity_both_scorers():
    for scorer in ("multifinben_default", "finmmeval_multiscript"):
        r = rouge1_prf(GOLD, GOLD, scorer)
        assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0


def test_analytical_case():
    # reference: 4 tokens; prediction: 2 tokens, both overlap
    ref = "revenue increased 20 percent"
    pred = "revenue increased"
    for scorer in ("multifinben_default", "finmmeval_multiscript"):
        r = rouge1_prf(ref, pred, scorer)
        assert r["precision"] == 1.0
        assert r["recall"] == 0.5
        assert math.isclose(r["f1"], 2 / 3, rel_tol=1e-9)


def test_analytical_multiset_clipping():
    # repeated token counts clip at reference multiplicity
    ref = "growth growth in cloud"          # growth x2, in, cloud
    pred = "growth growth growth cloud"     # growth x3, cloud
    r = rouge1_prf(ref, pred, "multifinben_default")
    assert r["overlap"] == 3                # min(2,3) growth + cloud
    assert math.isclose(r["precision"], 3 / 4)
    assert math.isclose(r["recall"], 3 / 4)


def test_empty_prediction_zero_not_nan():
    r = rouge1_prf("some reference", "", "finmmeval_multiscript")
    assert r == {**r, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_number_tokenization():
    # "$35.0B" -> latin: ['35','0b']; multiscript: ['35','0b'] after NFKC
    assert tokenize_latin_default("$35.0B") == ["35", "0b"]
    assert tokenize_multiscript("$35.0B") == ["35", "0b"]
    assert tokenize_latin_default("14.6%") == ["14", "6"]
    assert tokenize_latin_default("Q3 FY2020") == ["q3", "fy2020"]


def test_cjk_visibility_difference():
    zh = "微软第三财季营收达到350亿美元"
    assert tokenize_latin_default(zh) == ["350"]        # CJK invisible
    toks = tokenize_multiscript(zh)
    assert "350" in toks and len(toks) == 14            # 13 Han chars + '350'


def test_latin_default_matches_reference_package():
    rouge_score = pytest.importorskip("rouge_score.rouge_scorer")
    rs = rouge_score.RougeScorer(["rouge1"], use_stemmer=False)
    cases = [
        ("revenue increased 20 percent", "revenue increased"),
        (GOLD, "Answer: revenue increased to $35.0B in Q3 FY2020."),
        ("a b c", "a b c"),
    ]
    for ref, pred in cases:
        ours = rouge1_prf(ref, pred, "multifinben_default")
        theirs = rs.score(ref, pred)["rouge1"]
        assert math.isclose(ours["precision"], theirs.precision, rel_tol=1e-9)
        assert math.isclose(ours["recall"], theirs.recall, rel_tol=1e-9)
        assert math.isclose(ours["f1"], theirs.fmeasure, rel_tol=1e-9)


def test_cjk_quote_policy_case():
    """User-mandated Expert-relevant case: does including an original-language
    quote help, hurt, or do nothing under each scorer?

    Reference contains a Chinese quote (as Easy gold answers do). Compare a
    prediction WITH the quote vs the same prediction WITHOUT it.
    """
    ref = GOLD
    with_quote = ("Answer: Total revenue increased to $35.0B in Q3 FY2020, "
                  "a 14.6% YoY growth.\n"
                  "News Evidence: Chinese news: \"微软第三财季营收达到350亿美元\"")
    without_quote = ("Answer: Total revenue increased to $35.0B in Q3 FY2020, "
                     "a 14.6% YoY growth.\nNews Evidence: None")

    # Latin-only scorer: quote adds only '350' (already common); dropping the
    # quote must NOT increase F1 materially, and the quote never *hurts*.
    a_with = rouge1_prf(ref, with_quote, "multifinben_default")["f1"]
    a_without = rouge1_prf(ref, without_quote, "multifinben_default")["f1"]
    assert a_with >= a_without - 1e-9

    # Multiscript scorer: the quote contributes 13+ matching tokens -> strictly better.
    b_with = rouge1_prf(ref, with_quote, "finmmeval_multiscript")["f1"]
    b_without = rouge1_prf(ref, without_quote, "finmmeval_multiscript")["f1"]
    assert b_with > b_without + 0.05


def test_score_pairs_macro_average():
    refs = ["revenue increased 20 percent", "cash flow declined"]
    preds = ["revenue increased", "cash flow declined"]
    out = score_pairs(refs, preds, "multifinben_default")
    assert out["n"] == 2
    assert math.isclose(out["rouge1_f1"], (2 / 3 + 1.0) / 2, rel_tol=1e-9)
