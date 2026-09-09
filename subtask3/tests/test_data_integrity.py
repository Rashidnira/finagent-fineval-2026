"""Data-integrity and validator regression tests."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.answer_validator import validate
from src.question_router import route

ROOT = Path(__file__).resolve().parents[1]


def _train():
    return pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet")


def _test():
    return pd.read_parquet(ROOT / "data/manifests/official_finnlp_test.parquet")


def test_dataset_shapes():
    tr, te = _train(), _test()
    assert len(tr) == 76 and len(te) == 76
    assert te.id.is_unique and te.id.notna().all()
    assert set(tr.task_id) == set(te.task_id)
    assert tr.task_id.value_counts().eq(4).all()


def test_all_questions_route():
    tr, te = _train(), _test()
    assert {route(q) for q in tr.question.unique()} == {
        "Revenue", "BalanceSheet", "CashFlow", "RnD"}
    assert {route(q) for q in te.question.unique()} == {
        "RevenueFocus", "CapitalAllocation", "MarginStrategy", "Capex"}


def test_gold_answers_low_false_positive_rate():
    """Validator must not mass-flag the organizers' own gold answers."""
    tr = _train()
    flagged_quotes = flagged_nums = 0
    for _, r in tr.iterrows():
        v = validate(r.answer, r.query, "News Evidence:")
        flagged_quotes += v["n_unverbatim_quotes"] > 0
        flagged_nums += v["n_untraceable_numbers"] > 2
    assert flagged_quotes <= 8, f"quote check too strict: {flagged_quotes}/76"
    assert flagged_nums <= 8, f"numeric check too strict: {flagged_nums}/76"
