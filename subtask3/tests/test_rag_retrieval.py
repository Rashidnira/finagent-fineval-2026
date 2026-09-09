"""System D retrieval tests: BM25 determinism/isolation/ties, multilingual
queries, numeric preservation, hybrid fusion (with an injected fake encoder
— no model download in tests), gold-answer exclusion, and News Evidence:
None handling. Purely local, no network, no model, no API."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.bm25 import BM25Retriever, ChunkStore, normalize_source
from src.retrieval.gold_eval import (
    extract_gold_numbers, extract_quotes, number_covered, parse_gold_answer,
    chunk_number_cores, quote_hit,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.chunker import lexical_tokens

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def store():
    return ChunkStore("easy_train")


@pytest.fixture(scope="module")
def bm25(store):
    return BM25Retriever(store)


def _fake_encoder(dim=32):
    """Deterministic hash-based bag-of-tokens encoder for tests."""
    def encode(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in lexical_tokens(t):
                out[i, hash(tok) % dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out
    return encode


@pytest.fixture(scope="module")
def dense(store):
    return SemanticRetriever(store, encode_fn=_fake_encoder(), cache_path=None)


def test_deterministic_ranking(bm25):
    q = "What trends can be observed in the company's revenue amount?"
    a = bm25.retrieve("MSFT_20200429", q, "news", 8)
    b = bm25.retrieve("MSFT_20200429", q, "news", 8)
    assert [(h.chunk_id, h.score) for h in a] == [(h.chunk_id, h.score) for h in b]


def test_task_id_isolation(bm25, store):
    for task in ("MSFT_20200429", "UVV_20210208"):
        for src in ("news", "financial_statements"):
            hits = bm25.retrieve(task, "revenue growth cash flow", src, 50)
            assert hits and all(h.metadata["task_id"] == task for h in hits)
    with pytest.raises(KeyError):
        store.chunks("FAKE_TASK", "news")


def test_source_isolation(bm25):
    news = bm25.retrieve("JNJ_20200429", "revenue", "news", 50)
    fs = bm25.retrieve("JNJ_20200429", "revenue", "financial_statements", 50)
    assert all(h.metadata["source_type"] == "news" for h in news)
    assert all(h.metadata["source_type"] == "financial_statement" for h in fs)
    assert normalize_source("financial_statements") == "financial_statement"


def test_stable_tie_handling(bm25):
    # a query matching nothing gives all-zero scores → order must be chunk_id
    hits = bm25.retrieve("MSFT_20200429", "zzz qqq xxx", "news", 10)
    assert all(h.score == 0.0 for h in hits)
    assert [h.chunk_id for h in hits] == sorted(h.chunk_id for h in hits)


def test_multilingual_query(bm25):
    hits = bm25.retrieve("MSFT_20200429", "微软第三财季营收", "news", 3)
    assert hits[0].metadata["language"] == "zh" and hits[0].score > 0
    hits_el = bm25.retrieve("MSFT_20200429", "Microsoft πωλήσεις cloud", "news", 3)
    assert any(h.metadata["language"] == "el" for h in hits_el)


def test_numeric_token_preservation():
    toks = lexical_tokens("revenue $35.0B grew 14.6% vs (4,082) million")
    assert "14.6%" in toks and "35.0" in toks and "4,082" in toks


def test_gold_number_extraction_and_matching():
    nums = extract_gold_numbers(
        "Revenue was $143,015 million, up 14.6%; loss of (4,082); in 2020.")
    raws = {n.raw for n in nums}
    assert any("143,015" in r for r in raws)
    assert any("14.6%" in r for r in raws)
    assert any("4,082" in r for r in raws)
    assert not any(n.core == "2020" for n in nums)  # bare year excluded
    cores = chunk_number_cores(["| Revenue | $ | 143,015 | (4,082 | ) |"])
    by_core = {n.core: n for n in nums}
    assert number_covered(by_core["143015"], cores)
    assert number_covered(by_core["4082"], cores)
    # magnitude normalization: $35.0B matches 35,021 (millions) within 0.5%
    g = extract_gold_numbers("revenue of $35.0B")[0]
    assert g.magnitude == "B" and number_covered(g, {"35021"})
    assert not number_covered(g, {"36000"})


def test_evidence_none_handling():
    _, body, is_none = parse_gold_answer("Answer: stable.\nNews Evidence: None.")
    assert is_none
    _, body2, is_none2 = parse_gold_answer(
        'Answer: grew.\nNews Evidence: Chinese news: "营收350亿美元"')
    assert not is_none2 and "350" in body2
    assert parse_gold_answer("Answer: no label at all")[2] is True


def test_quote_extraction_and_hit():
    quotes = extract_quotes('Japanese news: "3月期の売上高は350億ドルで前年同期比15%増。"')
    assert len(quotes) == 1
    assert quote_hit(quotes[0], ["…3月期の売上高は350億ドルで前年同期比15%増。…"])
    assert not quote_hit(quotes[0], ["completely unrelated text"])


def test_hybrid_deterministic_and_isolated(bm25, dense):
    hy = HybridRetriever(bm25, dense, "rrf")
    a = hy.retrieve("HON_20200501", "capital expenditures", "financial_statements", 8)
    b = hy.retrieve("HON_20200501", "capital expenditures", "financial_statements", 8)
    assert [(h.chunk_id, round(h.score, 12)) for h in a] == \
           [(h.chunk_id, round(h.score, 12)) for h in b]
    assert all(h.metadata["task_id"] == "HON_20200501" and
               h.metadata["source_type"] == "financial_statement" for h in a)
    hw = HybridRetriever(bm25, dense, "weighted", alpha=0.7)
    c = hw.retrieve("HON_20200501", "capital expenditures", "financial_statements", 8)
    assert len(c) == 8 and c[0].score >= c[-1].score


def test_gold_answers_never_reach_retrieval(bm25, monkeypatch):
    """Behavioral guarantee: the eval harness queries with question text only."""
    from src.retrieval import easy_eval
    captured = []
    real = bm25.retrieve

    def spy(task_id, question, source_type, top_k):
        captured.append(question)
        return real(task_id, question, source_type, top_k)

    monkeypatch.setattr(bm25, "retrieve", spy)
    df = pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet",
                         columns=["task_id", "question", "answer"]).head(2)
    easy_eval.evaluate_retriever(bm25, df)
    assert captured
    for q, (_, row) in zip(captured[::2], df.iterrows()):
        assert q == row["question"]
    for q in captured:
        assert "News Evidence" not in q  # no gold-answer fragments in queries


def test_semantic_uses_injected_encoder_not_model(dense):
    hits = dense.retrieve("MSFT_20200429", "cloud revenue growth", "news", 5)
    assert len(hits) == 5 and hits[0].score >= hits[-1].score
    assert all(h.metadata["task_id"] == "MSFT_20200429" for h in hits)
