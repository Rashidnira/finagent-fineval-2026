"""System D prompt-builder tests (bm25 retriever — no model needed):
full-context preservation, evidence block structure, determinism, and the
no-answer guarantee (builder loads task_id/question/query columns only)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.bm25 import BM25Retriever, ChunkStore
from src.retrieval.build_system_d import (
    BLOCK_HEADER, DATASETS, build_dataset, compact_rows, evidence_block,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built():
    store = ChunkStore("expert_test")
    retriever = BM25Retriever(store)
    return build_dataset("expert_test", DATASETS["expert_test"], retriever, k=3)


def test_full_context_preserved(built):
    df = pd.read_parquet(DATASETS["expert_test"], columns=["task_id", "query"])
    queries = dict(zip(zip(df["task_id"], range(len(df))), df["query"]))
    by_order = list(df["query"])
    for row, orig in zip(built, by_order):
        ins, bl = row["insertion_offset"], row["block_len"]
        p = row["prompt"]
        assert p[:ins] + p[ins + bl:] == orig          # context byte-identical
        assert p[ins:ins + len(BLOCK_HEADER)] == BLOCK_HEADER
        assert p[ins + bl:].startswith("Question:")


def test_block_structure_and_task_isolation(built):
    for row in built:
        assert "[Financial statement excerpts]" in row["prompt"]
        assert "[News excerpts]" in row["prompt"]
        for src in ("news", "financial_statements"):
            for hit in row["retrieval"][src]:
                assert hit["chunk_id"].split(":")[1] == row["task_id"]


def test_deterministic_build(built):
    store = ChunkStore("expert_test")
    again = build_dataset("expert_test", DATASETS["expert_test"],
                          BM25Retriever(store), k=3)
    assert [r["prompt"] for r in built] == [r["prompt"] for r in again]


def test_compact_rows_preserve_figures():
    text = ("| Cash and cash equivalents | | $ | 11,710 | | | $ | 11,356 |\n"
            "|:---|:---|:---|:---|:---|:---|:---|:---|\n"
            "| Net loss | | (4,082 | ) | | | | |")
    out = compact_rows(text)
    assert "Cash and cash equivalents | $ | 11,710 | $ | 11,356" in out
    assert "(4,082" in out and ":---" not in out


def test_no_answer_columns_used():
    """Builder reads task_id/question/query only; expert file has no answers."""
    df = pd.read_parquet(DATASETS["expert_test"])
    assert "answer" not in df.columns
    import inspect
    from src.retrieval import build_system_d
    src_code = inspect.getsource(build_system_d)
    assert '"answer"' not in src_code and "'answer'" not in src_code
