"""System D retrieval data layer tests: parser coverage, chunk
losslessness, metadata completeness, determinism, and the no-gold-answer
guarantee. Purely local — no network, no model calls."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.context_parser import parse_context, LANG_CODE
from src.retrieval.chunker import (
    chunk_news_section, chunk_statement_block, sentence_spans, pack_spans,
    normalize_for_search, lexical_tokens, NEWS_HARD_MAX,
)

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "easy_train": ROOT / "data/raw/public-00000-of-00001.parquet",
    "expert_test": ROOT / "data/raw/PolyFiQA_test_participant.parquet",
}


@pytest.fixture(scope="module")
def contexts():
    """One (task_id, query) per task_id per dataset; answer column never read."""
    out = {}
    for name, path in FILES.items():
        df = pd.read_parquet(path, columns=["task_id", "query"])
        out[name] = list(df.drop_duplicates("task_id")[["task_id", "query"]]
                         .itertuples(index=False, name=None))
    return out


def test_parser_structure_all_tasks(contexts):
    for name, rows in contexts.items():
        assert len(rows) == 19
        for task_id, q in rows:
            p = parse_context(task_id, q)  # verify_coverage raises on lost text
            assert 3 <= len(p.statement_blocks) <= 4
            assert [s.language_name for s in p.news_sections] == \
                ["English", "Chinese", "Japanese", "Spanish", "Greek"]
            assert p.question.text.startswith("Question:")
            for blk in p.statement_blocks:
                assert q[blk.start:blk.end] == blk.text


def test_news_chunks_lossless_and_spans(contexts):
    for name, rows in contexts.items():
        for task_id, q in rows:
            p = parse_context(task_id, q)
            for sec in p.news_sections:
                cs = chunk_news_section(task_id, name, sec.language_name,
                                        sec.language, sec.body.text, sec.body.start)
                assert "".join(c.text for c in cs) == sec.body.text
                for c in cs:
                    assert q[c.char_start:c.char_end] == c.text
                    assert len(c.text) <= int(NEWS_HARD_MAX * 1.25) + 200


def test_statement_chunks_lossless_and_row_aware(contexts):
    for name, rows in contexts.items():
        for task_id, q in rows:
            p = parse_context(task_id, q)
            for ti, blk in enumerate(p.statement_blocks):
                cs = chunk_statement_block(task_id, name, ti, blk.text, blk.start)
                assert "\n".join(c.text for c in cs) == blk.text
                for c in cs:
                    assert q[c.char_start:c.char_end] == c.text
                    # never splits mid-row: chunk boundaries are line boundaries
                    assert not c.text.startswith(" |")


def test_statement_classification_covers_three_types(contexts):
    for name, rows in contexts.items():
        for task_id, q in rows:
            p = parse_context(task_id, q)
            types = set()
            for ti, blk in enumerate(p.statement_blocks):
                cs = chunk_statement_block(task_id, name, ti, blk.text, blk.start)
                types.add(cs[0].statement_type)
            assert {"balance_sheet", "cash_flow", "income_statement"} <= types, \
                f"{name}/{task_id}: {types}"


def test_metadata_complete_and_ids_unique(contexts):
    name = "expert_test"
    seen = set()
    for task_id, q in contexts[name]:
        p = parse_context(task_id, q)
        chunks = []
        for sec in p.news_sections:
            chunks += chunk_news_section(task_id, name, sec.language_name,
                                         sec.language, sec.body.text, sec.body.start)
        for ti, blk in enumerate(p.statement_blocks):
            chunks += chunk_statement_block(task_id, name, ti, blk.text, blk.start)
        for c in chunks:
            d = c.to_dict()
            assert d["task_id"] == task_id
            assert d["source_type"] in ("news", "financial_statement")
            assert d["language"] in LANG_CODE.values()
            assert isinstance(d["chunk_index"], int)
            assert d["text"] and d["search_text"] and d["sha1"]
            if d["source_type"] == "financial_statement":
                assert d["statement_type"] in (
                    "balance_sheet", "cash_flow", "income_statement", "unknown")
                assert d["statement_index"] is not None
            assert c.chunk_id not in seen
            seen.add(c.chunk_id)
            json.dumps(d, ensure_ascii=False)  # JSONL-serializable


def test_overlap_prefix_is_verbatim_previous_tail(contexts):
    name = "easy_train"
    for task_id, q in contexts[name]:
        p = parse_context(task_id, q)
        for sec in p.news_sections:
            cs = chunk_news_section(task_id, name, sec.language_name,
                                    sec.language, sec.body.text, sec.body.start)
            assert cs[0].overlap_prefix == ""
            for prev, cur in zip(cs, cs[1:]):
                assert cur.overlap_prefix
                assert prev.text.endswith(cur.overlap_prefix)


def test_determinism(contexts):
    task_id, q = contexts["easy_train"][0]
    p = parse_context(task_id, q)
    sec = p.news_sections[0]
    a = chunk_news_section(task_id, "easy_train", sec.language_name,
                           sec.language, sec.body.text, sec.body.start)
    b = chunk_news_section(task_id, "easy_train", sec.language_name,
                           sec.language, sec.body.text, sec.body.start)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_sentence_spans_partition_and_number_safety():
    text = "Revenue hit $760.98 billion. 微软营收增长。见财报！Growth was 14.6% YoY."
    spans = sentence_spans(text)
    assert "".join(text[s:e] for s, e in spans) == text
    joined = [text[s:e] for s, e in spans]
    assert any("760.98 billion" in s for s in joined)   # decimal not split
    assert len(spans) >= 3                              # CJK terminators split


def test_pack_spans_keeps_short_sections_whole():
    short = "Una frase corta en español."
    assert pack_spans(short) == [(0, len(short))]


def test_normalization_preserves_original_separately():
    orig = "Ｍicrosoft Ｒevenue  35,000  ΕΛΛΗΝΙΚΑ 微软"
    norm = normalize_for_search(orig)
    assert orig != norm and "microsoft" in norm and "ελληνικα" in norm
    toks = lexical_tokens("微软第三财季营收350亿美元 growth 14.6%")
    assert "微" in toks and "软" in toks and "growth" in toks and "14.6%" in toks


def test_no_answer_column_needed():
    """The chunker runs on the participant test file, which has no answer
    column at all — structural proof that gold answers are not used."""
    df = pd.read_parquet(FILES["expert_test"])
    assert "answer" not in df.columns
