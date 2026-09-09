"""Deterministic BM25 (Okapi) retrieval over System D chunk collections.

- One index per (task_id, source_type): retrieval NEVER crosses task_ids
  or mixes news with financial statements.
- Tokens come from the existing lexical layer (`lexical_tokens` over the
  stored `search_text`); the verbatim original `text` is returned untouched.
- Fully deterministic: pure-python scoring, ties broken by chunk_id.

No model, no network, no gold answers anywhere in this module.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.retrieval.chunker import lexical_tokens

ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "data/processed/rag_chunks"

# public API accepts either plural or singular spelling for statements
SOURCE_ALIASES = {
    "news": "news",
    "financial_statement": "financial_statement",
    "financial_statements": "financial_statement",
}


def normalize_source(source_type: str) -> str:
    try:
        return SOURCE_ALIASES[source_type]
    except KeyError:
        raise ValueError(f"unknown source_type {source_type!r}") from None


@dataclass
class RetrievedChunk:
    rank: int                 # 1-based
    chunk_id: str
    score: float
    text: str                 # verbatim original chunk text
    metadata: dict            # full chunk record (incl. periods, section, spans)


class ChunkStore:
    """Loads a dataset's chunk JSONL files and groups them per task/source."""

    def __init__(self, dataset: str, chunk_dir: Path = CHUNK_DIR):
        self.dataset = dataset
        self._groups: dict[tuple[str, str], list[dict]] = {}
        for src, fname in (("news", f"{dataset}.news.jsonl"),
                           ("financial_statement",
                            f"{dataset}.financial_statements.jsonl")):
            path = chunk_dir / fname
            with path.open(encoding="utf-8") as f:
                for line in f:
                    c = json.loads(line)
                    self._groups.setdefault((c["task_id"], src), []).append(c)
        for chunks in self._groups.values():
            chunks.sort(key=lambda c: c["chunk_id"])

    @property
    def task_ids(self) -> list[str]:
        return sorted({t for t, _ in self._groups})

    def chunks(self, task_id: str, source_type: str) -> list[dict]:
        key = (task_id, normalize_source(source_type))
        if key not in self._groups:
            raise KeyError(f"no chunks for {key}")
        return self._groups[key]


@dataclass
class _BM25Index:
    """Okapi BM25 with the standard smoothed idf: log(1 + (N-df+0.5)/(df+0.5))."""
    chunks: list[dict]
    k1: float = 1.5
    b: float = 0.75
    _tf: list[Counter] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _dl: list[int] = field(default_factory=list)
    _avgdl: float = 0.0

    def __post_init__(self):
        for c in self.chunks:
            toks = lexical_tokens(c["search_text"])
            tf = Counter(toks)
            self._tf.append(tf)
            self._dl.append(len(toks))
            for term in tf:
                self._df[term] += 1
        n = max(1, len(self.chunks))
        self._avgdl = sum(self._dl) / n if self._dl else 1.0

    def score(self, query_tokens: list[str]) -> list[float]:
        n_docs = len(self.chunks)
        out = []
        q_terms = Counter(query_tokens)
        for tf, dl in zip(self._tf, self._dl):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * (dl / self._avgdl or 1.0))
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n_docs - self._df[term] + 0.5)
                               / (self._df[term] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: ChunkStore, k1: float = 1.5, b: float = 0.75):
        self.store = store
        self.k1, self.b = k1, b
        self._indexes: dict[tuple[str, str], _BM25Index] = {}

    def _index(self, task_id: str, source: str) -> _BM25Index:
        key = (task_id, source)
        if key not in self._indexes:
            self._indexes[key] = _BM25Index(self.store.chunks(task_id, source),
                                            self.k1, self.b)
        return self._indexes[key]

    def retrieve(self, task_id: str, question: str, source_type: str,
                 top_k: int) -> list[RetrievedChunk]:
        source = normalize_source(source_type)
        idx = self._index(task_id, source)
        q_tokens = lexical_tokens(question)
        scores = idx.score(q_tokens)
        order = sorted(range(len(scores)),
                       key=lambda i: (-scores[i], idx.chunks[i]["chunk_id"]))
        return [RetrievedChunk(rank=r + 1,
                               chunk_id=idx.chunks[i]["chunk_id"],
                               score=scores[i],
                               text=idx.chunks[i]["text"],
                               metadata=idx.chunks[i])
                for r, i in enumerate(order[:top_k])]
