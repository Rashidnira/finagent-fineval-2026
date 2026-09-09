"""Hybrid fusion of the BM25 and dense retrievers (deterministic).

Two fusion methods over the FULL per-(task, source) ranking of both
retrievers (index groups are small, ≤ ~40 chunks, so no pooling cutoff):

- rrf:      reciprocal-rank fusion, score = Σ 1/(rrf_k + rank)
- weighted: min-max normalize each retriever's scores within the group,
            then alpha·bm25 + (1-alpha)·dense

Ties always break on chunk_id.
"""
from __future__ import annotations

from src.retrieval.bm25 import RetrievedChunk

FULL = 10_000  # retrieve the whole group from both parents


class HybridRetriever:
    def __init__(self, bm25, dense, method: str = "rrf",
                 alpha: float = 0.5, rrf_k: int = 60, name: str | None = None):
        assert method in ("rrf", "weighted")
        self.bm25, self.dense = bm25, dense
        self.method, self.alpha, self.rrf_k = method, alpha, rrf_k
        self.name = name or (f"hybrid_rrf{rrf_k}" if method == "rrf"
                             else f"hybrid_w{int(alpha * 100)}")

    @staticmethod
    def _minmax(hits) -> dict[str, float]:
        if not hits:
            return {}
        vals = [h.score for h in hits]
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return {h.chunk_id: 0.0 for h in hits}
        return {h.chunk_id: (h.score - lo) / (hi - lo) for h in hits}

    def retrieve(self, task_id: str, question: str, source_type: str,
                 top_k: int) -> list[RetrievedChunk]:
        a = self.bm25.retrieve(task_id, question, source_type, FULL)
        b = self.dense.retrieve(task_id, question, source_type, FULL)
        meta = {h.chunk_id: h for h in a}
        meta.update({h.chunk_id: h for h in b})

        fused: dict[str, float] = {}
        if self.method == "rrf":
            for hits in (a, b):
                for h in hits:
                    fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) \
                        + 1.0 / (self.rrf_k + h.rank)
        else:
            na, nb = self._minmax(a), self._minmax(b)
            for cid in meta:
                fused[cid] = self.alpha * na.get(cid, 0.0) \
                    + (1 - self.alpha) * nb.get(cid, 0.0)

        order = sorted(fused, key=lambda cid: (-fused[cid], cid))
        return [RetrievedChunk(rank=r + 1, chunk_id=cid, score=fused[cid],
                               text=meta[cid].text, metadata=meta[cid].metadata)
                for r, cid in enumerate(order[:top_k])]
