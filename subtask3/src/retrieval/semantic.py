"""Local dense retrieval (BGE-M3) for System D — offline, deterministic.

- The model runs LOCALLY on CPU via sentence-transformers; no generation
  API is ever called. Only embeddings are computed.
- Embeddings are cached on disk keyed by content sha1, so re-runs are
  byte-stable and the model is not needed once the cache is warm.
- `encode_fn` is injectable so tests run without the model.
- Embedded views: news chunks use the verbatim `text`; statement chunks use
  the compact `search_text` row rendering (markdown padding is noise for a
  sentence encoder). Originals are returned untouched.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.retrieval.bm25 import ChunkStore, RetrievedChunk, normalize_source

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data/processed/rag_embeddings/bge_m3_cache.npz"
MODEL_NAME = "BAAI/bge-m3"


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_model_encoder(model_name: str = MODEL_NAME, batch_size: int = 4):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)
    return encode


class SemanticRetriever:
    name = "bge_m3"

    def __init__(self, store: ChunkStore, encode_fn=None,
                 cache_path: Path | None = CACHE_PATH, model_name: str = MODEL_NAME):
        self.store = store
        self.model_name = model_name
        self._encode_fn = encode_fn          # lazy-load real model only if needed
        self.cache_path = cache_path
        self._cache: dict[str, np.ndarray] = {}
        if cache_path and cache_path.exists():
            with np.load(cache_path) as z:
                self._cache = {k: z[k] for k in z.files}
        self._matrices: dict[tuple[str, str], tuple[np.ndarray, list[dict]]] = {}

    # ------------------------------------------------------------ embedding

    def _encoder(self):
        if self._encode_fn is None:
            self._encode_fn = _load_model_encoder(self.model_name)
        return self._encode_fn

    @staticmethod
    def embed_view(chunk: dict) -> str:
        return chunk["text"] if chunk["source_type"] == "news" else chunk["search_text"]

    def _vectors_for(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if _key(t) not in self._cache]
        if missing:
            vecs = self._encoder()(missing)
            for t, v in zip(missing, vecs):
                self._cache[_key(t)] = v
            self._save_cache()
        return np.stack([self._cache[_key(t)] for t in texts])

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache_path, **self._cache)

    def warm(self) -> int:
        """Embed every chunk in the store; returns number embedded."""
        texts = []
        for task_id in self.store.task_ids:
            for src in ("news", "financial_statement"):
                texts += [self.embed_view(c) for c in self.store.chunks(task_id, src)]
        self._vectors_for(texts)
        return len(texts)

    # ------------------------------------------------------------ retrieval

    def _matrix(self, task_id: str, source: str):
        key = (task_id, source)
        if key not in self._matrices:
            chunks = self.store.chunks(task_id, source)
            mat = self._vectors_for([self.embed_view(c) for c in chunks])
            self._matrices[key] = (mat, chunks)
        return self._matrices[key]

    def retrieve(self, task_id: str, question: str, source_type: str,
                 top_k: int) -> list[RetrievedChunk]:
        source = normalize_source(source_type)
        mat, chunks = self._matrix(task_id, source)
        qv = self._vectors_for([question])[0]
        scores = mat @ qv                     # cosine (all vectors L2-normalized)
        order = sorted(range(len(chunks)),
                       key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]))
        return [RetrievedChunk(rank=r + 1, chunk_id=chunks[i]["chunk_id"],
                               score=float(scores[i]), text=chunks[i]["text"],
                               metadata=chunks[i])
                for r, i in enumerate(order[:top_k])]
