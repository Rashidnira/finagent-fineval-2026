"""Build System D prompts: full original context + highlighted evidence.

Run: python -m src.retrieval.build_system_d [--retriever <name>] [--k 5]

For every row of both datasets, retrieves top-k news and top-k statement
chunks for the row's QUESTION (never any answer; the participant test file
has no answer column), and inserts a "Highlighted Evidence" block into the
row's own query string immediately before its trailing `Question:` line.

The original context is preserved verbatim: prompt == query[:ins] + block +
query[ins:], verified per row. System C prompts/caches/outputs are not
touched — output goes to data/processed/system_d_prompts/. No generation
API is called.

Highlight rendering: news chunks verbatim; statement chunks as a compact
row view (non-empty cells joined with " | ", case and every figure/sign/$
preserved) since the full padded table remains authoritative in the context
above the block.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.question_router import route  # read-only import from System C
from src.retrieval.bm25 import ChunkStore
from src.retrieval.context_parser import QUESTION_RE
from src.retrieval.easy_eval import build_retrievers

OUT_DIR = ROOT / "data/processed/system_d_prompts"
DATASETS = {
    "easy_train": ROOT / "data/raw/public-00000-of-00001.parquet",
    "expert_test": ROOT / "data/raw/PolyFiQA_test_participant.parquet",
}

BLOCK_HEADER = (
    "Highlighted Evidence (automatically retrieved, verbatim excerpts from "
    "the context above; check these first, but the full context above "
    "remains authoritative):\n")


def compact_rows(chunk_text: str) -> str:
    """Case- and figure-preserving compact view of statement chunk rows."""
    rows = []
    for line in chunk_text.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set(":- ") for c in cells):
            continue
        compact = " | ".join(c for c in cells if c)
        if compact:
            rows.append(compact)
    return "\n".join(rows)


def evidence_block(news_hits, fs_hits) -> str:
    parts = [BLOCK_HEADER]
    parts.append("[Financial statement excerpts]\n")
    for h in fs_hits:
        md = h.metadata
        label = md["statement_type"] or "statement"
        periods = ", ".join(md["periods"]) if md["periods"] else "n/a"
        section = f", section: {md['section_label']}" if md["section_label"] else ""
        parts.append(f"({label}, periods: {periods}{section})\n"
                     f"{compact_rows(h.text)}\n")
    parts.append("[News excerpts]\n")
    for h in news_hits:
        parts.append(f"({h.metadata['language_name']} news)\n{h.text}\n")
    return "\n".join(parts) + "\n"


def build_dataset(name: str, path: Path, retriever, k: int) -> list[dict]:
    df = pd.read_parquet(path, columns=["task_id", "question", "query"])
    rows = []
    for _, r in df.iterrows():
        task_id, question, q = r["task_id"], r["question"], r["query"]
        news = retriever.retrieve(task_id, question, "news", k)
        fs = retriever.retrieve(task_id, question, "financial_statements", k)
        block = evidence_block(news, fs)

        matches = list(QUESTION_RE.finditer(q))
        ins = matches[-1].start()             # before the trailing Question: line
        prompt = q[:ins] + block + q[ins:]
        assert prompt[:ins] + prompt[ins + len(block):] == q, \
            f"{task_id}: original context not preserved"

        rows.append({
            "task_id": task_id, "question": question,
            "family": route(question), "prompt": prompt,
            "insertion_offset": ins, "block_len": len(block),
            "retrieval": {
                "retriever": retriever.name, "k": k,
                "news": [{"chunk_id": h.chunk_id, "score": round(h.score, 6)}
                         for h in news],
                "financial_statements": [
                    {"chunk_id": h.chunk_id, "score": round(h.score, 6)}
                    for h in fs],
            },
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default=None,
                    help="default: 'best' from experiments/rag_retrieval/easy_eval.json")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    name = args.retriever
    if name is None:
        name = json.loads((ROOT / "experiments/rag_retrieval/easy_eval.json")
                          .read_text(encoding="utf-8"))["best"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"retriever": name, "k": args.k, "datasets": {}}
    for ds, path in DATASETS.items():
        store = ChunkStore(ds)
        retriever = build_retrievers([name], store)[name]
        rows = build_dataset(ds, path, retriever, args.k)
        out = OUT_DIR / f"{ds}.system_d.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["datasets"][ds] = {
            "rows": len(rows), "file": out.name,
            "mean_prompt_chars": round(sum(len(r["prompt"]) for r in rows)
                                       / len(rows)),
            "mean_block_chars": round(sum(r["block_len"] for r in rows)
                                      / len(rows)),
        }
        print(ds, manifest["datasets"][ds])
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
