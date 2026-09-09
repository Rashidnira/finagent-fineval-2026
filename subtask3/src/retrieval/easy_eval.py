"""Easy-only retrieval evaluation harness for System D.

Run: python -m src.retrieval.easy_eval [--retrievers bm25 bge_m3 hybrid_rrf60 hybrid_w50 hybrid_w70]

Queries are built from `question` text ONLY (plus the task_id to select the
per-task index). Easy gold answers are read exclusively inside the metric
functions (src.retrieval.gold_eval). Expert answers do not exist locally in
the participant file and are never read. No generation API is called; the
only model use is LOCAL BGE-M3 embedding.

For every retriever and k ∈ {1,3,5,8}: gold-number coverage (statements),
news-evidence token recall + quote-hit rate (news, `None` rows excluded and
counted separately), and overall lexical support — overall and per Easy
question family (Revenue / BalanceSheet / CashFlow / RnD).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.question_router import route  # System C router, imported read-only
from src.retrieval.bm25 import BM25Retriever, ChunkStore
from src.retrieval.gold_eval import (
    chunk_number_cores, extract_gold_numbers, extract_quotes, number_covered,
    parse_gold_answer, quote_hit, token_set_recall,
)

KS = (1, 3, 5, 8)
FAMILIES = ("Revenue", "BalanceSheet", "CashFlow", "RnD")
EASY_PARQUET = ROOT / "data/raw/public-00000-of-00001.parquet"
OUT_DIR = ROOT / "experiments/rag_retrieval"


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def evaluate_row(retriever, task_id: str, question: str, answer: str,
                 ks=KS) -> dict:
    kmax = max(ks)
    news = retriever.retrieve(task_id, question, "news", kmax)
    fs = retriever.retrieve(task_id, question, "financial_statements", kmax)

    answer_sec, evidence_body, evidence_none = parse_gold_answer(answer)
    golds = extract_gold_numbers(answer_sec)
    quotes = [] if evidence_none else extract_quotes(evidence_body)

    per_k = {}
    for k in ks:
        news_texts = [h.text for h in news[:k]]
        fs_texts = [h.text for h in fs[:k]]
        cores = chunk_number_cores(fs_texts)
        covered = [g for g in golds if number_covered(g, cores)]
        per_k[k] = {
            "num_coverage": (len(covered) / len(golds)) if golds else None,
            "evidence_recall": (None if evidence_none
                                else token_set_recall(evidence_body, news_texts)),
            "quote_hit_rate": (None if (evidence_none or not quotes)
                               else sum(quote_hit(q, news_texts)
                                        for q in quotes) / len(quotes)),
            "overall_support": token_set_recall(answer, news_texts + fs_texts),
        }
    return {"evidence_none": evidence_none, "n_gold_numbers": len(golds),
            "n_quotes": len(quotes), "per_k": per_k,
            "news_hits": news, "fs_hits": fs, "golds": golds, "quotes": quotes,
            "evidence_body": evidence_body}


def evaluate_retriever(retriever, df: pd.DataFrame) -> tuple[dict, list[dict]]:
    rows = []
    for _, r in df.iterrows():
        res = evaluate_row(retriever, r["task_id"], r["question"], r["answer"])
        res["task_id"], res["family"] = r["task_id"], route(r["question"])
        res["question"] = r["question"]
        rows.append(res)

    agg: dict = {}
    for k in KS:
        agg[k] = {"overall": _agg(rows, k), "by_family": {
            fam: _agg([x for x in rows if x["family"] == fam], k)
            for fam in FAMILIES}}
    return agg, rows


def _agg(rows: list[dict], k: int) -> dict:
    ev_rows = [r for r in rows if not r["evidence_none"]]
    return {
        "n": len(rows),
        "n_evidence": len(ev_rows),
        "n_none": len(rows) - len(ev_rows),
        "num_coverage": _mean(r["per_k"][k]["num_coverage"] for r in rows),
        "evidence_recall": _mean(r["per_k"][k]["evidence_recall"] for r in ev_rows),
        "quote_hit_rate": _mean(r["per_k"][k]["quote_hit_rate"] for r in ev_rows),
        "overall_support": _mean(r["per_k"][k]["overall_support"] for r in rows),
    }


def composite_at5(agg: dict) -> float:
    """Deterministic model-selection criterion: mean of the four k=5 overall
    metrics (missing metrics count as 0)."""
    m = agg[5]["overall"]
    parts = [m["num_coverage"], m["evidence_recall"], m["quote_hit_rate"],
             m["overall_support"]]
    return round(sum(p or 0.0 for p in parts) / len(parts), 4)


def build_retrievers(names: list[str], store: ChunkStore) -> dict:
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.semantic import SemanticRetriever
    bm25 = BM25Retriever(store)
    out = {}
    dense = None
    for n in names:
        if n == "bm25":
            out[n] = bm25
        else:
            if dense is None:
                dense = SemanticRetriever(store)
            if n == "bge_m3":
                out[n] = dense
            elif n.startswith("hybrid_rrf"):
                out[n] = HybridRetriever(bm25, dense, "rrf",
                                         rrf_k=int(n.removeprefix("hybrid_rrf")))
            elif n.startswith("hybrid_w"):
                out[n] = HybridRetriever(bm25, dense, "weighted",
                                         alpha=int(n.removeprefix("hybrid_w")) / 100)
            else:
                raise ValueError(n)
    return out


def inspection_examples(rows: list[dict], k: int = 5) -> list[dict]:
    """First question of each family in dataset order — never hand-picked."""
    out = []
    for fam in FAMILIES:
        r = next(x for x in rows if x["family"] == fam)
        cores = chunk_number_cores([h.text for h in r["fs_hits"][:k]])
        news_texts = [h.text for h in r["news_hits"][:k]]
        out.append({
            "family": fam, "task_id": r["task_id"], "question": r["question"],
            "news_top5": [_hit_view(h) for h in r["news_hits"][:k]],
            "fs_top5": [_hit_view(h) for h in r["fs_hits"][:k]],
            "gold_numbers_covered": [g.raw for g in r["golds"]
                                     if number_covered(g, cores)],
            "gold_numbers_missed": [g.raw for g in r["golds"]
                                    if not number_covered(g, cores)],
            "evidence_none": r["evidence_none"],
            "quotes_hit": [q[:60] for q in r["quotes"]
                           if quote_hit(q, news_texts)],
            "quotes_missed": [q[:60] for q in r["quotes"]
                              if not quote_hit(q, news_texts)],
        })
    return out


def _hit_view(h) -> dict:
    md = h.metadata
    v = {"rank": h.rank, "chunk_id": h.chunk_id, "score": round(h.score, 4),
         "language": md["language"], "text_snippet": h.text[:160]}
    if md["source_type"] == "financial_statement":
        v |= {"statement_type": md["statement_type"],
              "section": md["section_label"], "periods": md["periods"]}
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", nargs="+",
                    default=["bm25", "bge_m3", "hybrid_rrf60",
                             "hybrid_w50", "hybrid_w70"])
    args = ap.parse_args()

    # Easy gold answers loaded ONLY for metric computation below.
    df = pd.read_parquet(EASY_PARQUET, columns=["task_id", "question", "answer"])
    store = ChunkStore("easy_train")
    retrievers = build_retrievers(args.retrievers, store)

    results, per_row = {}, {}
    for name, retr in retrievers.items():
        print(f"evaluating {name} ...", flush=True)
        agg, rows = evaluate_retriever(retr, df)
        results[name] = agg
        per_row[name] = rows

    ranking = sorted(results, key=lambda n: (-composite_at5(results[n]), n))
    best = ranking[0]

    payload = {
        "ks": list(KS),
        "composite_at5": {n: composite_at5(results[n]) for n in results},
        "best": best,
        "results": {n: {str(k): v for k, v in agg.items()}
                    for n, agg in results.items()},
        "inspection_best": inspection_examples(per_row[best]),
        "inspection_bm25": inspection_examples(per_row["bm25"])
        if "bm25" in per_row else None,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "easy_eval.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")

    lines = ["# EASY RETRIEVAL EVALUATION (System D)", "",
             f"Retrievers: {', '.join(results)}; best by composite@5: **{best}**",
             "", "| retriever | composite@5 |", "|---|---|"]
    lines += [f"| {n} | {composite_at5(results[n])} |" for n in ranking]
    for n in ranking:
        lines += ["", f"## {n}", "",
                  "| k | num_cov | evid_recall | quote_hit | support |",
                  "|---|---|---|---|---|"]
        for k in KS:
            m = results[n][k]["overall"]
            lines.append(f"| {k} | {m['num_coverage']} | {m['evidence_recall']} "
                         f"| {m['quote_hit_rate']} | {m['overall_support']} |")
        lines += ["", "By family @5:",
                  "| family | num_cov | evid_recall | quote_hit | support | none |",
                  "|---|---|---|---|---|---|"]
        for fam in FAMILIES:
            m = results[n][5]["by_family"][fam]
            lines.append(f"| {fam} | {m['num_coverage']} | {m['evidence_recall']} "
                         f"| {m['quote_hit_rate']} | {m['overall_support']} "
                         f"| {m['n_none']}/{m['n']} |")
    (ROOT / "reports/RAG_EASY_RETRIEVAL.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT_DIR / 'easy_eval.json'} and reports/RAG_EASY_RETRIEVAL.md")


if __name__ == "__main__":
    main()
