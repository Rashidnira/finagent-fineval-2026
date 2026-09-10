"""Length-compression pass over System D predictions (same pinned model) --
the 55-word-target variant used for submission Dc2 (public F1 0.3022).

Usage: python scripts/compress_d.py expert_test [--provider local_qwen3_32b]

Reads outputs/predictions/systemD_<provider>/<dataset>.predictions.jsonl,
rewrites each answer to <=55 words with the same pinned model, preserving
structure/figures/quotes. Writes <dataset>.compressed.jsonl next to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import Generator  # noqa: E402

INSTR = """You rewrite financial-QA answers to be shorter without losing substance.

Rewrite the answer below to AT MOST 55 words total. Rules:
- Keep the exact labeled structure of the original (e.g. "Answer: ..." and the "... Evidence: ..." line with its exact label).
- Keep every key figure exactly as written: numbers, units, currencies, percentages, periods.
- Keep quoted evidence verbatim; you may drop a less-relevant quote entirely to save words, but NEVER alter, translate, or paraphrase inside quotation marks.
- Reuse the question's key terms in the answer wording.
- Remove filler, hedging, repetition, and meta-commentary.
- Output ONLY the rewritten answer text, nothing else."""

PROMPT_VERSION = CONFIG_TAG = "compress_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["easy_train", "expert_test"])
    ap.add_argument("--provider", default="local_qwen3_32b")
    args = ap.parse_args()

    src = ROOT / f"outputs/predictions/systemD_{args.provider}/{args.dataset}.predictions.jsonl"
    dst = src.with_name(f"{args.dataset}.compressed.jsonl")
    if not src.exists():
        sys.exit(f"missing {src}; run src.retrieval.run_system_d first")

    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    gen = Generator(provider=args.provider)

    over = 0
    with dst.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            rec = gen.generate(
                experiment_id=f"systemD_compress_{args.dataset}",
                system=INSTR,
                user=f"Question: {r['question']}\n\nOriginal answer:\n{r['prediction']}",
                prompt_version=PROMPT_VERSION,
                task_id=r["task_id"],
                question=r["question"],
                config_tag=CONFIG_TAG,
            )
            pred = rec["raw_response"].strip()
            wc = len(pred.split())
            over += wc > 100
            f.write(json.dumps({"task_id": r["task_id"], "question": r["question"],
                                "family": r.get("family"), "prediction": pred,
                                "word_count": wc}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {r['task_id']} {len(r['prediction'].split())}w -> {wc}w"
                  + (" (cache)" if rec["from_cache"] else ""), flush=True)

    print(f"done -> {dst}; rows over 100 words after compression: {over}")


if __name__ == "__main__":
    main()
