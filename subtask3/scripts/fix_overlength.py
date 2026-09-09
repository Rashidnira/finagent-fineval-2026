"""Deterministic length guard: re-shorten any answer still above 100 words
with the SAME pinned model. Usage:

    python scripts/fix_overlength.py <predictions.jsonl> [--provider local_qwen3_32b]

Rows at or below 100 words are left untouched. For the best submission
(De2) this step ran on the 7 of 76 rows that were still above 100 words
after scripts/rewrite_en.py, and brought the maximum down to 99 words.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import Generator  # noqa: E402

INSTR = ("Rewrite the answer below to AT MOST 45 words total. Keep the exact "
         "labeled structure (Answer: / ... Evidence: lines), keep every figure "
         "exactly as written, keep at most ONE short verbatim quote (drop the "
         "rest; never alter or translate inside quotation marks). "
         "Output ONLY the rewritten answer.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--provider", default="local_qwen3_32b")
    args = ap.parse_args()

    p = Path(args.jsonl)
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    gen = Generator(provider=args.provider)

    fixed = 0
    for r in rows:
        if len(r["prediction"].split()) > 100:
            rec = gen.generate(
                experiment_id="systemD_compress_strict", system=INSTR,
                user=f"Question: {r['question']}\n\nOriginal answer:\n{r['prediction']}",
                prompt_version="compress_v2_strict", task_id=r["task_id"],
                question=r["question"], config_tag="compress_v2_strict")
            r["prediction"] = rec["raw_response"].strip()
            r["word_count"] = len(r["prediction"].split())
            print(f"{r['task_id']}: -> {r['word_count']}w")
            fixed += 1

    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mx = max(len(r["prediction"].split()) for r in rows)
    print(f"fixed {fixed}; max words now {mx}")
    if mx > 100:
        sys.exit("FATAL: an answer still exceeds 100 words")


if __name__ == "__main__":
    main()
