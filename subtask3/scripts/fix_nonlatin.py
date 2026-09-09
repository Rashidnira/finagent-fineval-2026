"""Deterministic language guard: rows that still contain non-Latin
characters after the English rewrite get one stricter English-only pass
with the SAME pinned model. Usage:

    python scripts/fix_nonlatin.py <predictions.jsonl> [--provider local_qwen3_32b]

Rows without such characters are left untouched. For the best submission
(De2) this touched 14 of 76 rows (config tag "fix_nonlatin_v1"); 10 rows
still match the detector afterwards, all of them curly quotes and dashes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import Generator  # noqa: E402

INSTR = ("The answer below contains non-English text (Chinese, Japanese, "
         "Greek, or Spanish). Rewrite it in ENGLISH ONLY, at most 70 words: "
         "translate the non-English parts faithfully into English prose "
         "(no quotation marks around translations), keep the labeled "
         "structure (Answer: / ... Evidence: lines), keep all figures "
         "exactly. Not a single non-Latin character may remain. "
         "Output ONLY the rewritten answer.")


def nonlatin(s: str) -> int:
    return sum(1 for ch in s if ord(ch) > 0x2000)


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
        if nonlatin(r["prediction"]) > 0:
            rec = gen.generate(
                experiment_id="fix_nonlatin", system=INSTR,
                user=f"Original answer:\n{r['prediction']}",
                prompt_version="fix_nonlatin_v1", task_id=r["task_id"],
                question=r["question"], config_tag="fix_nonlatin_v1")
            r["prediction"] = rec["raw_response"].strip()
            r["word_count"] = len(r["prediction"].split())
            r["nonlatin_chars"] = nonlatin(r["prediction"])
            print(f"{r['task_id']}: -> {r['word_count']}w "
                  f"nonlatin={r['nonlatin_chars']}"
                  + (" (cache)" if rec["from_cache"] else ""))
            fixed += 1

    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    left = sum(1 for r in rows if nonlatin(r["prediction"]) > 0)
    over = sum(1 for r in rows if len(r["prediction"].split()) > 100)
    print(f"fixed {fixed}; rows still with non-Latin chars: {left}; over_100: {over}")
    if over:
        sys.exit("FATAL: an answer exceeds 100 words; rerun fix_overlength.py")


if __name__ == "__main__":
    main()
