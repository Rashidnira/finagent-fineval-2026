"""Stage 3 of the best system: metric-aligned rewrite of System D answers
with the SAME pinned model (Qwen3-32B, self-hosted).

Usage:
    python scripts/rewrite_en.py [--provider local_qwen3_32b] [--dataset expert_test]

Reads  outputs/predictions/systemD_<provider>/<dataset>.predictions.jsonl
Writes outputs/predictions/systemD_<provider>/<dataset>.rewritten_en.jsonl

Three mechanical fixes, each motivated by verified behaviour of the official
scorer (no reference answers are used anywhere):
1) English only  - the scorer tokeniser keeps only [a-z0-9], so Chinese,
   Japanese and Greek text yields zero tokens; foreign quotes are rendered
   in English.
2) Compact money format ($27.2B / $416M) matching financial prose - only
   when the magnitude is unambiguous in the answer itself.
3) 55-65 word target (references are ~50 words; F1 penalises long answers).

Every call goes through src.generator.Generator, so responses are cached
under cache/llm/ and the run is byte-reproducible from a warm cache.
The prompt version and configuration tag are recorded with each generated response.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator import Generator  # noqa: E402

INSTR = """You rewrite financial-QA answers to a strict house style.

Rewrite the answer below. Rules:
- Target 55-65 words total, NEVER more than 75.
- ENGLISH ONLY. If the answer contains any Chinese, Japanese, Greek, or Spanish text (including quotes), replace it with a faithful English rendering of the same information. No non-Latin characters may remain.
- Keep the exact labeled structure of the original (e.g. "Answer: ..." and the "... Evidence: ..." line with its exact label).
- Write monetary amounts compactly: $27.2B, $416M, $3.5M — but ONLY when the magnitude is clear from the original wording (e.g. "US$14,014 million" -> "$14.0B"). If the scale of a raw number is not explicit, keep it unchanged. Never alter what a number means. Keep percentages as printed (e.g. 18.6%).
- Keep every distinct fact. Reuse the question's key terms in the wording.
- Remove filler, hedging, repetition, and meta-commentary.
- Output ONLY the rewritten answer text, nothing else."""

PROMPT_VERSION = CONFIG_TAG = "rewrite_en60_v1"


def nonlatin(s: str) -> int:
    """Count characters outside the Latin/ASCII range (rough diagnostic;
    curly quotes also count, so a small non-zero value is harmless)."""
    return sum(1 for ch in s if ord(ch) > 0x2000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="local_qwen3_32b")
    ap.add_argument("--dataset", default="expert_test",
                    choices=["expert_test", "easy_train"])
    args = ap.parse_args()

    src = ROOT / f"outputs/predictions/systemD_{args.provider}/{args.dataset}.predictions.jsonl"
    dst = src.with_name(f"{args.dataset}.rewritten_en.jsonl")
    if not src.exists():
        sys.exit(f"missing {src}; run stage 2 (src.retrieval.run_system_d) first")

    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    gen = Generator(provider=args.provider)

    with dst.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            rec = gen.generate(
                experiment_id=f"systemD_rewrite_en_{args.provider}",
                system=INSTR,
                user=f"Question: {r['question']}\n\nOriginal answer:\n{r['prediction']}",
                prompt_version=PROMPT_VERSION,
                task_id=r["task_id"],
                question=r["question"],
                config_tag=CONFIG_TAG,
            )
            pred = rec["raw_response"].strip()
            wc = len(pred.split())
            f.write(json.dumps({"task_id": r["task_id"], "question": r["question"],
                                "family": r.get("family"), "prediction": pred,
                                "word_count": wc, "nonlatin_chars": nonlatin(pred)},
                               ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {r['task_id']} "
                  f"{len(r['prediction'].split())}w -> {wc}w"
                  + (" (cache)" if rec["from_cache"] else ""), flush=True)

    wcs = [len(json.loads(l)["prediction"].split()) for l in dst.open(encoding="utf-8")]
    over = sum(w > 100 for w in wcs)
    print(f"done -> {dst}")
    print(f"mean={sum(wcs)/len(wcs):.1f}w max={max(wcs)} over_100={over}")
    if over:
        print("WARNING: some answers exceed 100 words; run "
              "scripts/fix_overlength.py before building the submission.")


if __name__ == "__main__":
    main()
