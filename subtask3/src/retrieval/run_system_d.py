"""Run System D prompts through a local provider (default: local_qwen3_32b).

Run: python -m src.retrieval.run_system_d --dataset easy_train [--limit N] [--dry-run]

Feeds the pre-built evidence-highlighted prompts
(data/processed/system_d_prompts/<dataset>.system_d.jsonl) through the
System C cached Generator front-end (imported READ-ONLY — same canonical
cache rules, separate fingerprint/namespace, so System C caches can never
be reused or overwritten). The prompt row already contains the full
original context + highlighted evidence + Question; the system message is
empty.

Safety: intended for the self-hosted local GGUF endpoint (LOCAL_OLLAMA_BASE,
no API key). --dry-run prints what would run without any network call.
Answers are validated with the System C answer_validator (read-only) against
the row's ORIGINAL query context, so evidence-block text cannot satisfy
quote-grounding by itself (it is a verbatim subset of the context anyway).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.answer_validator import validate      # read-only System C imports
from src.generator import Generator

PROMPT_DIR = ROOT / "data/processed/system_d_prompts"
OUT_ROOT = ROOT / "outputs/predictions"
RAW = {
    "easy_train": ROOT / "data/raw/public-00000-of-00001.parquet",
    "expert_test": ROOT / "data/raw/PolyFiQA_test_participant.parquet",
}
EVIDENCE_LABEL = {
    "easy_train": "News Evidence:",
    "expert_test": "Financial Statements Evidence:",
}


def load_rows(dataset: str) -> list[dict]:
    rows = []
    with (PROMPT_DIR / f"{dataset}.system_d.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(RAW), required=True)
    ap.add_argument("--provider", default="local_qwen3_32b")
    ap.add_argument("--config-tag", default="systemD_v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    # original contexts for quote-grounding validation
    ctx = dict(pd.read_parquet(RAW[args.dataset], columns=["task_id", "query"])
               .drop_duplicates("task_id").itertuples(index=False, name=None))

    if args.dry_run:
        print(f"dry-run: {len(rows)} prompts, provider={args.provider}, "
              f"tag={args.config_tag}, mean_chars="
              f"{round(sum(len(r['prompt']) for r in rows)/len(rows))}")
        return

    gen = Generator(provider=args.provider)
    out_dir = OUT_ROOT / f"systemD_{args.provider}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.predictions.jsonl"

    n_cached = n_new = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            rec = gen.generate(
                experiment_id=f"systemD_{args.provider}_{args.dataset}",
                system="",
                user=r["prompt"],
                prompt_version="systemD_v1",
                task_id=r["task_id"],
                question=r["question"],
                config_tag=args.config_tag,
            )
            n_cached += rec["from_cache"]
            n_new += not rec["from_cache"]
            pred = rec["raw_response"].strip()
            val = validate(pred, ctx[r["task_id"]],
                           EVIDENCE_LABEL[args.dataset])
            f.write(json.dumps({
                "task_id": r["task_id"], "question": r["question"],
                "family": r["family"], "prediction": pred,
                "validation": val, "from_cache": rec["from_cache"],
                "retrieval": r["retrieval"],
            }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {r['task_id']} "
                  f"{'cache' if rec['from_cache'] else 'new'}", flush=True)

    print(f"done: {n_new} generated, {n_cached} from cache -> {out_path}")


if __name__ == "__main__":
    main()
