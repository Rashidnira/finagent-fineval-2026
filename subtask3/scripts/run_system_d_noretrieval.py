"""System D ablation: per-question generation WITHOUT the retrieved evidence block.

Run: python scripts/run_system_d_noretrieval.py --dataset easy_train [--limit N] [--dry-run]

Purpose. The submitted results compare System C (grouped generation, no
retrieval) with System D (per-question generation + BGE-M3 evidence). That
single step changes two factors at once, so neither can be attributed the
gain on its own. This script supplies the missing cell of the 2x2: the same
per-question prompts as System D, with the evidence block removed.

The removal is exact and lossless. Each System D prompt row records the
`insertion_offset` and `block_len` at which the evidence block was spliced
into the original query, so deleting that span restores the prompt to
"original context + Question" byte-for-byte. Nothing else differs from
System D: same model, same rows, same validator, same output schema.

Cache/provenance safety: a distinct experiment_id namespace
(systemD_noretr_*) and prompt_version, so System C and System D caches can
never be read or overwritten by this run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.answer_validator import validate          # read-only, as System D
from src.generator import Generator
from src.retrieval.run_system_d import EVIDENCE_LABEL, PROMPT_DIR, RAW

OUT_ROOT = ROOT / "outputs/predictions"


def strip_evidence(row: dict) -> str:
    """Return the System D prompt with the evidence block removed."""
    p, off, ln = row["prompt"], row["insertion_offset"], row["block_len"]
    block = p[off:off + ln]
    if not block.startswith("Highlighted Evidence"):
        raise ValueError(f"{row['task_id']}: unexpected block at offset {off}")
    return p[:off] + p[off + ln:]


def load_rows(dataset: str) -> list[dict]:
    rows = []
    with (PROMPT_DIR / f"{dataset}.system_d.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(RAW), default="easy_train")
    ap.add_argument("--provider", default="local_qwen3_32b")
    ap.add_argument("--config-tag", default="systemD_noretr_v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    for r in rows:
        r["prompt_noretr"] = strip_evidence(r)

    ctx = dict(pd.read_parquet(RAW[args.dataset], columns=["task_id", "query"])
               .drop_duplicates("task_id").itertuples(index=False, name=None))

    if args.dry_run:
        full = sum(len(r["prompt"]) for r in rows) / len(rows)
        cut = sum(len(r["prompt_noretr"]) for r in rows) / len(rows)
        print(f"dry-run: {len(rows)} prompts, provider={args.provider}, "
              f"tag={args.config_tag}, mean_chars {round(full)} -> {round(cut)} "
              f"(evidence block removed)")
        return

    gen = Generator(provider=args.provider)
    out_dir = OUT_ROOT / f"systemD_noretr_{args.provider}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.predictions.jsonl"

    n_cached = n_new = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            rec = gen.generate(
                experiment_id=f"systemD_noretr_{args.provider}_{args.dataset}",
                system="",
                user=r["prompt_noretr"],
                prompt_version="systemD_noretr_v1",
                task_id=r["task_id"],
                question=r["question"],
                config_tag=args.config_tag,
            )
            n_cached += rec["from_cache"]
            n_new += not rec["from_cache"]
            pred = rec["raw_response"].strip()
            val = validate(pred, ctx[r["task_id"]], EVIDENCE_LABEL[args.dataset])
            f.write(json.dumps({
                "task_id": r["task_id"], "question": r["question"],
                "family": r["family"], "prediction": pred,
                "validation": val, "from_cache": rec["from_cache"],
                "retrieval": None,
            }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(rows)}] {r['task_id']} "
                  f"{'cache' if rec['from_cache'] else 'new'}", flush=True)

    print(f"done: {n_new} generated, {n_cached} from cache -> {out_path}")


if __name__ == "__main__":
    main()
