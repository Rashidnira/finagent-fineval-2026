"""Run one cell of the retrieval x granularity grid at one seed.

    python scripts/run_grid.py --cell gr_retr --seed 1
    python scripts/run_grid.py --cell pq_noretr --seed 1 --limit 2   # smoke test

Cells are built by src.retrieval.build_grid and share a byte-identical raw
context; they differ only in generation granularity and whether the retrieved
evidence block is present.

Decoding is the submitted system's sampling (temperature 0.7, top-p 0.8,
top-k 20) with an explicit seed, so a cell is reproducible and several seeds
give a spread. The seed and context length enter the provider fingerprint, so
these runs get their own cache keys and cannot collide with the runs behind
the submitted results.

No rewriting, length guard or language guard is applied: the grid measures
generation, not post-processing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.answer_validator import validate                        # noqa: E402
from src.generator import Generator                              # noqa: E402
from src.grouped_runner import parse_grouped                     # noqa: E402
from src.retrieval.build_grid import CELLS, OUT_DIR              # noqa: E402
from src.retrieval.run_system_d import EVIDENCE_LABEL, RAW       # noqa: E402

PRED_DIR = ROOT / "outputs/predictions/grid"
NUM_CTX = 40960          # largest grouped+retrieval prompt is ~32.7k tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", default="easy_train")
    ap.add_argument("--provider", default="local_qwen3_32b")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = OUT_DIR / f"{args.dataset}.{args.cell}.jsonl"
    if not path.exists():
        sys.exit(f"missing {path}; run: python -m src.retrieval.build_grid "
                 f"--dataset {args.dataset}")
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    grouped = args.cell.startswith("gr_")
    ctx = dict(pd.read_parquet(RAW[args.dataset], columns=["task_id", "query"])
               .drop_duplicates("task_id").itertuples(index=False, name=None))

    if args.dry_run:
        mean = sum(len(r["prompt"]) for r in rows) / len(rows)
        print(f"dry-run: cell={args.cell} seed={args.seed} prompts={len(rows)} "
              f"grouped={grouped} mean_chars={mean:.0f} num_ctx={args.num_ctx}")
        return

    gen = Generator(provider=args.provider, seed=args.seed, num_ctx=args.num_ctx)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRED_DIR / f"{args.dataset}.{args.cell}.seed{args.seed}.jsonl"

    n_new = n_cached = n_unparsed = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            rec = gen.generate(
                experiment_id=f"grid_{args.cell}_seed{args.seed}_{args.dataset}",
                system="",
                user=r["prompt"],
                prompt_version=f"grid_{args.cell}_v1",
                task_id=r["task_id"],
                question=r["question"],
                config_tag=f"grid_{args.cell}_seed{args.seed}",
            )
            n_cached += rec["from_cache"]
            n_new += not rec["from_cache"]
            raw = rec["raw_response"].strip()

            if grouped:
                try:
                    answers = parse_grouped(raw, r["uids"])
                except ValueError as e:
                    n_unparsed += 1
                    print(f"[{i}/{len(rows)}] {r['task_id']} UNPARSED: {e}", flush=True)
                    answers = {u: "" for u in r["uids"]}
                items = [(q, fam, answers[u])
                         for q, fam, u in zip(r["questions"], r["families"], r["uids"])]
            else:
                items = [(r["question"], r["family"], raw)]

            for question, family, pred in items:
                f.write(json.dumps({
                    "task_id": r["task_id"], "question": question, "family": family,
                    "prediction": pred,
                    "validation": validate(pred, ctx[r["task_id"]],
                                           EVIDENCE_LABEL[args.dataset]),
                    "cell": args.cell, "seed": args.seed,
                    "from_cache": rec["from_cache"],
                }, ensure_ascii=False) + "\n")

            print(f"[{i}/{len(rows)}] {r['task_id']} "
                  f"{'cache' if rec['from_cache'] else 'new'}", flush=True)

    n_answers = sum(1 for _ in out_path.open(encoding="utf-8"))
    print(f"done: {n_new} generated, {n_cached} cached, {n_unparsed} unparsed "
          f"-> {n_answers} answers in {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
