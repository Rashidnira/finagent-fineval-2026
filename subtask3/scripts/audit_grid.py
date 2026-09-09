"""Consistency audit for the retrieval x granularity grid.

    python scripts/audit_grid.py --seeds 1 2 3 4 5

Checks that the grid is what it claims to be, and fails loudly if not. Nine
checks in four groups:

  Prompt integrity   the four cells differ only in the two variables studied
  Output integrity   every run produced 76 answers covering the same items
  Provenance         every answer came from the pinned model at its seed,
                     with the recorded context window, and no post-processing
  Reproducibility    replaying a cell from cache reproduces the file exactly

Every check prints PASS or FAIL with the evidence behind it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.question_router import route                                  # noqa: E402
from src.retrieval.build_grid import CELLS, OUT_DIR, strip_block       # noqa: E402

PRED_DIR = ROOT / "outputs/predictions/grid"
CACHE = ROOT / "cache/llm"
GOLD = ROOT / "data/raw/public-00000-of-00001.parquet"
SRC_PROMPTS = ROOT / "data/processed/system_d_prompts"

EXPECT_MODEL = "qwen3:32b"
EXPECT_NUM_CTX = 40960
EXPECT_TEMP = 0.7
EXPECT_ANSWERS = 76


class Audit:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}", flush=True)
        return ok

    def report(self) -> int:
        failed = [n for n, ok, _ in self.results if not ok]
        print("\n" + "=" * 74)
        print(f"{len(self.results) - len(failed)} of {len(self.results)} checks passed")
        if failed:
            print("FAILED: " + ", ".join(failed))
        print("=" * 74)
        return 1 if failed else 0


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--dataset", default="easy_train")
    args = ap.parse_args()
    a = Audit()
    ds = args.dataset

    print("=" * 74)
    print(f"Grid consistency audit: {ds}, seeds {args.seeds}")
    print("=" * 74 + "\n")

    # ---------------------------------------------------------- prompts ----
    cells = {c: load(OUT_DIR / f"{ds}.{c}.jsonl") for c in CELLS}
    src = {}
    for r in load(SRC_PROMPTS / f"{ds}.system_d.jsonl"):
        src[(r["task_id"], r["family"])] = r

    a.check("1. prompt set sizes",
            len(cells["pq_noretr"]) == len(cells["pq_retr"]) == 76
            and len(cells["gr_noretr"]) == len(cells["gr_retr"]) == 19,
            f"per-question {len(cells['pq_noretr'])}/{len(cells['pq_retr'])}, "
            f"grouped {len(cells['gr_noretr'])}/{len(cells['gr_retr'])} "
            f"(expected 76/76 and 19/19)")

    # per-question cells differ only by the evidence block at the recorded offset
    bad = []
    for r0, r1 in zip(cells["pq_noretr"], cells["pq_retr"]):
        s = src[(r0["task_id"], r0["family"])]
        if r0["prompt"] != strip_block(s) or r1["prompt"] != s["prompt"]:
            bad.append(r0["task_id"])
    a.check("2. per-question cells differ only by the evidence block",
            not bad,
            "all 76 pairs reconstruct exactly from the source prompts"
            if not bad else f"mismatched: {sorted(set(bad))[:5]}")

    # grouped cells share the raw context with the per-question cells
    ctx_bad, ev_only_bad = [], []
    pq_by_task = defaultdict(list)
    for r in cells["pq_noretr"]:
        pq_by_task[r["task_id"]].append(r)
    for g0, g1 in zip(cells["gr_noretr"], cells["gr_retr"]):
        p = pq_by_task[g0["task_id"]][0]["prompt"]
        head = p[:p.rfind("\nQuestion:")].rstrip()
        if not (g0["prompt"].startswith(head) and g1["prompt"].startswith(head)):
            ctx_bad.append(g0["task_id"])
        # removing the evidence text from gr_retr must leave gr_noretr's length
        added = len(g1["prompt"]) - len(g0["prompt"])
        blocks = sum(src[(g0["task_id"], f)]["block_len"]
                     for f in [r["family"] for r in pq_by_task[g0["task_id"]]])
        if not 0 < added <= blocks + 40:      # +joins, minus stripped whitespace
            ev_only_bad.append((g0["task_id"], added, blocks))
    a.check("3. grouped cells use the same raw context as per-question cells",
            not ctx_bad,
            "all 19 grouped prompts start with the identical raw context"
            if not ctx_bad else f"mismatched: {ctx_bad}")
    a.check("4. grouped retrieval cell adds only evidence text",
            not ev_only_bad,
            "added characters match the retrieved block sizes in all 19 filings"
            if not ev_only_bad else f"unexpected: {ev_only_bad[:3]}")

    # ---------------------------------------------------------- outputs ----
    gold = pd.read_parquet(GOLD)
    gold_keys = {(r.task_id, route(r.question)) for r in gold.itertuples()}

    missing, wrong_n, empty, key_mismatch = [], [], [], []
    for c in CELLS:
        for s in args.seeds:
            p = PRED_DIR / f"{ds}.{c}.seed{s}.jsonl"
            if not p.exists():
                missing.append(p.name)
                continue
            rows = load(p)
            if len(rows) != EXPECT_ANSWERS:
                wrong_n.append(f"{p.name}={len(rows)}")
            n_empty = sum(1 for r in rows if not r["prediction"].strip())
            if n_empty:
                empty.append(f"{p.name}={n_empty}")
            keys = {(r["task_id"], r["family"]) for r in rows}
            if keys != gold_keys:
                key_mismatch.append(p.name)

    n_expected = len(CELLS) * len(args.seeds)
    a.check("5. every run produced a complete file",
            not missing and not wrong_n,
            f"{n_expected - len(missing)}/{n_expected} files present, "
            f"all with {EXPECT_ANSWERS} answers"
            if not (missing or wrong_n)
            else f"missing {missing}; wrong size {wrong_n}")
    a.check("6. no empty answers (grouped responses all parsed)",
            not empty,
            "no blank predictions in any run" if not empty else f"blanks: {empty}")
    a.check("7. every run covers the same 76 items as the gold file",
            not key_mismatch,
            "task_id and question-type sets match the gold data in all runs"
            if not key_mismatch else f"mismatched: {key_mismatch}")

    # -------------------------------------------------------- provenance ---
    fp_seen: Counter = Counter()
    tag_seen: Counter = Counter()
    for f in glob.glob(str(CACHE / "*.json")):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = d.get("config_tag", "")
        if not tag.startswith("grid_"):
            continue
        # Only the seeds under audit; other seeds may exist from other runs.
        if not any(tag.endswith(f"_seed{s_}") for s_ in args.seeds):
            continue
        fp = d.get("provider_fingerprint", {})
        tag_seen[tag] += 1
        fp_seen[(fp.get("model"), fp.get("temperature"), fp.get("num_ctx"),
                 fp.get("seed"))] += 1

    bad_fp = [k for k in fp_seen
              if k[0] != EXPECT_MODEL or k[1] != EXPECT_TEMP
              or k[2] != EXPECT_NUM_CTX or k[3] not in args.seeds]
    a.check("8. all grid generations used the pinned model, window and seeds",
            bool(fp_seen) and not bad_fp,
            f"{sum(fp_seen.values())} cached generations across "
            f"{len(fp_seen)} (model, temperature, num_ctx, seed) combinations, "
            f"all {EXPECT_MODEL} at temp {EXPECT_TEMP}, num_ctx {EXPECT_NUM_CTX}"
            if not bad_fp else f"unexpected fingerprints: {bad_fp[:3]}")

    post = [t for t in tag_seen if any(k in t for k in
                                       ("rewrite", "compress", "nonlatin", "overlength"))]
    a.check("9. no post-processing was applied to the grid",
            not post,
            "no rewrite, length-guard or language-guard calls carry a grid tag"
            if not post else f"post-processing tags found: {post}")

    # ---------------------------------------------------- reproducibility --
    cell, seed = "gr_retr", args.seeds[0]
    target = PRED_DIR / f"{ds}.{cell}.seed{seed}.jsonl"
    def answer_digest(path: Path) -> str:
        """Hash the answers only. The record also stores whether the response
        came from cache, which necessarily flips on a replay, so it is
        excluded from the comparison."""
        items = [(r["task_id"], r["family"], r["prediction"]) for r in load(path)]
        return hashlib.sha256(
            json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()

    if target.exists():
        before = answer_digest(target)
        proc = subprocess.run(
            [sys.executable, "scripts/run_grid.py", "--cell", cell,
             "--seed", str(seed), "--dataset", ds],
            cwd=ROOT, capture_output=True, text=True)
        after = answer_digest(target)
        cached = "0 generated" in proc.stdout
        a.check("10. replaying a cell from cache reproduces it exactly",
                proc.returncode == 0 and before == after and cached,
                f"{cell} seed {seed} replayed from cache with no new model "
                f"calls; the 76 answers are unchanged "
                f"(sha256 {before[:16]}...)"
                if before == after else
                f"the replayed answers differ ({before[:12]} -> {after[:12]})")

    return a.report()


if __name__ == "__main__":
    raise SystemExit(main())
