"""Score the retrieval x granularity grid and report mean and standard deviation.

    python scripts/score_grid.py --seeds 1 2 3 4 5

Scores every cell at every seed on the PolyFiQA-Easy development set with both
scorers, then reports per-cell mean and standard deviation across seeds, the
two main effects, and the interaction.

Effects are computed PAIRED: within each seed, the retrieval effect, the
generation-strategy effect and their interaction are computed from that seed's
four cells, and the mean and standard deviation are taken over those per-seed
differences. Pairing removes the part of the run-to-run variation that moves
all four cells together, so it is a tighter estimate than comparing the
standard deviations of the four cell scores separately.

Five seeds support description, not inference. The report therefore states
whether an estimated effect is larger or smaller than the variation across the
five runs, and never claims statistical significance or an absence of effect.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.question_router import route                              # noqa: E402
from src.retrieval.score_easy import score_file                    # noqa: E402

PRED_DIR = ROOT / "outputs/predictions/grid"
GOLD = ROOT / "data/raw/public-00000-of-00001.parquet"
CELLS = ["gr_noretr", "gr_retr", "pq_noretr", "pq_retr"]
LABEL = {"gr_noretr": "Grouped, no retrieval",
         "gr_retr": "Grouped, retrieval",
         "pq_noretr": "Per-question, no retrieval",
         "pq_retr": "Per-question, retrieval"}
SCORERS = [("multifinben_default", "default"), ("finmmeval_multiscript", "multiscript")]
FAMILIES = ["Revenue", "BalanceSheet", "CashFlow", "RnD"]


def msd(xs: list[float]) -> tuple[float, float]:
    return stats.mean(xs), (stats.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--dataset", default="easy_train")
    ap.add_argument("--out", default=str(ROOT / "reports/GRID_RESULTS.md"))
    args = ap.parse_args()

    gold = pd.read_parquet(GOLD)
    gold_words = {}
    for r in gold.itertuples():
        gold_words[(r.task_id, route(r.question))] = len(str(r.answer).split())
    gold_mean_words = sum(gold_words.values()) / len(gold_words)

    scores: dict[tuple[str, int], dict] = {}
    words: dict[tuple[str, int], float] = {}
    missing = []
    for cell in CELLS:
        for seed in args.seeds:
            path = PRED_DIR / f"{args.dataset}.{cell}.seed{seed}.jsonl"
            if not path.exists():
                missing.append(path.name)
                continue
            rows = [json.loads(l) for l in path.open(encoding="utf-8")]
            if len(rows) != 76:
                missing.append(f"{path.name} ({len(rows)} rows, expected 76)")
                continue
            scores[(cell, seed)] = score_file(path, gold)
            words[(cell, seed)] = sum(len(r["prediction"].split())
                                      for r in rows) / len(rows)
    if missing:
        print("missing or incomplete:\n  " + "\n  ".join(missing))
        if not scores:
            return 1

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"# Retrieval x granularity grid, PolyFiQA-Easy ({len(args.seeds)} seeds)")
    out()
    out("Temperature 0.7, top-p 0.8, top-k 20, one fixed seed per run. All four "
        "cells use the same raw context, the same model and quantization, the "
        "same output limit, the same computing backend and the same scoring "
        "code. No rewriting or normalization is applied.")
    out()
    out("This grid is a new controlled analysis. It ran with a larger context "
        "window and a different compute backend from the submitted system, so "
        "it is not a reproduction of the submitted runs; see "
        "reports/GRID_PROVENANCE.md. Using one environment for all four "
        "cells supports internally consistent comparisons.")
    out()

    for key, pretty in SCORERS:
        out(f"## ROUGE-1 F1, {pretty} scorer")
        out()
        out("| Configuration | " + " | ".join(f"seed {s}" for s in args.seeds)
            + " | mean | sd |")
        out("|---|" + "---|" * (len(args.seeds) + 2))
        cell_mean = {}
        for cell in CELLS:
            vals = [scores[(cell, s)][key]["overall"] for s in args.seeds
                    if (cell, s) in scores]
            if not vals:
                continue
            m, sd = msd(vals)
            cell_mean[cell] = m
            out(f"| {LABEL[cell]} | " + " | ".join(f"{v:.4f}" for v in vals)
                + f" | **{m:.4f}** | {sd:.4f} |")
        out()

        complete = [s for s in args.seeds
                    if all((c, s) in scores for c in CELLS)]
        if complete:
            # Paired: compute each effect within a seed, then summarise the
            # per-seed differences.
            per_seed = {"Retrieval, under grouped generation": [],
                        "Retrieval, under per-question generation": [],
                        "Per-question generation, without retrieval": [],
                        "Per-question generation, with retrieval": [],
                        "Interaction": []}
            for s_ in complete:
                v = {c: scores[(c, s_)][key]["overall"] for c in CELLS}
                r_gr = v["gr_retr"] - v["gr_noretr"]
                r_pq = v["pq_retr"] - v["pq_noretr"]
                per_seed["Retrieval, under grouped generation"].append(r_gr)
                per_seed["Retrieval, under per-question generation"].append(r_pq)
                per_seed["Per-question generation, without retrieval"].append(
                    v["pq_noretr"] - v["gr_noretr"])
                per_seed["Per-question generation, with retrieval"].append(
                    v["pq_retr"] - v["gr_retr"])
                per_seed["Interaction"].append(r_pq - r_gr)

            out(f"### Paired effects, computed within each seed "
                f"(n = {len(complete)} seeds)")
            out()
            out("| Effect | " + " | ".join(f"seed {s}" for s in complete)
                + " | mean | sd |")
            out("|---|" + "---|" * (len(complete) + 2))
            summary = {}
            for name, vals in per_seed.items():
                m, sd = msd(vals)
                summary[name] = (m, sd)
                out(f"| {name} | " + " | ".join(f"{v:+.4f}" for v in vals)
                    + f" | **{m:+.4f}** | {sd:.4f} |")
            out()
            for name, (m, sd) in summary.items():
                if name == "Interaction":
                    continue
                if sd == 0:
                    out(f"- {name}: {m:+.4f}.")
                elif abs(m) > sd:
                    out(f"- The estimated effect of {name.lower()} was "
                        f"{m:+.4f}, larger than the variation across the "
                        f"{len(complete)} runs (sd {sd:.4f}).")
                else:
                    out(f"- The estimated effect of {name.lower()} was "
                        f"{m:+.4f}, smaller than the variation across the "
                        f"{len(complete)} runs (sd {sd:.4f}).")
            out()
            out(f"With {len(complete)} seeds these figures describe the spread "
                f"observed across runs. They are not a significance test, and a "
                f"small estimate should be read as an effect we could not "
                f"resolve at this sample size rather than as an absence of one.")
            out()

    out("## Answer length")
    out()
    out("| Configuration | mean words | ratio to gold |")
    out("|---|---|---|")
    out(f"| Gold answers | {gold_mean_words:.1f} | 1.00 |")
    for cell in CELLS:
        vals = [words[(cell, s)] for s in args.seeds if (cell, s) in words]
        if vals:
            m, _ = msd(vals)
            out(f"| {LABEL[cell]} | {m:.1f} | {m / gold_mean_words:.2f} |")
    out()

    out("## Per question type, default scorer, mean across seeds")
    out()
    out("| Configuration | " + " | ".join(FAMILIES) + " |")
    out("|---|" + "---|" * len(FAMILIES))
    for cell in CELLS:
        vals = {f: [scores[(cell, s)]["multifinben_default"][f]
                    for s in args.seeds if (cell, s) in scores] for f in FAMILIES}
        if all(vals.values()):
            out(f"| {LABEL[cell]} | "
                + " | ".join(f"{msd(vals[f])[0]:.3f}" for f in FAMILIES) + " |")
    out()

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[saved] {Path(args.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
