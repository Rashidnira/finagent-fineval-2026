"""Tables 9 to 12 - Controlled development analysis (Appendix C.3).

WHAT THIS REPRODUCES
    The four-configuration grid that separates retrieval from generation
    strategy, on the 76-instance PolyFiQA-Easy development set, with five
    seeds each:

        G     grouped generation,       no retrieval
        G+R   grouped generation,       BGE-M3 evidence beside each question
        Q     question-level generation, no retrieval
        Q+R   question-level generation, BGE-M3 evidence block  (submitted design)

    All four prompt sets are built from one source, so the cells differ only
    in retrieval and generation strategy. Effects are computed within each
    seed and then summarised, which accounts for seed-level variation shared
    across the four configurations.

HOW LONG        about a minute from cache, no GPU
                (without cache: 20 runs, 3800 generations, needs Ollama)
WHAT IT NEEDS   cache/llm/, data/processed/grid_prompts/
OUTPUT          reproduce/results/tables_09_to_13_controlled_analysis.txt
                reports/GRID_RESULTS.md
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce._common import Report                       # noqa: E402

SEEDS = ["1", "2", "3", "4", "5"]
CELLS = ["pq_noretr", "pq_retr", "gr_noretr", "gr_retr"]
PRED = ROOT / "outputs/predictions/grid"
PROMPTS = ROOT / "data/processed/grid_prompts"


def run(cmd: list[str], rep: Report) -> bool:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        rep.p("command failed: " + " ".join(cmd))
        rep.pre("\n".join((proc.stdout + proc.stderr).splitlines()[-12:]))
        return False
    return True


def main() -> int:
    rep = Report("tables_09_to_13_controlled_analysis")
    rep.head("Tables 9 to 13 - Controlled development analysis")

    if not (PROMPTS / "easy_train.gr_retr.jsonl").exists():
        rep.p("Building the four prompt sets ...")
        if not run([sys.executable, "-m", "src.retrieval.build_grid",
                    "--dataset", "easy_train"], rep):
            rep.save()
            return 1

    missing = [(c, s) for c in CELLS for s in SEEDS
               if not (PRED / f"easy_train.{c}.seed{s}.jsonl").exists()]
    for cell, seed in missing:
        rep.p(f"Generating cell {cell}, seed {seed} ...")
        if not run([sys.executable, "scripts/run_grid.py", "--cell", cell,
                    "--seed", seed, "--dataset", "easy_train"], rep):
            rep.save()
            return 1

    proc = subprocess.run(
        [sys.executable, "scripts/score_grid.py", "--seeds", *SEEDS],
        cwd=ROOT, capture_output=True, text=True)
    rep.pre(proc.stdout.strip())
    if proc.returncode != 0:
        rep.pre(proc.stderr.strip()[-800:])
    rep.save()
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
