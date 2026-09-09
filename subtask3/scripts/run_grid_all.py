"""Run every cell of the grid at every seed.

    python scripts/run_grid_all.py --seeds 1 2 3

Runs 4 cells x N seeds. Responses are cached as they complete, so an
interrupted run resumes where it stopped rather than starting over.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ["pq_noretr", "pq_retr", "gr_noretr", "gr_retr"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--dataset", default="easy_train")
    ap.add_argument("--cells", nargs="+", default=CELLS, choices=CELLS)
    args = ap.parse_args()

    jobs = [(c, s) for s in args.seeds for c in args.cells]
    results = []
    t0 = time.time()

    for n, (cell, seed) in enumerate(jobs, 1):
        print(f"\n=== [{n}/{len(jobs)}] cell={cell} seed={seed} ===", flush=True)
        started = time.time()
        proc = subprocess.run(
            [sys.executable, "scripts/run_grid.py", "--cell", cell,
             "--seed", str(seed), "--dataset", args.dataset],
            cwd=ROOT, capture_output=True, text=True)
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        print("\n".join(tail[-3:]), flush=True)
        results.append((cell, seed, "ok" if proc.returncode == 0 else "FAILED",
                        time.time() - started))

    print("\n" + "=" * 62)
    print(f"{'cell':<12}{'seed':>5}{'status':>9}{'seconds':>10}")
    print("=" * 62)
    for cell, seed, status, secs in results:
        print(f"{cell:<12}{seed:>5}{status:>9}{secs:>10.0f}")
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0 if all(r[2] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
