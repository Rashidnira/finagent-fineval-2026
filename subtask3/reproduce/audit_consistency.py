"""Consistency audit - checks that the released analysis is what it claims.

WHAT THIS CHECKS
    Ten checks over the controlled analysis: that the four configurations
    differ only in retrieval and generation strategy, that every run produced
    76 answers over the same items, that all generations used the pinned
    model at the recorded seeds and context window with no post-processing,
    and that replaying a run from cache reproduces its answers exactly.

    It also regenerates the record of the computing environment used.

HOW LONG        under a minute, no GPU
OUTPUT          reproduce/results/audit_consistency.txt
                reports/GRID_PROVENANCE.md
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce._common import Report                       # noqa: E402


def main() -> int:
    rep = Report("audit_consistency")
    rep.head("Consistency audit")

    audit = subprocess.run(
        [sys.executable, "scripts/audit_grid.py", "--seeds", "1", "2", "3", "4", "5"],
        cwd=ROOT, capture_output=True, text=True)
    rep.pre(audit.stdout.strip())
    if audit.stderr.strip():
        rep.pre(audit.stderr.strip()[-600:])

    prov = subprocess.run([sys.executable, "scripts/record_grid_provenance.py"],
                          cwd=ROOT, capture_output=True, text=True)
    rep.p("Computing environment written to reports/GRID_PROVENANCE.md."
          if prov.returncode == 0 else "Could not write the environment record.")

    rep.save()
    return audit.returncode


if __name__ == "__main__":
    raise SystemExit(main())
