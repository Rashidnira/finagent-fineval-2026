"""Reproduce every table in the Subtask 3 sections of the paper, in order.

Run this first. It executes each table script, prints a summary, and leaves
one report file per table in reproduce/results/.

    python reproduce/run_all_tables.py

Total time is roughly five minutes on a laptop with no GPU, because every
model response is replayed from cache/llm/. Nothing here calls a model API.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

SCRIPTS = [
    ("Table 3    official results and submission integrity",
     "table_03_official_submission.py"),
    ("Table 4    system variants on the public split",
     "table_04_system_variants.py"),
    ("Table 8    model eligibility and parameter counts",
     "table_08_model_parameters.py"),
    ("Tables 9-13  controlled development analysis",
     "tables_09_to_13_controlled_analysis.py"),
    ("Tables 14-15 stage-wise validator flags",
     "tables_14_15_error_attribution.py"),
    ("Audit        consistency checks and environment record",
     "audit_consistency.py"),
]


def main() -> int:
    results = []
    for title, script in SCRIPTS:
        print("\n" + "#" * 78)
        print("# " + title)
        print("#" * 78 + "\n", flush=True)
        started = time.time()
        rc = subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT).returncode
        results.append((title, "ok" if rc == 0 else "FAILED", time.time() - started))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    width = max(len(t) for t, _, _ in results)
    for title, status, secs in results:
        print(f"{title:<{width}}  {status:>6}  {secs:6.1f}s")
    print(f"\nReports written to {(HERE / 'results').relative_to(ROOT)}/")

    failed = [t for t, s, _ in results if s != "ok"]
    if failed:
        print("\nFailed: " + ", ".join(failed))
        return 1
    print("\nAll tables reproduced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
