"""Table 3 - Official results, and the submitted answer file.

WHAT THIS REPRODUCES
    The submitted prediction file itself. The official ROUGE-1 scores in
    Table 4 were computed by the organisers against references that are not
    public, so they cannot be recomputed here. What can be checked is that
    this repository rebuilds the exact file that produced them, byte for
    byte, from the cached model responses.

HOW LONG        about one minute, no GPU
WHAT IT NEEDS   cache/llm/ and data/processed/system_d_prompts/
OUTPUT          reproduce/results/table_03_official_submission.txt
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce._common import Report, sha256_file          # noqa: E402

SUBMITTED = ROOT / "outputs/submission/submission_qwen32b_De2.csv"
EXPECTED_SHA = "9c6e3341d3f3158499926772b491cf12e63efcde696de13158c8e7793ec9a4dd"

# Official scores as returned by the organisers' leaderboard. Recorded here
# for reference; they are not recomputed by this script.
OFFICIAL = [
    ("Public (24 instances)", "0.3023", "0.3040", "0.3528", "10 / 15"),
    ("Private (52 instances)", "0.3419", "0.3227", "0.3899", "9 / 15"),
]


def main() -> int:
    rep = Report("table_03_official_submission")
    rep.head("Table 3 - Official results and submission integrity")

    rep.p("Official scores (from the organisers' leaderboard, not recomputed):")
    rep.table(["Split", "ROUGE-1 F1", "Precision", "Recall", "Rank"], OFFICIAL)

    rep.p("Rebuilding the submission from the cached model responses...")
    cmd = [sys.executable, "scripts/run_best_system.py", "--skip-retrieval", "--tag", "repro_check"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        rep.p("FAILED to rebuild. Last lines of output:")
        rep.pre("\n".join((proc.stdout + proc.stderr).splitlines()[-15:]))
        rep.save()
        return 1

    rebuilt = ROOT / "outputs/submission/submission_repro_check.csv"
    got = sha256_file(rebuilt)
    ref = sha256_file(SUBMITTED)

    rep.table(["File", "SHA-256"],
              [("submitted (shipped)", ref[:32] + "..."),
               ("rebuilt now", got[:32] + "...")])

    ok = got == EXPECTED_SHA == ref
    rep.p("RESULT: rebuilt file is byte-identical to the submitted file."
          if ok else
          "RESULT: MISMATCH. The rebuilt file differs from the submitted file.")
    rep.p("Note: answers are sampled at temperature 0.7 with no fixed seed, so a full "
          "regeneration without the cache will not match byte for byte. The pipeline, "
          "prompts and post-processing are identical either way.")
    rep.save()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
