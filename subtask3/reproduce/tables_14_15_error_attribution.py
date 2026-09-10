"""Tables 13 and 14 - Do the remaining errors come from generation or from post-processing.

WHAT THIS REPRODUCES
    Appendix C.4 of the paper. The pipeline applies three post-processing
    stages to the generated answers. This script re-runs them in order from
    the cache, takes a snapshot after each, and applies the same
    reference-free validator to every snapshot, so each error class can be
    attributed to the stage that introduced it.

        stage 0   raw generation
        stage 1   + rewrite          (all 76 answers)
        stage 2   + length guard     (the 7 answers over 100 words)
        stage 3   + language guard   (the 14 answers with non-Latin text)

    Stage 3 must reproduce the shipped file byte for byte; the script checks
    this and restores the original file when it finishes.

HOW LONG        about two minutes from cache, no GPU
WHAT IT NEEDS   cache/llm/, data/raw/PolyFiQA_test_participant.parquet
OUTPUT          reproduce/results/tables_14_15_error_attribution.txt
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce._common import Report, sha256_file          # noqa: E402
from src.answer_validator import validate                  # noqa: E402

PRED_DIR = ROOT / "outputs/predictions/systemD_local_qwen3_32b"
RAW = PRED_DIR / "expert_test.predictions.jsonl"
WORKING = PRED_DIR / "expert_test.rewritten_en.jsonl"
TEST_PARQUET = ROOT / "data/raw/PolyFiQA_test_participant.parquet"
LABEL = "Financial Statements Evidence:"
# The same detector the pipeline uses: scripts/fix_nonlatin.py counts
# characters with ord(ch) > 0x2000, so it catches CJK text and typographic
# punctuation but not Greek or Cyrillic. Counting every non-ASCII character
# instead would not describe the stage that actually ran.
NONLATIN = re.compile(r"[^\u0000-\u2000]")

STAGES = [
    ("0  raw generation", None),
    ("1  + rewrite", [sys.executable, "scripts/rewrite_en.py",
                      "--provider", "local_qwen3_32b"]),
    ("2  + length guard", [sys.executable, "scripts/fix_overlength.py",
                           str(WORKING), "--provider", "local_qwen3_32b"]),
    ("3  + language guard", [sys.executable, "scripts/fix_nonlatin.py",
                             str(WORKING), "--provider", "local_qwen3_32b"]),
]


def audit(path: Path, ctx: dict) -> dict:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    out = {"untraceable_numbers": 0, "unverbatim_quotes": 0,
           "missing_evidence_label": 0, "any": 0,
           "words": [], "nonlatin": 0, "over100": 0}
    for r in rows:
        v = validate(r["prediction"], ctx[r["task_id"]], LABEL)
        for flag in v["flags"]:
            if flag in out:
                out[flag] += 1
        out["any"] += bool(v["flags"])
        out["words"].append(v["word_count"])
        out["nonlatin"] += bool(NONLATIN.search(r["prediction"]))
        out["over100"] += v["word_count"] > 100
    return out


def main() -> int:
    rep = Report("tables_14_15_error_attribution")
    rep.head("Tables 14 and 15 - Stage-wise error attribution")

    if not WORKING.exists():
        rep.p("Missing " + str(WORKING.relative_to(ROOT)) + ". Run the submission "
              "reproduction first (reproduce/table_03_official_submission.py).")
        rep.save()
        return 1

    ctx = dict(pd.read_parquet(TEST_PARQUET, columns=["task_id", "query"])
               .drop_duplicates("task_id").itertuples(index=False, name=None))

    backup = Path(tempfile.mkdtemp()) / "shipped_final.jsonl"
    shutil.copy2(WORKING, backup)
    shipped_sha = sha256_file(backup)

    try:
        results = []
        for name, cmd in STAGES:
            if cmd is None:
                results.append((name, audit(RAW, ctx)))
                continue
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                rep.p("stage failed: " + name)
                rep.pre("\n".join((proc.stdout + proc.stderr).splitlines()[-12:]))
                rep.save()
                return 1
            results.append((name, audit(WORKING, ctx)))

        final_sha = sha256_file(WORKING)
    finally:
        shutil.copy2(backup, WORKING)

    rep.p("Table 14 - validator flags after each stage (counts are rows out of 76)")
    rep.table(["Error class"] + [n for n, _ in results],
              [("Untraceable numbers",) + tuple(r["untraceable_numbers"] for _, r in results),
               ("Non-verbatim quotations",) + tuple(r["unverbatim_quotes"] for _, r in results),
               ("Missing evidence label",) + tuple(r["missing_evidence_label"] for _, r in results),
               ("Rows with any flag",) + tuple(r["any"] for _, r in results)])

    rep.p("The final row reports the union of all flagged rows, including checks not "
          "listed separately, chiefly answers longer than 110 words at raw generation. "
          "Because one answer may carry multiple flags, this value need not equal the "
          "sum of the three displayed classes.")

    rep.p("Table 15 - answer length and script composition after each stage")
    rep.table(["Stage", "Mean words", "Max words", "Rows over 100 words",
               "Rows flagged for script"],
              [(n, f"{sum(r['words']) / len(r['words']):.1f}", max(r["words"]),
                r["over100"], r["nonlatin"]) for n, r in results])

    ok = final_sha == shipped_sha
    rep.p("Integrity: the final stage reproduces the shipped file byte for byte."
          if ok else
          "Integrity: WARNING, the final stage does not match the shipped file.")

    rep.p("Reading of the result. Untraceable numbers come from generation and are partly "
          "repaired by post-processing. Non-verbatim quotations come almost entirely from "
          "generation. Every missing evidence label is introduced by the length guard, even "
          "though its prompt asks for the labeled structure to be kept, so the failure is one "
          "of instruction-following under a 45-word target rather than a missing instruction. "
          "Post-processing is also required rather than optional: 23 of the 76 raw answers exceed "
          "the 100-word limit, the longest at 153 words, and seven still exceed it after the "
          "rewrite, which is why the length guard is applied to seven answers.")
    rep.save()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
