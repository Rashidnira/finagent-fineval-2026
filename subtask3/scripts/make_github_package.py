"""Build the public GitHub package for the Subtask 3 reproduction.

    python scripts/make_github_package.py --out ../FinAgent_Task3_public

Copies the files needed to reproduce the paper into a clean directory, so the
result can be pushed to a public repository without dragging along working
files, notes, or anything that should not be published.

Two things need a decision before publishing, and both are flagged by the
script rather than decided for you.

DATA. The PolyFiQA files under data/raw/ were released by the shared task
organisers under their own terms. Redistributing them in a public repository
may not be permitted. By default they are EXCLUDED and a download note is
written in their place. Pass --include-data to include them.

CACHE. cache/llm/ holds every model response used in the paper, which is what
makes reproduction exact and free. Each entry also embeds the prompt, and the
prompts contain the organisers' source contexts, so publishing the cache
publishes that text indirectly. By default the cache is EXCLUDED and the
package documents how to regenerate. Pass --include-cache to include it.

Running with neither flag produces a package that is safe to publish but
requires a GPU to reproduce. Running with both produces the package that
reproduces exactly in one minute with no GPU.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (source path, required) - directories are copied whole
ALWAYS = [
    ("src", True),
    ("scripts", True),
    ("reproduce", True),
    ("tests", True),
    ("prompts", True),
    ("legacy", False),
    ("requirements.txt", True),
    ("REPRODUCE.md", False),
    ("BEST_SYSTEM.md", False),
    ("LEAKAGE_GUARD.md", False),
    ("data/manifests", True),
    ("data/processed", True),
    ("outputs/submission", True),
    ("outputs/predictions", True),
    ("experiments/easy_benchmark", False),
    ("reports", False),
]

EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store"}

# Exploratory runs kept out of the public package by default. These models were
# tried during development but are not reported in the paper, so nothing in
# reproduce/ needs them. Pass --include-exploratory to ship them anyway.
EXPLORATORY_DIRS = [
    "outputs/predictions/systemD_local_gemma4_31b",
    "outputs/predictions/systemD_local_nemotron_49b",
    "outputs/predictions/expert_C_gemma4_31b_grouped_v1",
    "outputs/predictions/expert_C_gemma4_31b_grouped_v1_smoke",
    "outputs/predictions/expert_C_local_gemma4_31b_grouped_v1",
    "experiments/easy_benchmark/easy_C_gemma4_31b_grouped_v1",
    "experiments/easy_benchmark/easy_C_gemma4_31b_grouped_v1_smoke",
    "experiments/easy_benchmark/easy_C_local_gemma4_31b_grouped_v1",
]
PAPER_MODEL = "qwen3:32b"   # the only model reported in the paper

# Removed from the public package regardless of flags: transient outputs of the
# reproduction scripts themselves, and internal working notes that would only
# confuse a reader.
ALWAYS_DROP_GLOBS = [
    "outputs/submission/*repro_check*",
    "reports/CURRENT_STATE.md",          # dated internal status note
]

# Dropped with the other exploratory material unless --include-exploratory.
EXPLORATORY_GLOBS = [
    "outputs/submission/*gemma31b*",
    "outputs/submission/*nemotron49b*",
    "outputs/submission/*mbr_v1*",
    "reports/EASY_BENCHMARK_GEMMA.md",
]

DATA_NOTE = """# Data

The PolyFiQA files that belong here were released by the FinEval 2026 shared
task organisers and are not redistributed in this repository.

Place the following two files in this directory before running anything:

    public-00000-of-00001.parquet        development split (PolyFiQA-Easy)
    PolyFiQA_test_participant.parquet    test split as distributed to participants

Both come from the shared task materials. The development split is also
available from the MultiFinBen release on the Hugging Face Hub.
"""

CACHE_NOTE = """# Model response cache

This directory holds one JSON file per model call made for the paper, which is
what allows the results to be reproduced without a GPU.

It is not included in this repository, because each entry embeds the full
prompt and the prompts contain the shared task source documents.

Without the cache, the reproduction scripts will call the model instead. See
reproduce/README.md, section "Full regeneration", for how to set that up.
"""


def ignore(_dir, names):
    return {n for n in names if n in EXCLUDE_NAMES or n.endswith(".pyc")}


def copy(src: Path, dst: Path) -> int:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
        return sum(1 for _ in dst.rglob("*") if _.is_file())
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to create")
    ap.add_argument("--include-data", action="store_true",
                    help="include data/raw (check the organisers' terms first)")
    ap.add_argument("--include-cache", action="store_true",
                    help="include cache/llm (embeds source contexts in prompts)")
    ap.add_argument("--include-exploratory", action="store_true",
                    help="also ship the Gemma and Nemotron runs, which the paper "
                         "does not report and reproduce/ does not use")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        print(f"refusing to write into non-empty directory: {out}")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for rel, required in ALWAYS:
        src = ROOT / rel
        if not src.exists():
            if required:
                print(f"MISSING required path: {rel}")
                return 1
            continue
        n = copy(src, out / rel)
        if not args.include_exploratory:
            for rel_ex in EXPLORATORY_DIRS:
                if rel_ex.startswith(rel + "/") or rel_ex == rel:
                    victim = out / rel_ex
                    if victim.exists():
                        shutil.rmtree(victim)
            n = sum(1 for _ in (out / rel).rglob("*") if _.is_file()) if src.is_dir() else n
        total += n
        print(f"  {rel:<34} {n:>5} files")

    raw_dst = out / "data/raw"
    raw_dst.mkdir(parents=True, exist_ok=True)
    if args.include_data:
        total += copy(ROOT / "data/raw", raw_dst)
        print(f"  {'data/raw':<34} included")
    else:
        (raw_dst / "README.md").write_text(DATA_NOTE, encoding="utf-8")
        print(f"  {'data/raw':<34} EXCLUDED (note written)")

    cache_dst = out / "cache/llm"
    cache_dst.mkdir(parents=True, exist_ok=True)
    if args.include_cache:
        import json as _json
        kept = dropped = 0
        for src_file in sorted((ROOT / "cache/llm").glob("*.json")):
            if not args.include_exploratory:
                try:
                    model = _json.loads(src_file.read_text(encoding="utf-8")).get("model", "")
                except Exception:
                    model = ""
                if model != PAPER_MODEL:
                    dropped += 1
                    continue
            shutil.copy2(src_file, cache_dst / src_file.name)
            kept += 1
        total += kept
        note = f" ({dropped} exploratory entries dropped)" if dropped else ""
        print(f"  {'cache/llm':<34} {kept:>5} files{note}")
    else:
        (cache_dst / "README.md").write_text(CACHE_NOTE, encoding="utf-8")
        print(f"  {'cache/llm':<34} EXCLUDED (note written)")

    # Public front page: README_PUBLIC.md becomes README.md; the working notes
    # in the development README are shipped separately so nothing is lost.
    copy(ROOT / "README_PUBLIC.md", out / "README.md")
    if (ROOT / "README.md").exists():
        copy(ROOT / "README.md", out / "legacy" / "DEVELOPMENT_NOTES.md")
    total += 2
    print(f"  {'README.md (from README_PUBLIC)':<34} written")

    globs = list(ALWAYS_DROP_GLOBS)
    if not args.include_exploratory:
        globs += EXPLORATORY_GLOBS
    removed = 0
    for pattern in globs:
        for victim in out.glob(pattern):
            victim.unlink()
            removed += 1
    if removed:
        total -= removed
        print(f"  {'pruned transient/exploratory files':<34} {removed:>5} removed")

    (out / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.pytest_cache/\n.venv/\nreproduce/results/\n",
        encoding="utf-8")

    print(f"\nWrote {total} files to {out}")
    if not (args.include_data and args.include_cache):
        print("\nThis package cannot reproduce the published numbers as it stands.")
        if not args.include_data:
            print("  - data/raw is excluded, so nothing will run until the data is added")
        if not args.include_cache:
            print("  - cache/llm is excluded, so a GPU and Ollama are required")
        print("  Re-run with --include-data and/or --include-cache to change this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
