"""One-command reproduction of the Fin-Agent best submission
(System D + Qwen3-32B + English/number rewrite; public ROUGE-1 F1 0.3023,
private 0.3419).

    python scripts/run_best_system.py [--tag qwen32b_De2_repro] [--skip-retrieval]

Stages (each is idempotent and cached; re-running never regenerates a
cached model response):

  1. build_chunks      -> data/processed/rag_chunks/            (no model)
  2. build_system_d    -> data/processed/system_d_prompts/      (BGE-M3, CPU)
  3. run_system_d      -> outputs/predictions/systemD_local_qwen3_32b/
                          expert_test.predictions.jsonl         (Qwen3-32B)
  4. rewrite_en        -> expert_test.rewritten_en.jsonl        (Qwen3-32B)
  5. fix_overlength    -> same file, only rows > 100 words      (Qwen3-32B)
  6. fix_nonlatin      -> same file, only rows still matching the
                          code-point trigger                    (Qwen3-32B)
  7. systemd_to_submission + src.submission
                       -> outputs/submission/submission_<tag>.csv

Requirements: an Ollama server with the `qwen3:32b` model reachable at
LOCAL_OLLAMA_BASE (default http://127.0.0.1:11500). See BEST_SYSTEM.md.
With the shipped cache/llm/ and data/processed/ directories, stages 2-5 hit
the cache and the run completes offline in under a minute.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PROVIDER = "local_qwen3_32b"
PRED_DIR = ROOT / f"outputs/predictions/systemD_{PROVIDER}"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def run(*cmd: str) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=ENV)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen32b_De2_repro",
                    help="suffix for outputs/submission/submission_<tag>.csv")
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="reuse existing data/processed/ chunks and prompts")
    args = ap.parse_args()

    if not args.skip_retrieval:
        run(PY, "-m", "src.retrieval.build_chunks")
        run(PY, "-m", "src.retrieval.build_system_d", "--retriever", "bge_m3", "--k", "5")
    elif not (ROOT / "data/processed/system_d_prompts/expert_test.system_d.jsonl").exists():
        sys.exit("--skip-retrieval given but System D prompts are missing")

    run(PY, "-m", "src.retrieval.run_system_d", "--dataset", "expert_test",
        "--provider", PROVIDER)
    run(PY, "scripts/rewrite_en.py", "--provider", PROVIDER, "--dataset", "expert_test")
    rewritten = PRED_DIR / "expert_test.rewritten_en.jsonl"
    run(PY, "scripts/fix_overlength.py", str(rewritten), "--provider", PROVIDER)
    run(PY, "scripts/fix_nonlatin.py", str(rewritten), "--provider", PROVIDER)

    tmp_csv = PRED_DIR / f"submission_{args.tag}.csv"
    run(PY, "scripts/systemd_to_submission.py", str(rewritten), str(tmp_csv))
    run(PY, "-m", "src.submission", "--predictions", str(tmp_csv), "--tag", args.tag)
    print(f"\nSUBMISSION READY: outputs/submission/submission_{args.tag}.csv")


if __name__ == "__main__":
    main()
