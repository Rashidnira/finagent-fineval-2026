# Reproduction package — Fin-Agent, FinEval 2026 Task 3 (PolyFiQA)

Final submission: `outputs/submission/submission_qwen32b_De2.csv`
SHA-256: `9c6e3341d3f3158499926772b491cf12e63efcde696de13158c8e7793ec9a4dd`
Official scores: public 0.3023 F1 (rank 10/15), private 0.3419 F1 (rank 9/15).

## 1. Exact replay (no GPU, ~1 minute)

Every model response used for the submission is in `cache/llm/`. The driver
replays the cache and rebuilds the CSV.

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_best_system.py --skip-retrieval --tag repro
Get-FileHash -Algorithm SHA256 outputs\submission\submission_repro.csv   # must equal the hash above
.\.venv\Scripts\python.exe -m pytest tests -q                            # 83 tests
```

macOS / Linux:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/run_best_system.py --skip-retrieval --tag repro
sha256sum outputs/submission/submission_repro.csv    # must equal the hash above
./.venv/bin/python -m pytest tests -q                # 83 tests
```

`--skip-retrieval` reuses `data/processed/system_d_prompts/` (the retrieval
output shipped here). The result is byte-identical to the submitted file.

## 2. Full regeneration (GPU)

Delete `cache/llm/` and, optionally, `data/processed/` to regenerate everything.

```bash
# Ollama >= 0.11 with the pinned model
ollama pull qwen3:32b
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 OLLAMA_HOST=127.0.0.1:11500 ollama serve &
export LOCAL_OLLAMA_BASE=http://127.0.0.1:11500     # default if unset

python scripts/run_best_system.py --tag regen        # runs all stages
```

Stages: `src.retrieval.build_chunks` -> `src.retrieval.build_system_d --retriever bge_m3 --k 5`
(BGE-M3 on CPU; embeddings cached in `data/processed/rag_embeddings/`) ->
`src.retrieval.run_system_d --dataset expert_test --provider local_qwen3_32b`
(Qwen3-32B, T=0.7/top-p 0.8/top-k 20, thinking off) -> `scripts/rewrite_en.py`
-> `scripts/fix_overlength.py` -> `scripts/fix_nonlatin.py` ->
`scripts/systemd_to_submission.py` -> `src.submission`.

Sampling at temperature 0.7 has no fixed seed, so regenerated answers differ
from the cached ones; the pipeline, prompts and post-processing are identical.

## 3. Development-set scoring (references available)

```bash
python -m src.retrieval.run_system_d --dataset easy_train --provider local_qwen3_32b   # cached
python -m src.retrieval.score_easy outputs/predictions/systemD_local_qwen3_32b/easy_train.predictions.jsonl
```
Expected: 0.458 (default scorer) / 0.440 (multiscript), per-type as in the paper.

## Contents

| Path | Purpose |
|---|---|
| `src/`, `scripts/`, `tests/`, `prompts/`, `requirements.txt` | pipeline code |
| `legacy/` | development-era records (an old config and phase notes); nothing here describes the submitted system |
| `data/raw/*.parquet` | organiser-supplied train (Easy) and test (Expert) files |
| `data/manifests/official_finnlp_test.parquet` | official id manifest (maps task_id+question to submission id) |
| `data/processed/rag_chunks/`, `rag_embeddings/`, `system_d_prompts/` | retrieval outputs (lets stage 1-2 be skipped) |
| `cache/llm/` | every cached model response (canonical, keyed by prompt hash) |
| `outputs/predictions/systemD_local_qwen3_32b/` | intermediate JSONL of the final run |
| `outputs/submission/submission_qwen32b_De2.csv` | the submitted file |
| `BEST_SYSTEM.md` | system description and per-stage commands |

Models: Qwen3-32B (32.8B, Apache-2.0) served locally as 4-bit GGUF weights
(`qwen3:32b`, Q4_K_M);
BAAI/bge-m3 (568M) via sentence-transformers. No API keys are needed.
