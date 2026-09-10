# Reproducing the Subtask 3 results

Every number reported for Subtask 3 can be regenerated from this repository.
All model responses used in the paper are stored in `cache/llm/`, so nothing
here calls a model API and no GPU is needed.

```bash
python -m venv .venv
```

Then, on Windows (PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce/run_all_tables.py
```

or on macOS and Linux:

```bash
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python reproduce/run_all_tables.py
```

Calling the virtual environment's Python directly means there is no environment to
activate. That avoids PowerShell execution-policy errors and behaves the same whether
or not you use Anaconda. Let `pip install` finish completely before running anything
else -- if the terminal is closed or reloaded while pip is working, the packages are
silently not installed.

Runtime is under a minute. Each script prints its results and also writes a
report to `reproduce/results/`.

## What each file does

| File | Reproduces | Needs a GPU? |
|---|---|---|
| `run_all_tables.py` | Everything below, in order, with a summary | no |
| `table_03_official_submission.py` | Table 3. Rebuilds the submitted answer file and checks it byte for byte | no |
| `table_04_system_variants.py` | Table 4. Verifies the variant files and their answer lengths | no |
| `table_08_model_parameters.py` | Table 8. Parameter count of every learned component | no |
| `tables_09_to_13_controlled_analysis.py` | Tables 9–13. Retrieval vs generation strategy, four configurations x five seeds | no |
| `tables_14_15_error_attribution.py` | Tables 14–15. Which pipeline stage raised each class of validator flag | no |
| `audit_consistency.py` | Ten consistency checks over the analysis, plus the environment record | no |
| `_common.py` | Shared reporting helpers, not run directly | – |
| `results/` | Where the reports are written | – |

## What can and cannot be recomputed

**Recomputed from scratch.** Tables 8 to 14. These are measured on the
development set, where gold answers are public, or computed from files and
model weights, so the scripts recompute every number.

**Verified but not recomputed.** Tables 4 and 5. Those ROUGE scores were
produced by the organisers' leaderboard against references that were never
released. No script can recompute them. What the scripts verify instead is
that this repository rebuilds the exact prediction files that were scored,
checked by SHA-256.

## Two ways to run

**Cache replay (default).** Every model response used in the paper is stored
in `cache/llm/`, keyed by the provider fingerprint, the prompt and the
configuration tag. The scripts replay these and reproduce the published
numbers exactly.

**Full regeneration.** Delete `cache/llm/` and the scripts will call the model
instead. This needs a local GGUF server hosting `qwen3:32b`:

```bash
ollama pull qwen3:32b
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_HOST=127.0.0.1:11500 ollama serve &
export LOCAL_OLLAMA_BASE=http://127.0.0.1:11500
```

Answers are sampled at temperature 0.7 with no fixed seed, so regenerated
answers will not match the cached ones word for word, and scores will move by
a small amount. The pipeline, prompts and post-processing are identical. Use
cache replay to check the published numbers, and full regeneration to check
that the pipeline runs end to end.

## Expected results

```
Table 4    submitted file SHA-256 9c6e3341...a4dd, public F1 0.3023, private 0.3419
Table 5    four variant files, 76 rows each, no answer over 100 words
Table 8    Qwen3-32B 32.8B + BGE-M3 0.57B = 33.3B, within the 70B limit
Table 9    G 0.4547  G+R 0.4525  Q 0.4564  Q+R 0.4559   (default, mean of 5 seeds)
Table 10   every paired effect smaller than its seed-to-seed spread
Table 11   reference 47.9 words, G 33.3, G+R 35.3, Q 51.6, Q+R 55.8
Table 12   per question type, default scorer
Audit      10 of 10 checks pass; 950 grid generations verified
Table 13   untraceable numbers 5 -> 2, non-verbatim quotations 4 -> 5,
           missing evidence labels 0 -> 6
Table 14   mean length 91.8 -> 73.7 -> 69.4, answers over 100 words 23 -> 7 -> 0
```

## Notes

### A note on exact reproduction

Replaying the cache reproduces every published number exactly, on any machine.
Regenerating from scratch will not match word for word: a fixed seed makes
generation deterministic only for the same backend, build and GPU. Anyone
re-running without the cache should expect small differences, and can compare
their environment against `reports/GRID_PROVENANCE.md`.

`tables_14_15_error_attribution.py` re-runs the post-processing scripts, which
edit `outputs/predictions/systemD_local_qwen3_32b/expert_test.rewritten_en.jsonl`
in place. It backs the file up first and restores it afterwards, and checks
that the final stage reproduces the shipped file byte for byte.

The unit tests are separate from this directory and cover the pipeline itself:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q       # Windows
```

```bash
./.venv/bin/python -m pytest tests -q                 # macOS / Linux
```
