# Subtask 3: Multilingual Financial QA

This directory contains the implementation used for the multilingual financial
question-answering system.

## Method

The pipeline splits each filing into financial-statement and news chunks. BGE-M3
retrieves the five most relevant chunks of each type for every question. The
original context and highlighted evidence are then passed to Qwen3-32B, served
locally through Ollama as a 4-bit GGUF model. Answers are generated separately for
each question and then pass through the language, length, and formatting checks.

The learned components contain approximately 33.3B parameters in total:
Qwen3-32B for generation and BGE-M3 for retrieval. This is the model-based route
and remains below the shared task's 70B limit.

## Data layout

Obtain the official Subtask 3 files from the shared-task distribution and place
them here:

```text
subtask3/data/raw/
  public-00000-of-00001.parquet
  PolyFiQA_test_participant.parquet
```

The first file must contain `task_id`, `query`, `question`, and `answer`. The
participant test file must contain `task_id`, `query`, and `question` only.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r subtask3/requirements.txt
ollama pull qwen3:32b
```

Ollama is expected at `http://127.0.0.1:11500` by default. Set
`LOCAL_OLLAMA_BASE` if it is served elsewhere.

## Run the final pipeline

```bash
cd subtask3
python scripts/run_best_system.py --tag reproduction
```

This command builds chunks, retrieves evidence with BGE-M3, generates answers
with the local Qwen3-32B model, applies the post-processing checks, and writes a
new CSV under `outputs/submission/`.

Generation uses temperature 0.7, top-p 0.8, top-k 20, and thinking disabled.
Because the submitted run did not use a fixed sampling seed, a fresh run is not
expected to reproduce its wording byte for byte.

## Controlled comparison

The retrieval-by-generation-granularity comparison can be rerun with:

```bash
python -m src.retrieval.build_chunks
python -m src.retrieval.build_grid
python scripts/run_grid_all.py --seeds 1 2 3 4 5
python scripts/score_grid.py --seeds 1 2 3 4 5
python scripts/audit_grid.py --seeds 1 2 3 4 5
```

These commands create the four configurations (grouped/per-question, with/without
retrieval) using the same Qwen3-32B model and a 40,960-token context window.

## Tests

```bash
python -m pytest -q
```

Some integrity tests require the official data files described above.
