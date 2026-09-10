# Fin-Agent — best system (FinNLP 2026 Subtask 3, PolyFiQA)

Team: **Fin-Agent** (Tanzina Hossain, University of North Texas,
tanzinahossain@my.unt.edu)

Final submission: `outputs/submission/submission_qwen32b_De2.csv`
(76 rows, `id,prediction`, every answer <= 100 words).

| Split | ROUGE-1 F1 | P | R |
|---|---|---|---|
| Public (24 instances) | 0.3023 | 0.3040 | 0.3528 |
| Private (52 instances) | 0.3419 | 0.3227 | 0.3899 |

Eligibility: model-based route. The only generative model is **Qwen3-32B**
(32.8B parameters, Apache-2.0), served locally as 4-bit GGUF weights (`qwen3:32b`,
GGUF Q4_K_M, thinking disabled, temperature 0.7 / top-p 0.8 / top-k 20) on
one NVIDIA H100. Retrieval uses **BGE-M3** embeddings (568M, CPU). No
third-party inference API and no reference answers are used at any stage.

## Pipeline

```
raw parquet ──1──> chunks ──2──> System D prompts ──3──> Qwen3-32B answers
            ──4──> English/number rewrite (Qwen3-32B) ──5──> length guard
            ──6──> id,prediction CSV + integrity check
```

| # | Command | Model | Output |
|---|---|---|---|
| 1 | `python -m src.retrieval.build_chunks` | none | `data/processed/rag_chunks/` |
| 2 | `python -m src.retrieval.build_system_d --retriever bge_m3 --k 5` | BGE-M3 (CPU) | `data/processed/system_d_prompts/expert_test.system_d.jsonl` |
| 3 | `python -m src.retrieval.run_system_d --dataset expert_test --provider local_qwen3_32b` | Qwen3-32B | `outputs/predictions/systemD_local_qwen3_32b/expert_test.predictions.jsonl` |
| 4 | `python scripts/rewrite_en.py --provider local_qwen3_32b` | Qwen3-32B | `.../expert_test.rewritten_en.jsonl` |
| 5a | `python scripts/fix_overlength.py <rewritten_en.jsonl>` | Qwen3-32B (only rows > 100 words; 7 of 76 for De2) | same file |
| 5b | `python scripts/fix_nonlatin.py <rewritten_en.jsonl>` | Qwen3-32B (only rows with non-Latin characters; 14 of 76 for De2) | same file |
| 6 | `python scripts/systemd_to_submission.py <rewritten_en.jsonl> <out.csv>` then `python -m src.submission --predictions <out.csv> --tag <tag>` | none | `outputs/submission/submission_<tag>.csv` + audit CSV |

One command runs all stages:

```bash
python scripts/run_best_system.py --tag qwen32b_De2_repro
```

Verified 25 Aug 2026: with the shipped cache, this command rebuilds
`submission_qwen32b_De2.csv` byte-for-byte (0 of 76 rows differ).

Stage details:

- **Retrieval (stages 1-2).** Contexts are split losslessly into statement
  chunks and news chunks. BGE-M3 dense retrieval selects the top-5 news and
  top-5 statement chunks for the question. The full original context is
  kept; the retrieved chunks are inserted as a "Highlighted Evidence" block
  immediately before the trailing `Question:` line
  (`src/retrieval/build_system_d.py`).
- **Generation (stage 3).** One prompt per question, empty system message,
  through `src/generator.py` (`src/providers.py::LocalQwen3Provider`).
  Every response is checked to come from the pinned model and cached under
  `cache/llm/` keyed by (provider fingerprint, prompt, config tag). The
  first cached response is canonical and is never regenerated.
- **Post-processing (stages 4-5).** The same model rewrites each answer to
  55-65 words, renders non-English text in English and compacts
  unambiguous monetary figures (`scripts/rewrite_en.py`, config tag
  `rewrite_en60_v1`). A guard re-shortens anything still above 100 words.
- **Submission (stage 6).** Rows are mapped to official ids by exact
  `(task_id, question)` match; `src/submission.py` validates row count, id
  set, duplicates, empties, and read-back.

## Environment

```bash
pip install -r requirements.txt          # pandas, pyarrow, regex, numpy, requests, sentence-transformers ...
# Ollama server with the pinned model:
ollama pull qwen3:32b
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 OLLAMA_HOST=127.0.0.1:11500 ollama serve
export LOCAL_OLLAMA_BASE=http://127.0.0.1:11500   # default if unset
```

Reproducibility: `cache/llm/` in this bundle contains every model response
used for the final submission, so `run_best_system.py --skip-retrieval`
rebuilds `submission_qwen32b_De2.csv` byte-for-byte without a GPU. Delete
`cache/llm/` to regenerate from scratch (sampling at T=0.7 means fresh
generations differ slightly).

## Tests

```bash
python -m pytest tests/ -q
```
