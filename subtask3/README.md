# Fin_Agent — FinEval 2026 Subtask 3 (PolyFiQA)

Code and data to reproduce our Subtask 3 results at the FinEval 2026 shared
task on multilingual financial language understanding, co-located with FinNLP
at EMNLP 2026.

The task: answer a financial question from a long, mixed-language context —
SEC filing statements plus contemporaneous news in English, Chinese, Japanese,
Spanish and Greek — in at most 100 words, citing supporting evidence. Scored
with ROUGE-1 F1 against expert-written references.

| Split | ROUGE-1 F1 | Precision | Recall | Rank |
|---|---|---|---|---|
| Public (24 instances) | 0.3023 | 0.3040 | 0.3528 | 10 / 15 |
| Private (52 instances) | 0.3419 | 0.3227 | 0.3899 | 9 / 15 |

## Reproduce everything

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

The reproduction itself takes under a minute and needs no GPU, no API key and no
network: every model response used in the paper is stored in `cache/llm/` and
replayed, so the published numbers come out exactly. Reports are written to
`reproduce/results/`.

Installing the dependencies is the one step that does need the internet. It
downloads roughly 2 GB, mostly PyTorch, and takes a few minutes. Use
`requirements-lock.txt` instead of `requirements.txt` to install the exact
versions this release was verified with.

To check the pipeline itself:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q       # Windows, 83 tests
```

```bash
./.venv/bin/python -m pytest tests -q                 # macOS / Linux
```

## What the system does

For each question, BGE-M3 retrieves the five most relevant financial-statement
chunks and the five most relevant news chunks. These are inserted as a
highlighted evidence block into the original context, which is kept in full,
and Qwen3-32B generates one answer per question. Three post-processing stages
follow: an English rewrite targeting 55–65 words, a length guard for answers
still over 100 words, and a language guard for answers still containing
non-Latin characters. Validation rules then flag unsupported numbers and
non-verbatim quotations without altering the text.

Both models are open-weight and run locally. The submitted pipeline uses
33.3B parameters in total, within the shared task's 70B limit.

**A finding worth flagging.** A controlled analysis (Tables 9–13) compares all
four combinations of retrieval and generation strategy on the development set,
with five seeds each. The four configurations fall within 0.004 ROUGE-1 F1 of
one another, and every estimated effect is smaller than the variation across
seeds. We therefore report no resolvable advantage for either component, rather
than a gain attributable to retrieval.

## Layout

| Path | Contents |
|---|---|
| `reproduce/` | One script per paper table. **Start at `reproduce/README.md`** |
| `src/` | Pipeline: retrieval, generation, scoring, validation |
| `scripts/` | Runnable stages and the end-to-end driver |
| `prompts/` | Every prompt used, by stage |
| `tests/` | Unit tests |
| `data/raw/` | Shared task development and test files |
| `data/processed/` | Retrieval chunks, embeddings and built prompts |
| `cache/llm/` | Every model response used in the paper and the analysis, one JSON per call |
| `outputs/submission/` | Submitted answer file and the variants compared in the paper |
| `outputs/predictions/` | Per-stage model outputs |
| `reports/` | Data audit, retrieval evaluation, scorer provenance |
| `BEST_SYSTEM.md` | The submitted pipeline, stage by stage |
| `legacy/` | Development-era records; nothing there describes the submitted system |
| `REPRODUCE.md` | Rebuilding the submission specifically |
| `LEAKAGE_GUARD.md` | Data-integrity policy followed during development |

## How the cache works

Each file in `cache/llm/` records one model call: the prompt, the response, the
model, and a provider fingerprint. Lookups are keyed on all of these, so a
cached response is only reused for an identical call to the identical model.
The first response is canonical and is never regenerated.

Delete `cache/llm/` to regenerate from the model instead. That needs a local
GGUF server hosting `qwen3:32b`; see `reproduce/README.md`. Answers are sampled at temperature
0.7 with no fixed seed, so a regeneration will not match word for word and
scores will shift slightly. The pipeline and prompts are identical either way.

## Data

The PolyFiQA files in `data/raw/` are redistributed from the organisers' public
Hugging Face release,
`FinNLP-Multilingual-Understanding/finnlp2026-subtask3-polyfiqa` (commit
`8377c6e8`), under the **Apache-2.0** licence. PolyFiQA is part of MultiFinBen.
They are included here so the results can be reproduced without hunting for the
inputs. Please cite the MultiFinBen paper if you use them.

The test file carries no reference answers -- the organisers withheld them -- so
nothing in this repository discloses the evaluation key.

Reference answers for the test split were never released, which is why the
official scores above cannot be recomputed locally. `reproduce/` verifies
instead that this repository rebuilds the exact scored file, by SHA-256, and
recomputes every other table from scratch.

## Citation

```bibtex
@inproceedings{hossain2026finagent,
  title     = {Fin\_Agent at FinEval 2026: Task-Aware Methods for
               Multilingual Financial NLP},
  author    = {Hossain, Tanzina},
  booktitle = {Proceedings of the FinNLP Workshop at EMNLP 2026},
  year      = {2026}
}
```

Contact: tanzinahossain@my.unt.edu
