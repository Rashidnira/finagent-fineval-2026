# Fin_Agent: FinEval 2026

This repository contains the code needed to rerun our three FinEval 2026 systems.

| Subtask | Main method |
|---|---|
| 1: Greek financial NER | rules for numeric entities and candidate classification for textual entities |
| 2: Japanese ICR | weighted Japanese cue lexicon and ordinal scoring |
| 3: Multilingual financial QA | BGE-M3 retrieval and local Qwen3-32B generation |

## Repository structure

```text
subtask1/       Greek NER code and instructions
subtask2/       Japanese ICR code and instructions
subtask3/       Multilingual financial QA code and instructions
```

Each subtask directory has its own README with the required data layout and commands.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Subtask 1 uses an OpenAI-compatible local endpoint. Our submitted run used `Qwen/Qwen2.5-32B-Instruct-AWQ` with vLLM.

```bash
export FINNLP_BASE_URL=http://127.0.0.1:8770/v1
export FINNLP_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
```

Subtask 2 is rule-based and does not require a language model or GPU.
Subtask 3 uses BGE-M3 for retrieval and `qwen3:32b` served locally with Ollama.

## Track eligibility

We entered the model-based category. Subtask 1 uses a 32B model. Subtask 3
uses Qwen3-32B together with BGE-M3, approximately 33.3B parameters in total.
Both are below the 70B limit. The OpenRouter requirement applies to the
API-based agent/workflow category and therefore did not apply to these local
model runs.

## Contributors

- [tanzinahossain](https://github.com/tanzinahossain)

## AI Assistance

**AI Assistance in Code Development:** *The authors utilized Claude Code (Anthropic, 2026) to assist in drafting evaluation scripts and automated test suites for this repository. All generated functions were manually verified, debugged, and tested by the authors to guarantee accuracy.*

## Citation

Please cite the Fin_Agent system paper if you use this code.


## Reproduction

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for data setup and commands. Each
subtask also has its own README.
