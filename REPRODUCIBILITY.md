# Reproducibility

The repository contains code for all three subtasks. Obtain the data through
the FinNLP 2026 shared-task distribution and place both dataset directories under a
single source directory.

## Environment and data

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_data.py /path/to/source/data/hf
```

The setup command validates that each dataset contains `train.parquet` and
`test.parquet`, then creates non-destructive symbolic links. Use `--mode copy`
if links are unsuitable. Existing destinations are never replaced.

## Subtask 1

Run the numeric rules, candidate extraction, local-model classification, decoding,
and deterministic assembly stages described in `subtask1/README.md`. The textual
classification stage requires the 32B local model:

```bash
cd subtask1
# Follow the commands in README.md in order.
```

## Subtask 2

Subtask 2 is fully offline and CPU-only:

```bash
mkdir -p subtask2/out
python subtask2/scripts/s2_rules.py \
  --data subtask2/data/hf/finnlp2026-subtask2-japanese-icr/test.parquet \
  --out subtask2/out/scores.json
python subtask2/scripts/s2_submit.py \
  --scores subtask2/out/scores.json --rule sinkhorn --lam 1.0 --strength 1.0 \
  --out subtask2/out/submission.csv
```

## Subtask 3

Place the two official QA parquet files under `subtask3/data/raw`, start a local
Ollama server containing `qwen3:32b`, and run:

```bash
cd subtask3
python scripts/run_best_system.py --tag reproduction
```

See `subtask3/README.md` for the data schema, dependencies, controlled comparison,
and validation commands. Fresh sampled generations need not be byte-identical to
the original run.

Run behavioral checks with:

```bash
python -m unittest discover -s tests -v
```
