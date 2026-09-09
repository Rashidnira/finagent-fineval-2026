# Subtask 1: Greek Financial Named-Entity Recognition

This directory contains the implementation used for the Greek financial NER subtask. Numeric and textual entities are handled with separate procedures and are combined in the final step.

## Numeric entities

Numeric entities are extracted with the rules in `scripts/s1_rules.py`. The rules cover:

- money values
- percentages
- dates and other time expressions
- quantities
- law, article, and registry references

The extractor follows the dataset format. For example, dates may be divided into separate numeric spans, money entities contain the numeric part, and percentage entities normally retain the percent sign. Repeated entities are preserved because the output is evaluated as a multiset.

## Textual entities

Textual entities are produced in two stages.

First, `scripts/s1_candidate_spans.py` finds possible person, organisation, and location spans directly in each passage. Candidates include capitalised token sequences, names ending in common legal-form markers, and place names from the gazetteer.

Second, `scripts/s1_text_candclf.py` sends each candidate to the local language model for classification. The model assigns one of the textual entity types or rejects the candidate. This approach keeps the predicted text tied to spans that already occur in the passage.

The submitted run used `Qwen/Qwen2.5-32B-Instruct-AWQ` through a local vLLM endpoint. The model has 32 billion parameters.

## Post-processing

The textual output is cleaned before assembly:

- entity strings must occur verbatim in the source passage;
- English type names are mapped to the required Greek labels;
- each prediction is split on the last comma so commas inside an entity are retained;
- generic references such as `η Εταιρεία` and `ο Όμιλος` are removed; and
- duplicate predictions are handled according to the required output format.

`scripts/s1_assemble.py` regenerates the numeric predictions, loads the textual predictions, restores the original row order, and writes the final CSV. It also checks the row IDs, permitted entity types, UTF-8 encoding, line endings, and span validity.

## Setup

Install the packages from the repository root and start the local model endpoint. Then set:

```bash
export FINNLP_BASE_URL=http://127.0.0.1:8770/v1
export FINNLP_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
export FINNLP_ROOT=/path/to/finagent-fineval-2026/subtask1
```

The submitted inference configuration used vLLM with AWQ-Marlin quantisation, float16 computation, a maximum model length of 4096, seed 0, tensor parallel size 1, and one GPU.

## Run the pipeline

From the `subtask1` directory, run:

```bash
mkdir -p out/alt
python scripts/s1_gazetteer.py
python scripts/s1_candidate_spans.py
python scripts/s1_text_candclf.py --workers 16
python scripts/s1_text_candclf_decode.py --scores out/alt/candclf_scores.json --out_prefix candclf
cp out/alt/s1_text_candclf_maximal_pref_t09.csv out/s1_text_candclf.csv
python scripts/s1_assemble.py
```

The candidate decoder uses the maximal-preference rule with a confidence threshold of 0.90. Paths and output names can be changed through the script arguments.

## Local evaluation

The numeric rules can be checked against the released labelled data with:

```bash
python scripts/s1_metric.py --pred PRED.csv --gold GOLD.parquet
```

The scorer performs exact entity-level micro-F1 using multiset comparison. It also provides optional normalised, type-insensitive, and seqeval-style diagnostics.


## Reproducibility

Provide the official parquet files and external Greek resources, set
`FINNLP_EXTERNAL_DATA` if they are not under `data/external`, start vLLM, and
run the pipeline above.
