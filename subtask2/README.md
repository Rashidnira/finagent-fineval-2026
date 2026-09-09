# Subtask 2: Japanese Financial Implicit Communication Recognition

This directory contains the rule-based implementation for the Japanese implicit communication recognition subtask. It does not require a language model or GPU.

## Labels

The response is assigned one of five ordinal labels:

- `+2`: definite commitment or completed decision
- `+1`: positive intention with some uncertainty
- `0`: neutral, procedural, or non-committal response
- `-1`: soft refusal or reluctance
- `-2`: direct refusal or firm rejection

The implementation keeps this label order throughout the pipeline:

```text
[+2, +1, 0, -1, -2]
```

## Input preparation

`scripts/s2_rules.py` reads a parquet file. If separate `Q` and `R` columns are unavailable, it extracts the financial question and company response from the `query` field. Only the company response is used for cue matching.

For development checks, `scripts/s2_build_split.py` creates a class-balanced development split and a separate pool of examples. The script uses seed `20260818` and reads data from `data/hf/finnlp2026-subtask2-japanese-icr` under `FINNLP_ROOT`.

## Cue extraction

The scorer counts regular-expression matches for the following cue families:

| Family | Purpose |
|---|---|
| `DECIDED` | board approval, formal decision, agreement, or contract completion |
| `DONE` | an action that has been or will be carried out directly |
| `VOLIT` | intention, target, effort, or forward-looking commitment |
| `COND` | conditions such as market state, progress, timing, or approval |
| `NEUTRAL` | analysis, verification, preparation, or later explanation |
| `NOPROMISE` | language that avoids a guarantee or confirmed decision |
| `NEG_SOFT` | difficulty, delay, caution, or a soft refusal |
| `NEG_HARD` | explicit lack of plans, cancellation, refusal, or no comment |
| `REOPEN` | a clause that leaves open future reconsideration |
| `REAFFIRM` | language stating that an existing plan has not changed |
| `NUM` and `DATE` | numeric and time details supporting specificity |

The complete Japanese expressions are defined in `scripts/s2_rules.py`. Cue variants were added after inspecting the unlabeled response text.

Before hard-negation matching, reaffirmation phrases are masked. This prevents expressions such as `変更はありません` from being treated as refusals when they confirm an existing plan. Hard-negation patterns are also kept specific because a general factual expression ending in `ありません` is not necessarily a refusal.

## Scoring procedure

`scripts/s2_rules.py` creates one score for each label. Match counts are capped where repeated expressions should not dominate the prediction.

1. Decision and completion cues primarily increase `+2`, with smaller support for `+1`. Numeric and date details add specificity. If the same response contains a hard negative action, this evidence is assigned to the negative labels instead.
2. Hard-negative cues increase `-2` and `-1`. Soft-negative cues mainly increase `-1`. A reopening clause shifts part of the negative evidence from `-2` toward `-1`.
3. Volitional and conditional cues increase `+1`. Their weight is reduced when a hard negative is also present. A no-promise cue lowers `+2`, and unsettled intent receives a further `+2` penalty.
4. Reaffirmation without hard negation increases `+2` and gives smaller support to `0`.
5. Neutral cues increase `0`. A response without decision, intention, condition, or negative cues also receives a neutral default score. Negative evidence lowers the neutral score.

The exact weights are implemented directly in the `score_row` function in `scripts/s2_rules.py`.

## Calibration and prediction

`scripts/s2_submit.py` loads the five scores for each example and subtracts the null-prompt bias using the `--lam` value. The default prediction rule applies Sinkhorn calibration with a uniform target prior of `0.2` for each label and then takes the largest calibrated score.

The script also supports uncalibrated argmax, Hungarian assignment, and capped-majority prediction through the `--rule` option. Before writing the CSV, it checks that all 50 IDs are present once and that every prediction belongs to the five-label set.

## Run the pipeline

From the `subtask2` directory, run:

```bash
mkdir -p out
python scripts/s2_build_split.py
python scripts/s2_rules.py --data data/hf/finnlp2026-subtask2-japanese-icr/test.parquet --out scores.json
python scripts/s2_submit.py --scores scores.json --rule sinkhorn --lam 1.0 --strength 1.0 --out submission.csv
```

The input and output paths can be changed through the command-line arguments.

## Local evaluation

Predictions can be checked against labelled development data with:

```bash
python scripts/s2_metric.py --pred PRED.csv --gold GOLD.parquet
```

The evaluator reports accuracy, macro recall, and a confusion matrix.


## Reproducibility

The complete Subtask 2 path is CPU-only. After supplying the official parquet
files, run the commands above to regenerate the scores and prediction CSV.
