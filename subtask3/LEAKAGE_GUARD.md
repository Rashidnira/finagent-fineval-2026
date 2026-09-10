# LEAKAGE GUARD — FinNLP 2026 Subtask 3 (PolyFiQA)

## Identified leakage risk

The FinNLP 2026 Subtask 3 hidden test set is derived from the PolyFiQA-Expert
benchmark. A public Hugging Face dataset (`TheFinAI/PolyFiQA-Expert`) is known
to contain Expert-style questions **with reference answers** that overlap the
competition test set. Public web copies (papers, mirrors, benchmark repos) may
also contain those reference answers.

## Forbidden sources (never downloaded, inspected, queried, joined, or used)

- `TheFinAI/PolyFiQA-Expert` (any revision, any mirror)
- `TheFinAI/PolyFiQA-Easy` beyond what the organizers themselves re-published
  as the official train split (see below)
- Any private organizer variants (e.g. `*-July` refreshes)
- Any repository, paper appendix, dataset viewer, SQL console, or web search
  result exposing Expert reference answers
- Use of such answers for: prompt tuning, few-shot examples, validation,
  scoring, candidate selection, or test prediction generation

## Safeguards implemented

1. **Quarantine deletion (2026-08-18):** files previously downloaded to a
   temporary scratchpad during an earlier exploratory session
   (`expert.parquet`, `st_test_with_expert_answers.parquet`, `easy.parquet`,
   `st_train.parquet`, `st_test.parquet`) were deleted before any content was
   read into this pipeline. No Expert reference answer text was inspected,
   printed, or stored in this project.
2. **Allowlisted data sources only.** The pipeline reads exclusively from:
   - `data/raw/public-00000-of-00001.parquet` (organizer-supplied train)
   - `data/raw/PolyFiQA_test_participant.parquet` (organizer-supplied test)
   - `data/manifests/official_finnlp_train.parquet` and
     `official_finnlp_test.parquet`, downloaded from the official competition
     dataset `FinNLP-Multilingual-Understanding/finnlp2026-subtask3-polyfiqa`
     (commit `8377c6e8e2d10279f1e857e37ceab3ee313104ac`). The official train
     file is byte-identical (SHA256) to the supplied train file; the official
     test file contains no answer column.
3. **No fetch code targeting forbidden repos.** No script in `src/` references
   `TheFinAI` or any non-official dataset repository.
4. **Validation policy.** All system selection uses the 76 Easy/public labeled
   examples (with gold answers hidden from the model at generation time).
   Expert/test predictions are validated only by grounding checks against the
   supplied context, never against any external reference answer.

## Allowed sources used

- Organizer-supplied train/test parquet files and the official competition
  dataset repo (documented above)
- Official competition Space endpoints (`/competition_info`, `/dataset_info`,
  `/submission_info`, `/rules`) and the official shared-task website
- PolyFiQA / MultiFinBen / FinMMEval methodology papers and prior shared-task
  system papers (methods only, no reference answers)
- General financial NLP literature and pretrained-model knowledge

## Statement for the system paper

> No PolyFiQA-Expert reference answers were used at any stage. All model
> selection was performed on the organizer-released public (Easy) split;
> hidden-test predictions were validated exclusively by automated grounding
> checks against the supplied context. Publicly available copies of Expert
> reference answers were identified as a leakage hazard, quarantined, and
> deleted; the safeguard is documented in the project's LEAKAGE_GUARD.md.
