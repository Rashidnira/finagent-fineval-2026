# FinNLP 2026 Subtask 3 — PolyFiQA competition pipeline

Reproducible pipeline for the Multilingual Financial Q&A shared task
(FinNLP @ EMNLP-2026). See `config.yaml` for competition facts with
provenance, `LEAKAGE_GUARD.md` for the data-integrity policy, and
`reports/DATA_AUDIT.md` for the verified dataset structure.

## Status

- [x] Phase 0 — rules, submission schema, official test file, environment
- [x] Phase 1 — full data audit
- [x] Phase 2 — scorer verification (both candidates; quoting ambiguity bounded)
- [x] Phase 3 — Easy gold structural audit (per-type length bands)
- [x] Phase 4 — mechanical ROUGE perturbations (metric diagnostics)
- [ ] Phase 5 — Easy benchmark (System C grouped, Gemma 4 31B) — **BLOCKED on GEMINI_API_KEY (free Google AI Studio key)**
- [ ] Phase 7–8 — Expert generation + validation — prompts + runner + validator ready
- [ ] Phase 10 — validated submission checkpoint — builder + integrity tests ready

## Commands

```bash
python -m src.data_audit                                  # Phase 1 audit
python -m src.structural_audit                            # Phase 3
python -m src.perturbations                               # Phase 4
python -m src.grouped_runner --dataset easy --system C --count-tokens-only  # token check
python -m src.grouped_runner --dataset easy --system C --smoke MSFT_20200429  # smoke test
python -m src.grouped_runner --dataset easy --system C     # Phase 5 (19 calls)
python -m src.grouped_runner --dataset expert --system C   # Phase 7 (19 calls)
python -m src.submission --predictions <preds.csv> --tag baseline  # Phase 10
python -m pytest tests/ -q                                # 26 tests
```

## Layout

- `data/raw/` — organizer-supplied parquet files. **Read-only.**
- `data/manifests/` — official downloads (with commit hashes) + human-saved
  official materials; `official_finnlp_test.parquet` is the submission master.
- `src/` — pipeline modules; `reports/` — audit and experiment reports;
  `cache/llm/` — canonical cached LLM responses; `outputs/submission/` —
  immutable validated submission checkpoints.
