# SCORER PROVENANCE — Phase 2

## Provenance hierarchy search results

1. **Exact FinNLP 2026 competition scorer** — NOT AVAILABLE. Lives in the
   private competition repo (huggingface/competitions framework; only
   `/submission_info` etc. are public). Space commit `34ece895` contains no
   scoring code.
2. **Scorer distributed with the Space/shared task** — NONE (Space repo =
   Dockerfile + README only).
3. **Official MultiFinBen / PolyFiQA scorer** — FOUND. Repo
   `xueqingpeng/MultiFinBen`, default branch `main`, HEAD commit
   `a3fc54082d0bcc6f9cc4e3491222a9b405ee35db` (fetched 2026-08-18), file
   `tasks/multilingual/ml_utils.py` (archived at
   `data/manifests/multifinben_ml_utils.py`). Computes
   `evaluate.load("rouge").compute(...)["rouge1"]` = google `rouge_score`
   defaults.
4. **Published FinMMEval scoring specification** — FOUND. arXiv:2607.19867
   (FinMMEval 2026 Task 2 overview): Unicode NFKC normalization, lowercasing,
   regex tokenizing "words, numbers, CJK characters, Japanese kana,
   Arabic-script characters, and Greek/Latin ranges"; item-level unigram
   multiset P/R, F1 = 2PR/(P+R), **F1 = 0 when P+R = 0**; leaderboard =
   macro-average of item-level F1; the scorer consumes "the complete string
   stored in each answer field, including section labels and evidence text."
   Exact regex NOT published — our Scorer B is a reconstruction.
5. **Local reconstruction** — both candidates implemented in
   `src/tokenizer.py` + `src/scorer.py`.

## Implemented candidate scorers

| | Scorer A `multifinben_default` | Scorer B `finmmeval_multiscript` |
|---|---|---|
| Normalization | lowercase | NFKC + lowercase |
| Tokenization | split on non-`[a-z0-9]` (rouge_score default, no stemming) | Han/Hiragana/Katakana as single-char tokens; Latin/Greek/Cyrillic/Arabic/digit runs as words |
| CJK / Greek text | **deleted (metric-invisible)** | counted (CJK per character) |
| Numbers | `$35.0B`→`35,0b`; `14.6%`→`14,6`; `7,059`→`7,059m` split | identical to A (verified) |
| P/R/F1 | unigram multiset overlap, F1=2PR/(P+R), 0 when P+R=0 | identical formulas |
| Aggregation | macro-average of item F1 | macro-average of item F1 |

Punctuation, `$`, `%`, `.`, `,` are token separators under BOTH scorers;
currency symbols and percent signs are never tokens. Decimal numbers split at
the point (`14.6` → `14`, `6`), thousands separators split at the comma —
**copy figures exactly as printed in the reference-style wording; formatting
differences change unigrams.**

## Validation performed (tests/test_scorer.py — 9 passed)

- Identity: gold vs itself = P/R/F1 1.0 under both.
- Analytical: "revenue increased 20 percent" vs "revenue increased" →
  P=1, R=0.5, F1=2/3 (hand-computed), plus a multiset-clipping case.
- Fidelity: Scorer A matches `rouge_score.RougeScorer(["rouge1"],
  use_stemmer=False)` to 1e-9 on mixed-script cases (the reference package
  MultiFinBen's `evaluate` wrapper calls).
- Edge: empty prediction → 0.0, never NaN.
- **CJK-quote policy case (user-mandated):** prediction with vs without an
  original-language Chinese quote, reference containing that quote.

## Resolution of the quoting-policy ambiguity (pre-Phase 5 requirement)

Empirical facts (76 Easy gold answers):

- 26/76 gold answers contain non-Latin tokens; mean non-Latin share of
  Scorer-B token mass = **10.3%** (max 115 tokens in one answer).
- Simulated policy "drop the News Evidence section" vs full gold answer:
  F1 −0.105 under Scorer A, −0.150 under Scorer B.
- Under Scorer A a CJK quote contributes only its embedded numerals, but
  **cannot hurt**: CJK characters are invisible in BOTH prediction and
  reference, so they never enter the precision denominator.
- Under Scorer B the same quote is strictly rewarded (test case: +>0.05 F1).

**Conclusion: original-language quoting is weakly dominant under Scorer A and
strictly beneficial under Scorer B. The quoting policy does NOT flip on the
unresolved scorer identity — quote in the original language, never translate
quotes** (translation would add English tokens absent from the reference,
hurting precision under both scorers). The remaining scorer uncertainty
affects only the *measured level* of dev scores, not system ranking policy;
per the disagreement rule, all Phase 5 experiments report both scorers and
flag any conclusion that changes direction between them.

## Expert-tier note

Expert instructions require quoting *financial statements* (English/numeric)
rather than news, so non-Latin mass in test references is expected to be lower
than the 10.3% measured on Easy. The dominance argument is unchanged.
