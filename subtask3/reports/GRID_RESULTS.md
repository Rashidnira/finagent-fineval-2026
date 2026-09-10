# Retrieval x granularity grid, PolyFiQA-Easy (5 seeds)

Temperature 0.7, top-p 0.8, top-k 20, one fixed seed per run. All four cells use the same raw context, the same model and quantization, the same output limit, the same computing backend and the same scoring code. No rewriting or normalization is applied.

This grid is a new controlled analysis. It ran with a larger context window and a different compute backend from the submitted system, so it is not a reproduction of the submitted runs; see reports/GRID_PROVENANCE.md. Using one environment for all four cells supports internally consistent comparisons.

## ROUGE-1 F1, default scorer

| Configuration | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | sd |
|---|---|---|---|---|---|---|---|
| Grouped, no retrieval | 0.4361 | 0.4678 | 0.4722 | 0.4456 | 0.4518 | **0.4547** | 0.0151 |
| Grouped, retrieval | 0.4589 | 0.4356 | 0.4498 | 0.4606 | 0.4574 | **0.4525** | 0.0103 |
| Per-question, no retrieval | 0.4654 | 0.4517 | 0.4552 | 0.4570 | 0.4528 | **0.4564** | 0.0054 |
| Per-question, retrieval | 0.4607 | 0.4566 | 0.4480 | 0.4663 | 0.4479 | **0.4559** | 0.0080 |

### Paired effects, computed within each seed (n = 5 seeds)

| Effect | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | sd |
|---|---|---|---|---|---|---|---|
| Retrieval, under grouped generation | +0.0228 | -0.0322 | -0.0224 | +0.0150 | +0.0056 | **-0.0022** | 0.0239 |
| Retrieval, under per-question generation | -0.0047 | +0.0049 | -0.0072 | +0.0093 | -0.0049 | **-0.0005** | 0.0072 |
| Per-question generation, without retrieval | +0.0293 | -0.0161 | -0.0170 | +0.0114 | +0.0010 | **+0.0017** | 0.0195 |
| Per-question generation, with retrieval | +0.0018 | +0.0210 | -0.0018 | +0.0057 | -0.0095 | **+0.0034** | 0.0113 |
| Interaction | -0.0275 | +0.0371 | +0.0152 | -0.0057 | -0.0105 | **+0.0017** | 0.0250 |

- The estimated effect of retrieval, under grouped generation was -0.0022, smaller than the variation across the 5 runs (sd 0.0239).
- The estimated effect of retrieval, under per-question generation was -0.0005, smaller than the variation across the 5 runs (sd 0.0072).
- The estimated effect of per-question generation, without retrieval was +0.0017, smaller than the variation across the 5 runs (sd 0.0195).
- The estimated effect of per-question generation, with retrieval was +0.0034, smaller than the variation across the 5 runs (sd 0.0113).

With 5 seeds these figures describe the spread observed across runs. They are not a significance test, and a small estimate should be read as an effect we could not resolve at this sample size rather than as an absence of one.

## ROUGE-1 F1, multiscript scorer

| Configuration | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | sd |
|---|---|---|---|---|---|---|---|
| Grouped, no retrieval | 0.4209 | 0.4539 | 0.4646 | 0.4236 | 0.4350 | **0.4396** | 0.0191 |
| Grouped, retrieval | 0.4458 | 0.4229 | 0.4371 | 0.4454 | 0.4444 | **0.4391** | 0.0097 |
| Per-question, no retrieval | 0.4531 | 0.4378 | 0.4364 | 0.4303 | 0.4341 | **0.4383** | 0.0087 |
| Per-question, retrieval | 0.4412 | 0.4387 | 0.4325 | 0.4560 | 0.4348 | **0.4406** | 0.0092 |

### Paired effects, computed within each seed (n = 5 seeds)

| Effect | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | sd |
|---|---|---|---|---|---|---|---|
| Retrieval, under grouped generation | +0.0249 | -0.0310 | -0.0275 | +0.0218 | +0.0094 | **-0.0005** | 0.0269 |
| Retrieval, under per-question generation | -0.0119 | +0.0009 | -0.0039 | +0.0257 | +0.0007 | **+0.0023** | 0.0141 |
| Per-question generation, without retrieval | +0.0322 | -0.0161 | -0.0282 | +0.0067 | -0.0009 | **-0.0013** | 0.0231 |
| Per-question generation, with retrieval | -0.0046 | +0.0158 | -0.0046 | +0.0106 | -0.0096 | **+0.0015** | 0.0110 |
| Interaction | -0.0368 | +0.0319 | +0.0236 | +0.0039 | -0.0087 | **+0.0028** | 0.0273 |

- The estimated effect of retrieval, under grouped generation was -0.0005, smaller than the variation across the 5 runs (sd 0.0269).
- The estimated effect of retrieval, under per-question generation was +0.0023, smaller than the variation across the 5 runs (sd 0.0141).
- The estimated effect of per-question generation, without retrieval was -0.0013, smaller than the variation across the 5 runs (sd 0.0231).
- The estimated effect of per-question generation, with retrieval was +0.0015, smaller than the variation across the 5 runs (sd 0.0110).

With 5 seeds these figures describe the spread observed across runs. They are not a significance test, and a small estimate should be read as an effect we could not resolve at this sample size rather than as an absence of one.

## Answer length

| Configuration | mean words | ratio to gold |
|---|---|---|
| Gold answers | 47.9 | 1.00 |
| Grouped, no retrieval | 33.3 | 0.70 |
| Grouped, retrieval | 35.3 | 0.74 |
| Per-question, no retrieval | 51.6 | 1.08 |
| Per-question, retrieval | 55.8 | 1.16 |

## Per question type, default scorer, mean across seeds

| Configuration | Revenue | BalanceSheet | CashFlow | RnD |
|---|---|---|---|---|
| Grouped, no retrieval | 0.422 | 0.460 | 0.312 | 0.625 |
| Grouped, retrieval | 0.420 | 0.464 | 0.309 | 0.617 |
| Per-question, no retrieval | 0.424 | 0.483 | 0.340 | 0.578 |
| Per-question, retrieval | 0.442 | 0.482 | 0.308 | 0.591 |

