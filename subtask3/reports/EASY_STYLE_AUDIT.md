# EASY GOLD STRUCTURAL AUDIT (Phase 3)

76 answers; question types: {'Revenue': 19, 'BalanceSheet': 19, 'CashFlow': 19, 'RnD': 19}

## 3.1-3.2 Global token-mass decomposition (Scorer B tokens)
- total tokens: 4884
- numeric: 1075 (22.0%)
- non-Latin: 847 (17.3%)
- labels: 286 (5.9%)
- Answer-section words: 2449 (67.3%); Evidence-section words: 1189 (32.7%)

## 3.3 Per-question-type distributions

**words**
| qtype        |   mean |   50% |   25% |   75% |   min |   max |
|:-------------|-------:|------:|------:|------:|------:|------:|
| BalanceSheet |   36   |    33 |  27.5 |  43   |    15 |    68 |
| CashFlow     |   60.9 |    60 |  51.5 |  72   |     5 |   118 |
| Revenue      |   76.1 |    77 |  47   | 100.5 |    31 |   142 |
| RnD          |   18.5 |    14 |   5   |  21   |     5 |    60 |

**answer_sec_words**
| qtype        |   mean |   50% |   25% |   75% |   min |   max |
|:-------------|-------:|------:|------:|------:|------:|------:|
| BalanceSheet |   31.3 |    28 |  24.5 |  35.5 |    12 |    65 |
| CashFlow     |   44.8 |    47 |  41.5 |  53.5 |     2 |    61 |
| Revenue      |   38.8 |    36 |  31.5 |  40.5 |    18 |    97 |
| RnD          |   13.9 |    10 |   2   |  19   |     2 |    60 |

**evidence_sec_words**
| qtype        |   mean |   50% |   25% |   75% |   min |   max |
|:-------------|-------:|------:|------:|------:|------:|------:|
| BalanceSheet |    4.7 |     3 |   3   |   3   |     0 |    23 |
| CashFlow     |   16.1 |     3 |   3   |  20.5 |     0 |    74 |
| Revenue      |   37.3 |    46 |  11.5 |  56.5 |     0 |   101 |
| RnD          |    4.5 |     3 |   3   |   3   |     0 |    29 |

**numeric_tokens**
| qtype        |   mean |   50% |   25% |   75% |   min |   max |
|:-------------|-------:|------:|------:|------:|------:|------:|
| BalanceSheet |   11.1 |    11 |     5 |  15   |     4 |    21 |
| CashFlow     |   17.2 |    18 |    11 |  20.5 |     0 |    51 |
| Revenue      |   24.1 |    26 |    14 |  33   |     4 |    40 |
| RnD          |    4.3 |     4 |     0 |   8   |     0 |    16 |

**nonlatin_tokens**
| qtype        |   mean |   50% |   25% |   75% |   min |   max |
|:-------------|-------:|------:|------:|------:|------:|------:|
| BalanceSheet |    5.5 |     0 |     0 |   0   |     0 |    56 |
| CashFlow     |    4.9 |     0 |     0 |   0   |     0 |    44 |
| Revenue      |   32.6 |    25 |     9 |  49.5 |     0 |   110 |
| RnD          |    1.6 |     0 |     0 |   0   |     0 |    23 |

## 3.4 Evidence = None stratification
| qtype        | evidence_none   |   n |   mean_words |
|:-------------|:----------------|----:|-------------:|
| BalanceSheet | False           |   4 |         50   |
| BalanceSheet | True            |  15 |         32.3 |
| CashFlow     | False           |  11 |         69.1 |
| CashFlow     | True            |   8 |         49.6 |
| Revenue      | False           |  18 |         78.4 |
| Revenue      | True            |   1 |         35   |
| RnD          | False           |   4 |         37.5 |
| RnD          | True            |  15 |         13.4 |

Overall Evidence=None: 39/76

## Length-calibration recommendation (updates the 40-60 prior)
| qtype        |   mean |   median |   q25 |   q75 |
|:-------------|-------:|---------:|------:|------:|
| BalanceSheet |     36 |       33 |    28 |    43 |
| CashFlow     |     61 |       60 |    52 |    72 |
| Revenue      |     76 |       77 |    47 |   100 |
| RnD          |     18 |       14 |     5 |    21 |

Target band per type = [q25, q75] above; global prior 40-60 retained only for types whose IQR overlaps it.
