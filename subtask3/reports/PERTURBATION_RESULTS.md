# PERTURBATION RESULTS (Phase 4 — metric diagnostic only)

Prediction = perturbed gold, reference = original gold. Interpretation is limited to: which lexical properties matter in the references. It does NOT rank generation systems (see §0.3 of the project spec).

## Overall (macro means)
| variant              | scorer                |   precision |   recall |     f1 |   delta_f1 |
|:---------------------|:----------------------|------------:|---------:|-------:|-----------:|
| A_original           | finmmeval_multiscript |      1      |   1      | 1      |     0      |
| A_original           | multifinben_default   |      1      |   1      | 1      |     0      |
| B_labels_removed     | finmmeval_multiscript |      1      |   0.8914 | 0.9312 |    -0.0688 |
| B_labels_removed     | multifinben_default   |      1      |   0.8869 | 0.9288 |    -0.0712 |
| C_evidence_removed   | finmmeval_multiscript |      1      |   0.6801 | 0.7778 |    -0.2222 |
| C_evidence_removed   | multifinben_default   |      1      |   0.7326 | 0.8234 |    -0.1766 |
| D_numbers_removed    | finmmeval_multiscript |      1      |   0.7262 | 0.8271 |    -0.1729 |
| D_numbers_removed    | multifinben_default   |      1      |   0.7648 | 0.8609 |    -0.1391 |
| E_numbers_normalized | finmmeval_multiscript |      0.9858 |   0.972  | 0.9786 |    -0.0214 |
| E_numbers_normalized | multifinben_default   |      0.9849 |   0.9702 | 0.9772 |    -0.0228 |
| F_nonenglish_removed | finmmeval_multiscript |      1      |   0.8785 | 0.9161 |    -0.0839 |
| F_nonenglish_removed | multifinben_default   |      1      |   0.9462 | 0.9677 |    -0.0323 |

## F1 by question type (per scorer)

**multifinben_default**
| variant              |   BalanceSheet |   CashFlow |   Revenue |    RnD |
|:---------------------|---------------:|-----------:|----------:|-------:|
| A_original           |         1      |     1      |    1      | 1      |
| B_labels_removed     |         0.9654 |     0.9589 |    0.9827 | 0.8084 |
| C_evidence_removed   |         0.9428 |     0.8647 |    0.6944 | 0.7917 |
| D_numbers_removed    |         0.849  |     0.8435 |    0.8384 | 0.9126 |
| E_numbers_normalized |         0.9588 |     0.9743 |    0.9838 | 0.992  |
| F_nonenglish_removed |         0.9909 |     0.9777 |    0.9197 | 0.9825 |

**finmmeval_multiscript**
| variant              |   BalanceSheet |   CashFlow |   Revenue |    RnD |
|:---------------------|---------------:|-----------:|----------:|-------:|
| A_original           |         1      |     1      |    1      | 1      |
| B_labels_removed     |         0.9668 |     0.9609 |    0.9866 | 0.8105 |
| C_evidence_removed   |         0.9108 |     0.832  |    0.5987 | 0.7695 |
| D_numbers_removed    |         0.8274 |     0.8377 |    0.7223 | 0.9212 |
| E_numbers_normalized |         0.9588 |     0.9741 |    0.9874 | 0.9941 |
| F_nonenglish_removed |         0.959  |     0.943  |    0.8024 | 0.9603 |

## Evidence stratification (variants C and F, F1)
|                                                   |   evidence_present |   evidence_none |
|:--------------------------------------------------|-------------------:|----------------:|
| ('C_evidence_removed', 'finmmeval_multiscript')   |             0.6152 |          0.8898 |
| ('C_evidence_removed', 'multifinben_default')     |             0.727  |          0.8898 |
| ('F_nonenglish_removed', 'finmmeval_multiscript') |             0.7944 |          1      |
| ('F_nonenglish_removed', 'multifinben_default')   |             0.9208 |          1      |
