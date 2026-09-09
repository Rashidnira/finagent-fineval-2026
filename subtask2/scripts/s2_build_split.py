import os
#!/usr/bin/env python
"""
Build the BALANCED dev split + few-shot pool for Subtask 2.

Why balanced: train prior is +1:139 / 0:72 / +2:29 / -1:11 / -2:2, but the hidden test
set is exactly 10 per label. Accuracy on a random dev split rewards the majority prior and
is therefore meaningless. We sample equal counts per label so that dev accuracy == dev
macro-recall == an unbiased estimate of balanced-test accuracy.

Outputs:
  out/s2_dev_balanced.parquet   (id, query, answer, Q, R, resp_len)
  out/s2_pool.parquet           (same cols; everything not in dev -> few-shot pool)
  out/s2_train_parsed.parquet   (all 253 rows, parsed into Q/R)
  out/s2_test_parsed.parquet    (all 50 test rows, parsed into Q/R)
"""
import re, sys
import pandas as pd

ROOT = os.environ.get("FINNLP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = f"{ROOT}/data/hf/finnlp2026-subtask2-japanese-icr"
OUT = f"{ROOT}/out"
SEED = 20260818
LABELS = ["+2", "+1", "0", "-1", "-2"]
# n per class in dev. -2 has only 2 examples total -> take both.
# -1 has 11 -> take 6, leaving 5 real -1 exemplars in the few-shot pool.
DEV_N = {"+2": 6, "+1": 6, "0": 6, "-1": 6, "-2": 2}

PAT = re.compile(r"Financial Question:\s*(.*?)\nCompany Response:\s*(.*?)\n\nDirectly output", re.S)


def parse(q):
    m = PAT.search(q)
    if not m:
        return pd.Series(["", ""])
    return pd.Series([m.group(1).strip(), m.group(2).strip()])


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_parquet(f"{DATA}/train.parquet")
    te = pd.read_parquet(f"{DATA}/test.parquet")
    for df in (tr, te):
        df[["Q", "R"]] = df["query"].apply(parse)
        df["resp_len"] = df["R"].str.len()
        df["q_len"] = df["Q"].str.len()

    tr.to_parquet(f"{OUT}/s2_train_parsed.parquet", index=False)
    te.to_parquet(f"{OUT}/s2_test_parsed.parquet", index=False)

    dev_parts = []
    for lab in LABELS:
        sub = tr[tr["answer"] == lab]
        k = min(DEV_N[lab], len(sub))
        dev_parts.append(sub.sample(n=k, random_state=SEED))
    dev = pd.concat(dev_parts).sort_values("id").reset_index(drop=True)
    pool = tr[~tr["id"].isin(set(dev["id"]))].sort_values("id").reset_index(drop=True)

    dev.to_parquet(f"{OUT}/s2_dev_balanced.parquet", index=False)
    pool.to_parquet(f"{OUT}/s2_pool.parquet", index=False)

    print("DEV  n=%d" % len(dev))
    print(dev["answer"].value_counts().reindex(LABELS).to_string())
    print("\nPOOL n=%d" % len(pool))
    print(pool["answer"].value_counts().reindex(LABELS).fillna(0).astype(int).to_string())
    print("\nresp_len median: train=%.0f dev=%.0f TEST=%.0f (test range %d-%d)"
          % (tr.resp_len.median(), dev.resp_len.median(), te.resp_len.median(),
             te.resp_len.min(), te.resp_len.max()))
    print("dev ids:", list(dev["id"]))
    # standard error of a per-class recall estimate at n per class
    for lab in LABELS:
        n = int((dev.answer == lab).sum())
        print(f"  {lab}: n={n}  max SE of recall = {0.5/ (n**0.5):.3f}  "
              f"(granularity 1/{n} = {1/n:.3f})")


if __name__ == "__main__":
    main()
