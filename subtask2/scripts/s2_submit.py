#!/usr/bin/env python
"""Turn a test score JSON (from s2_rules.py) into the Subtask-2 submission CSV.

Submission format (SPEC.md): exactly 50 rows, columns id,prediction; prediction is exactly
one of +2,+1,0,-1,-2. Every test id 253..302 exactly once.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_calibrate import LABELS, load, softmax, sinkhorn, hungarian, cap_majority


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--stat", default="sum", choices=["sum", "mean"])
    ap.add_argument("--lam", type=float, default=1.0, help="contextual-calibration strength")
    ap.add_argument("--rule", default="sinkhorn",
                    choices=["argmax", "sinkhorn", "hungarian", "cap"])
    ap.add_argument("--strength", type=float, default=1.0, help="sinkhorn strength")
    ap.add_argument("--cap", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="CSV of analyst hypothesis for agreement stat")
    a = ap.parse_args()

    o, ids, gold, S, b = load(a.scores, a.stat)
    L = S - a.lam * b
    if a.rule == "argmax":
        idx = L.argmax(1)
    elif a.rule == "sinkhorn":
        idx = sinkhorn(softmax(L), np.full(5, 0.2), strength=a.strength).argmax(1)
    elif a.rule == "hungarian":
        idx = hungarian(L, len(ids) // 5)
    else:
        idx = cap_majority(L, a.cap)
    pred = [LABELS[i] for i in idx]

    df = pd.DataFrame({"id": ids, "prediction": pred})
    assert len(df) == 50, f"expected 50 rows, got {len(df)}"
    assert df["id"].nunique() == 50 and set(df["id"]) == set(range(253, 303)), "id coverage broken"
    assert set(df["prediction"]) <= set(LABELS), "bad label emitted"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)

    hist = {l: int(sum(p == l for p in pred)) for l in LABELS}
    print(f"wrote {a.out}  rule={a.rule} lam={a.lam} stat={a.stat} strength={a.strength}")
    print("predicted histogram  " + "  ".join(f"{l}:{hist[l]}" for l in LABELS))
    if a.compare and os.path.exists(a.compare):
        h = pd.read_csv(a.compare, dtype=str)
        col = "prediction" if "prediction" in h.columns else h.columns[1]
        m = df.astype(str).merge(h.astype(str)[["id", col]], on="id")
        agree = (m["prediction"] == m[col]).mean()
        print(f"agreement with analyst hypothesis ({a.compare}): {agree:.3f}  ({int((m['prediction']==m[col]).sum())}/{len(m)})")
        dis = m[m["prediction"] != m[col]]
        print("disagreements id(model/analyst): " +
              ", ".join(f"{r.id}({r.prediction}/{getattr(r, col)})" for r in dis.itertuples()))


if __name__ == "__main__":
    main()
