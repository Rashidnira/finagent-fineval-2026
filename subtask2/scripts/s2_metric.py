#!/usr/bin/env python
"""
Subtask 2 (Japanese ICR) scorer.

Usage:
  s2_metric.py --pred preds.csv --gold gold.parquet [--gold-col answer] [--json out.json]

pred CSV : columns id,prediction   (prediction in {+2,+1,0,-1,-2})
gold     : parquet with columns id, <gold-col>

Prints:
  * exact-label accuracy          (the official metric)
  * MACRO-RECALL                  (primary dev statistic: unbiased proxy for accuracy
                                   on the balanced 10-per-label hidden test set)
  * per-class precision / recall / F1 / support
  * confusion matrix (rows = gold, cols = pred)
  * gold vs predicted label distribution  (prior-collapse detector)
"""
import argparse, json, sys
import pandas as pd

LABELS = ["+2", "+1", "0", "-1", "-2"]

_ALIAS = {"+2": "+2", "2": "+2", "＋2": "+2", "+1": "+1", "1": "+1", "＋1": "+1",
          "0": "0", "-0": "0", "０": "0", "-1": "-1", "−1": "-1", "ー1": "-1",
          "-2": "-2", "−2": "-2", "ー2": "-2"}


def norm(x):
    """Normalise a raw prediction string to a canonical label, else None."""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip("'").strip()
    s = s.replace("＋", "+").replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("０", "0").replace("１", "1").replace("２", "2")
    if s in _ALIAS:
        return _ALIAS[s]
    # last resort: first label-looking token anywhere in the string
    for lab in ["+2", "-2", "+1", "-1"]:
        if lab in s:
            return lab
    if s and s[0] == "0":
        return "0"
    return None


def score(pred_df, gold_df, gold_col="answer", pred_col="prediction", verbose=True):
    g = gold_df[["id", gold_col]].copy()
    g["gold"] = g[gold_col].map(norm)
    p = pred_df[["id", pred_col]].copy()
    p["pred_raw"] = p[pred_col]
    p["pred"] = p[pred_col].map(norm)

    g["id"] = g["id"].astype(str)
    p["id"] = p["id"].astype(str)

    missing = set(g["id"]) - set(p["id"])
    extra = set(p["id"]) - set(g["id"])
    dupes = p["id"][p["id"].duplicated()].tolist()

    m = g.merge(p[["id", "pred", "pred_raw"]], on="id", how="left")
    n_unparsed = int(m["pred"].isna().sum())
    m["pred"] = m["pred"].fillna("<UNPARSED>")

    n = len(m)
    correct = int((m["pred"] == m["gold"]).sum())
    acc = correct / n if n else 0.0

    rows, recalls = [], []
    for lab in LABELS:
        sup = int((m["gold"] == lab).sum())
        tp = int(((m["gold"] == lab) & (m["pred"] == lab)).sum())
        pp = int((m["pred"] == lab).sum())
        rec = tp / sup if sup else float("nan")
        prec = tp / pp if pp else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (sup and pp and (prec + rec) > 0) else 0.0
        rows.append((lab, sup, pp, prec, rec, f1))
        if sup:
            recalls.append(rec)
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(r[5] for r in rows if r[1]) / max(1, sum(1 for r in rows if r[1]))

    if verbose:
        print(f"n scored              : {n}")
        print(f"exact-label ACCURACY  : {acc:.4f}  ({correct}/{n})")
        print(f"MACRO-RECALL (primary): {macro_recall:.4f}   <- unbiased proxy for balanced-test accuracy")
        print(f"macro-F1              : {macro_f1:.4f}")
        if n_unparsed:
            print(f"!! UNPARSEABLE predictions: {n_unparsed}")
        if missing:
            print(f"!! ids in gold missing from pred: {len(missing)} e.g. {sorted(missing)[:5]}")
        if extra:
            print(f"!! ids in pred not in gold: {len(extra)} e.g. {sorted(extra)[:5]}")
        if dupes:
            print(f"!! duplicate ids in pred: {len(dupes)} e.g. {dupes[:5]}")

        print("\nper-class")
        print(f"{'label':>6} {'supp':>5} {'npred':>6} {'prec':>7} {'recall':>7} {'F1':>7}")
        for lab, sup, pp, prec, rec, f1 in rows:
            fmt = lambda v: "  n/a  " if v != v else f"{v:7.3f}"
            print(f"{lab:>6} {sup:5d} {pp:6d} {fmt(prec)} {fmt(rec)} {f1:7.3f}")

        cols = LABELS + (["<UNPARSED>"] if n_unparsed else [])
        print("\nconfusion matrix  (rows = GOLD, cols = PRED)")
        print(f"{'':>6}" + "".join(f"{c:>11}" for c in cols))
        for gl in LABELS:
            line = f"{gl:>6}"
            for pl in cols:
                line += f"{int(((m['gold'] == gl) & (m['pred'] == pl)).sum()):>11}"
            print(line)

        print("\nlabel distribution (gold -> pred)")
        for lab in cols:
            print(f"{lab:>6}  gold {int((m['gold']==lab).sum()):3d}   pred {int((m['pred']==lab).sum()):3d}")

    return {"n": n, "accuracy": acc, "macro_recall": macro_recall, "macro_f1": macro_f1,
            "n_unparsed": n_unparsed, "n_missing": len(missing), "n_extra": len(extra),
            "per_class": {r[0]: {"support": r[1], "n_pred": r[2], "precision": r[3],
                                 "recall": r[4], "f1": r[5]} for r in rows}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions CSV with columns id,prediction")
    ap.add_argument("--gold", required=True, help="gold parquet (or csv) with id + label column")
    ap.add_argument("--gold-col", default="answer")
    ap.add_argument("--pred-col", default="prediction")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pred = pd.read_csv(a.pred, dtype=str, keep_default_na=False)
    gold = pd.read_parquet(a.gold) if a.gold.endswith(".parquet") else pd.read_csv(a.gold, dtype=str)
    for c in (a.pred_col,):
        if c not in pred.columns:
            sys.exit(f"pred CSV missing column '{c}'; has {list(pred.columns)}")
    res = score(pred, gold, a.gold_col, a.pred_col)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
