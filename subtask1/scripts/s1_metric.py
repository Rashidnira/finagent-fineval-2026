#!/usr/bin/env python
"""
Subtask 1 (Greek Financial NER) -- official-style metric harness.

PRIMARY METRIC (what the FinNLP-2026 leaderboard is documented to use):
    exact entity-level MICRO F1 over the newline-separated "entity, TYPE" pairs,
    compared as a MULTISET (bag) per row, TP/FP/FN summed globally.

    - gold legitimately repeats the same pair (e.g. "01, ΧΡΟΝΙΚΑ" twice for 01.01),
      so a set comparison would be wrong -> we use collections.Counter.
    - a literal "None" prediction (or empty cell / NaN) == the empty bag.
    - whitespace is normalised (NBSP -> space, collapse runs, strip);
      case and Greek accents are NOT normalised in the primary metric.

SECONDARY / DIAGNOSTIC METRICS
    * lenient        : casefold + strip Greek diacritics + drop spaces around
                       punctuation. Measures how much normalisation would buy.
    * type_insensitive : entity strings only, types ignored (upper bound on span
                       detection if typing were perfect).
    * span_only_micro (per-type breakdown also emitted).
    * plutus_seqeval : reproduction of the ORIGINAL Plutus/FinBen scorer
                       (The-FinAI/FinBen tasks/plutus/gr_utils.py):
                       both gold and pred answer strings are projected back onto
                       the passage as BIO tags by string search, then seqeval
                       entity F1 per row, MACRO-averaged (aggregation: mean).
                       This scorer is repetition-INSENSITIVE and order-insensitive.
                       Kept because the competition Space may reuse it.

CLI
    python s1_metric.py --pred preds.csv --gold validation.parquet
    python s1_metric.py --pred preds.csv --gold out/s1_dev_num.parquet --per-row misses.csv
"""
from __future__ import annotations

import argparse
import re
import string
import sys
import unicodedata
from collections import Counter, defaultdict

NUM_LABELS = ["ΧΡΗΜΑΤΑ", "ΠΟΣΟΣΤΑ", "ΧΡΟΝΙΚΑ", "ΠΟΣΟΤΗΤΕΣ", "ΑΛΛΑ"]
TEXT_LABELS = ["ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ"]
ALL_LABELS = set(NUM_LABELS + TEXT_LABELS)

_WS = re.compile(r"\s+")
NULL_TOKENS = {"none", "κανένα", "καμία", "kamia", "n/a", "-", "nan", ""}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def norm_ws(s: str) -> str:
    """Whitespace-only normalisation (primary metric)."""
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return _WS.sub(" ", s).strip()


def strip_accents(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in d if not unicodedata.combining(c)))


def norm_lenient(s: str) -> str:
    """Aggressive normalisation for the lenient variant."""
    s = norm_ws(s)
    s = strip_accents(s).casefold()
    # unify quote / dash / space-before-punct noise
    s = s.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+([.,;:%])", r"\1", s)
    s = s.rstrip(".,;: ")
    return s


def parse_answer(ans, lenient: bool = False):
    """
    Parse a newline-separated "entity, TYPE" block into a Counter of (entity, type).

    Type is taken as the text after the LAST comma (matches the original Plutus
    parser `val.split(", ")[-1]`), so entities that themselves contain commas
    (e.g. "4.387,15") survive as long as the type follows the final comma.
    A known-label check makes that robust even when a model appends junk.
    """
    bag = Counter()
    if ans is None:
        return bag
    s = str(ans)
    if s.lower().strip() in ("nan", "none"):
        return bag
    for raw in s.split("\n"):
        line = norm_ws(raw)
        if not line or line.lower() in NULL_TOKENS:
            continue
        if "," not in line:
            # malformed: no type -> counts as a wrong prediction with empty type
            ent, typ = line, ""
        else:
            head, tail = line.rsplit(",", 1)
            ent, typ = head.strip(), tail.strip()
            # tolerate trailing junk after the label ("2008, ΧΡΟΝΙΚΑ .")
            t_clean = typ.strip(" .;:'\"")
            if t_clean.upper() in ALL_LABELS:
                typ = t_clean.upper()
        if lenient:
            ent = norm_lenient(ent)
            typ = strip_accents(typ).casefold()
        if ent == "" and typ == "":
            continue
        bag[(ent, typ)] += 1
    return bag


# --------------------------------------------------------------------------- #
# micro F1 over multisets
# --------------------------------------------------------------------------- #
def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def micro_f1(golds, preds, lenient: bool = False, ignore_type: bool = False):
    """
    golds / preds: parallel iterables of raw answer strings.
    Returns dict with micro P/R/F1, counts, per-type breakdown, per-row rows.
    """
    tp = fp = fn = 0
    per_type = defaultdict(lambda: [0, 0, 0])  # type -> [tp, fp, fn]
    rows = []
    for g_raw, p_raw in zip(golds, preds):
        g = parse_answer(g_raw, lenient)
        p = parse_answer(p_raw, lenient)
        if ignore_type:
            g = Counter({(e, ""): c for (e, t), c in g.items()})
            p = Counter({(e, ""): c for (e, t), c in p.items()})
        inter = g & p          # multiset intersection == per-pair min(count)
        r_tp = sum(inter.values())
        r_fp = sum((p - g).values())
        r_fn = sum((g - p).values())
        tp += r_tp
        fp += r_fp
        fn += r_fn
        for (e, t), c in inter.items():
            per_type[t][0] += c
        for (e, t), c in (p - g).items():
            per_type[t][1] += c
        for (e, t), c in (g - p).items():
            per_type[t][2] += c
        rows.append(
            dict(tp=r_tp, fp=r_fp, fn=r_fn,
                 f1=prf(r_tp, r_fp, r_fn)[2],
                 missed="; ".join(f"{e}, {t}" for (e, t), c in (g - p).items() for _ in range(c)),
                 spurious="; ".join(f"{e}, {t}" for (e, t), c in (p - g).items() for _ in range(c)))
        )
    p, r, f = prf(tp, fp, fn)
    return dict(
        precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn,
        n_gold=tp + fn, n_pred=tp + fp,
        per_type={t: dict(zip(("precision", "recall", "f1"), prf(*v)),
                          tp=v[0], fp=v[1], fn=v[2], support=v[0] + v[2])
                  for t, v in sorted(per_type.items())},
        rows=rows,
    )


# --------------------------------------------------------------------------- #
# original Plutus / FinBen scorer (BIO projection + seqeval, macro mean)
# --------------------------------------------------------------------------- #
_SPLIT_RE = r"(\s+|[" + re.escape(string.punctuation).replace("%", "") + r"«»‘’“”€])"


def _plutus_process_text(entity_string: str, text: str):
    entity_list = [(", ".join(v.split(", ")[:-1]), v.split(", ")[-1])
                   for v in str(entity_string).split("\n")]
    text_words = list(filter(None, re.split(_SPLIT_RE, text)))
    labels = ["O"] * len(text_words)
    word_indices = [0]
    for w in text_words[:-1]:
        word_indices.append(word_indices[-1] + len(w))
    for entity, etype in entity_list:
        if not entity:
            continue
        start = 0
        while True:
            start = text.find(entity, start)
            if start == -1:
                break
            end = start + len(entity) - 1
            try:
                sw = next(i for i, ind in enumerate(word_indices) if ind >= start)
                ew = next(i for i, ind in enumerate(word_indices) if ind > end)
                labels[sw] = "B-" + etype
                for i in range(sw + 1, ew):
                    labels[i] = "I-" + etype
            except StopIteration:
                pass
            start = end + 1
    out = [l for w, l in zip(text_words, labels) if not re.search(_SPLIT_RE, w)]
    return out


def _bio_spans(tags):
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = [t[2:], i, i]
        elif t.startswith("I-"):
            if cur and cur[0] == t[2:]:
                cur[2] = i
            else:                      # seqeval default (IOB2-ish) -> new span
                if cur:
                    spans.append(cur)
                cur = [t[2:], i, i]
        else:
            if cur:
                spans.append(cur)
            cur = None
    if cur:
        spans.append(cur)
    return {(a, b, c) for a, b, c in spans}


def plutus_seqeval(golds, preds, texts):
    """Per-row entity F1 on BIO projections, MACRO-averaged (Plutus default)."""
    scores, tp_t = [], 0
    fp_t = fn_t = 0
    for g_raw, p_raw, txt in zip(golds, preds, texts):
        g_raw = "" if g_raw is None else str(g_raw)
        p_raw = "" if p_raw is None else str(p_raw)
        if p_raw.strip().lower() in ("none", "nan"):
            p_raw = ""
        gs = _bio_spans(_plutus_process_text(g_raw, txt))
        ps = _bio_spans(_plutus_process_text(p_raw, txt))
        tp = len(gs & ps)
        fp = len(ps - gs)
        fn = len(gs - ps)
        tp_t += tp
        fp_t += fp
        fn_t += fn
        scores.append(prf(tp, fp, fn)[2])
    macro = sum(scores) / len(scores) if scores else 0.0
    p, r, f = prf(tp_t, fp_t, fn_t)
    return dict(macro_f1=macro, micro_precision=p, micro_recall=r, micro_f1=f,
                tp=tp_t, fp=fp_t, fn=fn_t)


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def score_all(golds, preds, texts=None):
    out = {
        "primary_micro": {k: v for k, v in micro_f1(golds, preds).items() if k != "rows"},
        "lenient_micro": {k: v for k, v in micro_f1(golds, preds, lenient=True).items()
                          if k not in ("rows", "per_type")},
        "type_insensitive_micro": {k: v for k, v in
                                   micro_f1(golds, preds, ignore_type=True).items()
                                   if k not in ("rows", "per_type")},
    }
    if texts is not None:
        out["plutus_seqeval"] = plutus_seqeval(golds, preds, texts)
    return out


def _fmt(d, indent=0):
    pad = " " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_fmt(v, indent + 2))
        elif isinstance(v, float):
            lines.append(f"{pad}{k}: {v:.4f}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a subtask-1 predictions CSV against a gold parquet.")
    ap.add_argument("--pred", required=True, help="CSV with columns id,prediction (or gold-like parquet)")
    ap.add_argument("--gold", required=True, help="parquet with columns [id,] answer, text")
    ap.add_argument("--pred-col", default="prediction")
    ap.add_argument("--gold-col", default="answer")
    ap.add_argument("--per-row", default=None, help="optional CSV path for per-row TP/FP/FN + misses")
    a = ap.parse_args(argv)

    import pandas as pd

    gold = pd.read_parquet(a.gold)
    pred = pd.read_csv(a.pred, dtype=str, keep_default_na=False) if a.pred.endswith(".csv") \
        else pd.read_parquet(a.pred)

    if "id" in gold.columns and "id" in pred.columns:
        n_g = len(gold)
        merged = gold.merge(pred[["id", a.pred_col]], on="id", how="left")
        miss = merged[a.pred_col].isna().sum()
        if miss:
            print(f"WARNING: {miss}/{n_g} gold ids have no prediction (scored as empty)", file=sys.stderr)
        merged[a.pred_col] = merged[a.pred_col].fillna("")
        extra = set(pred["id"]) - set(gold["id"])
        if extra:
            print(f"WARNING: {len(extra)} prediction ids not in gold (ignored)", file=sys.stderr)
    else:
        if len(gold) != len(pred):
            sys.exit(f"row count mismatch and no id column: gold={len(gold)} pred={len(pred)}")
        merged = gold.copy()
        merged[a.pred_col] = pred[a.pred_col].values
        print("NOTE: no id column on both sides -> scored by row order", file=sys.stderr)

    texts = merged["text"].tolist() if "text" in merged.columns else None
    res = score_all(merged[a.gold_col].tolist(), merged[a.pred_col].tolist(), texts)
    print(f"rows scored: {len(merged)}")
    print(_fmt(res))

    if a.per_row:
        det = micro_f1(merged[a.gold_col].tolist(), merged[a.pred_col].tolist())["rows"]
        df = pd.DataFrame(det)
        if "id" in merged.columns:
            df.insert(0, "id", merged["id"].values)
        df.to_csv(a.per_row, index=False)
        print(f"per-row detail -> {a.per_row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
