"""Shared utilities for the subtask-1 TEXTUAL alternative-methods pass.

No gold labels for gr_text_* exist anywhere. Nothing here computes an F1 against
gold; every function is a *proxy* diagnostic or an agreement measure.
"""
import os, re, json, unicodedata
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.environ.get("FINNLP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = os.path.join(ROOT, "data/hf/finnlp2026-subtask1-greek-ner/test.parquet")
TYPES = ("ΠΡΟΣΩΠΟ", "ΟΡΓΑΝΙΣΜΟΣ", "ΤΟΠΟΘΕΣΙΑ")


def load_passages():
    t = pd.read_parquet(TEST)
    t = t[t.id.str.startswith("gr_text_")].reset_index(drop=True)
    return {r.id: r.text for r in t.itertuples()}


def parse_pred_csv(path):
    """-> {id: [(surface, TYPE), ...]} preserving order and duplicates."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for r in df.itertuples():
        rid = r.id
        if not rid.startswith("gr_text_"):
            continue
        cell = (r.prediction or "").strip()
        ents = []
        if cell and cell.lower() != "none":
            for line in cell.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # split on the LAST comma: entity surface may itself contain commas
                if "," not in line:
                    continue
                surf, typ = line.rsplit(",", 1)
                surf, typ = surf.strip(), typ.strip()
                if surf:
                    ents.append((surf, typ))
        out[rid] = ents
    return out


def write_pred_csv(path, preds, ids=None):
    ids = ids or sorted(preds)
    rows = []
    for rid in ids:
        ents = preds.get(rid, [])
        cell = "\n".join(f"{s}, {t}" for s, t in ents) if ents else "None"
        rows.append({"id": rid, "prediction": cell})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------- grounding ----------
_WORDCH = re.compile(r"[^\W_]", re.UNICODE)


def token_boundary_ok(surface, text):
    """True iff `surface` occurs in `text` at a word boundary on both sides."""
    if not surface:
        return False
    i = 0
    while True:
        i = text.find(surface, i)
        if i < 0:
            return False
        left_ok = i == 0 or not _WORDCH.match(text[i - 1])
        j = i + len(surface)
        right_ok = j >= len(text) or not _WORDCH.match(text[j])
        if left_ok and right_ok:
            return True
        i += 1


def substring_ok(surface, text):
    return bool(surface) and surface in text


def diagnostics(preds, passages, name=""):
    n_ent = sum(len(v) for v in preds.values())
    types = Counter(t for v in preds.values() for _, t in v)
    sub = tb = 0
    bad_tb, bad_sub = [], []
    ntok = Counter()
    empt = 0
    for rid, ents in preds.items():
        if not ents:
            empt += 1
        txt = passages[rid]
        for s, t in ents:
            ntok[min(len(s.split()), 6)] += 1
            if substring_ok(s, txt):
                sub += 1
            else:
                bad_sub.append((rid, s, t))
            if token_boundary_ok(s, txt):
                tb += 1
            else:
                bad_tb.append((rid, s, t))
    legal = sum(c for t, c in types.items() if t in TYPES)
    return {
        "method": name,
        "n_rows": len(preds),
        "n_entities": n_ent,
        "ent_per_row": round(n_ent / max(len(preds), 1), 2),
        "empty_rows": empt,
        "by_type": dict(types),
        "type_legality_rate": round(legal / n_ent, 4) if n_ent else None,
        "substring_grounding_rate": round(sub / n_ent, 4) if n_ent else None,
        "token_boundary_grounding_rate": round(tb / n_ent, 4) if n_ent else None,
        "ungrounded_token_boundary": bad_tb,
        "ungrounded_substring": bad_sub,
        "span_len_tokens_hist": dict(sorted(ntok.items())),
    }


# ---------- agreement ----------
def to_multiset(preds, typed=True):
    d = {}
    for rid, ents in preds.items():
        d[rid] = Counter((s, t) if typed else (s,) for s, t in ents)
    return d


def pair_agreement(a, b, ids, typed=True):
    A, B = to_multiset(a, typed), to_multiset(b, typed)
    inter = union = na = nb = 0
    for rid in ids:
        ca, cb = A.get(rid, Counter()), B.get(rid, Counter())
        inter += sum((ca & cb).values())
        union += sum((ca | cb).values())
        na += sum(ca.values())
        nb += sum(cb.values())
    return {
        "intersection": inter,
        "union": union,
        "jaccard": round(inter / union, 4) if union else 0.0,
        "n_a": na,
        "n_b": nb,
        "dice": round(2 * inter / (na + nb), 4) if (na + nb) else 0.0,
        "recall_of_a_by_b": round(inter / na, 4) if na else 0.0,
        "recall_of_b_by_a": round(inter / nb, 4) if nb else 0.0,
    }
