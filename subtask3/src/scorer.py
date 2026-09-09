"""ROUGE-1 scorers for FinNLP 2026 Subtask 3.

Two candidate implementations of the (private) competition metric; see
src/tokenizer.py and reports/SCORER_PROVENANCE.md for provenance. Both score
the COMPLETE answer string (labels + evidence included) and macro-average
item-level F1, matching both documented ancestors.

Usage:
    from src.scorer import score_pairs
    result = score_pairs(refs, preds, scorer="finmmeval_multiscript")
"""
from collections import Counter

from src.tokenizer import TOKENIZERS

SCORERS = tuple(TOKENIZERS)


def rouge1_prf(reference: str, prediction: str, scorer: str) -> dict:
    """Item-level unigram-multiset precision/recall/F1 (F1=0 when P+R=0)."""
    tok = TOKENIZERS[scorer]
    ref_counts = Counter(tok(reference))
    pred_counts = Counter(tok(prediction))
    overlap = sum((ref_counts & pred_counts).values())
    n_ref = sum(ref_counts.values())
    n_pred = sum(pred_counts.values())
    p = overlap / n_pred if n_pred else 0.0
    r = overlap / n_ref if n_ref else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "overlap": overlap, "ref_tokens": n_ref, "pred_tokens": n_pred}


def score_pairs(references: list, predictions: list, scorer: str) -> dict:
    """Macro-averaged item-level ROUGE-1 over aligned (reference, prediction) lists."""
    if len(references) != len(predictions):
        raise ValueError(f"length mismatch: {len(references)} refs vs "
                         f"{len(predictions)} preds")
    items = [rouge1_prf(ref, pred, scorer)
             for ref, pred in zip(references, predictions)]
    n = len(items)
    return {
        "scorer": scorer,
        "n": n,
        "rouge1_f1": sum(i["f1"] for i in items) / n if n else 0.0,
        "rouge1_p": sum(i["precision"] for i in items) / n if n else 0.0,
        "rouge1_r": sum(i["recall"] for i in items) / n if n else 0.0,
        "items": items,
    }


def score_all(references: list, predictions: list) -> dict:
    """Score under every candidate scorer (disagreement rule, Phase 2.6)."""
    return {s: {k: v for k, v in score_pairs(references, predictions, s).items()
                if k != "items"}
            for s in SCORERS}
