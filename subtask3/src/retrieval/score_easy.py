"""Score Easy-tier prediction files against Easy gold (dual ROUGE-1 scorers).

Run: python -m src.retrieval.score_easy pred.jsonl [pred2.jsonl ...]

Accepts one or more predictions JSONL files (rows need task_id, question,
prediction — the System D runner format; System C outputs can be converted
to the same three fields). Scores each against the supplied Easy gold
answers under BOTH scorer variants (multifinben_default and
finmmeval_multiscript, via the System C scorer imported read-only), overall
and per question family. Use with a System C file and a System D file to get
the C-vs-D A/B on the same backend.

Easy gold is used for SCORING ONLY. Expert gold does not exist locally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.question_router import route          # read-only System C imports
from src.scorer import score_pairs

FAMILIES = ("Revenue", "BalanceSheet", "CashFlow", "RnD")
SCORERS = ("multifinben_default", "finmmeval_multiscript")


def load_predictions(path: Path) -> dict[tuple[str, str], str]:
    """Keyed by (task_id, family) — unique per row in every Easy file.
    Accepts the System D runner JSONL (has question) and the System C
    grouped-runner predictions.csv (has qtype)."""
    out = {}
    if path.suffix == ".csv":
        for _, r in pd.read_csv(path).iterrows():
            out[(r["task_id"], r["qtype"])] = str(r["prediction"])
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["task_id"], route(r["question"]))] = r["prediction"]
    return out


def score_file(path: Path, gold: pd.DataFrame) -> dict:
    preds = load_predictions(path)
    rows = []
    for _, g in gold.iterrows():
        key = (g["task_id"], route(g["question"]))
        if key in preds:
            rows.append((g["answer"], preds[key], route(g["question"])))
    res = {"file": path.name, "n_scored": len(rows), "n_gold": len(gold)}
    for sc in SCORERS:
        refs = [r[0] for r in rows]
        hyps = [r[1] for r in rows]
        res[sc] = {"overall": round(score_pairs(refs, hyps, sc)["rouge1_f1"], 4)}
        for fam in FAMILIES:
            fr = [(a, p) for a, p, f in rows if f == fam]
            if fr:
                res[sc][fam] = round(score_pairs([a for a, _ in fr],
                                                 [p for _, p in fr], sc)["rouge1_f1"], 4)
    return res


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.retrieval.score_easy pred.jsonl [...]")
    gold = pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet",
                           columns=["task_id", "question", "answer"])
    results = [score_file(Path(p), gold) for p in sys.argv[1:]]
    print(json.dumps(results, indent=1, ensure_ascii=False))
    hdr = ["file", "n"] + [f"{sc.split('_')[0]}:{c}"
                           for sc in SCORERS for c in ("overall",) + FAMILIES]
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in results:
        cells = [r["file"], f"{r['n_scored']}/{r['n_gold']}"]
        for sc in SCORERS:
            cells += [str(r[sc].get(c, "-"))
                      for c in ("overall",) + FAMILIES]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
