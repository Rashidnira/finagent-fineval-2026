"""Phase 3 — Easy gold structural audit (deterministic, no LLM).

Decomposes every Easy gold answer into sections and token classes under both
candidate scorers; reports per-question-type distributions to refine the
40-60-word length prior.

Run: python -m src.structural_audit
"""
import io
import sys
from pathlib import Path

import pandas as pd
import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tokenizer import tokenize_latin_default, tokenize_multiscript

ROOT = Path(__file__).resolve().parents[1]

QUESTION_TYPE = {
    "What trends can be observed": "Revenue",
    "balance sheet reflect": "BalanceSheet",
    "operating, investing and financing cash flow": "CashFlow",
    "R&D ratio": "RnD",
}

EVIDENCE_LABEL = "News Evidence:"
_NONLATIN = re.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Greek}]")
_NUMERIC = re.compile(r"\d")


def qtype(question: str) -> str:
    for marker, name in QUESTION_TYPE.items():
        if marker in question:
            return name
    return "UNKNOWN"


def split_answer(ans: str):
    i = ans.find(EVIDENCE_LABEL)
    if i == -1:
        return ans, ""
    return ans[:i], ans[i:]


def features(ans: str) -> dict:
    answer_sec, evid_sec = split_answer(ans)
    tokA, tokB = tokenize_latin_default(ans), tokenize_multiscript(ans)
    evid_body = evid_sec[len(EVIDENCE_LABEL):].strip() if evid_sec else ""
    evid_none = evid_body.rstrip(". ").lower() in {"none", "”none”", '"none"'} or \
        evid_body.lower().startswith("none")
    return {
        "words": len(ans.split()),
        "tokens_A": len(tokA),
        "tokens_B": len(tokB),
        "answer_sec_words": len(answer_sec.split()),
        "evidence_sec_words": len(evid_sec.split()),
        "label_tokens": 1 + (3 if evid_sec else 0),  # answer / news evidence
        "numeric_tokens": sum(1 for t in tokB if _NUMERIC.search(t)),
        "nonlatin_tokens": sum(1 for t in tokB if _NONLATIN.search(t)),
        "evidence_passages": max(evid_sec.count("“"), evid_sec.count('"') // 2,
                                 evid_sec.count("「")),
        "evidence_none": evid_none,
        "has_evidence_label": bool(evid_sec),
    }


def main():
    out = io.StringIO()

    def w(line=""):
        print(line)
        out.write(line + "\n")

    df = pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet")
    feats = pd.DataFrame([features(a) for a in df.answer])
    feats.insert(0, "task_id", df.task_id.values)
    feats.insert(1, "qtype", df.question.map(qtype).values)
    feats["numeric_share_B"] = feats.numeric_tokens / feats.tokens_B
    feats["nonlatin_share_B"] = feats.nonlatin_tokens / feats.tokens_B

    w("# EASY GOLD STRUCTURAL AUDIT (Phase 3)")
    w(f"\n76 answers; question types: {feats.qtype.value_counts().to_dict()}")
    assert (feats.qtype != "UNKNOWN").all(), "unmapped question template"

    w("\n## 3.1-3.2 Global token-mass decomposition (Scorer B tokens)")
    tot = feats.tokens_B.sum()
    w(f"- total tokens: {tot}")
    for col, label in [("numeric_tokens", "numeric"), ("nonlatin_tokens", "non-Latin"),
                       ("label_tokens", "labels")]:
        w(f"- {label}: {feats[col].sum()} ({feats[col].sum()/tot:.1%})")
    aw, ew = feats.answer_sec_words.sum(), feats.evidence_sec_words.sum()
    w(f"- Answer-section words: {aw} ({aw/(aw+ew):.1%}); "
      f"Evidence-section words: {ew} ({ew/(aw+ew):.1%})")

    w("\n## 3.3 Per-question-type distributions")
    for col in ["words", "answer_sec_words", "evidence_sec_words",
                "numeric_tokens", "nonlatin_tokens"]:
        w(f"\n**{col}**")
        stats = (feats.groupby("qtype")[col]
                 .describe(percentiles=[.25, .5, .75])
                 [["mean", "50%", "25%", "75%", "min", "max"]].round(1))
        w(stats.to_markdown())

    w("\n## 3.4 Evidence = None stratification")
    strat = (feats.groupby(["qtype", "evidence_none"])
             .agg(n=("words", "size"), mean_words=("words", "mean"))
             .round(1).reset_index())
    w(strat.to_markdown(index=False))
    w(f"\nOverall Evidence=None: {feats.evidence_none.sum()}/76")

    w("\n## Length-calibration recommendation (updates the 40-60 prior)")
    rec = (feats.groupby("qtype")["words"]
           .agg(["mean", "median",
                 lambda s: s.quantile(.25), lambda s: s.quantile(.75)]))
    rec.columns = ["mean", "median", "q25", "q75"]
    w(rec.round(0).to_markdown())
    w("\nTarget band per type = [q25, q75] above; global prior 40-60 retained "
      "only for types whose IQR overlaps it.")

    outdir = ROOT / "experiments/structural_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    feats.to_csv(outdir / "easy_style_metrics.csv", index=False)
    (ROOT / "reports/EASY_STYLE_AUDIT.md").write_text(out.getvalue(), encoding="utf-8")
    w("\nWrote reports/EASY_STYLE_AUDIT.md and experiments/structural_audit/easy_style_metrics.csv")


if __name__ == "__main__":
    main()
