"""Phase 4 — mechanical ROUGE perturbations of Easy gold answers.

METRIC DIAGNOSTIC ONLY: quantifies which lexical properties of the reference
answers carry ROUGE-1 mass. Does not evaluate any generation system.

Run: python -m src.perturbations
"""
import io
import sys
from pathlib import Path

import pandas as pd
import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scorer import rouge1_prf, SCORERS
from src.structural_audit import qtype, split_answer, EVIDENCE_LABEL

ROOT = Path(__file__).resolve().parents[1]

_NONLATIN_CHAR = re.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Greek}]")
_NUM_TOKEN = re.compile(r"\S*\d\S*")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_CURRENCY = re.compile(r"[$€¥£]")


def v_original(ans):
    return ans


def v_labels_removed(ans):
    return ans.replace("Answer:", " ").replace(EVIDENCE_LABEL, " ")


def v_evidence_removed(ans):
    return split_answer(ans)[0]


def v_numbers_removed(ans):
    return _NUM_TOKEN.sub(" ", ans)


def v_numbers_normalized(ans):
    return _CURRENCY.sub(" ", _THOUSANDS.sub("", ans))


def v_nonenglish_removed(ans):
    """Remove evidence sentences containing non-Latin script; keep the rest."""
    answer_sec, evid_sec = split_answer(ans)
    if not evid_sec:
        return ans
    kept = [seg for seg in re.split(r"(?<=[”\"]) ", evid_sec)
            if not _NONLATIN_CHAR.search(seg)]
    return answer_sec + " ".join(kept)


VARIANTS = {
    "A_original": v_original,
    "B_labels_removed": v_labels_removed,
    "C_evidence_removed": v_evidence_removed,
    "D_numbers_removed": v_numbers_removed,
    "E_numbers_normalized": v_numbers_normalized,
    "F_nonenglish_removed": v_nonenglish_removed,
}


def main():
    out = io.StringIO()

    def w(line=""):
        print(line)
        out.write(line + "\n")

    df = pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet")
    df["qtype"] = df.question.map(qtype)
    df["evidence_none"] = df.answer.map(
        lambda a: split_answer(a)[1][len(EVIDENCE_LABEL):].strip().lower()
        .startswith("none") if split_answer(a)[1] else True)

    rows = []
    for variant, fn in VARIANTS.items():
        for scorer in SCORERS:
            for _, r in df.iterrows():
                s = rouge1_prf(r.answer, fn(r.answer), scorer)
                rows.append({"variant": variant, "scorer": scorer,
                             "qtype": r.qtype, "evidence_none": r.evidence_none,
                             **{k: s[k] for k in ("precision", "recall", "f1")}})
    res = pd.DataFrame(rows)

    w("# PERTURBATION RESULTS (Phase 4 — metric diagnostic only)")
    w("\nPrediction = perturbed gold, reference = original gold. Interpretation "
      "is limited to: which lexical properties matter in the references. It "
      "does NOT rank generation systems (see §0.3 of the project spec).")

    w("\n## Overall (macro means)")
    piv = (res.groupby(["variant", "scorer"])[["precision", "recall", "f1"]]
           .mean().round(4).reset_index())
    base = piv[piv.variant == "A_original"].set_index("scorer")["f1"]
    piv["delta_f1"] = piv.apply(lambda r: round(r.f1 - base[r.scorer], 4), axis=1)
    w(piv.to_markdown(index=False))

    w("\n## F1 by question type (per scorer)")
    for scorer in SCORERS:
        w(f"\n**{scorer}**")
        t = (res[res.scorer == scorer]
             .pivot_table(index="variant", columns="qtype", values="f1")
             .round(4))
        w(t.to_markdown())

    w("\n## Evidence stratification (variants C and F, F1)")
    t = (res[res.variant.isin(["C_evidence_removed", "F_nonenglish_removed"])]
         .pivot_table(index=["variant", "scorer"], columns="evidence_none",
                      values="f1").round(4)
         .rename(columns={False: "evidence_present", True: "evidence_none"}))
    w(t.to_markdown())

    outdir = ROOT / "experiments/perturbations"
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "perturbation_scores.csv", index=False)
    (ROOT / "reports/PERTURBATION_RESULTS.md").write_text(out.getvalue(), encoding="utf-8")
    w("\nWrote reports/PERTURBATION_RESULTS.md and experiments/perturbations/perturbation_scores.csv")


if __name__ == "__main__":
    main()
