"""Stage-3 candidate selection: Minimum Bayes Risk (MBR) consensus decoding.

Run: python -m src.mbr_select --candidates a.csv b.csv c.jsonl ... --out-tag mbr_v1

For every official test question, selects among N candidate answers (produced
by the defined generator configurations, e.g. System C / System D x model)
the one with the highest mean ROUGE-1 F1 against all OTHER candidates for
that question (the consensus answer). No reference/gold access, no human
selection: the utility is computed purely between candidates with the
project's own scorer, making the stage deterministic and reproducible.

Inputs may be runner predictions (.csv with id or task_id+qtype) or System D
jsonl (task_id+question). Provenance (chosen source, per-candidate utilities)
is written next to the submission input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.question_router import route
from src.scorer import rouge1_prf

OFFICIAL_TEST = ROOT / "data/manifests/official_finnlp_test.parquet"
UTILITY_SCORER = "multifinben_default"   # utility between candidates only


def load_candidates(path: Path) -> dict:
    """Return {(task_id, qtype): prediction} for one candidate file."""
    out = {}
    if path.suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8", dtype=str)
        for _, r in df.iterrows():
            out[(r["task_id"], r["qtype"])] = str(r["prediction"])
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[(r["task_id"], route(r["question"]))] = r["prediction"]
    return out


def mbr_pick(cands: list) -> tuple:
    """Return (winner_index, utilities): utility_i = mean ROUGE-1 F1 of
    candidate i against every other candidate."""
    utils = []
    for i, ci in enumerate(cands):
        scores = [rouge1_prf(cj, ci, UTILITY_SCORER)["f1"]
                  for j, cj in enumerate(cands) if j != i]
        utils.append(sum(scores) / len(scores))
    best = max(range(len(cands)), key=lambda i: utils[i])
    return best, utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="candidate prediction files (runner csv / D jsonl)")
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args()

    sources = [Path(p) for p in args.candidates]
    cand_maps = [load_candidates(p) for p in sources]
    official = pd.read_parquet(OFFICIAL_TEST)

    rows, prov, missing = [], [], 0
    for r in official.itertuples():
        key = (r.task_id, route(r.question))
        cands, names = [], []
        for m, p in zip(cand_maps, sources):
            if key in m and str(m[key]).strip():
                cands.append(m[key])
                names.append(p.parent.name + "/" + p.name)
        if not cands:
            missing += 1
            continue
        best, utils = mbr_pick(cands) if len(cands) > 1 else (0, [1.0])
        rows.append({"id": r.id, "prediction": cands[best]})
        prov.append({"id": r.id, "task_id": r.task_id, "qtype": key[1],
                     "chosen": names[best],
                     "utilities": {n: round(u, 4)
                                   for n, u in zip(names, utils)}})

    if missing:
        sys.exit(f"FATAL: {missing} official questions had no candidate")

    outdir = ROOT / "outputs/predictions/mbr"
    outdir.mkdir(parents=True, exist_ok=True)
    sub_in = outdir / f"{args.out_tag}_submission_input.csv"
    pd.DataFrame(rows).to_csv(sub_in, index=False, encoding="utf-8")
    (outdir / f"{args.out_tag}_provenance.json").write_text(
        json.dumps(prov, indent=1, ensure_ascii=False), encoding="utf-8")

    picks = pd.Series([p["chosen"] for p in prov]).value_counts()
    print(f"WROTE {sub_in} ({len(rows)} rows)")
    print("picks per source:")
    print(picks.to_string())


if __name__ == "__main__":
    main()
