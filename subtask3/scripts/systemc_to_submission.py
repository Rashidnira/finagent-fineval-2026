"""Convert a System C (no-retrieval, grouped) predictions.csv into the
official `id,prediction` submission CSV.

Usage:
    python scripts/systemc_to_submission.py <predictions.csv> <out.csv>

src/grouped_runner.py already writes the official row id per prediction
(matched during generation via the question router), so this script only
selects and orders the id,prediction columns and validates against the
official manifest -- it does not re-derive ids.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data/manifests/official_finnlp_test.parquet"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: systemc_to_submission.py <predictions.csv> <out.csv>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    official = pd.read_parquet(OFFICIAL, columns=["id"])
    preds = pd.read_csv(src, usecols=["id", "prediction"])

    off_ids, pred_ids = set(official.id), set(preds.id)
    if off_ids - pred_ids:
        sys.exit(f"FATAL: missing ids: {sorted(off_ids - pred_ids)[:10]}")
    if pred_ids - off_ids:
        sys.exit(f"FATAL: extra ids: {sorted(pred_ids - off_ids)[:10]}")
    if preds.id.duplicated().any():
        sys.exit(f"FATAL: duplicate ids: {preds.id[preds.id.duplicated()].tolist()}")
    over = preds[preds.prediction.astype(str).str.split().str.len() > 100].id.tolist()
    if over:
        sys.exit(f"FATAL: {len(over)} answers exceed 100 words: {over[:5]}")

    order = {i: n for n, i in enumerate(official.id)}
    preds = preds.sort_values("id", key=lambda s: s.map(order))
    dst.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(dst, index=False, encoding="utf-8",
                quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    print(f"WROTE {dst} ({len(preds)} rows, all ids matched, max words "
          f"{preds.prediction.astype(str).str.split().str.len().max()})")


if __name__ == "__main__":
    main()
