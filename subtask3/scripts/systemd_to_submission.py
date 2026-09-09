"""Stage 4 of the best system: convert a System D predictions JSONL into the
official `id,prediction` CSV.

Usage:
    python scripts/systemd_to_submission.py <predictions.jsonl> <out.csv>

Rows are mapped by exact (task_id, question) match against the official
test manifest (data/manifests/official_finnlp_test.parquet). The script
refuses to guess: any unmatched row or wrong row count aborts. Run
`python -m src.submission --predictions <out.csv> --tag <tag>` afterwards
for the full integrity check and the audited copy in outputs/submission/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data/manifests/official_finnlp_test.parquet"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: systemd_to_submission.py <predictions.jsonl> <out.csv>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    official = pd.read_parquet(OFFICIAL, columns=["id", "task_id", "question"])
    id_map = {(r.task_id, r.question): r.id for r in official.itertuples()}

    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    out, missing = [], []
    for r in rows:
        key = (r["task_id"], r["question"])
        if key not in id_map:
            missing.append(key)
            continue
        out.append({"id": id_map[key], "prediction": r["prediction"]})

    if missing:
        sys.exit(f"FATAL: {len(missing)} rows had no (task_id, question) match "
                 f"in the official manifest; first: {missing[0]}")
    if len(out) != len(official):
        sys.exit(f"FATAL: {len(out)} rows converted, expected {len(official)}")
    over = [o["id"] for o in out if len(str(o["prediction"]).split()) > 100]
    if over:
        sys.exit(f"FATAL: {len(over)} answers exceed 100 words: {over[:5]}")

    order = {i: n for n, i in enumerate(official.id)}
    out.sort(key=lambda o: order[o["id"]])
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(dst, index=False, encoding="utf-8",
                             quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    print(f"WROTE {dst} ({len(out)} rows, all ids matched, max words "
          f"{max(len(str(o['prediction']).split()) for o in out)})")


if __name__ == "__main__":
    main()
