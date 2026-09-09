"""Phase 10 — submission construction and integrity validation.

Official spec (Space /submission_info, archived in data/manifests/):
UTF-8 CSV, exactly 76 rows, columns `id,prediction`; complete answer in
`prediction`; CSV-quote cells containing commas or line breaks.

Usage:
    python -m src.submission --predictions <predictions.csv> --tag baseline
    python -m src.submission --predictions <predictions.csv> --tag v2
"""
import argparse
import csv
import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TEST = ROOT / "data/manifests/official_finnlp_test.parquet"
SUBDIR = ROOT / "outputs/submission"


class SubmissionError(RuntimeError):
    pass


def build_submission(preds: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Validate predictions against the official test file; return the
    submission frame in official row order. Raises SubmissionError on any
    integrity violation — never repairs silently."""
    problems = []
    if "id" not in preds.columns or "prediction" not in preds.columns:
        raise SubmissionError(f"predictions must have columns id,prediction; "
                              f"got {list(preds.columns)}")
    if preds.id.isna().any():
        problems.append("null ids in predictions")
    if preds.id.duplicated().any():
        problems.append(f"duplicate ids: {preds.id[preds.id.duplicated()].tolist()}")
    off_ids, pred_ids = set(official.id), set(preds.id)
    if off_ids - pred_ids:
        problems.append(f"missing ids: {sorted(off_ids - pred_ids)[:10]}")
    if pred_ids - off_ids:
        problems.append(f"extra ids: {sorted(pred_ids - off_ids)[:10]}")
    empty = preds[preds.prediction.isna() | (preds.prediction.astype(str).str.strip() == "")]
    if len(empty):
        problems.append(f"empty predictions for ids: {empty.id.tolist()}")
    if problems:
        raise SubmissionError("; ".join(problems))

    sub = (official[["id"]]
           .merge(preds[["id", "prediction"]], on="id", how="left"))
    if len(sub) != len(official):
        raise SubmissionError(f"row count {len(sub)} != {len(official)}")
    return sub


def write_submission(sub: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8",
               quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    # read-back verification
    back = pd.read_csv(path, encoding="utf-8", dtype=str)
    if list(back.columns) != ["id", "prediction"] or len(back) != len(sub):
        raise SubmissionError("read-back verification failed")
    if not (back.id.values == sub.id.astype(str).values).all():
        raise SubmissionError("read-back id order mismatch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="CSV with id,prediction columns (runner output)")
    ap.add_argument("--tag", required=True,
                    help="'baseline' (immutable) or a version tag like v2")
    args = ap.parse_args()

    preds = pd.read_csv(args.predictions, encoding="utf-8", dtype={"id": str})
    official = pd.read_parquet(OFFICIAL_TEST)

    sub = build_submission(preds, official)

    name = ("baseline_validated_submission.csv" if args.tag == "baseline"
            else f"submission_{args.tag}.csv")
    path = SUBDIR / name
    if args.tag == "baseline" and path.exists():
        raise SubmissionError(
            "baseline checkpoint already exists and is immutable; use a "
            "version tag (e.g. --tag v2) instead")
    write_submission(sub, path)

    # audit file
    audit = official[["id", "task_id", "question"]].merge(
        preds, on="id", how="left")
    audit["word_count"] = audit.prediction.astype(str).str.split().str.len()
    audit["generated_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    audit_path = path.with_name(path.stem.replace("submission", "audit") + ".csv")
    audit.drop(columns=["prediction"]).to_csv(audit_path, index=False,
                                              encoding="utf-8")

    print(f"WROTE {path} ({len(sub)} rows) and {audit_path}")
    print("All integrity checks passed: row count, id set, no duplicates, "
          "no empties, read-back verified.")


if __name__ == "__main__":
    main()
