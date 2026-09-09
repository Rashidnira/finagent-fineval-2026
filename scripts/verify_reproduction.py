#!/usr/bin/env python3
"""Validate submission shape and published checksums for the released subtasks."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
EXPECTED = {"subtask1": ("ee6c3c6577c04df9fb3b24608aeaf062", 200), "subtask2": ("7d4c73365c98c737a020fc8ee1802dca", 50)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subtask", choices=EXPECTED)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    digest = hashlib.md5(args.csv.read_bytes()).hexdigest()  # Published artifact identifier.
    expected_digest, expected_rows = EXPECTED[args.subtask]
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != ["id", "prediction"]:
        raise SystemExit("expected columns: id,prediction")
    if len(rows) != expected_rows or len({row["id"] for row in rows}) != expected_rows:
        raise SystemExit(f"expected {expected_rows} unique rows, got {len(rows)}")
    if digest != expected_digest:
        raise SystemExit(f"checksum mismatch: {digest} != {expected_digest}")
    print(f"PASS {args.subtask}: {expected_rows} rows, md5 {digest}")

if __name__ == "__main__":
    main()
