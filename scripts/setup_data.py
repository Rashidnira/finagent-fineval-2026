#!/usr/bin/env python3
"""Link or copy reviewer-supplied FinNLP parquet files into the repository layout."""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
DATASETS = {"subtask1": "finnlp2026-subtask1-greek-ner", "subtask2": "finnlp2026-subtask2-japanese-icr"}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="directory containing the two dataset directories")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--subtask", choices=("all", *DATASETS), default="all")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    selected = DATASETS if args.subtask == "all" else {args.subtask: DATASETS[args.subtask]}
    for subtask, dataset in selected.items():
        src = (args.source / dataset).resolve()
        missing = [name for name in ("train.parquet", "test.parquet") if not (src / name).is_file()]
        if missing:
            raise SystemExit(f"{src}: missing {', '.join(missing)}")
        dst = args.root / subtask / "data" / "hf" / dataset
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            raise SystemExit(f"refusing to replace existing path: {dst}")
        if args.mode == "copy":
            shutil.copytree(src, dst)
        else:
            os.symlink(src, dst, target_is_directory=True)
        print(f"{subtask}: {dst} -> {src} ({args.mode})")

if __name__ == "__main__":
    main()
