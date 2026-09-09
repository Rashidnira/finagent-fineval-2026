"""Build the four prompt sets of the retrieval x granularity grid.

    python -m src.retrieval.build_grid --dataset easy_train

The submitted paper compared System C (grouped generation, no retrieval) with
System D (per-question generation, BGE-M3 retrieval). That single step changes
two variables at once, and the two systems also differed in ways unrelated to
either: System C whitespace-compressed the context and wrapped it in our own
instructions, while System D used the organisers' query verbatim.

This module builds all four cells of the grid from one source, so the only
differences between cells are the two variables under study:

    cell            granularity     retrieval
    pq_noretr       per-question    none
    pq_retr         per-question    BGE-M3 top-5 + top-5
    gr_noretr       grouped         none
    gr_retr         grouped         BGE-M3 top-5 + top-5

Every cell starts from the same raw context. The System D prompt rows record
the offset and length of the evidence block that was spliced into the
organisers' query, so deleting that span recovers the original query exactly,
and the same span is what gets re-inserted for the retrieval cells. No cell
uses whitespace compression.

One difference cannot be removed: answering four questions in one call needs
an instruction saying how to separate the four answers, which per-question
generation does not need. That instruction is identical in both grouped cells,
so it cannot affect the retrieval contrast, and it is reported as inherent to
the granularity manipulation.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "data/processed/system_d_prompts"
OUT_DIR = ROOT / "data/processed/grid_prompts"

CELLS = ["pq_noretr", "pq_retr", "gr_noretr", "gr_retr"]

GROUPED_INSTRUCTION = """\
Answer all FOUR questions above, independently of one another. Follow the \
answer format required by the instructions at the top of this message for \
every answer. Output EXACTLY four blocks, in order, and nothing else:

<ANSWER_1 id="{id1}">
(answer to question 1)
</ANSWER_1>
<ANSWER_2 id="{id2}">
(answer to question 2)
</ANSWER_2>
<ANSWER_3 id="{id3}">
(answer to question 3)
</ANSWER_3>
<ANSWER_4 id="{id4}">
(answer to question 4)
</ANSWER_4>"""


def strip_block(row: dict) -> str:
    """The organisers' original query, with the evidence block removed."""
    p, off, ln = row["prompt"], row["insertion_offset"], row["block_len"]
    if not p[off:off + ln].startswith("Highlighted Evidence"):
        raise ValueError(f"{row['task_id']}: unexpected block at offset {off}")
    return p[:off] + p[off + ln:]


def evidence_block(row: dict) -> str:
    off, ln = row["insertion_offset"], row["block_len"]
    return row["prompt"][off:off + ln]


def split_head_tail(query: str) -> tuple[str, str]:
    """Split the query into (everything before the trailing question, question)."""
    i = query.rfind("\nQuestion:")
    if i == -1:
        raise ValueError("no trailing Question: block")
    return query[:i], query[i:].lstrip("\n")


def uid(row: dict) -> str:
    return f"{row['task_id']}__{row['family']}"


def build(dataset: str) -> dict[str, list[dict]]:
    rows = [json.loads(l) for l in
            (PROMPT_DIR / f"{dataset}.system_d.jsonl").open(encoding="utf-8")]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["task_id"]].append(r)

    out: dict[str, list[dict]] = {c: [] for c in CELLS}

    for task_id, rs in groups.items():
        if len(rs) != 4:
            raise ValueError(f"{task_id}: expected 4 questions, got {len(rs)}")

        bases = [strip_block(r) for r in rs]
        heads = {split_head_tail(b)[0] for b in bases}
        if len(heads) != 1:
            raise ValueError(f"{task_id}: the four questions do not share a context")
        head = heads.pop()
        tails = [split_head_tail(b)[1] for b in bases]
        uids = [uid(r) for r in rs]

        # ---- per-question cells: one prompt per question -------------------
        for r, base in zip(rs, bases):
            out["pq_noretr"].append({"task_id": r["task_id"], "question": r["question"],
                                     "family": r["family"], "prompt": base,
                                     "uids": [uid(r)]})
            out["pq_retr"].append({"task_id": r["task_id"], "question": r["question"],
                                   "family": r["family"], "prompt": r["prompt"],
                                   "uids": [uid(r)]})

        # ---- grouped cells: one prompt per filing --------------------------
        for cell, with_evidence in (("gr_noretr", False), ("gr_retr", True)):
            blocks = []
            for i, (r, tail) in enumerate(zip(rs, tails), 1):
                piece = f'--- QUESTION {i} (id="{uids[i - 1]}") ---\n{tail}'
                if with_evidence:
                    piece += "\n\n" + evidence_block(r).strip()
                blocks.append(piece)
            prompt = (head.rstrip() + "\n\n" + "\n\n".join(blocks) + "\n\n"
                      + GROUPED_INSTRUCTION.format(id1=uids[0], id2=uids[1],
                                                   id3=uids[2], id4=uids[3]))
            out[cell].append({"task_id": task_id,
                              "question": " | ".join(r["question"] for r in rs),
                              "family": "GROUPED", "prompt": prompt,
                              "uids": uids,
                              "questions": [r["question"] for r in rs],
                              "families": [r["family"] for r in rs]})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="easy_train")
    args = ap.parse_args()

    out = build(args.dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cell, rows in out.items():
        path = OUT_DIR / f"{args.dataset}.{cell}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        chars = sum(len(r["prompt"]) for r in rows) / len(rows)
        print(f"{cell:<10} {len(rows):>3} prompts   mean {chars:>7.0f} chars   -> "
              f"{path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
