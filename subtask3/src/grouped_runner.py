"""Grouped document-level generation: one call per task_id answers all four
questions sharing the same evidence context (verified identical in Phase 1).

Usage:
    python -m src.grouped_runner --dataset easy  --system C --smoke MSFT_20200429
    python -m src.grouped_runner --dataset easy  --system C
    python -m src.grouped_runner --dataset expert --system C
    python -m src.grouped_runner --dataset easy --system C --count-tokens-only

Parsing maps answers back to rows by explicit id attribute (never row
position alone). A group whose output cannot be parsed gets ONE regeneration
(config_tag retry1 — a new experiment per the canonical rule, permitted on
parse failure); if that also fails, the group falls back to per-question
full-context calls.
"""
import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import regex as re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.answer_validator import validate
from src.context_compress import compress_ws
from src.data_audit import split_query
from src.experiment_runner import EVIDENCE_LABEL, system_prompt
from src.generator import Generator
from src.question_router import route
from src.scorer import SCORERS, rouge1_prf

ROOT = Path(__file__).resolve().parents[1]

_ANSWER_TAG = re.compile(
    r'<ANSWER_(\d)\s+id="([^"]+)"\s*>(.*?)</ANSWER_\1\s*>', re.S)

PREAMBLE = """You are a senior financial analyst. Below is the full context for ONE company filing (financial statements plus news articles in English, Chinese, Japanese, Spanish, and Greek), followed by FOUR independent questions about it.

Rules:
- Answer each question independently. Never merge answers or refer from one answer to another.
- Each question has its own instruction block; follow it, and the answer format it specifies, exactly.
- Each answer must be 100 words or fewer.
- Every factual claim must be traceable to the supplied context. Copy figures exactly as printed.
- Output EXACTLY four blocks, one per question, in this format and nothing else:

<ANSWER_1 id="{id1}">
(answer to question 1 in its required format)
</ANSWER_1>
<ANSWER_2 id="{id2}">
(answer to question 2 in its required format)
</ANSWER_2>
<ANSWER_3 id="{id3}">
(answer to question 3 in its required format)
</ANSWER_3>
<ANSWER_4 id="{id4}">
(answer to question 4 in its required format)
</ANSWER_4>
"""


def row_uid(r) -> str:
    return r.id if hasattr(r, "id") and pd.notna(getattr(r, "id", None)) \
        else f"{r.task_id}__{route(r.question)}"


def build_grouped_prompt(group: pd.DataFrame, system: str, dataset: str):
    """Return (system_text, user_text, uid_order, prompt_versions)."""
    rows = list(group.itertuples())
    assert len(rows) == 4, f"group must have 4 rows, got {len(rows)}"
    # identical within group (Phase 1); losslessly whitespace-compressed to
    # fit the free-tier 16k-input-token admission bucket
    evidence = compress_ws(split_query(rows[0].query)[1])
    uids = [row_uid(r) for r in rows]

    qblocks, pvs = [], []
    for i, r in enumerate(rows, 1):
        qtype = route(r.question)
        instr, pv = system_prompt(system, qtype, dataset)
        pvs.append(pv)
        qblocks.append(
            f'--- QUESTION {i} (id="{uids[i-1]}") ---\n'
            f"Instructions for this question:\n{instr.strip()}\n\n"
            f"Question {i}: {r.question}\n")

    sys_text = PREAMBLE.format(id1=uids[0], id2=uids[1], id3=uids[2],
                               id4=uids[3])
    user_text = evidence.strip() + "\n\n" + "\n".join(qblocks)
    return sys_text, user_text, uids, pvs


def parse_grouped(text: str, uids: list) -> dict:
    """Return {uid: answer_text}; raise ValueError if any uid is missing."""
    found = {}
    for n, uid, body in _ANSWER_TAG.findall(text):
        found[uid] = body.strip()
    missing = [u for u in uids if u not in found]
    if missing:
        # positional fallback ONLY if all four tags exist with wrong ids
        tags = _ANSWER_TAG.findall(text)
        if len(tags) == 4 and len({t[0] for t in tags}) == 4:
            by_n = {int(t[0]): t[2].strip() for t in tags}
            return {uid: by_n[i + 1] for i, uid in enumerate(uids)}
        raise ValueError(f"unparsed answers for ids {missing}")
    return {u: found[u] for u in uids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["easy", "expert"])
    ap.add_argument("--system", default="C", choices=["B", "C"])
    ap.add_argument("--smoke", default=None, metavar="TASK_ID",
                    help="run only this one document group")
    ap.add_argument("--count-tokens-only", action="store_true",
                    help="token-check the largest grouped request, no generation")
    ap.add_argument("--provider", default="local_qwen3_32b",
                    choices=["local_qwen3_32b", "local_gemma4_31b",
                             "local_nemotron_49b", "openrouter_gemma",
                             "openrouter_gemma4_26b_a4b", "openrouter_qwen3_32b"],
                    help="generation provider; the submitted system used "
                         "local_qwen3_32b (the others were development candidates)")
    args = ap.parse_args()

    if args.dataset == "easy":
        df = pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet")
        outbase = ROOT / "experiments/easy_benchmark"
    else:
        df = pd.read_parquet(ROOT / "data/manifests/official_finnlp_test.parquet")
        outbase = ROOT / "outputs/predictions"

    evidence_label = EVIDENCE_LABEL[args.dataset]
    gen = Generator(provider=args.provider)
    # The provider's experiment_namespace keeps each candidate's runs fully
    # separate from other candidates and from the legacy *_gemma4_31b_*
    # direct-Google experiment ids and their results.
    exp_id = (f"{args.dataset}_{args.system}_"
              f"{gen.provider.experiment_namespace}_grouped_v1"
              + (f"_smoke" if args.smoke else ""))
    outdir = outbase / exp_id
    outdir.mkdir(parents=True, exist_ok=True)

    task_ids = [args.smoke] if args.smoke else sorted(df.task_id.unique())

    if args.count_tokens_only:
        biggest, big_tid = None, None
        for tid in sorted(df.task_id.unique()):
            group = df[df.task_id == tid]
            s, u, _, _ = build_grouped_prompt(group, args.system, args.dataset)
            if biggest is None or len(s) + len(u) > len(biggest):
                biggest, big_tid = s + "\n\n" + u, tid
        n_tok = gen.provider.count_tokens(biggest)
        limit = 262_144
        print(json.dumps({
            "largest_group": big_tid, "chars": len(biggest),
            "input_tokens": n_tok, "max_output_tokens": 4096,
            "context_limit": limit,
            "headroom_tokens": limit - n_tok - 4096,
            "ok": n_tok + 4096 < limit * 0.9}, indent=1))
        return

    # LEGACY google_gemma admission bucket (16,000 input tokens/min/model):
    # requests above it can never be admitted grouped and are routed to
    # per-question compressed calls. Applies ONLY when the provider sets
    # grouped_admission_limited (openrouter_gemma has no such bucket, so all
    # groups stay grouped). Routing is OFFLINE (no mid-run API dependency):
    # KNOWN_OVERSIZE holds the groups measured >15,700 tokens compressed
    # (count_tokens, 2026-08-19); the char-based estimator (~3.2 chars/token
    # on compressed text, deliberately conservative) is the backstop.
    admission_limited = getattr(gen.provider, "grouped_admission_limited", False)
    KNOWN_OVERSIZE = {"HON_20201030", "JNJ_20200429"}
    EST_CHARS_PER_TOKEN = 3.2
    EST_LIMIT = 17_000  # est-tokens; ≈16.1k real given the ~5% overestimate

    def single_calls(group, tid, tag):
        out = {}
        for r in group.itertuples():
            qtype = route(r.question)
            instr, pv = system_prompt(args.system, qtype, args.dataset)
            rec1 = gen.generate(experiment_id=exp_id, system=instr,
                                user=compress_ws(r.query), prompt_version=pv,
                                task_id=tid, question=r.question,
                                submission_id=row_uid(r),
                                config_tag=tag)
            out[row_uid(r)] = (rec1["raw_response"].strip(), rec1)
        return out

    rows_out, calls_before = [], len(list((ROOT / "cache/llm").glob("*.json")))
    for tid in task_ids:
        group = df[df.task_id == tid]
        sys_text, user_text, uids, pvs = build_grouped_prompt(
            group, args.system, args.dataset)

        est_tok = int(len(sys_text + user_text) / EST_CHARS_PER_TOKEN)
        answers, mode = None, "grouped"
        if admission_limited and (tid in KNOWN_OVERSIZE or est_tok > EST_LIMIT):
            print(f"[{exp_id}] {tid} oversize for 16k/min bucket "
                  f"(est {est_tok} tok) — routing to per-question calls")
            mode = "single_oversize"
            singles = single_calls(group, tid, "single_oversize_v1")
            answers = {u: t for u, (t, _) in singles.items()}
            rec = next(iter(singles.values()))[1]
        else:
            for tag in ["v1", "retry1"]:
                rec = gen.generate(experiment_id=exp_id, system=sys_text,
                                   user=user_text, prompt_version=";".join(pvs),
                                   task_id=tid, config_tag=f"grouped_{tag}")
                try:
                    answers = parse_grouped(rec["raw_response"], uids)
                    break
                except ValueError as e:
                    print(f"[{exp_id}] {tid} parse failure ({tag}): {e}")
            if answers is None:
                print(f"[{exp_id}] {tid} falling back to per-question calls")
                mode = "single_fallback"
                singles = single_calls(group, tid, "single_fallback_v1")
                answers = {u: t for u, (t, _) in singles.items()}
                rec = next(iter(singles.values()))[1]

        for r in group.itertuples():
            uid = row_uid(r)
            pred = answers[uid]
            val = validate(pred, r.query, evidence_label)
            row = {"task_id": tid, "qtype": route(r.question),
                   "prediction": pred, "generation_mode": mode,
                   "cache_key": rec["cache_key"],
                   **{f"val_{k}": v for k, v in val.items()
                      if k not in ("untraceable_numbers", "unverbatim_quotes")},
                   "val_flags": ";".join(val["flags"]),
                   "val_untraceable": ";".join(val["untraceable_numbers"]),
                   "val_unverbatim": ";".join(val["unverbatim_quotes"])}
            if hasattr(r, "id"):
                row = {"id": r.id, **row}
            if args.dataset == "easy":
                for s in SCORERS:
                    m = rouge1_prf(r.answer, pred, s)
                    row.update({f"{s}_p": m["precision"], f"{s}_r": m["recall"],
                                f"{s}_f1": m["f1"]})
            rows_out.append(row)
        note = ""
        if args.dataset == "easy":
            grp_rows = rows_out[-4:]
            note = (f"f1(A)={sum(x['multifinben_default_f1'] for x in grp_rows)/4:.3f} "
                    f"f1(B)={sum(x['finmmeval_multiscript_f1'] for x in grp_rows)/4:.3f}")
        print(f"[{exp_id}] {tid} {mode} {note}")

    res = pd.DataFrame(rows_out)
    res.to_csv(outdir / "predictions.csv", index=False, encoding="utf-8")

    calls = len(list((ROOT / "cache/llm").glob("*.json"))) - calls_before
    agg = {"experiment_id": exp_id, "n": len(res),
           "date": datetime.date.today().isoformat(),
           "model": gen.provider.model,
           "provider": gen.provider.name, "system_type": args.system,
           "dataset": args.dataset, "generation_mode": "grouped+routed",
           "new_api_calls": calls}
    if args.dataset == "easy":
        for s in SCORERS:
            agg[f"{s}_f1"] = res[f"{s}_f1"].mean()
            agg[f"{s}_p"] = res[f"{s}_p"].mean()
            agg[f"{s}_r"] = res[f"{s}_r"].mean()
    agg["format_pass_rate"] = (~res.val_flags.str.contains(
        "missing|empty|over_length|code_fence")).mean()
    agg["numeric_grounded_rate"] = (res.val_n_untraceable_numbers == 0).mean()
    agg["quote_verbatim_rate"] = (res.val_n_unverbatim_quotes == 0).mean()
    (outdir / "aggregate.json").write_text(json.dumps(agg, indent=1))

    if args.dataset == "easy":
        bytype = res.groupby("qtype")[[f"{s}_f1" for s in SCORERS]
                                      + ["val_word_count"]].mean().round(4)
    else:
        bytype = res.groupby("qtype")[["val_word_count", "val_n_quotes"]
                                      ].mean().round(4)
    bytype.to_csv(outdir / "by_qtype.csv")

    fail = (res[res.val_flags != ""]
            .groupby("task_id")
            .agg(n_flagged=("val_flags", "size"),
                 flags=("val_flags", lambda s: ";".join(sorted(set(
                     f for row in s for f in row.split(";") if f)))))
            .reset_index())
    fail.to_csv(outdir / "task_failure_list.csv", index=False)

    idx = ROOT / "experiments/EXPERIMENT_INDEX.csv"
    exists = idx.exists()
    with idx.open("a", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(agg), extrasaction="ignore")
        if not exists:
            wcsv.writeheader()
        wcsv.writerow(agg)

    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=1))
    print("\n=== PER-TASK FAILURE LIST ===")
    print(fail.to_string(index=False) if len(fail) else "(no flagged rows)")


if __name__ == "__main__":
    main()
