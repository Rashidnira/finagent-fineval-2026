"""Phase 5 experiment runner: Systems B/C on the Easy set (gold hidden).

Usage:
    python -m src.experiment_runner --system B --dataset easy
    python -m src.experiment_runner --system C --dataset easy [--limit N]

The model receives ONLY the supplied query (which embeds context + question).
Gold answers are used exclusively for post-hoc scoring.
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.answer_validator import validate
from src.generator import Generator
from src.question_router import route
from src.scorer import SCORERS, rouge1_prf

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"

EASY_PROMPTS_C = {
    "Revenue": "revenue_v1.txt",
    "BalanceSheet": "balance_sheet_v1.txt",
    "CashFlow": "cash_flow_v1.txt",
    "RnD": "rd_v1.txt",
}
EXPERT_PROMPTS_C = {
    "RevenueFocus": "revenue_focus_v2.txt",
    "CapitalAllocation": "capital_allocation_v2.txt",
    "MarginStrategy": "margin_strategy_v2.txt",
    "Capex": "capex_v2.txt",
}
EVIDENCE_LABEL = {"easy": "News Evidence:", "expert": "Financial Statements Evidence:"}


def system_prompt(system: str, qtype: str, tier: str) -> tuple:
    """Return (prompt_text, prompt_version)."""
    if system == "B":
        f = PROMPTS / tier / "structured_v1.txt"
        return f.read_text(encoding="utf-8"), f"{tier}/structured_v1"
    if system == "C":
        name = (EASY_PROMPTS_C if tier == "easy" else EXPERT_PROMPTS_C)[qtype]
        f = PROMPTS / tier / name
        return f.read_text(encoding="utf-8"), f"{tier}/{name.removesuffix('.txt')}"
    raise ValueError(f"unknown system {system!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["B", "C"])
    ap.add_argument("--dataset", required=True, choices=["easy", "expert"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", default="openrouter_gemma",
                    choices=["openrouter_gemma", "openrouter_gemma4_26b_a4b",
                             "openrouter_qwen3_32b", "local_qwen3_32b",
                             "local_gemma4_31b", "local_nemotron_49b"],
                    help="pinned OpenRouter competition candidate")
    args = ap.parse_args()

    if args.dataset == "expert" and args.system != "C":
        ap.error("expert generation uses the selected per-type system (C)")

    if args.dataset == "easy":
        df = pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet")
        outbase = ROOT / "experiments/easy_benchmark"
    else:
        df = pd.read_parquet(ROOT / "data/manifests/official_finnlp_test.parquet")
        outbase = ROOT / "outputs/predictions"
    if args.limit:
        # keep whole task_id groups (group-leakage rule)
        keep = sorted(df.task_id.unique())[: max(1, args.limit // 4)]
        df = df[df.task_id.isin(keep)]

    evidence_label = EVIDENCE_LABEL[args.dataset]
    gen = Generator(provider=args.provider)
    # The provider's experiment_namespace keeps each candidate's runs fully
    # separate from other candidates and from the legacy *_gemma4_31b_*
    # direct-Google experiment ids and their results.
    exp_id = (f"{args.dataset}_{args.system}_"
              f"{gen.provider.experiment_namespace}_single_v1")
    outdir = outbase / exp_id
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        qtype = route(r.question)
        sys_prompt, pv = system_prompt(args.system, qtype, args.dataset)
        rec = gen.generate(experiment_id=exp_id, system=sys_prompt,
                           user=r.query, prompt_version=pv,
                           task_id=r.task_id, question=r.question,
                           submission_id=getattr(r, "id", None))
        pred = rec["raw_response"]
        val = validate(pred, r.query, evidence_label)
        row = {"task_id": r.task_id, "qtype": qtype, "prediction": pred,
               "from_cache": rec["from_cache"], "stop_reason": rec["stop_reason"],
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
            score_note = (f"f1(A)={row['multifinben_default_f1']:.3f} "
                          f"f1(B)={row['finmmeval_multiscript_f1']:.3f} ")
        else:
            score_note = ""
        rows.append(row)
        print(f"[{exp_id}] {r.task_id} {qtype:16s} {score_note}"
              f"flags={row['val_flags'] or '-'} "
              f"{'(cache)' if rec['from_cache'] else ''}")

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "predictions.csv", index=False, encoding="utf-8")

    # aggregates
    agg = {"experiment_id": exp_id, "n": len(res),
           "date": datetime.date.today().isoformat(),
           "model": gen.provider.model,
           "provider": gen.provider.name, "system_type": args.system,
           "dataset": args.dataset}
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

    # per-question-type breakdown
    type_cols = ([f"{s}_f1" for s in SCORERS] if args.dataset == "easy"
                 else ["val_word_count", "val_n_quotes"])
    bytype = res.groupby("qtype")[type_cols].mean().round(4)
    bytype.to_csv(outdir / "by_qtype.csv")

    # per-task_id failure list (user rule: targeting map for Phase 8)
    fail = (res[res.val_flags != ""]
            .groupby("task_id")
            .agg(n_flagged=("val_flags", "size"),
                 flags=("val_flags", lambda s: ";".join(sorted(set(
                     f for row in s for f in row.split(";") if f)))))
            .reset_index())
    fail.to_csv(outdir / "task_failure_list.csv", index=False)

    # experiment index
    idx = ROOT / "experiments/EXPERIMENT_INDEX.csv"
    exists = idx.exists()
    with idx.open("a", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(agg))
        if not exists:
            wcsv.writeheader()
        wcsv.writerow(agg)

    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=1))
    print("\n=== PER-TASK FAILURE LIST ===")
    print(fail.to_string(index=False) if len(fail) else "(no flagged rows)")


if __name__ == "__main__":
    main()
