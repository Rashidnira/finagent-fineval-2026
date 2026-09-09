#!/usr/bin/env python
"""Decode candclf scores -> non-overlapping spans, with a threshold sweep.

Pass-1 (s1_text_candclf.py) gives p(A/B/C/D) per candidate span. Candidates
massively overlap (prefix/suffix/maximal variants of the same name), so a decode
step is required. Two decoders are compared:
  greedy_score  -- take highest p_entity first, then any non-overlapping span
  maximal_pref  -- among overlapping candidates above tau, keep the LONGEST
                   (Appendix E wants the full legal name incl. Α.Ε./LTD)
"""
import os, sys, json, argparse, re
from collections import defaultdict, Counter

ROOT = os.environ.get("FINNLP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/scripts")
from s1_text_altlib import load_passages, write_pred_csv

CH2TYPE = {"A": "ΠΡΟΣΩΠΟ", "B": "ΟΡΓΑΝΙΣΜΟΣ", "C": "ΤΟΠΟΘΕΣΙΑ"}
BAD_HEADS = {"στις", "στην", "στον", "στο", "στη", "της", "του", "των", "την", "τον", "το",
             "η", "ο", "οι", "τα", "και", "με", "για", "από", "που", "ως", "κατά", "μετά",
             "εταιρεία", "εταιρία", "όμιλος", "ομίλου", "όμιλο", "θυγατρική", "μητρική"}


def p_ent(c):
    return 1.0 - c["p"].get("D", 1.0)


def best_type(c):
    return max("ABC", key=lambda k: c["p"].get(k, 0.0))


# lowercase Greek function words. Appendix E rule 4 excludes the article from a span
# and rule 5 annotates consecutive entities SEPARATELY, so a *lowercase* function word
# in the interior of a span means the span has swallowed connective text. Uppercase
# ΚΑΙ is left alone: it really does occur inside Greek company names
# ("ΕΤΑΙΡΕΙΑ ΥΔΡΕΥΣΗΣ ΚΑΙ ΑΠΟΧΕΤΕΥΣΗΣ ΘΕΣΣΑΛΟΝΙΚΗΣ Α.Ε.").
FUNC_LC = {"και","ή","με","σε","στο","στη","στην","στον","στους","στις","στα","από","για",
           "προς","ως","που","το","τα","ο","η","οι","του","της","των","την","τον","τους",
           "τις","δια","κατά","μετά","μέσω","επί","υπό","περί","έναντι","ήτοι","καθώς",
           "στην","τη","θυγατρική","εταιρεία","εταιρία","όμιλο","ομίλου","τίτλο","επωνυμία",
           "διακριτικό","έδρα","οδό","οδός","αριθμό","ποσού","αξίας"}


def head_ok(c):
    t0 = c["surface"].split()[0].lower().strip(".,«»\"")
    return t0 not in BAD_HEADS


def interior_ok(c):
    toks = [t.strip(".,«»\"()") for t in c["surface"].split()]
    if len(toks) > 8:
        return False
    for i, t in enumerate(toks):
        if not (t.lower() in FUNC_LC and t.islower()):
            continue
        # Appendix E keeps the Greek patronymic inside a PERSON span
        # ("Γεώργιος Δημητρίου του Κωνσταντίνου"): allow a lowercase του/της
        # when it sits between two capitalised tokens.
        if (t.lower() in ("του", "της") and 0 < i < len(toks) - 1 and len(toks) <= 4
                and toks[i - 1][:1].isupper() and toks[i + 1][:1].isupper()):
            continue
        return False
    return True


def clean_surface(c, text):
    """Trim symmetric quotes/brackets and a dangling sentence-final period."""
    s, a, b = c["surface"], c["start"], c["end"]
    while s and s[0] in "«»\"'([":
        s = s[1:]; a += 1
    while s and s[-1] in "«»\"')],;:":
        s = s[:-1]; b -= 1
    # trailing period: keep it only when the final token is a DOTTED abbreviation
    # ("Α.Ε.", "S.A.", "A.B.E.E."); strip it when it is just the sentence period
    # after an undotted token ("... AG." -> "... AG").
    if s.endswith("."):
        last = s.split()[-1] if s.split() else ""
        if last.count(".") < 2:
            s = s[:-1]; b -= 1
    # unbalanced parenthesis: trim back to before the dangling "("
    if s.count("(") > s.count(")"):
        k = s.rfind("(")
        s = s[:k].rstrip(); b = a + len(s)
    if text[a:b] != s or len(s) < 2:
        return None
    d = dict(c); d["surface"], d["start"], d["end"] = s, a, b
    return d


def decode(cands, tau, mode, text):
    live = []
    for c in cands:
        if p_ent(c) < tau or not head_ok(c) or not interior_ok(c):
            continue
        cc = clean_surface(c, text)
        if cc is None or not head_ok(cc) or not interior_ok(cc):
            continue
        live.append(cc)
    if mode == "greedy_score":
        live.sort(key=lambda c: (-p_ent(c), -(c["end"] - c["start"]), c["start"]))
    else:  # maximal_pref
        live.sort(key=lambda c: (-(c["end"] - c["start"]), -p_ent(c), c["start"]))
    kept, taken = [], []
    for c in live:
        if any(not (c["end"] <= a or c["start"] >= b) for a, b in taken):
            continue
        kept.append(c); taken.append((c["start"], c["end"]))
    kept.sort(key=lambda c: c["start"])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=f"{ROOT}/out/alt/candclf_scores.json")
    ap.add_argument("--out_prefix", default="candclf")
    args = ap.parse_args()
    P = load_passages(); ids = sorted(P)
    S = json.load(open(args.scores))["rows"]
    summary = []
    for mode in ("maximal_pref", "greedy_score"):
        for tau in (0.5, 0.7, 0.9, 0.95, 0.99):
            preds, detail = {}, {}
            for rid in ids:
                kept = decode(S.get(rid, []), tau, mode, P[rid])
                preds[rid] = [(c["surface"], CH2TYPE[best_type(c)]) for c in kept]
                detail[rid] = [{"surface": c["surface"], "start": c["start"], "end": c["end"],
                                "type": CH2TYPE[best_type(c)], "p_ent": round(p_ent(c), 4),
                                "p": c["p"]} for c in kept]
            tag = f"{args.out_prefix}_{mode}_t{str(tau).replace('.','')}"
            write_pred_csv(f"{ROOT}/out/alt/s1_text_{tag}.csv", preds, ids)
            byt = Counter(t for v in preds.values() for _, t in v)
            n = sum(len(v) for v in preds.values())
            summary.append({"variant": tag, "mode": mode, "tau": tau, "n_entities": n,
                            "by_type": dict(byt),
                            "empty_rows": sum(1 for v in preds.values() if not v)})
            json.dump(detail, open(f"{ROOT}/out/alt/{tag}_detail.json", "w"), ensure_ascii=False)
            print(summary[-1], flush=True)
    json.dump(summary, open(f"{ROOT}/out/alt/{args.out_prefix}_decode_summary.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
