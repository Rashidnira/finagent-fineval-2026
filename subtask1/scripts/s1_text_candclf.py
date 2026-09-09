#!/usr/bin/env python
"""(Method C) Candidate-span CLASSIFICATION instead of free generation.

Every span comes from out/alt/candidate_spans.json, which was built from the
UNLABELLED test `text` column plus gazetteers mined from CLEAN external corpora.
The LLM never emits a string: it only assigns one of
  A=ΠΡΟΣΩΠΟ  B=ΟΡΓΑΝΙΣΜΟΣ  C=ΤΟΠΟΘΕΣΙΑ  D=none
to a span we already hold char offsets for. Verbatim grounding is therefore 100%
by construction. Scores come from top_logprobs on the single answer token.
"""
import os, sys, json, re, argparse, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("FINNLP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/scripts")
import finnlp_config as CFG
from s1_text_altlib import load_passages, write_pred_csv

CH2TYPE = {"A": "ΠΡΟΣΩΠΟ", "B": "ΟΡΓΑΝΙΣΜΟΣ", "C": "ΤΟΠΟΘΕΣΙΑ", "D": None}

SYSTEM = (
    "Είσαι ειδικός σχολιαστής ονομαστικών οντοτήτων σε ελληνικές οικονομικές εκθέσεις "
    "(ετήσιες οικονομικές καταστάσεις εισηγμένων εταιρειών). Απαντάς πάντα με ΕΝΑ κεφαλαίο "
    "λατινικό γράμμα και τίποτε άλλο."
)

USER = """Κείμενο:
{text}

Υποψήφιο απόσπασμα: «{span}»

Ερώτηση: Μέσα σε αυτό το κείμενο, τι είναι το απόσπασμα «{span}»;

A = ΠΡΟΣΩΠΟ — κύριο όνομα φυσικού προσώπου
B = ΟΡΓΑΝΙΣΜΟΣ — επωνυμία εταιρείας, τράπεζας, δημόσιου φορέα, χρηματιστηρίου ή θεσμού
C = ΤΟΠΟΘΕΣΙΑ — όνομα χώρας, πόλης, περιοχής ή διεύθυνσης
D = ΤΙΠΟΤΑ — δεν είναι ονομαστική οντότητα (π.χ. κοινό ουσιαστικό όπως «Εταιρεία», «Όμιλος»,
    λογιστικός όρος, αριθμός, ημερομηνία), Ή είναι ελλιπές/υπερβολικά μεγάλο κομμάτι
    και όχι ακριβώς η πλήρης επωνυμία/όνομα.

Απάντησε με ένα γράμμα (A, B, C ή D):"""


def score_one(client, model, text, span, retries=3):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER.format(text=text, span=span)}],
                max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=10,
            )
            lp = r.choices[0].logprobs.content[0].top_logprobs
            import math
            raw = {}
            for e in lp:
                t = e.token.strip().upper()
                if t in CH2TYPE:
                    raw[t] = max(raw.get(t, -1e9), e.logprob)
            if not raw:
                raw = {"D": 0.0}
            m = max(raw.values())
            ex = {k: math.exp(v - m) for k, v in raw.items()}
            z = sum(ex.values())
            return {k: ex.get(k, 0.0) / z for k in "ABCD"}
        except Exception as e:
            if a == retries - 1:
                return {"err": str(e)[:200]}
    return {"err": "unreachable"}


def prefilter(c):
    """Cheap, purely surface-level pruning of the 2,651-candidate inventory.
    Everything dropped here is dropped BEFORE the model sees it; the drop rules
    use no gold information."""
    s = c["surface"].strip()
    if len(s) < 2:
        return False
    if not re.search(r"[^\W\d_]", s):          # no letters at all
        return False
    if c["n_tokens"] > 12:
        return False
    if s[0].islower():                          # Greek entities are capitalised
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default="candclf")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    P = load_passages()
    inv = json.load(open(f"{ROOT}/out/alt/candidate_spans.json"))
    client = CFG.client()

    jobs = []
    for row in inv["rows"]:
        rid = row["id"]
        for c in row["candidates"]:
            if prefilter(c):
                jobs.append((rid, c))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"candidates total={sum(len(r['candidates']) for r in inv['rows'])} "
          f"after prefilter={len(jobs)}", flush=True)

    res = [None] * len(jobs)
    lock = threading.Lock(); done = [0]

    def work(i):
        rid, c = jobs[i]
        res[i] = score_one(client, CFG.MODEL, P[rid], c["surface"])
        with lock:
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"  {done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, range(len(jobs))))

    errs = sum(1 for r in res if r and "err" in r)
    out = defaultdict(list)
    for (rid, c), r in zip(jobs, res):
        if not r or "err" in r:
            continue
        out[rid].append({"surface": c["surface"], "start": c["start"], "end": c["end"],
                         "n_tokens": c["n_tokens"], "sources": c["sources"],
                         "p": {k: round(v, 5) for k, v in r.items()}})
    path = f"{ROOT}/out/alt/{args.out_prefix}_scores.json"
    json.dump({"meta": {"model": CFG.MODEL, "n_jobs": len(jobs), "n_errors": errs},
               "rows": out}, open(path, "w"), ensure_ascii=False)
    print(f"WROTE {path}  errors={errs}", flush=True)


if __name__ == "__main__":
    main()
