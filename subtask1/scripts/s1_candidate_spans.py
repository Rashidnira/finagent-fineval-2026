import os
#!/usr/bin/env python
"""Candidate-span inventory for the 100 subtask-1 TEST PASSAGES.

Legitimacy: the test passages are UNLABELLED INPUT we are given. This script
reads only `text` from data/hf/.../test.parquet and the gazetteers built in
out/alt/gazetteer.json (which come from external corpora verified CLEAN).
It never touches any gold label.

Produces a HIGH-RECALL pool for a later reranker:
  * every maximal capitalised token run, and every contiguous sub-run of it
  * every span ending in a corporate legal-form marker (Α.Ε., ΑΕ, Ε.Π.Ε., LTD,
    PLC, Α.Ν.Ε., Ο.Ε., ΑΕΒΕ, ΙΚΕ, S.A., GmbH, ...)
  * every gazetteer hit (PER / ORG / LOC) anchored on a capitalised token
  * person-name shaped bigrams (known Greek given name + surname-suffix token)

Output: out/alt/candidate_spans.json
"""
import json, os, re, sys, unicodedata
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.environ.get("FINNLP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = f"{ROOT}/data/hf/finnlp2026-subtask1-greek-ner/test.parquet"
GAZ = f"{ROOT}/out/alt/gazetteer.json"
OUT = f"{ROOT}/out/alt/candidate_spans.json"

MAX_SPAN_TOKENS = 6        # cap for exhaustive sub-run enumeration
MAX_SPAN_TOKENS_LONG = 16  # cap for whole runs / legal-form spans (long ΑΕ names)

# ---------------------------------------------------------------- tokenisation
TOKEN_RE = re.compile(
    r"(?:[^\W\d_]\.){2,}"          # dotted acronym: Α.Ε.  Ε.Π.Ε.  S.A.
    r"|[^\W\d_]{1,4}\.(?=\s*[^\W\d_])"  # initial/abbrev + dot: "Α." "ΧΑΡ." "ΗΛ."
    r"|[^\W\d_]+(?:['’][^\W\d_]+)*"  # word (κατ'αρχήν)
    r"|\d+(?:[.,]\d+)*"            # number
    r"|[^\s\w]",                   # single punctuation char
    re.UNICODE,
)
SENT_END = {".", ";", "!", "·", "\n"}

# Corporate legal-form markers, matched dot/case/accent-insensitively.
LEGAL_FORMS_RAW = [
    "Α.Ε.", "ΑΕ", "Α.Ε", "Ε.Π.Ε.", "ΕΠΕ", "Ο.Ε.", "ΟΕ", "Ε.Ε.",
    "Α.Ν.Ε.", "ΑΝΕ", "Α.Ε.Β.Ε.", "ΑΕΒΕ", "Α.Ε.Ε.", "ΑΕΕ", "Ι.Κ.Ε.", "ΙΚΕ",
    "Α.Ε.Ε.Α.Π.", "ΑΕΕΑΠ", "Α.Ε.Π.Ε.Υ.", "ΑΕΠΕΥ", "Α.Χ.Ε.Π.Ε.Υ.", "ΑΧΕΠΕΥ",
    "Α.Ε.Β.Ε.Ε.", "ΑΕΒΕΕ", "Α.Ε.Τ.Ε.", "Μ.Α.Ε.", "ΜΑΕ", "Μ.Ε.Π.Ε.", "ΜΕΠΕ",
    "Α.Ε.Γ.Α.", "ΑΕΓΑ", "Ν.Π.Ι.Δ.", "ΝΠΙΔ", "Ν.Π.Δ.Δ.", "ΝΠΔΔ",
    "LTD", "LTD.", "LIMITED", "PLC", "PLC.", "S.A.", "SA", "GMBH", "AG",
    "INC", "INC.", "CORP", "CORP.", "CO.", "N.V.", "B.V.", "S.P.A.", "SPA",
    "SRL", "S.R.L.", "LLC", "L.L.C.", "OYJ", "AB", "A/S", "HOLDINGS", "GROUP",
]


def deacc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def key(s):
    return re.sub(r"\s+", " ", deacc(str(s)).casefold()).strip()


def legal_key(s):
    return re.sub(r"[^\w]", "", deacc(str(s)).casefold())


LEGAL_KEYS = {legal_key(x) for x in LEGAL_FORMS_RAW}

# lowercase words allowed INSIDE a name run (Greek articles / prepositions,
# English/French connectors) — only when flanked by capitalised tokens.
INNER_JOINERS = {key(x) for x in [
    "της", "του", "των", "τη", "την", "το", "τα", "και", "στην", "στη", "στο",
    "στον", "στα", "επί", "and", "of", "the", "for", "de", "di", "da", "van",
    "von", "der", "den", "del", "la", "le", "el", "&", "-", "–",
]}

MONTHS = {key(x) for x in [
    "Ιανουάριος", "Ιανουαρίου", "Ιανουάριο", "Φεβρουάριος", "Φεβρουαρίου", "Φεβρουάριο",
    "Μάρτιος", "Μαρτίου", "Μάρτιο", "Απρίλιος", "Απριλίου", "Απρίλιο",
    "Μάιος", "Μαΐου", "Μάιο", "Ιούνιος", "Ιουνίου", "Ιούνιο",
    "Ιούλιος", "Ιουλίου", "Ιούλιο", "Αύγουστος", "Αυγούστου", "Αύγουστο",
    "Σεπτέμβριος", "Σεπτεμβρίου", "Σεπτέμβριο", "Οκτώβριος", "Οκτωβρίου", "Οκτώβριο",
    "Νοέμβριος", "Νοεμβρίου", "Νοέμβριο", "Δεκέμβριος", "Δεκεμβρίου", "Δεκέμβριο",
    "Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή",
]}

PERSON_TITLES = {key(x) for x in [
    "κ", "κ.", "κος", "κα", "κύριος", "κυρία", "Πρόεδρος", "Προέδρου", "Πρόεδρο",
    "Αντιπρόεδρος", "Διευθύνων", "Σύμβουλος", "Διευθυντής", "Γενικός",
    "Οικονομικός", "Δρ", "Dr", "Mr", "Ms", "Mrs",
]}

SURNAME_SUFFIXES = ("οπουλος", "οπουλου", "ιδης", "ιδη", "αδης", "αδη", "ακης",
                    "ακη", "ελης", "ελη", "ογλου", "ατος", "ατου", "ης", "η",
                    "ος", "ου", "as", "is", "os")


def is_upperish(tok):
    """First alphabetic char is uppercase."""
    for c in tok:
        if c.isalpha():
            return c.isupper()
    return False


def is_allcaps(tok):
    letters = [c for c in tok if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def tokenize(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


_RES = None


def load_resources():
    global _RES
    if _RES is not None:
        return _RES
    gaz = json.load(open(GAZ, encoding="utf-8"))
    common = set(gaz["common_words"])
    G = {t: set() for t in ("PER", "ORG", "LOC")}
    maxn = {t: 1 for t in G}
    for t in G:
        for e in gaz[t]:
            k = key(e)
            if not k:
                continue
            n = len(k.split())
            if n > MAX_SPAN_TOKENS:
                continue
            G[t].add(k)
            maxn[t] = max(maxn[t], n)
    # given-name lexicon = first token of multi-token PER gazetteer entries
    given = Counter()
    for e in gaz["PER"]:
        parts = key(e).split()
        if len(parts) >= 2:
            given[parts[0]] += 1
    GIVEN = {w for w, c in given.items() if c >= 2}
    _RES = (common, G, GIVEN)
    return _RES


def candidates_for(text):
    """All candidate spans for one passage. Pure function of `text` + gazetteers."""
    common, G, GIVEN = load_resources()
    if True:
        toks = tokenize(text)
        n = len(toks)
        upper = [is_upperish(t) for t, _, _ in toks]
        kk = [key(t) for t, _, _ in toks]
        # sentence-initial positions
        sent_init = [False] * n
        if n:
            sent_init[0] = True
        for i in range(1, n):
            if toks[i - 1][0] in SENT_END or toks[i - 1][0] in {"«", "\""}:
                sent_init[i] = True

        cands = defaultdict(lambda: {"sources": set(), "gaz": {}})

        def add(i, j, source, extra=None, maxlen=MAX_SPAN_TOKENS):
            """token slice [i, j) -> candidate."""
            if j <= i or (j - i) > maxlen:
                return
            s, e = toks[i][1], toks[j - 1][2]
            surf = text[s:e].strip()
            if not surf or not any(c.isalpha() for c in surf):
                return
            c = cands[(s, e)]
            c["surface"] = surf
            c["tok_i"], c["tok_j"] = i, j
            c["sentence_initial"] = sent_init[i]
            c["sources"].add(source)
            if extra:
                c["gaz"].update(extra)

        # ---- 1. maximal capitalised runs (+ every contiguous sub-run) -------
        i = 0
        runs = []
        while i < n:
            if not upper[i]:
                i += 1
                continue
            j = i + 1
            while j < n:
                if upper[j]:
                    j += 1
                elif kk[j] in INNER_JOINERS and j + 1 < n and upper[j + 1]:
                    j += 2
                elif (toks[j][1] == toks[j - 1][2] and j + 1 < n
                      and toks[j + 1][1] == toks[j][2] and upper[j + 1]):
                    j += 2      # glued: no whitespace between the pieces
                else:
                    break
            runs.append((i, j))
            i = j
        for (a, b) in runs:
            add(a, b, "cap_run_maximal", maxlen=MAX_SPAN_TOKENS_LONG)
            for x in range(a + 1, b):        # suffixes  (drop leading words)
                add(x, b, "cap_run_suffix", maxlen=MAX_SPAN_TOKENS_LONG)
            for y in range(a + 1, b):        # prefixes  (drop trailing words)
                add(a, y, "cap_run_prefix", maxlen=MAX_SPAN_TOKENS_LONG)
            for x in range(a, b):
                for y in range(x + 1, min(b, x + MAX_SPAN_TOKENS) + 1):
                    if (x, y) == (a, b):
                        continue
                    # drop sub-runs that are only a joiner / only a month
                    ks = kk[x:y]
                    if all(t in INNER_JOINERS for t in ks):
                        continue
                    add(x, y, "cap_run_sub")
            # sentence-initial run whose first token is a common Greek word:
            # also emit the run minus that token (e.g. "Στις 08.11.2019, η ΟΠΑΠ")
            if sent_init[a] and kk[a] in common and b - a > 1:
                add(a + 1, b, "cap_run_strip_sentence_initial", maxlen=MAX_SPAN_TOKENS_LONG)

        # ---- 1b. one preceding lowercase head noun ------------------------
        # Greek toponyms/orgs are often headed by a lowercase common noun that the
        # annotators include: "ρέμα Πικροδάφνης", "φαράγγι Ρίχτη", "νομό Αττικής".
        for (a, b) in runs:
            if a > 0 and not upper[a - 1] and toks[a - 1][0].isalpha() and len(toks[a - 1][0]) > 2:
                add(a - 1, b, "lowercase_head_ext", maxlen=MAX_SPAN_TOKENS_LONG)
                if b - a > 1:
                    add(a - 1, a + 1, "lowercase_head_ext")

        # ---- 1c. trailing-period variants ---------------------------------
        # gold in Greek corpora sometimes swallows the sentence period ("ΕΕ.")
        for (a, b) in runs:
            if b < n and toks[b][0] == "." and toks[b][1] == toks[b - 1][2]:
                add(a, b + 1, "trailing_period_variant", maxlen=MAX_SPAN_TOKENS_LONG)

        # ---- 1d. cap-run + trailing integer (street addresses: "Εγνατίας 127")
        for (a, b) in runs:
            if b < n and re.fullmatch(r"\d{1,5}", toks[b][0]):
                add(a, b + 1, "street_number_ext", maxlen=MAX_SPAN_TOKENS_LONG)
                add(b - 1, b + 1, "street_number_ext")

        # ---- 1e. run + parenthesised acronym: "QATAR FOUNDATION STADIUM (QFS)"
        for (a, b) in runs:
            if b + 2 < n and toks[b][0] == "(" and upper[b + 1] and toks[b + 2][0] == ")":
                add(a, b + 3, "paren_acronym_ext", maxlen=MAX_SPAN_TOKENS_LONG)

        # ---- 2. legal-form-marker terminated spans -------------------------
        for j in range(n):
            if legal_key(toks[j][0]) not in LEGAL_KEYS:
                continue
            end = j + 1
            # absorb a trailing bare '.' that belongs to the abbreviation
            for a in range(max(0, j - MAX_SPAN_TOKENS_LONG + 1), j + 1):
                # a company name starts at a capitalised token or an opening quote,
                # never mid-clause: this keeps the pool from swallowing whole sentences
                if not (upper[a] or toks[a][0] in {"«", '"', "\u201c"}):
                    continue
                if not any(upper[x] for x in range(a, j)):
                    continue
                add(a, end, "legal_form", maxlen=MAX_SPAN_TOKENS_LONG)
            add(j, end, "legal_form_marker_only")

        # ---- 3. gazetteer hits (anchored on a capitalised token) -----------
        for a in range(n):
            if not upper[a]:
                continue
            for b in range(a + 1, min(n, a + MAX_SPAN_TOKENS) + 1):
                k = " ".join(kk[a:b])
                hits = {t: True for t in G if k in G[t]}
                if hits:
                    add(a, b, "gazetteer", {t: True for t in hits})

        # ---- 4. person-shaped bigrams / title-anchored --------------------
        for a in range(n - 1):
            if not (upper[a] and upper[a + 1]):
                continue
            if kk[a] in MONTHS or kk[a + 1] in MONTHS:
                continue
            if kk[a] in GIVEN or deacc(kk[a + 1]).endswith(SURNAME_SUFFIXES):
                add(a, a + 2, "person_shape")
        for a in range(n):
            if kk[a].rstrip(".") in PERSON_TITLES:
                b = a + 1
                while b < n and upper[b] and b - a <= 3:
                    b += 1
                if b > a + 1:
                    add(a + 1, b, "person_title_anchored")

        out = []
        for (s, e), c in sorted(cands.items()):
            out.append({
                "surface": c["surface"], "start": s, "end": e,
                "n_tokens": c["tok_j"] - c["tok_i"],
                "sentence_initial": c["sentence_initial"],
                "sources": sorted(c["sources"]),
                "gazetteer_types": sorted(c["gaz"].keys()),
                "all_caps": is_allcaps(c["surface"].replace(" ", "")),
                "first_token_is_common_word": key(c["surface"].split()[0]) in common,
            })
        return out


def main():
    df = pd.read_parquet(TEST, columns=["id", "text"])
    df = df[df["id"].str.startswith("gr_text_")].reset_index(drop=True)
    rows, src_tally, len_tally = [], Counter(), Counter()
    for _, r in df.iterrows():
        text, rid = str(r["text"]), str(r["id"])
        out = candidates_for(text)
        for item in out:
            for sc in item["sources"]:
                src_tally[sc] += 1
            len_tally[item["n_tokens"]] += 1
        rows.append({"id": rid, "text": text, "n_candidates": len(out), "candidates": out})

    doc = {
        "meta": {
            "built_from": "data/hf/finnlp2026-subtask1-greek-ner/test.parquet (UNLABELLED `text` column only)",
            "gazetteer": "out/alt/gazetteer.json (external corpora, all verified CLEAN)",
            "uses_gold_labels": False,
            "n_passages": len(rows),
            "total_candidates": sum(r["n_candidates"] for r in rows),
            "candidates_per_passage_mean": round(sum(r["n_candidates"] for r in rows) / max(1, len(rows)), 1),
            "by_source": dict(src_tally.most_common()),
            "by_n_tokens": {str(k): v for k, v in sorted(len_tally.items())},
            "max_span_tokens": MAX_SPAN_TOKENS,
            "legal_form_markers": LEGAL_FORMS_RAW,
        },
        "rows": rows,
    }
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(doc["meta"], ensure_ascii=False, indent=2))
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
