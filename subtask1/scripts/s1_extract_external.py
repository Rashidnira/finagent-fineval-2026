#!/usr/bin/env python
"""Load every downloaded external Greek resource into a uniform
{name: {"sentences":[str], "entities":[(surface,TYPE)], "meta":{}}} view.

Every resource here was downloaded by this script's companion commands into
data/external/. Nothing is cited that does not load."""
import glob, json, os, re
import pandas as pd

ROOT = os.environ.get("FINNLP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.environ.get("FINNLP_EXTERNAL_DATA", os.path.join(ROOT, "data", "external"))


def _iob_spans(tokens, tags):
    out, cur, curtype = [], [], None
    for t, g in zip(tokens, tags):
        g = str(g)
        if g.startswith("B-"):
            if cur: out.append((" ".join(cur), curtype))
            cur, curtype = [t], g[2:]
        elif g.startswith("I-") and cur and g[2:] == curtype:
            cur.append(t)
        elif g.startswith("I-"):          # I- without B-: treat as new
            if cur: out.append((" ".join(cur), curtype))
            cur, curtype = [t], g[2:]
        else:
            if cur: out.append((" ".join(cur), curtype)); cur, curtype = [], None
    if cur: out.append((" ".join(cur), curtype))
    return out


def _pq(pat):
    fs = sorted(glob.glob(pat))
    if not fs: return None
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_iob2_file(path):
    sents, tagseqs, toks, tgs = [], [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if not line.strip():
            if toks: sents.append(toks); tagseqs.append(tgs); toks, tgs = [], []
            continue
        parts = line.split()
        if len(parts) < 2: continue
        toks.append(parts[0]); tgs.append(parts[-1])
    if toks: sents.append(toks); tagseqs.append(tgs)
    return sents, tagseqs


def wikiann_el():
    d = _pq(f"{EXT}/wikiann_el/*.parquet")
    if d is None: return None
    LAB = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC"]
    sents, ents = [], []
    for tk, tg in zip(d["tokens"], d["ner_tags"]):
        tk = list(tk); tg = [LAB[int(i)] for i in tg]
        sents.append(" ".join(tk)); ents += _iob_spans(tk, tg)
    return dict(sentences=sents, entities=ents, meta={"rows": len(d), "label_scheme": "PER/ORG/LOC"})


def elner(version):
    sents, ents = [], []
    per_split = {}
    for f in sorted(glob.glob(f"{EXT}/elNER_extract/{version}/IOB2/*.iob2")):
        s, t = load_iob2_file(f)
        per_split[os.path.basename(f)] = len(s)
        for tk, tg in zip(s, t):
            sents.append(" ".join(tk)); ents += _iob_spans(tk, tg)
    if not sents: return None
    return dict(sentences=sents, entities=ents, meta={"splits": per_split})


def greek_legal_ner():
    sents, ents = [], []
    per = {}
    for f in sorted(glob.glob(f"{EXT}/greek_legal_ner/*.jsonl")):
        n = 0
        for line in open(f, encoding="utf-8"):
            o = json.loads(line); tk = o["words"]; tg = o["ner"]
            sents.append(" ".join(tk)); ents += _iob_spans(tk, tg); n += 1
        per[os.path.basename(f)] = n
    return dict(sentences=sents, entities=ents, meta={"splits": per})


def polyglot_el():
    d = _pq(f"{EXT}/polyglot_ner_el/el/train/*.parquet")
    if d is None: return None
    sents, ents = [], []
    for tk, tg in zip(d["words"], d["ner"]):
        tk = list(tk); tg = list(tg)
        sents.append(" ".join(tk)); ents += _iob_spans(tk, ["B-"+x if x != "O" else "O" for x in tg])
    return dict(sentences=sents, entities=ents, meta={"rows": len(d)})


def mapa_el():
    d = _pq(f"{EXT}/mapa/*.parquet")
    if d is None: return None
    d = d[d["language"] == "el"]
    sents, ents = [], []
    for tk, tg in zip(d["tokens"], d["coarse_grained"]):
        tk = list(tk); tg = list(tg)
        sents.append(" ".join(tk)); ents += _iob_spans(tk, tg)
    return dict(sentences=sents, entities=ents, meta={"rows": len(d)})


def finerweb_ell():
    d = _pq(f"{EXT}/finerweb_ell/*.parquet")
    if d is None: return None
    ents = []
    for txt, cs in zip(d["text"], d["char_spans"]):
        txt = str(txt)
        try:
            for sp in cs:
                ents.append((txt[int(sp["start"]):int(sp["end"])], str(sp["label"])))
        except Exception:
            pass
    return dict(sentences=[str(x) for x in d["text"]], entities=ents, meta={"rows": len(d)})


def sredfm_el():
    d = _pq(f"{EXT}/sredfm_el/*.parquet")
    if d is None: return None
    ents = []
    for e in d["entities"]:
        try:
            for x in e: ents.append((str(x.get("surfaceform","")), str(x.get("type","?"))))
        except Exception: pass
    return dict(sentences=[str(x) for x in d["text"]], entities=ents, meta={"rows": len(d)})


# bltlab/open-ner-core-types ships ner_tags as ClassLabel ints. Verified against the
# raw elNER4 IOB2 counts: id 3 -> B-ORG (13526), 4 -> I-ORG (9402/9401), 5 -> B-PER
# (10878 == elNER4 B-PERSON), 6 -> I-PER (6262 == elNER4 I-PERSON).
OPENNER_CORE_LABELS = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER"]


def openner_elner():
    d = _pq(f"{EXT}/openner_elner_ell/*.parquet")
    if d is None: return None
    sents, ents = [], []
    for tk, tg in zip(d["tokens"], d["ner_tags"]):
        tk = list(tk); tg = [OPENNER_CORE_LABELS[int(x)] for x in tg]
        sents.append(" ".join(tk)); ents += _iob_spans(tk, tg)
    return dict(sentences=sents, entities=ents, meta={"rows": len(d)})


def humadex():
    d = _pq(f"{EXT}/humadex_greek_ner/*.parquet")
    if d is None: return None
    col = "text" if "text" in d.columns else d.columns[0]
    return dict(sentences=[str(x) for x in d[col]], entities=[], meta={"rows": len(d), "cols": list(d.columns)})


LOADERS = {
    "wikiann_el": wikiann_el,
    "elNER4": lambda: elner("elNER4"),
    "elNER18": lambda: elner("elNER18"),
    "openner_core_elNER_ell": openner_elner,
    "greek_legal_ner": greek_legal_ner,
    "polyglot_ner_el": polyglot_el,
    "mapa_el": mapa_el,
    "finerweb_ell": finerweb_ell,
    "sredfm_el": sredfm_el,
    "humadex_greek_ner": humadex,
}

if __name__ == "__main__":
    import collections, sys
    for name, fn in LOADERS.items():
        try:
            r = fn()
        except Exception as e:
            print(f"{name:<26} LOAD FAILED {type(e).__name__}: {e}"); continue
        if r is None:
            print(f"{name:<26} NOT DOWNLOADED"); continue
        c = collections.Counter(t for _, t in r["entities"])
        print(f"{name:<26} sents={len(r['sentences']):<8} ents={len(r['entities']):<8} types={dict(c.most_common(12))}")
        print(f"{'':<26} meta={r['meta']}")
