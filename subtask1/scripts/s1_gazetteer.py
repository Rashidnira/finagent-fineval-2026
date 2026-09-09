#!/usr/bin/env python
"""Build PER / ORG / LOC gazetteers + a Greek 'common word' list from the CLEAN
external corpora only (all verified CLEAN by scripts/s1_contam_all.py).

Nothing here touches the test gold. Output: out/alt/gazetteer.json
"""
import argparse, collections, json, os, re, sys, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finnlp_config as C
from s1_extract_external import LOADERS

DEFAULT_OUT = os.path.join(C.OUT, "alt", "gazetteer.json")

# Source corpus -> how its entity types map onto ΠΡΟΣΩΠΟ / ΟΡΓΑΝΙΣΜΟΣ / ΤΟΠΟΘΕΣΙΑ
TYPE_MAP = {
    "PER": "PER", "PERSON": "PER",
    "ORG": "ORG", "ORGANISATION": "ORG", "organization": "ORG",
    "LOC": "LOC", "GPE": "LOC", "LOCATION-UNK": "LOC", "LOCATION-NAT": "LOC",
    "location": "LOC", "location / city": "LOC", "location / country": "LOC",
    "country": "LOC", "person": "PER", "ADDRESS": "LOC",
}
# Corpora contributing gazetteer entries (all CLEAN, all Greek)
GAZ_SOURCES = ["wikiann_el", "elNER4", "elNER18", "openner_core_elNER_ell",
               "greek_legal_ner", "polyglot_ner_el", "mapa_el", "finerweb_ell"]
# Corpora contributing running text for the common-word frequency model
FREQ_SOURCES = ["polyglot_ner_el", "elNER4", "wikiann_el", "greek_legal_ner"]

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def deacc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def key(s):
    return re.sub(r"\s+", " ", deacc(str(s)).casefold()).strip()


def main():
    ap = argparse.ArgumentParser(description="Build the external-data Greek NER gazetteer.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    gaz = {"PER": collections.Counter(), "ORG": collections.Counter(), "LOC": collections.Counter()}
    ent_tokens = collections.Counter()   # token -> times seen inside ANY entity
    all_tokens = collections.Counter()   # token -> times seen at all
    provenance = collections.defaultdict(lambda: collections.Counter())

    loaded = {}
    for name in set(GAZ_SOURCES) | set(FREQ_SOURCES):
        r = LOADERS[name]()
        if r is None:
            print("skip (not downloaded):", name); continue
        loaded[name] = r

    for name in GAZ_SOURCES:
        r = loaded.get(name)
        if not r: continue
        n = 0
        for surface, typ in r["entities"]:
            m = TYPE_MAP.get(typ)
            if not m: continue
            s = re.sub(r"\s+", " ", str(surface)).strip()
            if not s or len(s) < 2: continue
            gaz[m][s] += 1
            provenance[m][name] += 1
            n += 1
            for w in WORD.findall(s):
                ent_tokens[key(w)] += 1
        print(f"{name:<26} contributed {n} gazetteer entries")

    for name in FREQ_SOURCES:
        r = loaded.get(name)
        if not r: continue
        for sent in r["sentences"]:
            for w in WORD.findall(sent):
                all_tokens[key(w)] += 1

    # A token is a "common Greek word" (bad NE evidence) when it is frequent in
    # running text yet is almost never part of an annotated entity.
    common = sorted(w for w, c in all_tokens.items()
                    if c >= 200 and ent_tokens.get(w, 0) / c < 0.02)

    out = {
        "note": "Built ONLY from external corpora verified CLEAN against the 100 test passages.",
        "sources_used": GAZ_SOURCES,
        "freq_sources": FREQ_SOURCES,
        "counts": {k: len(v) for k, v in gaz.items()},
        "provenance": {k: dict(v) for k, v in provenance.items()},
        "n_common_words": len(common),
        "PER": [w for w, _ in gaz["PER"].most_common()],
        "ORG": [w for w, _ in gaz["ORG"].most_common()],
        "LOC": [w for w, _ in gaz["LOC"].most_common()],
        "common_words": common,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print("counts:", out["counts"], "common_words:", len(common))
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
