#!/usr/bin/env python
"""Assemble + validate the final subtask-1 submission (200 rows)."""
import os, sys, csv
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finnlp_config as C
from s1_rules import extract

NUM_T = {"ΧΡΗΜΑΤΑ","ΠΟΣΟΣΤΑ","ΧΡΟΝΙΚΑ","ΠΟΣΟΤΗΤΕΣ","ΑΛΛΑ"}
TXT_T = {"ΠΡΟΣΩΠΟ","ΟΡΓΑΝΙΣΜΟΣ","ΤΟΠΟΘΕΣΙΑ"}

te = pd.read_parquet(os.path.join(C.S1_DIR, "test.parquet"))
txt = pd.read_csv(f"{C.OUT}/s1_text_candclf.csv", dtype=str, keep_default_na=False)
txtmap = dict(zip(txt.id, txt.prediction))

recs = []
for r in te.itertuples():
    if r.id.startswith("gr_num"):
        pairs = extract(r.text)                      # regenerate fresh from rules
        pred = "\n".join(f"{e}, {t}" for e, t in pairs) or "None"
    else:
        pred = txtmap[r.id]
    recs.append({"id": r.id, "prediction": pred})

out = f"{C.OUT}/submission_s1_v4.csv"
pd.DataFrame(recs).to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

# ---------- validation ----------
raw = open(out, "rb").read()
d = pd.read_csv(out, dtype=str, keep_default_na=False)
text = dict(zip(te.id, te.text))
errs = []
if raw[:3] == b"\xef\xbb\xbf": errs.append("file has a UTF-8 BOM")
if b"\r\n" in raw: errs.append("file has CRLF line endings")
if len(d) != 200: errs.append(f"row count {len(d)} != 200")
if list(d.columns) != ["id","prediction"]: errs.append(f"columns {list(d.columns)}")
if set(d.id) != set(te.id): errs.append("id set mismatch vs test parquet")
if d.id.duplicated().any(): errs.append("duplicate ids")
if (d.prediction.str.strip() == "").any(): errs.append("empty prediction cell(s)")

bad_type = bad_verbatim = 0
for r in d.itertuples():
    if r.prediction.strip() == "None": continue
    legal = NUM_T if r.id.startswith("gr_num") else TXT_T
    for l in r.prediction.split("\n"):
        if "," not in l: errs.append(f"{r.id}: unparseable line {l!r}"); continue
        e, t = l.rsplit(",", 1); e, t = e.strip(), t.strip()
        if t not in legal: bad_type += 1
        if e not in text[r.id]: bad_verbatim += 1
if bad_type: errs.append(f"{bad_type} illegal type(s) for their half")
if bad_verbatim: errs.append(f"{bad_verbatim} span(s) not verbatim in text")

# paired-row collision check (numeric vs textual label sets are disjoint)
def spans(p): return {l.rsplit(",",1)[0].strip() for l in p.split("\n") if "," in l}
pm = dict(zip(d.id, d.prediction))
coll = sum(1 for i in range(1,101)
           if spans(pm.get(f"gr_num_{i:03d}","")) & spans(pm.get(f"gr_text_{i:03d}","")))

print(f"=== submission_s1_v4.csv ===")
print(f"bytes {len(raw)}  rows {len(d)}  ids ok {set(d.id)==set(te.id)}")
print(f"paired-row span collisions: {coll} (expect 0)")
print("VALIDATION:", "PASS — no defects" if not errs else "FAIL")
for e in errs: print("  !", e)
