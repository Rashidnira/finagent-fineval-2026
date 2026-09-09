#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic rule/regex numeric NER for subtask 1 (GRFinNUM conventions).

extract(text) -> list of (entity, TYPE) in text order, repeated per occurrence.
Every emitted entity is a verbatim substring of `text`.
"""
from __future__ import annotations
import re, unicodedata

TYPES = ("ΧΡΗΜΑΤΑ", "ΠΟΣΟΣΤΑ", "ΧΡΟΝΙΚΑ", "ΠΟΣΟΤΗΤΕΣ", "ΑΛΛΑ")

MONTHS = ("Ιανουαρίου|Φεβρουαρίου|Μαρτίου|Απριλίου|Μαΐου|Μαίου|Ιουνίου|Ιουλίου|Αυγούστου|"
          "Σεπτεμβρίου|Οκτωβρίου|Νοεμβρίου|Δεκεμβρίου|Ιανουάριος|Φεβρουάριος|Μάρτιος|Απρίλιος|"
          "Μάιος|Ιούνιος|Ιούλιος|Αύγουστος|Σεπτέμβριος|Οκτώβριος|Νοέμβριος|Δεκέμβριος|"
          "Ιανουάριο|Φεβρουάριο|Μάρτιο|Απρίλιο|Μάιο|Ιούνιο|Ιούλιο|Αύγουστο|Σεπτέμβριο|"
          "Οκτώβριο|Νοέμβριο|Δεκέμβριο")

NUM = r"\d+(?:[.,]\d+)*"

PCT_RE = re.compile(rf"{NUM}(?:\s?[-–]\s?{NUM})?\s?%")
DATE3_RE = re.compile(r"(?<!\d)(\d{1,2})([./\-])(\d{1,2})\2(\d{2,4})(?!\d)")
DATE2_RE = re.compile(r"(?<!\d)(\d{1,2})([./])(\d{1,2})(?![\d./])")
DATE_MONTHNAME_RE = re.compile(rf"(?<!\d)(\d{{1,2}})(?:η|ης|ας|ο[υς]?)?\s+({MONTHS})\s*,?\s*(\d{{4}})?(?!\d)")
LAW_RE = re.compile(r"(?:[νΝ]\.\s?[ΔΠ]?\.?|ΝΟΜΟ[ΣΥ]|νόμο[υς]?|Κ\.?Ν\.?|Ν\.Δ\.)\s*(\d{2,5})\s*/\s*(\d{4})")
LAW_SHORT_RE = re.compile(r"(?:[νΝ]\.\s?|νόμου?\s+|ΝΟΜΟ[ΣΥ]\s+)(\d{2,5})\s*/\s*(\d{2})(?!\d)")
LAW2_RE = re.compile(r"(?<![\d/])(\d{3,5})\s*/\s*(\d{4})(?![\d/])")
ART_RE = re.compile(r"(?:άρθρ[α-ωΑ-Ω]{0,3}|Άρθρ[α-ωΑ-Ω]{0,3}|άρ\.|παρ\.|παράγραφο[υς]?|περ\.|εδάφιο|Κανονισμ[όoό]?[ςυ]?)\s*"
                    rf"(\d+[α-ωΑ-Ω]?)")
ID_RE = re.compile(r"(?:[ΦF]\.?[ΕE]\.?[ΚK]\.?|ΓΕ\.?ΜΗ\.?|Γ\.?Ε\.?ΜΗ\.?|Σ\.?Ο\.?Ε\.?Λ\.?|Α\.?Μ\.?\s?ΣΟΕΛ|"
                   r"αρ\.\s?πρωτ\.|Α\.?Δ\.?|ΑΦΜ|Α\.Φ\.Μ\.|Α\.Μ\.Α\.Ε\.|ΑΡ\.?Μ\.?Α\.?Ε\.?|Τ\.?Κ\.?|"
                   r"Δ\.?Σ\.?\s?αριθ|αριθμ?ό?ν?\.?\s|αριθ\.|ΔΠΧΑ|ΔΛΠ|ΔΠΧΠ|IFRS|IAS|ΕΛΠ|ΔΠΛ)"r"\s*[:]?\s*(\d[\d./]*)")
DUR_RE = re.compile(rf"(?<!\d)({NUM})\s*\)?\s*(?:-|–)?\s*(?:ετ(?:ών|ους|ίας|ή|ια)|έτ(?:η|ους|ος)|μην(?:ών|ιαία|ες|ας)|"
                    r"μήν(?:ες|α)|ημερ(?:ών|ες)|ημέρ(?:ες|α)|εβδομάδ(?:ες|ων)|τριμήν(?:ου|ων|α)|"
                    r"εξάμην(?:ου|α)|χρόνι(?:α|ων)|ώρ(?:ες|ών))")
GLUED_DUR_RE = re.compile(r"(?<!\d)(\d+)(?=ετ[οίηώ]|ετής|ετούς|ετία|μην|ήμερ|ωρ[οη])")
YEAR_RE = re.compile(r"(?<![\d.,/])(18\d{2}|19\d{2}|20\d{2})(?!\d|[.,/]\d|\s?%)")
QUARTER_RE = re.compile(r"(?:Q|Τ)(\d)(?=\s|/|$)")
TOKEN_RE = re.compile(rf"{NUM}")
PAREN_COUNT_RE = re.compile(r"\((\d{1,4})\)")

MONEY_CUES = ("€", "ευρώ", "Ευρώ", "ΕΥΡΩ", "EUR", "eur", "ποσό", "ποσού", "ποσά", "ποσών", "τίμημα",
              "τιμήματος", "αξία", "αξίας", "κεφάλαι", "δάνει", "μέρισμα", "μερίσμα", "τζίρο",
              "κέρδ", "ζημι", "ζημί", "πωλήσε", "έσοδ", "εσόδ", "κύκλος εργασιών", "EBITDA",
              "χιλ.", "χιλιάδ", "εκατ", "εκ.", "δισ", "$", "δολ", "τιμή", "αμοιβ", "χρηματοδότησ",
              "επένδυσ", "επενδύσ", "καταβολ", "οφειλ", "υποχρεώσε", "απαιτήσε", "φόρο", "πρόστιμ",
              "εισφορ", "μισθ", "προμήθει", "δαπάν", "κόστο", "κόστος", "αποζημίωσ", "ομολογιακ",
              "τραπεζικ", "ταμειακ", "ρευστ", "λογιστική αξία", "ονομαστικής αξίας")
QTY_CUES = ("τ.μ", "τετραγωνικ", "μ2", "MW", "MWh", "KW", "κυβικ", "στρέμμα", "στρεμμάτ",
            "μετοχές", "μετοχών", "μετοχες", "μετοχή", "τεμάχ", "εργαζόμεν", "υπάλληλ", "άτομα", "ατόμων", "προσωπικ",
            "κατάστημα", "καταστήμα", "χώρες", "χωρών", "σημεία", "υποκαταστήμα", "πλοί",
            "οχήμα", "μονάδ", "συνδρομητ", "πελάτ", "θέσεις", "τόνο", "κιλά", "λίτρα",
            "δικαιώματα ψήφου", "ομολογί", "ονομαστικών μετοχών", "κοινών μετοχών")
TIME_CUES = ("χρήση", "χρήσης", "χρήσεως", "έτος", "έτους", "ημερομηνία", "περίοδο", "περιόδου",
             "χρονιά", "τρίμηνο", "εξάμηνο", "χρήσεις")


def _win(text, s, e, back=45, fwd=40):
    return text[max(0, s - back):s], text[e:e + fwd]


def _has(cues, *chunks):
    for c in chunks:
        for q in cues:
            if q in c:
                return True
    return False


def extract(text: str):
    n = len(text)
    claimed = [None] * n          # char -> type assigned by a high-precision rule
    spans = []                    # (start, end, entity, type)

    def claim(s, e, typ, ent=None):
        if any(claimed[i] is not None for i in range(s, e)):
            return False
        ent = ent if ent is not None else text[s:e]
        for i in range(s, e):
            claimed[i] = typ
        spans.append((s, e, ent, typ))
        return True

    # 1) percentages (span includes % and its exact spacing)
    for m in PCT_RE.finditer(text):
        claim(m.start(), m.end(), "ΠΟΣΟΣΤΑ")

    # 4) full dates dd.mm.yyyy -> three ΧΡΟΝΙΚΑ components
    for m in DATE3_RE.finditer(text):
        for gi in (1, 3, 4):
            claim(m.start(gi), m.end(gi), "ΧΡΟΝΙΚΑ")
    # 2) explicit law references  N. 3556/2007
    for m in LAW_RE.finditer(text):
        claim(m.start(1), m.end(1), "ΑΛΛΑ")
        claim(m.start(2), m.end(2), "ΧΡΟΝΙΚΑ")
    for m in LAW_SHORT_RE.finditer(text):
        claim(m.start(1), m.end(1), "ΑΛΛΑ")
        claim(m.start(2), m.end(2), "ΑΛΛΑ")
    for m in re.finditer(r"(?<![\d/])((?:19|20)\d{2})\s*/\s*(\d{1,4})(?![\d/])", text):
        if not re.match(r"^(19|20)\d{2}$", m.group(2)):
            claim(m.start(1), m.end(1), "ΧΡΟΝΙΚΑ")
            claim(m.start(2), m.end(2), "ΑΛΛΑ")
    for m in LAW2_RE.finditer(text):      # bare 1234/2007 pattern
        claim(m.start(1), m.end(1), "ΑΛΛΑ")
        claim(m.start(2), m.end(2), "ΧΡΟΝΙΚΑ")

    # 3) article / paragraph numbers, registry ids
    for m in re.finditer(r"(?:άρθρ[α-ωΑ-Ω]{0,3}|παρ\.)\s*(\d+)\s*(?:έως|εώς|-|–|και)\s*(\d+)", text):
        claim(m.start(1), m.end(1), "ΑΛΛΑ")
        claim(m.start(2), m.end(2), "ΑΛΛΑ")
    for m in ART_RE.finditer(text):
        g = m.group(1)
        d = re.match(r"\d+", g)
        claim(m.start(1), m.start(1) + d.end(), "ΑΛΛΑ")
    for m in ID_RE.finditer(text):
        dm = re.match(r"\d+", m.group(1))
        if dm:
            claim(m.start(1), m.start(1) + dm.end(), "ΑΛΛΑ")

    # 5) day + month-name (+ year)
    for m in DATE_MONTHNAME_RE.finditer(text):
        claim(m.start(1), m.end(1), "ΧΡΟΝΙΚΑ")
        if m.group(3):
            claim(m.start(3), m.end(3), "ΧΡΟΝΙΚΑ")
    # 6) partial dates dd.mm / dd/mm with a date cue nearby
    for m in DATE2_RE.finditer(text):
        b, f = _win(text, m.start(), m.end(), 30, 20)
        if _has(("από", "έως", "μέχρι", "την", "Την", "στις", "Στις", "περίοδο", "χρήση",
                 "ημερομηνία", "1.1", "31.12"), b, f) or re.search(r"^\s*(έως|-|–)", f):
            claim(m.start(1), m.end(1), "ΧΡΟΝΙΚΑ")
            claim(m.start(3), m.end(3), "ΧΡΟΝΙΚΑ")

    # 7) durations
    for m in DUR_RE.finditer(text):
        claim(m.start(1), m.end(1), "ΧΡΟΝΙΚΑ")
    for m in GLUED_DUR_RE.finditer(text):
        claim(m.start(1), m.end(1), "ΧΡΟΝΙΚΑ")

    # 8) bare years
    for m in YEAR_RE.finditer(text):
        claim(m.start(1), m.end(1), "ΧΡΟΝΙΚΑ")

    # 9) everything else: context-based typing
    UNIT_AFTER = re.compile(r"^\s?(MWh|MW|KWh|KW|GW|kg|χλμ|km|τ\.?μ\.?|μ2|%)")
    LATIN_NAME_BEFORE = re.compile(r"([A-Z][A-Za-z&.\-]*|«|\u201c|\")\s?$")
    LATIN_NAME_AFTER = re.compile(r"^\s?(?:[A-Za-z][A-Za-z&.\-]*|Α\.Ε\.|ΑΕ|Α\.Ε|»|\u201d)")
    COUNT_NOUN = re.compile(
        r"^[^;]{0,34}?(μετοχ[έώήα]|μετοχές|μετοχών|ομολογ|δάνεια|δανείων|ακίνητ|έργα|έργων|"
        r"δικαιωμ|συνδρομ|πελάτ|εταιρε[ίι]|καταστήμα|κατάστημα|υποκαταστ|άτομ|υπάλληλ|εργαζ|"
        r"χώρ[εα]|πλοί|οχήμα|μονάδ|τεμάχ|θέσε|σημεί|νοικοκυρ|προϊόντ|συμβάσε|μέλη|μέλος|"
        r"ξενοδοχ|στρέμ|όροφ|αίθουσ|φοιτητ|πελατ|πολίτ|κατοίκ|επιβάτ|ασθεν|μαθητ|χρήστ|"
        r"συνδέσε|δωμάτ|κλίν|τμχ)")
    COUNT_NOUN_BACK = re.compile(
        r"(νοικοκυρ|μετοχές|μετοχών|συνδρομ|πελάτ|ακίνητ|εργαζ|υπάλληλ|καταστήμα|τεμάχ|"
        r"δικαιωμάτων|ομολογ)[^;.]{0,34}$")
    SHARE_CTX = ("άμεσα", "έμμεσα", "δικαιωμάτων", "δικαιώματα", "προαίρεσης", "ενεχυρ")
    for m in TOKEN_RE.finditer(text):
        s0, e0 = m.start(), m.end()
        if any(claimed[i] is not None for i in range(s0, e0)):
            continue
        tok = m.group(0)
        b, f = _win(text, s0, e0)
        unit = UNIT_AFTER.match(f)
        if not unit:
            if re.search(r"[A-Za-z]$", b[-1:]) or re.match(r"^[A-Za-z]", f):
                continue
            if re.search(r"[A-Za-zΑ-Ωα-ω]-$", b):          # COVID-19, Τ-2
                continue
            if LATIN_NAME_BEFORE.search(b) and LATIN_NAME_AFTER.match(f):
                continue
        if re.search(r"(?:^|[^Α-Ωα-ωίϊΐόάέύϋΰήώ])(Λεωφ|Λεωφόρο[υς]?|οδο[ύς]|Οδό[ςυ]|Οδού)\b[^,]{0,30}$", b):
            continue
        near_b, near_f = text[max(0, s0 - 14):s0], text[e0:e0 + 22]
        bare_int = tok.isdigit()
        money_near = _has(("€", "ευρώ", "Ευρώ", "ΕΥΡΩ", "EUR", "$", "δολ", "ποσό", "ποσού"),
                          near_b, near_f)
        typ = None
        if unit:
            typ = "ΠΟΣΟΣΤΑ" if unit.group(1) == "%" else "ΠΟΣΟΤΗΤΕΣ"
        elif text[max(0, s0 - 1):s0] == "(" and text[e0:e0 + 1] == ")" and bare_int and not money_near:
            typ = "ΠΟΣΟΤΗΤΕΣ"
        elif (COUNT_NOUN.match(f) or COUNT_NOUN_BACK.search(b)) and not money_near:
            typ = "ΠΟΣΟΤΗΤΕΣ"
        elif money_near:
            typ = "ΧΡΗΜΑΤΑ"
        elif _has(("χιλ.", "χιλιάδ", "εκατ", "εκ.", "δισ"), near_b, near_f):
            typ = "ΧΡΗΜΑΤΑ"
        elif _has(QTY_CUES, b, f) or _has(SHARE_CTX, b, f):
            typ = "ΠΟΣΟΤΗΤΕΣ"
        elif _has(MONEY_CUES, b, f):
            typ = "ΧΡΗΜΑΤΑ" if not bare_int else "ΠΟΣΟΤΗΤΕΣ"
        elif _has(TIME_CUES, b, f):
            typ = "ΧΡΟΝΙΚΑ"
        else:
            if "," in tok or "." in tok:
                money_doc = any(q in text for q in ("€", "ευρώ", "Ευρώ", "ποσό", "ποσού",
                                                    "αξίας", "τίμημα", "τιμήματος", "$"))
                qty_doc = any(q in text for q in ("μετοχ", "δικαιωμ", "ομολογ", "τ.μ", "MW"))
                typ = "ΧΡΗΜΑΤΑ" if (money_doc or not qty_doc) else "ΠΟΣΟΤΗΤΕΣ"
            else:
                typ = "ΠΟΣΟΤΗΤΕΣ"
        claim(s0, e0, typ)

    spans.sort()
    return [(ent, typ) for _, _, ent, typ in spans]


def predict_str(text):
    p = extract(text)
    return "\n".join(f"{e}, {t}" for e, t in p) if p else "None"
