#!/usr/bin/env python
"""
Subtask 2 - cue-based rule scorer (NO GPU).

Implements the decision procedure from subtask2/s2_label_guide.md directly as weighted regex cues,
emitting a 5-way score vector with the score JSON schema used by the calibration tools so it can be fed to
s2_calibrate.py / s2_submit.py unchanged, and so it can be averaged with the LLM logits.

Purpose: (a) a submission that exists even if the shared endpoint stays down, (b) an
independent signal to ensemble with / sanity-check the LLM.
"""
import argparse, json, os, re, sys
import numpy as np, pandas as pd

LABELS = ["+2", "+1", "0", "-1", "-2"]
QPAT = re.compile(r"Financial Question:\s*(.*?)\nCompany Response:\s*(.*?)\n\nDirectly output", re.S)

# --- cue lexicon. weight, regex, label it pushes toward -------------------------------
# NB: these were revised after reading the raw Japanese of the test items the v1 rules got
# obviously wrong (id 290 "決議済み" was not matched by 決議(し|いた); id 262/295 "考えていません"
# was matched by neither NEG_HARD nor NEG_SOFT; id 253 "想定しています" was not a +1 cue). The
# fixes come from the cue lists in subtask2/s2_label_guide.md, not from any hypothesised label.
DECIDED = r"(決議(し|いた|済|さ)|決定(し|いた|済|さ)|承認(し|いた|済)|締結(し|済|いた)|確定(して|しま|済)|正式に決|可決|合意に達し|署名し|決まりました|取締役会で(決議|承認|確認|決定))"
DONE    = r"(完了(します|しました|いたします)|開始します|実施します|移行します|適用します|充当します|返済します|継続します|実行します|導入します|払い戻します|上場します|着工します)"
# deliberately specific: a bare ありません/ございません also fires on factual statements
# ("交渉中の案件はありません") which are not refusals of the proposed action.
NEG_HARD = r"(^いいえ|予定(は|も)(あり|ござい)ません|計画(は|も)(あり|ござい)ません|(こと|事)(は|も)(あり|ござい)ません|する考えは(あり|ござい)ません|考えはありません|意思は(あり|ござい)ません|方針は(あり|ござい)ません|しません|いたしません|行いません|実施しません|提供しない|検討して(い)?ません|中止(しま|いた|し)|終了します|撤回(しま|いた)|除外して|白紙(に|と)|社内規程|お答えできません|回答(は|を)差し控え|開示する考えはな|ノーコメント)"
NEG_SOFT = r"(難しい|困難と|見送る|見送り|時期尚早|優先順位は低い|考えて(い|お)?(り)?ません|踏み切る考えは|慎重に|控えます|差し控え)"
REOPEN  = r"(ただし|但し|一方で|改めて(検討|判断|見直)|再検討|見直します|余地(は|が)あ|状況が変われば|環境が変わ|整った段階|数年後|将来的に(は)?検討|選択肢の一つ|高まれば|柔軟に)"
VOLIT   = r"(たいと考え|たい(と思|です|。)|努めて|目指し|目指す|注力して|進めてまいり|拡大したい|引き上げたい|検討してまいり|検討を進め|前向き|想定して(い|お)|判断します|排除して(い)?(お|ま)?らず)"
COND    = r"(れば|次第|整えば|得られれば|できれば|想定どおり|条件が|確認できれば|進捗次第|市況次第|場合には|可能性があります|応じて)"
NEUTRAL = r"(検証して|精査して|分析して|策定過程|策定中|評価して(い|お)|確認して(いる|おり)|段階です|据え置|次回(の)?(ご)?説明|発表時に(ご)?説明|お示しします|公表します|双方|総合的に(勘案|判断)|継続的に評価|検討(して|を進めて)いる段階)"
NUM     = r"(\d[\d,\.]*\s*(億円|百万円|万株|株|円|%|％|拠点|店|名|件|年|月|日))"
DATE    = r"(20\d\d\s*年|\d{1,2}\s*月\s*\d{0,2}\s*日?|今期末|来年度?下?期|本日|来月|次期|期限まで)"
NOPROMISE = r"(確約は(でき|致し)|約束(は|でき)|保証(は|でき)|決まって(い|お)りません|確定して(い|お)りません|決定した事実はあり|決定して(い|お)りません)"
# "(方針に)変更はありません" re-affirms an ALREADY-ANNOUNCED promise; the guide calls that a
# commitment, not a refusal, so it must not be counted as a hard negation.
REAFFIRM = r"(変更(は|も)?(あり|ござい)ません|変更は(ない|無い)|変更せず|変更しません|計画どおり継続|従来(の|どおり)[^。]*変更)"

def cues(text):
    f = {}
    f["decided"] = len(re.findall(DECIDED, text))
    f["done"] = len(re.findall(DONE, text))
    # strip re-affirmation phrases before counting hard negations: "(方針に)変更はありません"
    # is a commitment to an already-announced promise, not a refusal of the proposed action.
    # But a genuine refusal in the SAME response ("停止する予定はありません") must still count,
    # which is why we mask only the re-affirmation span rather than gating on its presence.
    f["neg_hard"] = len(re.findall(NEG_HARD, re.sub(REAFFIRM, "", text), flags=re.M))
    f["neg_soft"] = len(re.findall(NEG_SOFT, text))
    f["reopen"] = len(re.findall(REOPEN, text))
    f["volit"] = len(re.findall(VOLIT, text))
    f["cond"] = len(re.findall(COND, text))
    f["neutral"] = len(re.findall(NEUTRAL, text))
    f["num"] = len(re.findall(NUM, text))
    f["date"] = len(re.findall(DATE, text))
    f["noprom"] = len(re.findall(NOPROMISE, text))
    f["reaffirm"] = len(re.findall(REAFFIRM, text))
    return f


def score_row(Q, R):
    """Return a 5-vector of pseudo-logits, following s2_label_guide.md sec.4 as soft evidence."""
    f = cues(R)
    s = dict.fromkeys(LABELS, 0.0)
    reaff = f["reaffirm"] > 0
    hard_neg = f["neg_hard"] > 0
    neg = hard_neg or f["neg_soft"] > 0
    reopen = f["reopen"] > 0

    # 1. completed decision act
    dec = min(f["decided"], 2) * 2.0 + min(f["done"], 2) * 0.8
    if dec:
        spec = 0.7 * min(f["num"], 2) + 0.7 * min(f["date"], 2)
        if hard_neg:                       # decided NOT to act ("正式に中止しました")
            s["-2"] += dec + spec
            s["-1"] += 0.4 * dec
        else:
            s["+2"] += dec + spec
            s["+1"] += 0.5 * dec
    # 2. explicit negative about the action
    if hard_neg:
        s["-2"] += 2.2 * min(f["neg_hard"], 2)
        s["-1"] += 0.9 * min(f["neg_hard"], 2)
    if f["neg_soft"]:
        s["-1"] += 2.0 * min(f["neg_soft"], 2)
        s["-2"] += 0.7 * min(f["neg_soft"], 2)
    if neg and reopen:                     # the -1/-2 boundary is the reopening clause
        s["-1"] += 1.8
        s["-2"] -= 1.4
    if neg and not reopen:
        s["-2"] += 0.9
    # 3. positive intent, unsettled.  NOT gated on hard_neg: a response can state positive
    # intent and separately negate a FACT ("前向きに検討します。一方で交渉中の案件はありません"),
    # which v1 mis-scored as -2.
    w = 0.5 if hard_neg else 1.0
    s["+1"] += w * (1.5 * min(f["volit"], 2) + 0.9 * min(f["cond"], 2))
    if f["noprom"]:
        s["+1"] += 1.0
        s["+2"] -= 1.5
    if (f["volit"] or f["cond"]) and not dec:
        s["+2"] -= 1.0
    if reaff and not hard_neg:
        s["+2"] += 1.2
        s["0"] += 0.6
    # 4. neutral / process
    s["0"] += 1.6 * min(f["neutral"], 3)
    if not neg and not f["volit"] and not dec:
        s["0"] += 1.5
    if neg:
        s["0"] -= 1.2
    # NOTE on polarity: s2_label_guide.md sec.0 says that when the QUESTION proposes a negative
    # action ("will you STOP buybacks?"), a firm denial of stopping is still labelled -2, i.e. the
    # label tracks the negation in the RESPONSE, not the direction of the underlying business
    # action. So no sign flip is applied here. (An earlier version flipped +2/-2 in that case and
    # it disagreed with the guide on exactly the item the guide uses as its example, id 256.)
    return np.array([s[l] for l in LABELS], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    df = pd.read_parquet(a.data)
    if "Q" not in df.columns:
        qr = df["query"].apply(lambda q: pd.Series(QPAT.search(q).groups() if QPAT.search(q) else ("", "")))
        df["Q"], df["R"] = qr[0], qr[1]
    S = np.array([score_row(q, r) for q, r in zip(df["Q"], df["R"])]) * a.scale
    gold = df["answer"].astype(str).tolist() if "answer" in df.columns else [None] * len(df)
    rows = [{"id": int(df["id"].iloc[i]), "gold": gold[i],
             "scores": {l: {"sum": float(S[i, j]), "mean": float(S[i, j]), "ntok": 1}
                        for j, l in enumerate(LABELS)}} for i in range(len(df))]
    nb = {"rules": {l: {"sum": 0.0, "mean": 0.0, "ntok": 1} for l in LABELS}}
    json.dump({"config": {"data": a.data, "guide": "rules", "shots": 0, "order": "-",
                          "exemplars": "rules", "model": "regex-cues", "prompt_chars": 0},
               "labels": LABELS, "null_bias": nb, "prompt_sample": "", "rows": rows},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
