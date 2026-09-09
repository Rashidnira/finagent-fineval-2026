# Subtask 2 — Japanese Financial ICR: label-semantics guide
Evidence base: all 253 train rows (parsed into Q/R) + all 50 test rows read by hand.
Every quote below is real text from the dataset; train quotes carry their **gold** label,
test quotes are marked `[test, unlabeled]`.

---
## 0. The decision axis (read this first)

The label is **not sentiment**. It is the company's *intent toward the specific action the
question proposes*, on a commitment↔refusal scale:

```
+2  the action is DECIDED / already executed / promised          "we will, and it is settled"
+1  the action is INTENDED but CONDITIONAL or aspirational       "we want to / we will if X"
 0  NO stance is taken — facts, process, "still evaluating"      "we are looking into it"
-1  SOFT NO — declined for now, door explicitly left open        "difficult at this point, but…"
-2  HARD NO — decided not to, or a flat refusal to engage        "we will not / no comment"
```

**Polarity flips with the question.** If the question proposes a *negative* action
("will you STOP buybacks?", "will you cut jobs?"), then a firm *denial* is `-2`, not `+2`:
> Q 自己株式取得を停止しますか。 / A 自己株式取得を停止する予定はありません。すでに公表した
> 取得枠は変更せず、期限まで計画どおり継続します。 `[test 256, unlabeled → -2]`

Always resolve: *what action is being asked about* → *does the answer commit to it, refuse it,
or neither*.

---
## 1. Two register shifts you must be aware of

**(a) Train is long real IR transcript text; test is short synthetic vignettes.**
train response length: median **240** chars (27–1174). test: median **58** chars (42–78), extremely
uniform. Test items are clean, single-intent, purpose-written. Any heuristic that keys on
"long rambling = commitment, short = refusal" (true in train: −2 median 84 chars, +2 median 298)
will be *actively harmful* on test, where every item is short.

**(b) The negative classes mean different things in train vs test.**
In train, almost all `-1`/`-2` are **refusals to DISCLOSE** (差し控え / ノーコメント / 非開示 —
present in 27% of train `-1` and 50% of train `-2`, and ~0–4% of every other class).
In test, the negatives are **refusals to ACT** (しません / 予定はありません / 難しい).
→ *Do not draw few-shot `-1`/`-2` exemplars from train verbatim; they teach the wrong surface form.*

**(c) Train labels contain outright noise.** Train `id=100` (`+2`) and `id=129` (`-2`) have
essentially the *same* response text ("処理であるため、資金的な問題は生じない…新たな資金調達は
計画していない。"). One of the two -2 examples in the entire train set is therefore probably
mislabeled. Do not hard-fit to train labels.

---
## 2. Per-label cues, with the hard boundaries

### `+2` Strong Commitment — the decision has already been made
The response contains a **completed decision act** and usually **hard specifics** (a date, an
amount, a share count, a board resolution). It is reporting, not aspiring.

Marker verbs (past/perfective, decision-act):
`決議しました / 決定しました / 承認しました / 締結しました / 確定しています / 織り込みました /
公表しました / 実行します / 完了します`
Also: `変更はない / 変更する予定はありません` when it re-affirms an *already-announced* promise
(累進配当, 取得枠, 配当計画), and `約束している通り`.

> 2024 年 5 月 15 日開催の**取締役会において**…普通株式 1 株を 3 株に分割することを**決議いたし
> ました**。 `[train 18, gold +2]`
> 現中経の21年度までは**約束している通り**、減配しないという、累進配当の方針に**変更は無い**。
> `[train 136, gold +2]`
> 本日の取締役会で、取得総額上限120億円…**決議しました**。明日から12月末までの期間に市場買付け
> を実施します。 `[test 260, unlabeled → +2]`
> 2027年4月1日に国内全42拠点で新システムへ一斉移行します。移行日と予算は**確定しており**…
> `[test 283, unlabeled → +2]`

Notice: in test, all ten `+2` candidates carry a **number + a date** (120億円/800万株, 2027年10月
着工/350億円, 60円/45円, 平均5％/来年4月, 180億円/3月末, 20店/今期末, 2027年2月1日).
**"Concrete figure + fixed timing + past-tense decision verb" is the single best `+2` detector.**

### `+1` Weak / Qualified Commitment — intent, but hedged or conditional
Same *direction* as `+2`, but the commitment is **volitional, aspirational, or gated on a
condition**. Nothing is settled yet.

Marker endings:
`〜たいと考えております / 〜していきたい / 〜に努めてまいります` (volitional 〜たい)
`〜を目指します / 目指してまいります` (目指す = target, explicitly *not* a promise)
`〜と考えています / 〜と見ています` (belief, not decision)
Conditionals: `〜が得られれば / 〜が整えば / 〜できれば / 〜次第では / 想定どおり進めば`

> 株主優待制度について、現状**具体的に決定していることはございませんが**…安定的に引き上げること
> により、株主還元に**努めていきたいと考えております**。 `[train 19, gold +1]`
> まずは中期経営計画2022期間中に連結純資産比20%以内という目標を**確実に達成したい**。その先に
> ついては現時点で確定している方針はないが、**状況に応じて検討していく**。 `[train 227, gold +1]`
> 自治体の許認可が計画どおり**得られれば**、来年下期の着工を**想定しています**。
> `[test 253, unlabeled → +1]`
> 安定したフリーキャッシュフローを**確認できれば**、40％程度への引き上げを**目指したい**と考えて
> います。**ただし**、投資案件や市況によって最終判断は変わります。 `[test 281, unlabeled → +1]`

#### ► HARD BOUNDARY `+2` vs `+1`
| test | `+2` | `+1` |
|---|---|---|
| Has the decision happened? | yes — 決議/決定/確定/締結 (past) | no — 目指す/〜たい/検討 |
| Timing | a specific date already fixed | "if…", "when conditions allow", "next FY onward" |
| Numbers | binding amounts/counts | targets, or none |
| Reversibility language | absent | 〜次第では見直す / 確約はできない / ただし |
| Test-set feel | "…しました。…します。" | "…れば、…たいと考えています。" |

Trap: `目指す` alone is **never** enough for `+2` — it is 21% of train `+2` but 17% of train `+1`,
so it is *not discriminative*. The decision verb is.
> ROE10％は目指すべき水準として掲げています。市場環境に左右される部分もあるため**確約はできま
> せんが**、収益改善と資本効率向上を進めます。 `[test 288, unlabeled → +1]` — 目指す + explicit
> refusal to promise = `+1`, not `+2`, and not `-1` (the direction is still positive).

### `0` Neutral / Hedged — no stance is expressed at all
The company answers *around* the question: gives facts, describes a process, says it is
still evaluating, or defers the answer to a future occasion. Crucially **neither direction is
chosen** — you cannot tell from the answer whether they will or will not.

Markers:
`検証しています / 精査しています / 分析している段階です / 策定過程にあります / 確認しています`
`〜については次回（本決算時／計画発表時）に説明します／公表します` (defers the *answer*, not the action)
`両者の影響 / 双方のシナリオ / 一方で…一方で` (balanced both-sides framing)
`現時点では〜を据え置いています` (status quo maintained, no new intent)
Pure factual reporting with no forward verb at all:
> 発注者との協議については概ね終了したものと考えている。 `[train 217, gold 0]`
> パパングループ内販売金融会社から自動車事業への配当があったため、ネットキャッシュを維持できた。
> `[train 170, gold 0]`
> 円安による増益効果がある**一方**、輸入原材料コストは増加します。両者の影響を**精査しており**、
> 現時点では通期予想を**据え置いています**。 `[test 264, unlabeled → 0]`
> 現行計画では…を主要指標としています。次期計画の指標体系は**策定過程にあります**。
> `[test 294, unlabeled → 0]`

#### ► HARD BOUNDARY `0` vs `-1`
Both look evasive. The separator is **whether a negative answer to the question is implied**.

* `0` = *"we haven't decided / here are the facts"* — the question is left genuinely open,
  and the response contains **no negative word about the action**.
* `-1` = *"no, not now"* — there **is** a negative, but it is softened by a hedge and an
  explicit re-opening (`ただし〜すれば再検討 / 環境が変われば`).

| | `0` | `-1` |
|---|---|---|
| Negative predicate present? | no | yes: 難しい / 見送る / 考えていません / 時期尚早 / 予定はありません |
| Time framing | "we are in the middle of X" | "not *now* / not *this term*" (現時点では、当面) |
| What follows | a schedule for *answering* (次回説明します) | a condition for *reconsidering* (再検討します) |

> `0`: 対象不動産については継続保有と売却の**双方から価値を検証**しています。鑑定評価や事業上の
> 必要性を確認している段階です。 `[test 274 → 0]` — no "no" anywhere.
> `-1`: 現在の収益規模では準備開始は**時期尚早**と考えています。数年後に事業基盤が整った段階で、
> 資金調達手段の一つとして検討します。 `[test 296 → -1]` — a "no", then a re-opening.

### `-1` Weak Refusal — soft no, door left open
Markers (negative predicate + softener + re-opening):
`〜は難しいと考えています / 難しい状況です`
`見送る方向です / 見送ります`
`現時点では〜は考えていません / 当面の優先順位は低い / 時期尚早`
`現段階で〜に踏み切る考えはありません`
Almost always followed by `ただし / 〜すれば / 〜が変化すれば` + `再検討します / 見直します`.
In **train** the same class appears as a soft *disclosure* refusal that still gives partial info:
> 社内でストレステストはやっていますが、試算値については**開示を差し控えさせていただきます**。
> 2020年9月末のTier1比率は19.3%… `[train 117, gold -1]`
> NASDAQ もニューヨークもロンドンも**検討していますが、まだ決断はしていません**。今のところ、
> 何の決定もなされていません。 `[train 110, gold -1]`
> `[test]` 今期中の追加増配は**現時点では難しい**と考えています。**ただし**、年度末のキャッシュ
> フローが想定を上回れば、改めて**検討する余地**はあります。 `[test 298 → -1]`

### `-2` Strong Refusal — settled no, or a flat non-answer
Two surface forms, both `-2`:
1. **Decided negative action** (dominant in test): a categorical negation, typically reinforced
   by a second sentence that *closes off* the future.
   `〜しません / 〜する考えはありません / 行いません / 中止しました / 除外しています /
   予定もありません / 社内規程です / 取締役会で確認しています`
   > 当社は個人向け暗号資産取引には**参入しません**。事業リスクが当社方針に合わないため、将来の
   > 事業候補からも**除外しています**。 `[test 255 → -2]`
   > 海外工場計画は**正式に中止しました**。用地契約も解除済みであり、同地域で建設計画を再開する
   > **予定はありません**。 `[test 292 → -2]`
2. **Flat refusal to answer, giving nothing** (dominant in train):
   > 同社はまだ決算公表を行っていないため、**回答は差し控えさせて頂く**。 `[train 87, gold -2]`

#### ► HARD BOUNDARY `-1` vs `-2`
`-2` is **terminal**: no condition under which the answer changes is offered, and the negation
is categorical (しません / 中止しました / 意思はありません). `-1` **always** offers a reopening
clause. If you can find a `ただし…すれば…検討/再検討/見直し` tail, it is `-1`.
In train, the same split holds for disclosure refusals: `[train 87, -2]` gives *nothing*;
`[train 117, -1]` refuses but hands over real numbers.

---
## 3. The five sharpest cues (ranked by discriminative power)

1. **Past-tense decision verb (決議しました／決定しました／確定しています／締結しました) +
   a concrete figure + a fixed date ⇒ `+2`.** Nothing else in the label set carries a completed
   decision act. Frequency in train: 決議/決定/正式に決/承認 appears in 7% of `+2` and ≤1% of every
   other class.
2. **Conditional clause (〜れば／〜次第／〜が整えば) or volitional ending (〜たい／目指す) with no
   completed decision ⇒ `+1`.** 〜たい appears in 27% of `+1` vs 6% of `0`; conditionals in 19% of
   `+1` vs 7% of `0`.
3. **A negative predicate that is *categorical and terminal* (しません／する考えはありません／
   中止しました／除外しています／意思はありません) ⇒ `-2`; the same negative *softened*
   (難しい／見送る方向／現時点では考えていません／時期尚早) and paired with a `ただし…再検討`
   reopening ⇒ `-1`.** The reopening clause is the whole difference.
4. **No negative predicate anywhere AND no forward-looking volitional verb — only 検証/精査/
   分析/策定過程/次回説明します/据え置き ⇒ `0`.** "We are in the middle of it" is `0`; "not now"
   is `-1`. In train, 目指す is essentially absent from `0` (3%) while 検証/factual reporting
   dominates it.
5. **差し控えます／ノーコメント／非開示／回答を控えます ⇒ negative class in the *train* register**
   (27% of `-1`, 50% of `-2`, ~0% elsewhere): giving *nothing* = `-2`, refusing but still handing
   over substantive figures = `-1`. **In the test register this cue is rare and can instead sit at
   `0`** when the company declines to comment on a *specific deal* but describes its general
   process (see `[test 289]`, flagged low-confidence). Weight this cue lower on short test-style
   items than on long transcript-style items.

---
## 4. Ready-to-paste decision procedure (for the prompt)

```
1. Identify the ACTION the question proposes.
2. Does the response report that the action (or its refusal) has ALREADY been decided,
   resolved, contracted, or scheduled — with a figure or a fixed date?
     yes, action  -> +2      yes, refusal -> -2
3. Otherwise, is there an explicit negative about the action
   (しません / 難しい / 見送る / 考えていません / 予定はありません)?
     yes -> is a reopening condition offered (ただし… / …すれば再検討)?
              yes -> -1        no -> -2
4. Otherwise, is there positive intent (〜たい / 目指す / 〜すれば…します / 前向きに検討)?
     yes -> +1
5. Otherwise (facts only, still evaluating, defers the answer, both-sides framing) -> 0
```
