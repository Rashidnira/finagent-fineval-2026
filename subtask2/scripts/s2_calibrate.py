#!/usr/bin/env python
"""
Subtask 2 - calibration + balanced-assignment layer.

Reads a score JSON written by s2_rules.py (per-row 5-way label logprobs) and evaluates,
on the balanced dev split:
  1. plain argmax
  2. contextual calibration  (subtract lambda * logp(label | content-free input))
  3. train-prior divide-out   (subtract lambda * log pi_train(label))
  4. batch calibration        (subtract the per-label mean logit over the batch)
  5. Sinkhorn prior matching  (rescale column marginals toward uniform, partial strength)
  6. Hungarian hard assignment (exactly n/5 per label)

Because dev is 6/6/6/6/2 (only two -2 rows exist in the whole train set) rather than exactly
uniform, the assignment rules (5,6) are ALSO evaluated by bootstrap: repeatedly draw a
class-balanced batch (k per label, with replacement) from dev, run the rule on that batch with a
uniform target, and average. That is a faithful simulation of what happens on the 50 balanced
test rows, which the raw dev number is not.
"""
import argparse, json, math, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

LABELS = ["+2", "+1", "0", "-1", "-2"]
TRAIN_PRIOR = np.array([29, 139, 72, 11, 2], float)
TRAIN_PRIOR /= TRAIN_PRIOR.sum()


# ------------------------------------------------------------------ io
def load(path, stat="sum"):
    o = json.load(open(path))
    ids = [r["id"] for r in o["rows"]]
    gold = [r["gold"] for r in o["rows"]]
    S = np.array([[r["scores"][l][stat] for l in LABELS] for r in o["rows"]], float)
    nb = o["null_bias"]
    B = np.array([[nb[k][l][stat] for l in LABELS] for k in nb], float)
    b = B.mean(0)
    return o, ids, gold, S, b


def softmax(X, axis=-1):
    X = X - X.max(axis=axis, keepdims=True)
    E = np.exp(X)
    return E / E.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------ decision rules
def sinkhorn_offsets(L, target, iters=80):
    """Log-space per-label offsets c such that softmax(L + c) has column marginals == target.
    Returning the offsets (rather than the rescaled matrix) lets us interpolate: applying
    alpha*c gives a genuine soft->hard continuum, which repeatedly rescaling does not
    (any strength>0 converges to the same fixed point given enough iterations)."""
    c = np.zeros(L.shape[1])
    for _ in range(iters):
        col = softmax(L + c).mean(0)
        c = c + np.log(target / np.maximum(col, 1e-12))
    return c


def sinkhorn(P, target, iters=80, strength=1.0):
    """Convenience wrapper kept for callers that hand in a probability matrix."""
    L = np.log(np.maximum(P, 1e-12))
    c = sinkhorn_offsets(L, target, iters)
    return softmax(L + strength * c)


def hungarian(L, per_label):
    """Hard assignment: exactly per_label rows to each label, maximising total logit."""
    n, k = L.shape
    assert n == per_label * k, f"{n} != {per_label}*{k}"
    cost = -np.repeat(L, 1, axis=0)
    big = np.repeat(cost, 1, axis=1)
    big = np.concatenate([cost[:, [j]] for j in range(k) for _ in range(per_label)], axis=1)
    r, c = linear_sum_assignment(big)
    out = np.empty(n, int)
    for ri, ci in zip(r, c):
        out[ri] = ci // per_label
    return out


def cap_majority(L, cap):
    """Softer alternative to hard assignment: no label may be predicted more than `cap` times.
    Over-quota rows with the smallest top1-vs-runnerup margin are re-assigned to their best
    label that still has room. Uses no knowledge beyond 'the test set is balanced'."""
    n, k = L.shape
    order = np.argsort(-L, axis=1)
    pred = order[:, 0].copy()
    for _ in range(n * k):
        cnt = np.bincount(pred, minlength=k)
        over = [j for j in range(k) if cnt[j] > cap]
        if not over:
            break
        j = max(over, key=lambda x: cnt[x])
        rows = np.where(pred == j)[0]
        # margin between current label and the best label that still has room
        best_alt, margin = {}, {}
        for i in rows:
            for alt in order[i]:
                if alt != j and np.bincount(pred, minlength=k)[alt] < cap:
                    best_alt[i] = alt
                    margin[i] = L[i, j] - L[i, alt]
                    break
        if not margin:
            break
        i = min(margin, key=margin.get)
        pred[i] = best_alt[i]
    return pred


# ------------------------------------------------------------------ scoring
def metrics(pred_idx, gold):
    pred = [LABELS[i] for i in pred_idx]
    n = len(gold)
    acc = sum(p == g for p, g in zip(pred, gold)) / n
    recs = []
    for l in LABELS:
        sup = sum(g == l for g in gold)
        if sup:
            recs.append(sum(p == g == l for p, g in zip(pred, gold)) / sup)
    hist = [sum(p == l for p in pred) for l in LABELS]
    return acc, sum(recs) / len(recs), hist


def hist_str(h):
    return "/".join(str(x) for x in h)


def bootstrap_assign(L, gold, k_per=2, B=1500, seed=7):
    """Simulate the balanced 50-row test batch: draw k_per rows per gold label (with
    replacement), then compare argmax / soft-Sinkhorn / hard-Hungarian on that batch."""
    rng = np.random.default_rng(seed)
    gi = np.array([LABELS.index(g) for g in gold])
    idx_by = [np.where(gi == j)[0] for j in range(5)]
    res = {"argmax": [], "sink0.25": [], "sink0.5": [], "sink1.0": [], "hungarian": [], "cap": []}
    for _ in range(B):
        sel = np.concatenate([rng.choice(idx_by[j], k_per, replace=True) for j in range(5)])
        Lb, gb = L[sel], [gold[i] for i in sel]
        P = softmax(Lb)
        tgt = np.full(5, 0.2)
        res["argmax"].append(metrics(Lb.argmax(1), gb)[0])
        res["sink0.5"].append(metrics(sinkhorn(P, tgt, strength=0.5).argmax(1), gb)[0])
        res["sink1.0"].append(metrics(sinkhorn(P, tgt, strength=1.0).argmax(1), gb)[0])
        res["sink0.25"].append(metrics(sinkhorn(P, tgt, strength=0.25).argmax(1), gb)[0])
        res["hungarian"].append(metrics(hungarian(Lb, k_per), gb)[0])
        res["cap"].append(metrics(cap_majority(Lb, int(round(1.5 * k_per))), gb)[0])
    return {k: (float(np.mean(v)), float(np.std(v) / math.sqrt(B))) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--stat", default="sum", choices=["sum", "mean"])
    ap.add_argument("--boot", type=int, default=1500)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    o, ids, gold, S, b = load(a.scores, a.stat)
    cfg = o["config"]
    print(f"=== {a.scores}")
    print(f"    config: guide={cfg['guide']} shots={cfg['shots']}/label order={cfg['order']} "
          f"exemplars={cfg['exemplars']} stat={a.stat} n={len(gold)} prompt_chars={cfg['prompt_chars']}")
    print(f"    null-input bias b (logp of each label on content-free input): "
          + "  ".join(f"{l}={v:.2f}" for l, v in zip(LABELS, b)))

    rows = []
    def rep(name, pred, extra=""):
        acc, mr, h = metrics(pred, gold)
        rows.append((name, acc, mr, h))
        print(f"  {name:<34s} acc={acc:.4f}  macroR={mr:.4f}  pred {hist_str(h)} {extra}")

    print("\n-- step 1/2: plain argmax on the scored prompt")
    rep("argmax", S.argmax(1))

    print("-- step 3a: contextual calibration  (S - lam*b_null)")
    best = (None, -1)
    for lam in [0.25, 0.5, 0.75, 1.0]:
        pred = (S - lam * b).argmax(1)
        rep(f"ctx-calib lam={lam}", pred)
        mr = metrics(pred, gold)[1]
        if mr > best[1]:
            best = (lam, mr)
    print("-- step 3b: train-prior divide-out  (S - lam*log pi_train)")
    for lam in [0.5, 1.0]:
        rep(f"prior-divout lam={lam}", (S - lam * np.log(TRAIN_PRIOR)).argmax(1))
    print("-- step 3c: batch calibration (subtract per-label batch mean)")
    rep("batch-calib", (S - S.mean(0)).argmax(1))

    print("-- step 4: explicit balanced assignment on the whole dev batch")
    print("   (dev marginals are 6/6/6/6/2, NOT uniform, so these numbers are pessimistic;")
    print("    the bootstrap below is the faithful simulation of the balanced 50-row test set)")
    P = softmax(S)
    tgt = np.array([6, 6, 6, 6, 2], float) / len(gold)
    unif = np.full(5, 0.2)
    rep("sinkhorn->uniform s=1.0", sinkhorn(P, unif, strength=1.0).argmax(1))
    rep("sinkhorn->uniform s=0.5", sinkhorn(P, unif, strength=0.5).argmax(1))
    rep("sinkhorn->uniform s=0.25", sinkhorn(P, unif, strength=0.25).argmax(1))
    rep("sinkhorn->dev-marginals", sinkhorn(P, tgt, strength=1.0).argmax(1), "(oracle marginals)")
    rep("cap-majority cap=8", cap_majority(S, 8))

    lam = best[0]
    Sc = S - lam * b
    print(f"\n-- step 4 (bootstrap) on ctx-calibrated logits lam={lam}: "
          f"balanced batches drawn from dev, uniform target")
    for k_per in (2, 4):
        r = bootstrap_assign(Sc, gold, k_per=k_per, B=a.boot)
        print(f"   batch = {k_per} per label ({5*k_per} rows), B={a.boot}")
        for k, (m, se) in r.items():
            print(f"      {k:<12s} mean acc = {m:.4f} +/- {se:.4f}")

    if a.json:
        json.dump({"config": cfg, "stat": a.stat,
                   "variants": [{"name": n, "acc": ac, "macro_recall": mr, "hist": h}
                                for n, ac, mr, h in rows]},
                  open(a.json, "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
