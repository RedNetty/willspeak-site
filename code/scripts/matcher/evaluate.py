#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _cli


def l2n(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def protocol(V, words, k, rng, reps=20):
    hits = {"proto": [0, 0], "max": [0, 0]}; n = 0
    for _ in range(reps):
        by = collections.defaultdict(list)
        for i, w in enumerate(words):
            by[w].append(i)
        ex, te = [], []
        for w, idx in by.items():
            idx = list(idx); rng.shuffle(idx); ex += idx[:k]
            if len(idx) >= k + 1:
                te += idx[k:]
        if not te:
            return None
        E, el = V[ex], words[ex]
        labs = sorted(set(el))
        P = l2n(np.stack([E[el == w].mean(0) for w in labs]))
        for i in te:
            s = V[i] @ E.T
            best = {}
            for sc, w in zip(s, el):
                best[w] = max(best.get(w, -9), sc)
            for mode, scores in (("max", best), ("proto", dict(zip(labs, V[i] @ P.T)))):
                ranked = [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
                hits[mode][0] += ranked[0] == words[i]; hits[mode][1] += words[i] in ranked[:3]
            n += 1
    return {m: (hits[m][0] / n, hits[m][1] / n, n // reps) for m in hits}



def evaluate_embeddings(paths: list[str], ks: list[int], dim: int, alpha: float) -> int:
    from willspeak.matcher import training as S
    from willspeak.torgo import SEVERITY

    d = S.load_embeddings(paths)
    speakers = sorted(np.unique(d["speakers"]))
    print(f"{len(d['raw'])} takes · {len(set(d['labels']))} labels · {len(speakers)} speakers: {' '.join(speakers)}")
    print("projection fitted on the other speakers for every row (leave-one-speaker-out)\n")
    rows = []
    for held in speakers:
        m_te = d["speakers"] == held
        m_tr = ~m_te
        if m_tr.sum() < 50 or m_te.sum() < 10:
            print(f"  {held}: too few takes to evaluate"); continue
        X, lab, sess = d["raw"][m_te], d["labels"][m_te], d["sessions"][m_te]
        Xtr = S.session_center(d["raw"][m_tr], d["sessions"][m_tr])
        W, mu = S.lda_fit(Xtr, d["labels"][m_tr], dim=dim, alpha=alpha)
        Xc = S.session_center(X, sess)
        spaces = {"raw": l2n(X.astype(np.float64)), "sessnorm": l2n(Xc), "projected": l2n((Xc - mu) @ W)}
        r = {"speaker": held, "severity": SEVERITY.get(held, "?"), "n": int(m_te.sum()),
             "sessions": len(np.unique(sess))}
        for k in ks:
            for name, V in spaces.items():
                res = protocol(V, lab, k, np.random.default_rng(0))
                r[f"k{k}_{name}_top1"] = res["proto"][0] if res else float("nan")
                r[f"k{k}_{name}_top3"] = res["proto"][1] if res else float("nan")
        for name, V in spaces.items():
            acc, n = S.cross_session_top1(V, lab, sess)
            r[f"cross_{name}"] = acc; r["cross_n"] = n
        causal, causal_n = S.causal_cross_session_top1(X, lab, sess, project=lambda values, mu=mu, W=W: (values - mu) @ W)
        r["cross_causal_projected"] = causal
        rows.append(r)
        k0 = ks[0]
        print(f"  {held:5s} {r['severity']:16s} n={r['n']:4d} sessions={r['sessions']}  "
              f"k={k0} top-1 raw {r[f'k{k0}_raw_top1']:.3f} sessnorm {r[f'k{k0}_sessnorm_top1']:.3f} "
              f"projected {r[f'k{k0}_projected_top1']:.3f} | top-3 projected {r[f'k{k0}_projected_top3']:.3f}"
              + (f" | cross-session raw {r['cross_raw']:.3f} projected {r['cross_projected']:.3f} (n={r['cross_n']})"
                 + f" | causal mean, all enrollment takes {causal:.3f} (query uses={causal_n})"
                 if r["sessions"] > 1 else " | cross-session: one session only"))
    if not rows:
        return 1

    def macro(sel, key):
        v = [x[key] for x in sel if not np.isnan(x[key])]
        return (sum(v) / len(v)) if v else float("nan")

    k0 = ks[0]
    groups = {"all speakers": rows, "severe": [x for x in rows if x["severity"] == "severe"]}
    print()
    for gname, sel in groups.items():
        if not sel:
            continue
        print(f"macro {gname} ({len(sel)}): k={k0} top-1 raw {macro(sel, f'k{k0}_raw_top1'):.3f} -> "
              f"sessnorm {macro(sel, f'k{k0}_sessnorm_top1'):.3f} -> projected {macro(sel, f'k{k0}_projected_top1'):.3f}"
              f" | cross-session raw {macro(sel, 'cross_raw'):.3f} -> projected {macro(sel, 'cross_projected'):.3f}"
              f" | causal mean {macro(sel, 'cross_causal_projected'):.3f}")
    print("\nThe session-normalized/projected legacy rows use complete test-session means, including future clips.")
    print("The causal row uses the bank prior and only current/earlier query clips; it also enrolls all takes.")
    print("Query uses include repeated enrollment folds. Closed-set accuracy does not measure rejection or spoken controls.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--cross-session", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--emb", nargs="+", help="embeddings .npz from embed_manifest.py: leave-one-speaker-out mode")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    if args.emb:
        return evaluate_embeddings(args.emb, args.k, args.dim, args.alpha)

    from willspeak.db import Store
    from willspeak.matcher.engine import Matcher, Projection
    store = Store(args.data)
    m = Matcher(store, Path(args.data))
    if not m.available():
        print("torch/transformers are not installed in THIS Python. Run:\n"
              "  python -m pip install -r requirements.txt")
        return 1
    print("Embedding accepted clips (cached after the first run)…")
    r = m.rebuild()
    print(f"  {r}")
    if len(m.raw) < 10:
        print("Fewer than 10 usable takes -record some critical/functional prompts first.")
        return 1
    R = np.stack(m.raw); words = np.array(m.labels); sess = np.array(m.sessions)
    cnt = collections.Counter(words)
    print(f"\n{len(words)} takes · {len(cnt)} labels · {sum(v >= 2 for v in cnt.values())} labels with >= 2 takes · sessions: {dict(collections.Counter(sess))}")
    print(f"projection: {m.proj.describe()}")
    Rn = R.copy()
    for s in np.unique(sess):
        Rn[sess == s] -= R[sess == s].mean(0)
    spaces = {"raw WavLM cosine": l2n(R), "session-normalised (full session)": l2n(Rn),
              "projected (full session)": l2n(m.proj(Rn))}
    if m.proj.T is not None:
        noT = Projection(m.proj.path); noT.T = None
        spaces["projected, W only (no session transform)"] = l2n(noT(Rn))
    print("\nclosed vocabulary = every label above; k = accepted takes per label; "
          "prototype = mean of the k takes (live page) | max = best single take")
    for k in args.k:
        for name, V in spaces.items():
            res = protocol(V, words, k, np.random.default_rng(0))
            if res is None:
                print(f"k={k}: not enough labels with {k+1}+ takes"); break
            pr, mx = res["proto"], res["max"]
            print(f"  k={k}  {name:32s} prototype top-1 {pr[0]:.3f} top-3 {pr[1]:.3f} | max top-1 {mx[0]:.3f} top-3 {mx[1]:.3f}   (n={pr[2]})")

    if args.cross_session and len(np.unique(sess)) > 1:
        print("\ncross-session: enrol on one session, test on the others (same labels)")
        V = spaces["projected (full session)"]; Vr = spaces["raw WavLM cosine"]
        for s in np.unique(sess):
            en = np.where(sess == s)[0]; te = [i for i in np.where(sess != s)[0] if words[i] in set(words[en])]
            if len(te) < 3:
                continue
            for name, VV in (("raw", Vr), ("projected", V)):
                h1 = h3 = 0
                for i in te:
                    sc = VV[i] @ VV[en].T
                    best = {}
                    for v, w in zip(sc, words[en]):
                        best[w] = max(best.get(w, -9), v)
                    ranked = [w for w, _ in sorted(best.items(), key=lambda kv: -kv[1])]
                    h1 += ranked[0] == words[i]; h3 += words[i] in ranked[:3]
                print(f"  enrol {s} ({len(set(words[en]))} labels) -> others: {name:9s} "
                      f"top-1 {h1/len(te):.3f} top-3 {h3/len(te):.3f} (n={len(te)})")
        from willspeak.matcher.training import causal_cross_session_top1

        causal, n = causal_cross_session_top1(R, words, sess, project=m.proj, warmup=m.warmup)
        print(f"  causal session mean, prototype scoring, all enrollment takes: top-1 {causal:.3f} (query uses={n})")


    print("\nFull-session means include test clips. The causal row uses only current/earlier query clips.")
    print("Closed-set results omit unknown-label queries from accuracy and do not measure rejection, ASR fusion or control actions.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _cli.PAUSE_ALWAYS = True
    sys.exit(_cli.run(main, "match_eval"))
