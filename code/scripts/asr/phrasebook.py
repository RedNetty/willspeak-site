#!/usr/bin/env python3
"""Replace each saved hypothesis with the nearest phrase from a phrasebook."""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak.g2p import text_phones
from willspeak.normalize import normalize

try:
    from rapidfuzz.distance import Levenshtein as _RFL

    def lev(a, b):
        return _RFL.distance(a, b)
except Exception:  # pragma: no cover
    def lev(a, b):
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def kind_of(r):
    if "kind" in r:
        return r["kind"]
    return "sentence" if len(str(r.get("text", "")).split()) >= 3 else "word"


def phones(text):
    p, _oov = text_phones(text)
    return tuple(x for x in p if x != "_")


class Item:
    def __init__(self, text):
        self.norm = normalize(text)
        self.words = tuple(self.norm.split())
        self.phones = phones(self.norm)


def distance(a: Item, b: Item) -> float:
    c = lev(a.norm, b.norm) / max(len(a.norm), len(b.norm), 1)
    w = lev(a.words, b.words) / max(len(a.words), len(b.words), 1)
    p = lev(a.phones, b.phones) / max(len(a.phones), len(b.phones), 1)
    return (c + w + p) / 3.0


def hyp_map(folder):
    out = {}
    with open(Path(folder) / "hypotheses.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[str(r["id"])] = r["hyp"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-glob", nargs="+", required=True, help="manifest files whose sentence prompts form the pool")
    ap.add_argument("--speaker", action="append", required=True, help="NAME=test manifest, repeatable")
    ap.add_argument("--model", action="append", required=True,
                    help="label=SPK:report folder,SPK:report folder ... one entry per speaker, repeatable")
    ap.add_argument("--kinds", default="sentence",
                    help="which clips are scored and which prompts form the pool: sentence (default), word, "
                         "or sentence,word")
    ap.add_argument("--sizes", default="10,25,50,100", help="phrasebook sizes; the whole pool is always added")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--thresholds", default="",
                    help="comma list of distances; also report the rule 'keep the match only if its distance is at most t, "
                         "otherwise keep the raw hypothesis'. Empty: skip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    kinds = [k for k in a.kinds.split(",") if k]
    if not kinds or any(k not in ("sentence", "word") for k in kinds):
        sys.exit("--kinds must be sentence, word or sentence,word")
    noun = " and ".join(kinds)
    sizes = [int(x) for x in a.sizes.split(",") if x]
    thresholds = [float(x) for x in a.thresholds.split(",") if x]

    files = sorted({p for g in a.pool_glob for p in glob.glob(g)})
    if not files:
        sys.exit("No manifest files match --pool-glob")
    seen = {}
    for p in files:
        for r in read_rows(p):
            t = str(r.get("text", ""))
            if "(live)" in t or kind_of(r) not in kinds:
                continue
            n = normalize(t)
            if n and n not in seen:
                seen[n] = Item(t)
    pool = list(seen.values())
    pool_index = {it.norm: i for i, it in enumerate(pool)}
    print("phrasebook pool: %d distinct %s prompts from %d manifest files" % (len(pool), noun, len(files)))

    speakers = {}
    for s in a.speaker:
        name, path = s.split("=", 1)
        speakers[name] = path
    models = {}
    for m in a.model:
        label, rest = m.split("=", 1)
        per = {}
        for part in rest.split(","):
            spk, folder = part.split(":", 1)
            per[spk] = folder
        missing = [s for s in speakers if s not in per]
        if missing:
            sys.exit("model %s has no folder for %s" % (label, ", ".join(missing)))
        models[label] = per
    labels = list(models)

    clips = {}
    for spk, path in speakers.items():
        rows = [r for r in read_rows(path) if kind_of(r) in kinds and "(live)" not in str(r.get("text", ""))]
        clips[spk] = [(str(r["id"]), Item(r["text"]), kind_of(r)) for r in rows]
        print("%s: %d %s clips of %d in the test manifest" % (spk, len(rows), noun, len(read_rows(path))))
        for _id, it, _k in clips[spk]:
            if it.norm not in pool_index:
                sys.exit("test prompt missing from the pool (the pool glob must include the test manifests): %s" % it.norm)

    dist = {l: {} for l in labels}
    raw = {l: {} for l in labels}
    for l in labels:
        for spk in speakers:
            hyps = hyp_map(models[l][spk])
            ds, rs = [], []
            for cid, it, _k in clips[spk]:
                if cid not in hyps:
                    sys.exit("no hypothesis for %s in %s" % (cid, models[l][spk]))
                h = Item(hyps[cid])
                rs.append(h)
                ds.append([distance(h, ph) for ph in pool])
            dist[l][spk] = ds
            raw[l][spk] = rs
        print("distances done for %s" % l, flush=True)

    P = len(pool)
    results = {"pool": P, "sizes": sizes, "repeats": a.repeats, "speakers": {}, "pooled": {}}
    keys = [str(s) for s in sizes] + ["all"]

    def new():
        return {"exact": 0, "err": 0, "words": 0, "clips": 0}

    acc = {}
    tacc = {}
    bestd = {}
    rawacc = {}
    rawk = {}
    kacc = {}
    for spk in speakers:
        for l in labels:
            c = new()
            for (cid, ref, _k), h in zip(clips[spk], raw[l][spk]):
                rk = rawk.setdefault((l, _k), {"err": 0, "words": 0})
                rk["err"] += lev(ref.words, h.words)
                rk["words"] += len(ref.words)
                c["clips"] += 1
                c["words"] += len(ref.words)
                c["err"] += lev(ref.words, h.words)
                c["exact"] += int(ref.norm == h.norm)
            rawacc[(l, spk)] = c
        for cond in ("closed", "open"):
            for key in keys:
                reps = 1 if key == "all" else a.repeats
                for l in labels:
                    acc[(l, spk, cond, key)] = new()
                    bestd[(l, spk, cond, key)] = []
                    for t in thresholds:
                        tacc[(l, spk, cond, key, t)] = {"err": 0, "words": 0, "clips": 0, "accepted": 0, "accepted_exact": 0, "accepted_wrong": 0}
                for rep in range(reps):
                    for ci, (cid, ref, kind) in enumerate(clips[spk]):
                        tr = pool_index[ref.norm]
                        others = [i for i in range(P) if i != tr]
                        if key == "all":
                            sub = list(range(P)) if cond == "closed" else others
                        else:
                            n = int(key)
                            rng = random.Random("%s|%s|%d|%s" % (spk, key, rep, cid))
                            if cond == "closed":
                                sub = [tr] + rng.sample(others, min(n - 1, len(others)))
                                rng.shuffle(sub)
                            else:
                                sub = rng.sample(others, min(n, len(others)))
                        for l in labels:
                            d = dist[l][spk][ci]
                            best = sub[min(range(len(sub)), key=lambda k: (d[sub[k]], k))]
                            c = acc[(l, spk, cond, key)]
                            c["clips"] += 1
                            c["words"] += len(ref.words)
                            c["err"] += lev(ref.words, pool[best].words)
                            c["exact"] += int(best == tr)
                            bd = d[best]
                            bestd[(l, spk, cond, key)].append(bd)
                            if key == "all" and len(kinds) > 1:
                                for th in [None] + thresholds:
                                    kx = kacc.setdefault((l, cond, kind, th), {"err": 0, "words": 0})
                                    kx["words"] += len(ref.words)
                                    if th is None or bd <= th:
                                        kx["err"] += lev(ref.words, pool[best].words)
                                    else:
                                        kx["err"] += lev(ref.words, raw[l][spk][ci].words)
                            for th in thresholds:
                                x = tacc[(l, spk, cond, key, th)]
                                x["clips"] += 1
                                x["words"] += len(ref.words)
                                if bd <= th:
                                    x["accepted"] += 1
                                    x["accepted_exact"] += int(best == tr)
                                    x["accepted_wrong"] += int(best != tr)
                                    x["err"] += lev(ref.words, pool[best].words)
                                else:
                                    x["err"] += lev(ref.words, raw[l][spk][ci].words)

    def pooled(l, cond, key):
        c = new()
        for spk in speakers:
            x = acc[(l, spk, cond, key)]
            for k in c:
                c[k] += x[k]
        return c

    def pct(c):
        return 100.0 * c["err"] / max(c["words"], 1), 100.0 * c["exact"] / max(c["clips"], 1)

    for spk in list(speakers) + ["ALL"]:
        print("\n######## %s" % spk)
        if spk == "ALL":
            rw = {l: new() for l in labels}
            for l in labels:
                for s in speakers:
                    for k, v in rawacc[(l, s)].items():
                        rw[l][k] += v
        else:
            rw = {l: rawacc[(l, spk)] for l in labels}
        nclip = max(rw[labels[0]]["clips"], 1)
        nword = rw[labels[0]]["words"]
        print("%d %s clips, %d reference words; each cell is spoken WER %% / exact-phrase %%" % (nclip, noun, nword))
        for cond in ("closed", "open"):
            print("-- %s phrasebook (%s)" % (cond, "true phrase always on the list" if cond == "closed" else "true phrase removed"))
            print("%-34s %-13s " % ("model", "raw hypothesis") + " ".join("%-13s" % ("size " + k if k != "all" else "all %d" % (P - (cond == "open"))) for k in keys))
            for l in labels:
                w, e = pct(rw[l])
                line = "%-34s %5.1f / %5.1f  " % (l, w, e)
                for k in keys:
                    c = pooled(l, cond, k) if spk == "ALL" else acc[(l, spk, cond, k)]
                    w2, e2 = pct(c)
                    line += "%5.1f / %5.1f  " % (w2, e2)
                print(line)
        results["speakers" if spk != "ALL" else "pooled"][spk] = {
            l: {"raw": rw[l], **{cond: {k: (pooled(l, cond, k) if spk == "ALL" else acc[(l, spk, cond, k)]) for k in keys} for cond in ("closed", "open")}}
            for l in labels}

    if thresholds:
        def tsum(l, spk, cond, key, th):
            x = {"err": 0, "words": 0, "clips": 0, "accepted": 0, "accepted_exact": 0, "accepted_wrong": 0}
            for s_ in (speakers if spk == "ALL" else [spk]):
                for k, v in tacc[(l, s_, cond, key, th)].items():
                    x[k] += v
            return x

        def quart(vals):
            v = sorted(vals)
            if not v:
                return "n/a"
            return "%.2f / %.2f / %.2f" % (v[len(v) // 4], v[len(v) // 2], v[(3 * len(v)) // 4])

        print("\n######## rejection rule: keep the nearest phrase only if its distance is at most t, otherwise keep the raw hypothesis")
        print("distance of the nearest phrase, quartiles (25% / median / 75%), pooled over speakers, whole pool")
        for cond in ("closed", "open"):
            for l in labels:
                vals = [v for sp in speakers for v in bestd[(l, sp, cond, "all")]]
                print("  %-7s %-34s %s" % (cond, l, quart(vals)))
        ths = ["%.2f" % t for t in thresholds]
        for key in ("50", "all"):
            if key != "all" and int(key) not in sizes:
                continue
            for spk in list(speakers) + ["ALL"]:
                print("\n-- %s, phrasebook %s; cell is spoken WER %% / share of clips where the match was kept" % (spk, "of 50" if key == "50" else "of the whole pool"))
                print("%-7s %-24s %-13s " % ("", "model", "raw") + " ".join("t=%-9s" % t for t in ths) + " no rule")
                for cond in ("closed", "open"):
                    for l in labels:
                        rw = (lambda r: 100.0 * r["err"] / max(r["words"], 1))(
                            rawacc[(l, spk)] if spk != "ALL" else {k: sum(rawacc[(l, sp)][k] for sp in speakers) for k in ("err", "words")})
                        line = "%-7s %-24s %5.1f         " % (cond, l, rw)
                        for th in thresholds:
                            x = tsum(l, spk, cond, key, th)
                            line += "%5.1f / %3.0f%%  " % (100.0 * x["err"] / max(x["words"], 1), 100.0 * x["accepted"] / max(x["clips"], 1))
                        c = pooled(l, cond, key) if spk == "ALL" else acc[(l, spk, cond, key)]
                        line += "%5.1f" % pct(c)[0]
                        print(line)
        print("\n-- a list that covers only some of what the speaker says, pooled over speakers, whole pool: spoken WER = coverage x closed + (1 - coverage) x open")
        print("%-8s %-24s %-9s " % ("coverage", "model", "no rule") + " ".join("t=%-6s" % t for t in ths))
        for cov in (1.0, 0.9, 0.8, 0.5):
            for l in labels:
                line = "%-8s %-24s " % ("%d%%" % round(100 * cov), l)
                c_, o_ = pct(pooled(l, "closed", "all"))[0], pct(pooled(l, "open", "all"))[0]
                line += "%7.1f   " % (cov * c_ + (1 - cov) * o_)
                for th in thresholds:
                    xc, xo = tsum(l, "ALL", "closed", "all", th), tsum(l, "ALL", "open", "all", th)
                    wc, wo = 100.0 * xc["err"] / max(xc["words"], 1), 100.0 * xo["err"] / max(xo["words"], 1)
                    line += "%7.1f " % (cov * wc + (1 - cov) * wo)
                print(line)
        if len(kinds) > 1:
            print("\n-- by kind of clip, pooled over speakers, whole pool: spoken WER % (the sentence and word rows add up to the ALL rows above)")
            print("%-8s %-7s %-24s %-9s %-8s " % ("kind", "cond", "model", "words", "raw") + " ".join("t=%-6s" % t for t in ths) + " no rule")
            for kd in kinds:
                for cond in ("closed", "open"):
                    for l in labels:
                        r_ = rawk[(l, kd)]
                        line = "%-8s %-7s %-24s %-9d %7.1f " % (kd, cond, l, r_["words"], 100.0 * r_["err"] / max(r_["words"], 1))
                        for th in thresholds:
                            x = kacc[(l, cond, kd, th)]
                            line += "%7.1f " % (100.0 * x["err"] / max(x["words"], 1))
                        x = kacc[(l, cond, kd, None)]
                        line += "%7.1f" % (100.0 * x["err"] / max(x["words"], 1))
                        print(line)
            results["by_kind"] = {"|".join(map(str, k)): v for k, v in kacc.items()}
        results["rejection"] = {"thresholds": thresholds, "counts": {"|".join(map(str, k)): v for k, v in tacc.items()}}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\nwrote %s" % (out / "results.json"))


if __name__ == "__main__":
    main()
