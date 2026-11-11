#!/usr/bin/env python3
"""How much could combining the systems we already have gain? Lab test 3, CPU only, no new decoding."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import labkit as lab
from willspeak.evaluate import DEL, INS, align
from willspeak.normalize import normalize


def vote(hypotheses: list[str]) -> str:
    words = [normalize(h).split() for h in hypotheses]
    pivot = words[0]
    slots: list[list[str | None]] = [[w] for w in pivot]
    gaps: list[list[tuple[str, ...]]] = [[()] for _ in range(len(pivot) + 1)]
    for other in words[1:]:
        position, inserted = 0, []
        gap_for = {g: () for g in range(len(pivot) + 1)}
        for op, _, hyp_word in align(pivot, other):
            if op == INS:
                inserted.append(hyp_word)
                continue
            if inserted:
                gap_for[position] = tuple(inserted)
                inserted = []
            slots[position].append(None if op == DEL else hyp_word)
            position += 1
        if inserted:
            gap_for[position] = tuple(inserted)
        for g in range(len(pivot) + 1):
            gaps[g].append(gap_for[g])

    def winner(candidates):
        counts: dict = {}
        for c in candidates:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=counts.get)

    out: list[str] = []
    for g in range(len(pivot) + 1):
        out.extend(winner(gaps[g]))
        if g < len(pivot):
            chosen = winner(slots[g])
            if chosen is not None:
                out.append(chosen)
    return " ".join(out)


def group_report(labels: tuple[str, ...], systems: dict[str, lab.Pooled], manifests: dict[str, dict[str, dict]],
                 hyps: dict[str, dict[str, dict[str, str]]], pivot: str) -> dict:
    ordered = [pivot] + [x for x in labels if x != pivot]
    oracle, voted, words = {}, {}, {}
    for speaker, manifest in manifests.items():
        o = v = n = 0
        counts = [systems[x].counts[speaker] for x in ordered]
        for clip_id, row in manifest.items():
            o += min(c[clip_id][0] for c in counts)
            n += counts[0][clip_id][1]
            combined = vote([hyps[x][speaker][clip_id] for x in ordered])
            v += lab.clip_counts({clip_id: row}, {clip_id: combined})[clip_id][0]
        oracle[speaker], voted[speaker], words[speaker] = o, v, n
    return {"systems": list(labels), "pivot": pivot, "oracle": oracle, "vote": voted,
            "oracle_pooled": sum(oracle.values()), "vote_pooled": sum(voted.values()), "words": words,
            "pooled_words": sum(words.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, metavar="LABEL=SPK:PATH,SPK:PATH")
    ap.add_argument("--speakers", default=",".join(lab.SEVERE_TEST))
    ap.add_argument("--export", default=lab.EXPORT)
    ap.add_argument("--max-subset", type=int, default=4, help="largest group of systems to try (default 4)")
    ap.add_argument("--out", required=True, help="new folder to write results.json into (refuses to overwrite)")
    a = ap.parse_args()
    if len(a.system) < 2:
        ap.error("need at least two --system")
    if a.max_subset < 2:
        ap.error("--max-subset must be at least 2")
    lab.refuse_existing(a.out)
    speakers = [s.strip() for s in a.speakers.split(",") if s.strip()]
    manifests = {s: lab.read_manifest(lab.test_manifest(s, a.export)) for s in speakers}
    systems, hyps, sources = {}, {}, {}
    for text in a.system:
        label, folders = lab.parse_system(text)
        if label in systems:
            raise ValueError(f"System {label!r} is listed twice")
        systems[label] = lab.load_pooled(label, folders, manifests)
        hyps[label] = {s: lab.read_hyps(folders[s]) for s in speakers}
        sources.update({f"{label}/{s}": folders[s] for s in speakers})
    labels = list(systems)
    singles = {x: systems[x].pooled_wrong() for x in labels}
    best = min(labels, key=lambda x: singles[x])
    total = systems[best].pooled_words()
    bar = lab.bar_words(total)
    print(f"pooled test words {total}; 35% bar is {bar} wrong or fewer")
    for x in labels:
        print(f"  {x:<28} {singles[x]:>4} wrong ({100 * singles[x] / total:.1f}%)")
    results = []
    for size in range(2, min(a.max_subset, len(labels)) + 1):
        for group in itertools.combinations(labels, size):
            pivot = min(group, key=lambda x: singles[x])
            results.append(group_report(group, systems, manifests, hyps, pivot))
    if len(labels) > a.max_subset:
        results.append(group_report(tuple(labels), systems, manifests, hyps, best))
    results.sort(key=lambda r: (r["oracle_pooled"], r["vote_pooled"]))
    everything = next(r for r in results if len(r["systems"]) == len(labels))
    print(f"\nbest single system: {best}, {singles[best]} wrong")
    print("lowest oracle ceilings (best system chosen per clip) and what a plain vote reaches:")
    for r in results[:8]:
        print(f"  {' + '.join(r['systems']):<60} oracle {r['oracle_pooled']:>4} ({100 * r['oracle_pooled'] / total:.1f}%)   "
              f"vote {r['vote_pooled']:>4} ({100 * r['vote_pooled'] / total:.1f}%)")
    reachable = min(r["oracle_pooled"] for r in results)
    print(f"\nlowest oracle over every group tried: {reachable} wrong ({100 * reachable / total:.1f}%); "
          f"{'below' if reachable <= bar else 'above'} the 35% bar ({bar})")
    lab.save_json(Path(a.out) / "results.json", {
        "singles": singles, "best_single": best, "pooled_words": total, "bar_wrong": bar,
        "groups": results, "everything": everything, "speakers": speakers,
        "provenance": lab.provenance(__file__, sources)})
    print(f"wrote {lab.here(a.out) / 'results.json'}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "hyp_bounds"))
