#!/usr/bin/env python3
"""Score several systems on the severe speakers' test sets and pool them."""
from __future__ import annotations

import argparse
import sys

import labkit as lab

NOISE_POINTS = 4.5


def verdict(pooled_gain_points: float, speaker_gain_points: dict[str, float], noise_points: float = NOISE_POINTS) -> str:
    if abs(pooled_gain_points) <= noise_points:
        return "unclear"
    sign = 1 if pooled_gain_points > 0 else -1
    agreeing = sum(1 for g in speaker_gain_points.values() if g * sign > 0)
    needed = 2 if len(speaker_gain_points) >= 3 else len(speaker_gain_points)
    if agreeing < needed:
        return "unclear"
    return "better" if sign > 0 else "worse"


def build(systems: dict[str, lab.Pooled], baseline: str, by: str = "clip", noise_points: float = NOISE_POINTS) -> list[dict]:
    if baseline not in systems:
        raise ValueError(f"Baseline {baseline!r} is not one of {list(systems)}")
    base = systems[baseline]
    rows = []
    for label, system in systems.items():
        row = {
            "system": label,
            "wrong": {s: system.wrong(s) for s in system.speakers},
            "words": {s: system.words(s) for s in system.speakers},
            "pooled_wrong": system.pooled_wrong(),
            "pooled_words": system.pooled_words(),
            "pooled_wer": system.pooled_wrong() / system.pooled_words(),
            "empty_clips": system.pooled_empty()[0],
            "clips": system.pooled_empty()[1],
            "meets_bar": system.pooled_wrong() <= lab.bar_words(system.pooled_words()),
            "interval": lab.pooled_interval(system, by=by),
        }
        if label != baseline:
            gains = {s: 100.0 * (base.wrong(s) - system.wrong(s)) / system.words(s) for s in system.speakers}
            pooled = 100.0 * (base.pooled_wrong() - system.pooled_wrong()) / system.pooled_words()
            row["against"] = baseline
            row["gain_points"] = pooled
            row["gain_points_by_speaker"] = gains
            row["gain_interval"] = lab.pooled_interval(system, base, by=by)
            row["verdict"] = verdict(pooled, gains, noise_points)
        rows.append(row)
    return rows


def render(rows: list[dict], speakers: list[str], by: str, kind: str = "all") -> str:
    width = max(len(r["system"]) for r in rows)
    head = f"{'system':<{width}}  " + " ".join(f"{s:>9}" for s in speakers) + "   pooled  WER    95% CI         empty  bar"
    what = "every clip" if kind == "all" else f"{kind} clips only"
    out = [f"{what}: wrong words / reference words per speaker; intervals resample {by}s, 2000 draws, seed 0", head]
    for r in rows:
        cells = " ".join(f"{r['wrong'][s]:>4}/{r['words'][s]:<4}" for s in speakers)
        lo, hi = r["interval"]["ci"]
        out.append(f"{r['system']:<{width}}  {cells}  {r['pooled_wrong']:>3}/{r['pooled_words']:<3} {100 * r['pooled_wer']:5.1f}%  "
                   f"{100 * lo:5.1f} to {100 * hi:5.1f}  {r['empty_clips']:>3}/{r['clips']:<3} {'meets' if r['meets_bar'] else 'no'}")
    for r in rows:
        if "gain_points" not in r:
            continue
        lo, hi = r["gain_interval"]["gain_ci"]
        per = ", ".join(f"{s} {g:+.1f}" for s, g in r["gain_points_by_speaker"].items())
        out.append(f"{r['system']} against {r['against']}: pooled gain {r['gain_points']:+.1f} points "
                   f"(95% CI {100 * lo:+.1f} to {100 * hi:+.1f}); by speaker {per}; verdict {r['verdict']}")
    out.append(f"bar: 35% of the pooled words is {lab.bar_words(rows[0]['pooled_words'])} wrong or fewer")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, metavar="LABEL=SPK:PATH,SPK:PATH",
                    help="one system's hypotheses per speaker; repeat for each system")
    ap.add_argument("--baseline", help="label of the system the others are compared with (default: the first)")
    ap.add_argument("--speakers", default=",".join(lab.SEVERE_TEST), help="speakers to pool (default M01,M02,M04)")
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--export", default=lab.EXPORT)
    ap.add_argument("--by", default="clip", choices=["clip", "prompt"], help="resampling unit for the intervals")
    ap.add_argument("--noise", type=float, default=NOISE_POINTS, help="smallest pooled change in WER points that can count")
    ap.add_argument("--kind", default="all", choices=["all", *lab.KINDS],
                    help="clips in the main table (default all); with 'all' a sentences-only table follows it")
    ap.add_argument("--out", help="also write the table as .json (refuses to overwrite)")
    a = ap.parse_args()
    if a.out:
        lab.refuse_existing(a.out)
    speakers = [s.strip() for s in a.speakers.split(",") if s.strip()]
    full = {s: lab.read_manifest(lab.here(f"{a.export}/{s}/{a.split}.jsonl")) for s in speakers}
    hyps: dict[str, dict[str, dict[str, str]]] = {}
    sources: dict[str, str] = {}
    for text in a.system:
        label, folders = lab.parse_system(text)
        if label in hyps:
            raise ValueError(f"System {label!r} is listed twice")
        hyps[label] = {s: lab.read_hyps(folders[s]) for s in full if s in folders}
        for s, folder in folders.items():
            sources[f"{label}/{s}"] = folder
    baseline = a.baseline or next(iter(hyps))

    def table(kind: str) -> list[dict]:
        manifests = {s: lab.only_kind(m, kind) for s, m in full.items()}
        systems = {label: lab.Pooled(label, manifests, per) for label, per in hyps.items()}
        return build(systems, baseline, a.by, a.noise)

    rows = table(a.kind)
    print(render(rows, speakers, a.by, a.kind))
    sentence_rows = None
    if a.kind == "all" and any(lab.clip_kind(r) == "sentence" for m in full.values() for r in m.values()):
        sentence_rows = table("sentence")
        print()
        print("-- secondary score: sentences only (single-word clips left out)")
        print(render(sentence_rows, speakers, a.by, "sentence"))
    if a.out:
        lab.save_json(a.out, {"rows": rows, "kind": a.kind, "sentences": sentence_rows, "baseline": baseline, "speakers": speakers,
                              "split": a.split, "unit": a.by, "noise_points": a.noise, "provenance": lab.provenance(__file__, sources)})
        print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "pool_table"))
