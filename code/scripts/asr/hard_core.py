"""Where the remaining errors are."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
import labkit  # noqa: E402

DURATION_BINS = ((0.0, 1.5, "under 1.5 s"), (1.5, 3.0, "1.5 to 3 s"), (3.0, float("inf"), "over 3 s"))


def duration_bin(seconds: float) -> str:
    for low, high, label in DURATION_BINS:
        if low <= seconds < high:
            return label
    return DURATION_BINS[-1][2]


def analyse(systems: dict[str, dict[str, str]], export_root: str, top: int = 20) -> dict:
    labels = list(systems)
    speakers = sorted({s for per in systems.values() for s in per})
    for label, per in systems.items():
        if set(per) != set(speakers):
            raise ValueError(f"system {label!r} does not cover every speaker ({sorted(per)} against {speakers})")
    clips: list[dict] = []
    for s in speakers:
        manifest = labkit.read_manifest(labkit.test_manifest(s, export_root))
        counts = {label: labkit.clip_counts(manifest, labkit.read_hyps(systems[label][s])) for label in labels}
        hyps = {label: labkit.read_hyps(systems[label][s]) for label in labels}
        for clip_id, row in manifest.items():
            errors = {label: counts[label][clip_id][0] for label in labels}
            words = counts[labels[0]][clip_id][1]
            clips.append({"id": clip_id, "speaker": s, "kind": row.get("kind") or str(row.get("prompt_set", "")).replace("torgo-", "") or "unknown",
                          "duration_s": float(row.get("duration_s", 0.0)), "words": words, "reference": row["text"],
                          "errors": errors, "oracle": min(errors.values()), "answers": {label: hyps[label][clip_id] for label in labels}})
    words_total = sum(c["words"] for c in clips)
    pooled = {label: sum(c["errors"][label] for c in clips) for label in labels}
    best = min(labels, key=lambda label: pooled[label])
    oracle = sum(c["oracle"] for c in clips)
    all_wrong = [c for c in clips if all(c["errors"][label] >= c["words"] for label in labels)]
    all_right = [c for c in clips if all(c["errors"][label] == 0 for label in labels)]
    fixable = sum(c["errors"][best] - c["oracle"] for c in clips)

    def breakdown(key) -> dict:
        groups: dict[str, dict] = {}
        for c in clips:
            g = groups.setdefault(key(c), {"clips": 0, "words": 0, "oracle": 0, "best": 0, "every_system_wrong": 0})
            g["clips"] += 1
            g["words"] += c["words"]
            g["oracle"] += c["oracle"]
            g["best"] += c["errors"][best]
            g["every_system_wrong"] += int(all(c["errors"][label] >= c["words"] for label in labels))
        return groups

    hardest = sorted(clips, key=lambda c: (-c["oracle"] / max(1, c["words"]), -sum(c["errors"].values()) / max(1, c["words"]), c["id"]))[:top]
    return {"systems": labels, "speakers": speakers, "clips": len(clips), "words": words_total, "pooled": pooled, "best_system": best,
            "oracle": oracle, "best_errors_fixable_by_another_system": fixable, "best_errors_no_system_fixes": oracle,
            "clips_every_system_fully_wrong": len(all_wrong), "words_in_them": sum(c["words"] for c in all_wrong),
            "clips_every_system_right": len(all_right), "words_in_those": sum(c["words"] for c in all_right),
            "by_kind": breakdown(lambda c: c["kind"]), "by_duration": breakdown(lambda c: duration_bin(c["duration_s"])),
            "by_speaker": breakdown(lambda c: c["speaker"]), "hardest": hardest}


def show(result: dict) -> None:
    labels = result["systems"]
    print(f"{result['clips']} clips, {result['words']} words, {len(labels)} systems; best single system: {result['best_system']}")
    print(f"{'system':<32} {'wrong':>6} {'WER':>7}")
    for label in labels:
        print(f"{label:<32} {result['pooled'][label]:>6} {100 * result['pooled'][label] / result['words']:>6.1f}%")
    print(f"{'oracle (best system per clip)':<32} {result['oracle']:>6} {100 * result['oracle'] / result['words']:>6.1f}%")
    print()
    print(f"of the best system's {result['pooled'][result['best_system']]} wrong words, another saved system gets {result['best_errors_fixable_by_another_system']} right "
          f"and no system gets {result['best_errors_no_system_fixes']} right")
    print(f"clips every system gets fully wrong: {result['clips_every_system_fully_wrong']} ({result['words_in_them']} words); "
          f"every system gets fully right: {result['clips_every_system_right']} ({result['words_in_those']} words)")
    for title, key in (("by kind of clip", "by_kind"), ("by duration", "by_duration"), ("by speaker", "by_speaker")):
        print()
        print(f"-- {title}: clips, words, the best system's wrong words, the oracle's, clips every system gets fully wrong")
        print(f"{'group':<16} {'clips':>5} {'words':>6} {'best':>6} {'oracle':>7} {'all wrong':>9}")
        for group, g in sorted(result[key].items()):
            print(f"{group:<16} {g['clips']:>5} {g['words']:>6} {g['best']:>6} {g['oracle']:>7} {g['every_system_wrong']:>9}")
    print()
    print(f"-- the {len(result['hardest'])} hardest clips (highest share of words no system gets); test material, never tune on it")
    for c in result["hardest"]:
        print(f"{c['id']}  {c['speaker']}  {c['kind']}  {c['duration_s']:.1f} s  oracle {c['oracle']}/{c['words']}  ref: {c['reference']!r}")
        for label in labels:
            print(f"    {label:<30} {c['errors'][label]:>2}  {c['answers'][label]!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, metavar="LABEL=SPK:path,...", help="a system's saved answers per speaker")
    ap.add_argument("--export-root", default=labkit.EXPORT)
    ap.add_argument("--top", type=int, default=20, help="how many of the hardest clips to list")
    ap.add_argument("--out", required=True, help="results .json to write")
    a = ap.parse_args(argv)
    systems = dict(labkit.parse_system(text) for text in a.system)
    if len(systems) != len(a.system):
        raise SystemExit("two systems have the same label")
    result = analyse(systems, a.export_root, a.top)
    result["provenance"] = labkit.provenance(__file__, {f"{label}:{s}": p for label, per in systems.items() for s, p in per.items()})
    labkit.save_json(a.out, result)
    show(result)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
