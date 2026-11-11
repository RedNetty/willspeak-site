#!/usr/bin/env python3
"""Rescore every saved test decode on sentences only."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import labkit as lab

SEED = re.compile(r"(?:-s|-seed|seed)(1[7-9]|2[0-4])(?=\b|-|$)")
TAIL = re.compile(r"(/hypotheses)?(-test-nemo|-test)?(\.jsonl)?$")
FALLBACK_WORDS = {"M01": {"all": 169, "sentence": 122, "word": 47},
                  "M02": {"all": 95, "sentence": 68, "word": 27},
                  "M04": {"all": 161, "sentence": 113, "word": 48}}


def run_key(path: Path, reports: Path) -> str:
    rel = path.relative_to(reports).as_posix()
    for suffix in ("/hypotheses.jsonl", "/results.json", ".jsonl"):
        if rel.endswith(suffix):
            return rel[: -len(suffix)]
    return rel


def family_of(key: str, speaker: str) -> tuple[str, str]:
    seed_match = SEED.search(key)
    seed = seed_match.group(1) if seed_match else "-"
    family = SEED.sub("{seed}", key) if seed_match else key
    family = TAIL.sub("", family.replace(speaker, "{S}"))
    return family, seed


def from_decodes(reports: Path, manifests: dict[str, dict[str, dict]]) -> dict[str, dict]:
    ids = {s: set(m) for s, m in manifests.items()}
    kinds = {s: {k: lab.only_kind(m, k) for k in ("all", *lab.KINDS)} for s, m in manifests.items()}
    out: dict[str, dict] = {}
    for path in sorted(reports.rglob("*.jsonl")):
        if "manifest" in path.name or path.name.endswith("provenance.jsonl"):
            continue
        key = run_key(path, reports)
        if key in out:
            continue
        try:
            hyps = lab.read_hyps(path)
        except (ValueError, KeyError, json.JSONDecodeError, OSError):
            continue
        speaker = next((s for s, wanted in ids.items() if set(hyps) == wanted), None)
        if speaker is None:
            continue
        out[key] = {"speaker": speaker,
                    **{k: lab.totals(lab.clip_counts(rows, hyps)) for k, rows in kinds[speaker].items()}}
    return out


def from_results(reports: Path, words: dict[str, dict[str, int]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(reports.rglob("results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "word" not in data or not isinstance(data.get("by_group"), dict):
            continue
        groups = {str(k).replace("torgo-", ""): v for k, v in data["by_group"].items()}
        if not all(k in groups for k in lab.KINDS):
            continue
        key = run_key(path, reports)
        speaker = next((s for s in words if s in key), None)
        if speaker is None or data["word"].get("n_ref") != words[speaker]["all"]:
            continue
        if groups["sentence"].get("n_ref") != words[speaker]["sentence"]:
            continue
        out[key] = {"speaker": speaker, "all": (data["word"]["errors"], data["word"]["n_ref"]),
                    **{k: (groups[k]["errors"], groups[k]["n_ref"]) for k in lab.KINDS}}
    return out


def families(scored: dict[str, dict], speakers: list[str], partial: bool = False) -> list[dict]:
    grouped: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for key, row in scored.items():
        family, seed = family_of(key, row["speaker"])
        grouped[family][seed][row["speaker"]] = row
    rows: list[dict] = []
    for family, seeds in grouped.items():
        full = {seed: per for seed, per in seeds.items() if all(s in per for s in speakers)}
        if partial:
            if full:
                continue
            have = [s for s in speakers if all(s in per for per in seeds.values())]
            if not have:
                continue
            complete, use = seeds, have
        else:
            complete, use = full, speakers
        if not complete:
            continue
        speakers_here = use
        runs = []
        for seed in sorted(complete):
            per = complete[seed]
            runs.append({"seed": seed,
                         **{k: sum(per[s][k][0] for s in speakers_here) for k in ("all", *lab.KINDS)},
                         "sentence_by_speaker": {s: per[s]["sentence"][0] for s in speakers_here}})
        first = complete[next(iter(sorted(complete)))]
        words = {k: sum(first[s][k][1] for s in speakers_here) for k in ("all", *lab.KINDS)}
        sentence_words = {s: first[s]["sentence"][1] for s in speakers_here}
        mean = {k: sum(r[k] for r in runs) / len(runs) for k in ("all", *lab.KINDS)}
        rows.append({"family": family, "speakers": speakers_here, "seeds": [r["seed"] for r in runs], "runs": runs, "words": words,
                     "sentence_words_by_speaker": sentence_words,
                     "sentence_mean": mean["sentence"], "sentence_wer": mean["sentence"] / words["sentence"],
                     "sentence_by_seed": [r["sentence"] for r in runs],
                     "sentence_wer_by_speaker": {s: sum(r["sentence_by_speaker"][s] for r in runs) / len(runs) / sentence_words[s]
                                                 for s in speakers_here},
                     "all_mean": mean["all"], "all_wer": mean["all"] / words["all"],
                     "word_mean": mean["word"], "word_wer": mean["word"] / words["word"]})
    merged: dict[tuple, dict] = {}
    for row in sorted(rows, key=lambda r: r["family"]):
        fingerprint = tuple((r["seed"], r["all"], tuple(sorted(r["sentence_by_speaker"].items()))) for r in row["runs"])
        if fingerprint in merged:
            merged[fingerprint].setdefault("same_runs_as", []).append(row["family"])
        else:
            merged[fingerprint] = row
    return sorted(merged.values(), key=lambda r: (r["sentence_mean"], r["all_mean"]))


def render(rows: list[dict], speakers: list[str], source: str) -> str:
    if not rows:
        return "no complete decode of the test sets was found"
    words = rows[0]["words"]
    if any(r.get("speakers", speakers) != speakers for r in rows):
        out = ["recipes run on only some speakers (each scored on its own speakers; not comparable with the main table):",
               f"{'sent. wrong':>11} {'/words':>6} {'sent. WER':>9}  {'speakers':<9} {'seeds':>5}  {'by seed':<18} {'all-clip WER':>12}  recipe"]
        for r in rows:
            out.append(f"{r['sentence_mean']:>11.1f} {r['words']['sentence']:>6} {100 * r['sentence_wer']:>8.1f}%  {'+'.join(r['speakers']):<9} "
                       f"{len(r['seeds']):>5}  {' '.join(str(x) for x in r['sentence_by_seed']):<18} {100 * r['all_wer']:>11.1f}%  {r['family']}")
        return "\n".join(out)
    bar = lab.bar_words(words["sentence"])
    out = [f"sentences only: {words['sentence']} reference words over {', '.join(speakers)}; 35% or lower is {bar} wrong or fewer",
           f"source: {source}; means over the seeds listed; lower is better",
           f"{'#':>3}  {'sent. wrong':>11} {'sent. WER':>9}  " + " ".join(f"{s:>6}" for s in speakers)
           + f"  {'seeds':>5}  {'by seed':<24} {'all clips':>14} {'words only':>10}  recipe"]
    for i, r in enumerate(rows, 1):
        per = " ".join(f"{100 * r['sentence_wer_by_speaker'][s]:5.1f}%" for s in speakers)
        seeds = " ".join(str(x) for x in r["sentence_by_seed"])
        name = r["family"] + (f"  (same runs: {', '.join(r['same_runs_as'])})" if r.get("same_runs_as") else "")
        out.append(f"{i:>3}  {r['sentence_mean']:>11.1f} {100 * r['sentence_wer']:>8.1f}%  {per}  {len(r['seeds']):>5}  {seeds:<24} "
                   f"{r['all_mean']:>6.1f} {100 * r['all_wer']:>5.1f}% {100 * r['word_wer']:>9.1f}%  {name}"
                   + ("  meets 35%" if r["sentence_mean"] <= bar else ""))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--export", default=lab.EXPORT)
    ap.add_argument("--speakers", default=",".join(lab.SEVERE_TEST))
    ap.add_argument("--source", choices=["auto", "decodes", "results"], default="auto",
                    help="auto: score the decodes when the test manifests are here, else read results.json")
    ap.add_argument("--out", help="also write the table as .json (refuses to overwrite)")
    a = ap.parse_args()
    if a.out:
        lab.refuse_existing(a.out)
    speakers = [s.strip() for s in a.speakers.split(",") if s.strip()]
    reports = lab.here(a.reports)
    have_manifests = all(lab.test_manifest(s, a.export).is_file() for s in speakers)
    use_decodes = a.source == "decodes" or (a.source == "auto" and have_manifests)
    if use_decodes:
        if not have_manifests:
            raise FileNotFoundError(f"test manifests missing under {a.export}; use --source results")
        manifests = {s: lab.read_manifest(lab.test_manifest(s, a.export)) for s in speakers}
        scored = from_decodes(reports, manifests)
        source = f"{len(scored)} saved decodes scored against the test manifests"
    else:
        words = {s: FALLBACK_WORDS[s] for s in speakers}
        scored = from_results(reports, words)
        source = f"{len(scored)} results.json files (by_group counts)"
    rows = families(scored, speakers)
    partial = families(scored, speakers, partial=True)
    print(render(rows, speakers, source))
    if partial:
        print()
        print(render(partial, speakers, source))
    if a.out:
        lab.save_json(a.out, {"kind": "sentence", "speakers": speakers, "source": source, "rows": rows, "partial_rows": partial,
                              "provenance": lab.provenance(__file__, {})})
        print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "sentence_rescore"))
