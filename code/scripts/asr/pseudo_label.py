#!/usr/bin/env python3
"""Turn a model's own answers into training labels, without ever reading the real ones."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import labkit as lab
import phrasebook
from willspeak.normalize import normalize

SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70)


def read_phrases(files: list[str], manifests: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for path in files:
        for line in lab.here(path).read_text(encoding="utf-8").splitlines():
            if normalize(line):
                seen.setdefault(normalize(line))
    for path in manifests:
        with lab.here(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    text = str(json.loads(line).get("text", ""))
                    if "(live)" not in text and normalize(text):
                        seen.setdefault(normalize(text))
    return list(seen)


def gate(hyp: str, phrases: list[phrasebook.Item], threshold: float, margin: float) -> dict:
    if len(phrases) < 2:
        raise ValueError("The gate compares the two nearest phrases, so the list needs at least two")
    if not normalize(hyp).split():
        return {"accepted": False, "reason": "empty answer", "phrase": None, "distance": None, "gap": None}
    item = phrasebook.Item(hyp)
    ranked = sorted((phrasebook.distance(item, p), i) for i, p in enumerate(phrases))
    best, index = ranked[0]
    gap = ranked[1][0] - best
    reason = "ok" if best <= threshold and gap >= margin else ("too far from every phrase" if best > threshold
                                                              else "two phrases about equally near")
    return {"accepted": reason == "ok", "reason": reason, "phrase": phrases[index].norm, "distance": best, "gap": gap}


def dev_phrases(phrases: list[str], share: float) -> set[str]:
    ordered = sorted(phrases, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    return set(ordered[: max(1, round(share * len(ordered)))])


def audit(decisions: dict[str, dict], truth: dict[str, str], phrases: set[str], margin: float) -> dict:
    def at(threshold: float) -> dict:
        kept = [c for c, d in decisions.items() if d["distance"] is not None and d["distance"] <= threshold and d["gap"] >= margin]
        right = sum(1 for c in kept if decisions[c]["phrase"] == normalize(truth[c]))
        return {"threshold": threshold, "kept": len(kept), "right": right, "precision": right / len(kept) if kept else None}

    kept = [c for c, d in decisions.items() if d["accepted"]]
    wrong = [c for c in kept if decisions[c]["phrase"] != normalize(truth[c])]
    return {"kept": len(kept), "right": len(kept) - len(wrong), "wrong": len(wrong),
            "wrong_because_true_phrase_not_on_list": sum(1 for c in wrong if normalize(truth[c]) not in phrases),
            "precision": (len(kept) - len(wrong)) / len(kept) if kept else None,
            "true_phrase_on_list": sum(1 for c in decisions if normalize(truth[c]) in phrases), "clips": len(decisions),
            "by_threshold": [at(t) for t in SWEEP], "note": "read after the choice was made; not used to choose anything"}


def write_export(rows: list[dict], decisions: dict[str, dict], audio_root: Path, out: Path, dev_share: float) -> dict:
    kept = [r for r in rows if decisions[str(r["id"])]["accepted"]]
    phrases = sorted({decisions[str(r["id"])]["phrase"] for r in kept})
    if len(phrases) < 5:
        raise ValueError(f"Only {len(phrases)} distinct phrases were kept; too few to hold some out as a dev set")
    held_out = dev_phrases(phrases, dev_share)
    split_rows: dict[str, list[dict]] = {"train": [], "dev": []}
    for row in kept:
        phrase = decisions[str(row["id"])]["phrase"]
        split = "dev" if phrase in held_out else "train"
        name = f"{split}/{row['id']}.wav"
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lab.here(audio_root / row["audio"]), dst)
        split_rows[split].append({
            "id": str(row["id"]), "audio": name, "text": phrase, "duration_s": row["duration_s"], "condition": "unknown",
            "speaker": row.get("speaker", "unknown"), "kind": "sentence" if len(phrase.split()) >= 3 else "word",
            "prompt_set": "pseudo-label"})
    for split, items in split_rows.items():
        with (out / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(r) + "\n" for r in items))
    return {"train_clips": len(split_rows["train"]), "dev_clips": len(split_rows["dev"]), "distinct_phrases": len(phrases)}


def read_audio_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append({"id": str(row["id"]), "audio": row["audio"], "duration_s": row["duration_s"],
                             "speaker": row.get("speaker", "unknown")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="his earlier recordings (a train.jsonl); only id, audio and duration are read")
    ap.add_argument("--audio-root", help="where those audio paths start (default: the manifest's folder)")
    ap.add_argument("--hyps", required=True, help="the current model's answers for those clips (folder or .jsonl)")
    ap.add_argument("--phrases", nargs="*", default=[], help="text files, one phrase per line")
    ap.add_argument("--phrases-from", nargs="*", default=[], metavar="MANIFEST", help="manifests whose text column is the phrase list")
    ap.add_argument("--test-manifest", help="refuse to run if any phrase is one of its prompts")
    ap.add_argument("--threshold", type=float, default=0.50, help="largest distance to the nearest phrase that is kept")
    ap.add_argument("--margin", type=float, default=0.10, help="how much nearer the best phrase must be than the second best")
    ap.add_argument("--dev-share", type=float, default=0.12, help="share of phrases held out as the dev split")
    ap.add_argument("--audit", action="store_true", help="after choosing, count right and wrong labels against the true text")
    ap.add_argument("--out", required=True, help="new export folder (refuses to overwrite)")
    a = ap.parse_args()
    if not a.phrases and not a.phrases_from:
        ap.error("give a phrase list: --phrases FILE or --phrases-from MANIFEST")
    if not 0 < a.dev_share <= 0.5 or a.threshold <= 0 or a.margin < 0:
        ap.error("--dev-share must be above 0 and at most 0.5, --threshold above 0, --margin not negative")
    lab.refuse_existing(a.out)
    manifest = lab.here(a.manifest)
    audio_root = lab.here(a.audio_root) if a.audio_root else manifest.parent
    rows = read_audio_rows(manifest)
    hyps = lab.read_hyps(a.hyps)
    missing = [r["id"] for r in rows if r["id"] not in hyps]
    if missing:
        raise ValueError(f"--hyps has no answer for {len(missing)} clips, first {missing[:3]}")
    phrase_list = read_phrases(a.phrases, a.phrases_from)
    if len(phrase_list) < 10:
        raise ValueError(f"A phrase list of {len(phrase_list)} is too short to tell phrases apart")
    if a.test_manifest:
        test = {normalize(r["text"]) for r in lab.read_manifest(a.test_manifest).values()}
        shared = sorted(set(phrase_list) & test)
        if shared:
            raise ValueError(f"The phrase list contains {len(shared)} test prompts, first {shared[:3]}; use other speakers' train/dev prompts")
    else:
        print("note: no --test-manifest, so test prompts leaking in through the phrase list is not checked")
    items = [phrasebook.Item(p) for p in phrase_list]
    decisions = {r["id"]: {"hyp": hyps[r["id"]], **gate(hyps[r["id"]], items, a.threshold, a.margin)} for r in rows}
    out = lab.here(a.out)
    out.mkdir(parents=True)
    counts = write_export(rows, decisions, audio_root, out, a.dev_share)
    (out / "phrases.txt").write_text("".join(p + "\n" for p in phrase_list), encoding="utf-8")
    with (out / "decisions.jsonl").open("x", encoding="utf-8") as handle:
        handle.write("".join(json.dumps({"id": c, **d}) + "\n" for c, d in decisions.items()))
    kept = [c for c, d in decisions.items() if d["accepted"]]
    reasons: dict[str, int] = {}
    for d in decisions.values():
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    summary = {
        "format": "willspeak-pseudo-labels-v1", "clips": len(rows), "kept": len(kept), "kept_share": len(kept) / len(rows),
        "reasons": reasons, "threshold": a.threshold, "margin": a.margin, "dev_share": a.dev_share, "phrases_on_list": len(phrase_list),
        "kept_that_equal_the_raw_answer": sum(1 for c in kept if normalize(decisions[c]["hyp"]) == decisions[c]["phrase"]),
        **counts, "audit": None}
    if a.audit:
        truth = {str(r["id"]): r["text"] for r in lab.read_manifest(manifest).values()}
        summary["audit"] = audit(decisions, truth, set(phrase_list), a.margin)
    inputs = {"manifest": a.manifest, "hypotheses": a.hyps}
    lab.save_json(out / "selection.json", {**summary, "provenance": lab.provenance(__file__, inputs)})
    print(f"kept {len(kept)} of {len(rows)} clips ({100 * summary['kept_share']:.1f}%): train {counts['train_clips']}, "
          f"dev {counts['dev_clips']}, {counts['distinct_phrases']} phrases; rejected for: {reasons}")
    if summary["audit"]:
        s = summary["audit"]
        print(f"audit (not used to choose): {s['right']} of {s['kept']} kept labels were right "
              f"({100 * s['precision']:.1f}%); {s['wrong_because_true_phrase_not_on_list']} of the {s['wrong']} wrong ones were "
              f"phrases that were not on the list; true phrase on the list for {s['true_phrase_on_list']} of {s['clips']} clips")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "pseudo_label"))
