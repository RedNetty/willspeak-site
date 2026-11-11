#!/usr/bin/env python3
"""Scan TORGO and write the manifests and export folders the experiments use."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _cli


def project_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def export_rows(utts, torgo_root: Path, out_dir: Path) -> dict[str, int]:
    import soundfile as sf

    from willspeak import audio as A
    from willspeak.torgo import write_jsonl

    counts = {"train": 0, "dev": 0, "test": 0}
    rows = {"train": [], "dev": [], "test": []}
    for u in utts:
        dst = out_dir / u.split / f"{u.id}.wav"
        if not dst.exists():
            x, sr = sf.read(str(torgo_root / u.audio), dtype="float32")
            y = A.to_mono_16k(x, sr)
            dst.parent.mkdir(parents=True, exist_ok=True)
            A.write_wav(str(dst), y, 16000, subtype="PCM_16")
        info = sf.info(str(dst))
        rows[u.split].append({
            "id": u.id, "audio": f"{u.split}/{u.id}.wav", "text": u.text, "prompt_text": u.text,
            "duration_s": round(info.duration, 3), "prompt_set": f"torgo-{u.kind}",
            "condition": "unknown", "speaker": u.speaker, "severity": u.severity,
            "session": u.session, "kind": u.kind, "mic": u.mic,
        })
    for split, rs in rows.items():
        counts[split] = write_jsonl(rs, out_dir / f"{split}.jsonl")
    return counts


def write_test_subsets(export_dir: Path, severe) -> None:
    with open(export_dir / "test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    subsets = {
        "test-sentences.jsonl": lambda r: r["kind"] == "sentence",
        "test-severe-sentences.jsonl": lambda r: r["kind"] == "sentence" and r["speaker"] in severe,
    }
    for name, keep in subsets.items():
        kept = [r for r in rows if keep(r)]
        with open(export_dir / name, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in kept))
        print(f"    {name}: {len(kept)} clips")


def pooled_rows(utts, present, held) -> tuple[list, int]:
    from willspeak.torgo import session_holdout, split_key

    rows, test_keys = [], set()
    for spk in present:
        mine = [u for u in utts if u.speaker == spk]
        if spk in held:
            kept, _ = session_holdout(mine)
            rows += [u for u in kept if u.split in ("train", "dev")]
            test_keys |= {split_key(u.text) for u in kept if u.split == "test"}
        else:
            rows += [u for u in mine if u.split == "train"]
    overlap = sum(1 for u in rows if split_key(u.text) in test_keys)
    return rows, overlap


def describe_export(folder: Path) -> dict:
    speakers: dict[str, dict] = {}
    totals = {"train_clips": 0, "dev_clips": 0, "train_minutes": 0.0, "dev_minutes": 0.0}
    for split in ("train", "dev"):
        path = folder / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            entry = speakers.setdefault(row["speaker"], {"train_clips": 0, "dev_clips": 0, "train_minutes": 0.0, "dev_minutes": 0.0})
            minutes = float(row["duration_s"]) / 60
            for record in (entry, totals):
                record[f"{split}_clips"] += 1
                record[f"{split}_minutes"] += minutes
    for record in (*speakers.values(), totals):
        for key in ("train_minutes", "dev_minutes"):
            record[key] = round(record[key], 2)
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    return {**totals, "speakers": dict(sorted(speakers.items())), "bytes": size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/public/torgo", help="folder holding TORGO/ (from fetch_torgo.py)")
    ap.add_argument("--torgo", help="TORGO root if not <out>/TORGO")
    ap.add_argument("--export", action="store_true", help="write train/dev/test export folders")
    ap.add_argument("--missing-only", action="store_true",
                    help="with --export, create absent export folders and preserve existing ones")
    ap.add_argument("--speakers", nargs="*", help="limit exports to these speakers (default: every dysarthric speaker)")
    ap.add_argument("--curve", help="also write export/<SPEAKER>-min1|5|15 with the training set cut to N minutes")
    ap.add_argument("--curve-minutes", type=float, nargs="+", default=[1, 5, 15])
    ap.add_argument("--severe-loso", action="store_true",
                    help="also write export/loso-severe-<SPEAKER> for each severe speaker among the exported ones")
    ap.add_argument("--severe-base", action="store_true",
                    help="also write export/severe-base: every severe speaker's train and dev clips, no test clips. "
                         "It trains the shared base under a personal adapter for someone outside TORGO "
                         "(scripts/asr/personal.py base)")
    ap.add_argument("--pooled", action="store_true",
                    help="also write export/pooled-<SPEAKERS>: every dysarthric speaker's training prompts in one set, the "
                         "named speakers' own held-out session excluded, their dev clips as dev, no test clips")
    ap.add_argument("--only", nargs="*", metavar="NAME",
                    help="with --export, write only these export folders (for example controls, or loso-M01)")
    a = ap.parse_args()

    from willspeak.torgo import CITATION, CONTROLS, DYSARTHRIC, SEVERE, check_disjoint, scan, summarize, write_jsonl

    out = project_path(a.out)
    torgo = project_path(a.torgo) if a.torgo else out / "TORGO"
    if not torgo.is_dir():
        print(f"TORGO not found at {torgo}; run scripts/corpus/fetch_torgo.py first", file=sys.stderr)
        return 1
    utts = scan(torgo)
    if not utts:
        print(f"no usable utterances under {torgo}", file=sys.stderr)
        return 1
    check_disjoint(utts)

    man = out / "manifests"
    n_all = write_jsonl([u.row() for u in utts], man / "all.jsonl")
    n_sev = write_jsonl([u.row() for u in utts if u.speaker in SEVERE and u.kind == "sentence"],
                        man / "severe-sentences.jsonl")
    n_sent = write_jsonl([u.row() for u in utts if u.speaker in DYSARTHRIC and u.kind == "sentence"],
                         man / "all-sentences.jsonl")
    words = [u for u in utts if u.speaker in DYSARTHRIC and u.kind == "word"]
    with open(man / "words.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "session", "speaker"])
        for u in words:
            w.writerow([f"../TORGO/{u.audio}", u.text.lower(), f"{u.speaker}_{u.session}", u.speaker])

    table = summarize(utts)
    lines = ["# TORGO manifest summary", "", f"Utterances kept: {n_all} (one file each; head mic preferred).",
             f"Severe-speaker sentences: {n_sev}. Dysarthric sentences: {n_sent}. Dysarthric single words: {len(words)}.",
             "", "| speaker | severity | sessions | words | sentences | train | dev | test | head mic | array mic |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spk, s in table.items():
        lines.append(f"| {spk} | {s['severity']} | {s['sessions']} | {s['word']} | {s['sentence']} | "
                     f"{s['train']} | {s['dev']} | {s['test']} | {s['headMic']} | {s['arrayMic']} |")
    lines += ["", "Split by normalised prompt text, identical for every speaker (70/10/20 by prompt).",
              "", f"Cite: {CITATION}"]
    (man / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    print(f"manifests -> {man}")

    if a.export:
        import soundfile as sf

        from willspeak.torgo import session_holdout

        present = [s for s in DYSARTHRIC if any(u.speaker == s for u in utts)]
        targets: dict[str, list] = {}
        chosen = tuple(a.speakers) if a.speakers else present
        for spk in chosen:
            mine = [u for u in utts if u.speaker == spk]
            if not mine:
                continue
            others = [u for u in utts if u.speaker in present and u.speaker != spk and u.split != "test"]
            targets[f"loso-{spk}"] = others + [u for u in mine if u.split == "test"]
            if a.severe_loso and spk in SEVERE:
                severe_others = [u for u in others if u.speaker in SEVERE]
                targets[f"loso-severe-{spk}"] = severe_others + [u for u in mine if u.split == "test"]
            targets[spk], note = session_holdout(mine)
            print(f"  {spk}: {note}")
        if a.severe_base:
            targets["severe-base"] = [u for u in utts if u.speaker in SEVERE and u.split != "test"]
        if not a.speakers:
            targets["controls"] = [u for u in utts if u.speaker in CONTROLS]
            targets["all-dysarthric"] = [u for u in utts if u.speaker in present]
        pooled_name, pooled_held = None, []
        if a.pooled:
            held = [s for s in chosen if s in present]
            if not held:
                print("--pooled: none of the named speakers is downloaded", file=sys.stderr)
                return 1
            rows, overlap = pooled_rows(utts, present, held)
            if overlap:
                print(f"--pooled: {overlap} training or dev clips share a prompt with a held-out test clip; nothing written",
                      file=sys.stderr)
                return 1
            pooled_name, pooled_held = "pooled-" + "-".join(held), held
            targets[pooled_name] = rows
        if a.curve:
            base, _ = session_holdout([u for u in utts if u.speaker == a.curve])
            train = [u for u in base if u.split == "train"]
            train.sort(key=lambda u: hashlib.sha256(u.id.encode()).hexdigest())
            for minutes in a.curve_minutes:
                keep, total = set(), 0.0
                for u in train:
                    if total >= minutes * 60:
                        break
                    keep.add(u.id)
                    total += sf.info(str(torgo / u.audio)).duration
                label = f"{minutes:g}"
                targets[f"{a.curve}-min{label}"] = [u for u in base if u.split != "train" or u.id in keep]
        if a.only:
            unknown = [n for n in a.only if n not in targets]
            if unknown:
                print(f"--only: no export folder named {', '.join(unknown)}; this command would write: {', '.join(targets)}", file=sys.stderr)
                return 1
            targets = {n: r for n, r in targets.items() if n in a.only}
        for name, rows in targets.items():
            if not rows:
                print(f"  {name}: nothing to export (speaker not downloaded?)")
                continue
            destination = out / "export" / name
            if a.missing_only and destination.exists() and any(destination.iterdir()):
                print(f"  export/{name}: kept existing folder (--missing-only)")
                continue
            counts = export_rows(rows, torgo, destination)
            print(f"  export/{name}: train {counts['train']}  dev {counts['dev']}  test {counts['test']}")
            if name == "all-dysarthric":
                write_test_subsets(out / "export" / name, SEVERE)
            if name == pooled_name:
                record = {"targets": pooled_held, "speakers_present": present, "test_prompt_overlap": 0,
                          "rule": "train: every dysarthric speaker's train-prompt clips, the targets' from their training "
                                  "sessions only; dev: the targets' own dev clips; test: none (each target is scored on "
                                  "export/<speaker>/test.jsonl)",
                          **describe_export(destination)}
                (destination / "pooled.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                for spk, entry in record["speakers"].items():
                    print(f"    {spk}: train {entry['train_clips']} clips / {entry['train_minutes']} min, "
                          f"dev {entry['dev_clips']} clips / {entry['dev_minutes']} min")
                print(f"    total: train {record['train_clips']} clips / {record['train_minutes']} min, "
                      f"dev {record['dev_clips']} clips; {record['bytes'] / 2**20:.0f} MB on disk; record in {name}/pooled.json")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "torgo_manifest"))
