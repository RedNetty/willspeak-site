#!/usr/bin/env python3
"""Prompt-grouped cross-validation for the personal models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import labkit as lab  # noqa: E402
from willspeak.normalize import normalize  # noqa: E402

EXPORT = "data/public/torgo/export"
VERSION = 1


def prompt_key(row: dict) -> str:
    return normalize(str(row.get("prompt_text") or row.get("text") or ""), us_spelling=False)


def stable(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    for attempt in (os.link, os.symlink):
        try:
            attempt(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            continue
    shutil.copyfile(src, dst)


def assign_folds(rows: list[dict], folds: int, seed: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("cross-validation needs at least 2 folds")
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = prompt_key(r)
        if not key:
            raise ValueError(f"clip {r.get('id')} has an empty transcript")
        groups.setdefault(key, []).append(r)
    if len(groups) < folds:
        raise ValueError(f"{len(groups)} distinct prompts cannot fill {folds} folds")
    load = [0] * folds
    out: dict[str, int] = {}
    order = sorted(groups, key=lambda k: (-len(k.split()), stable(seed, k)))
    for key in order:
        fold = min(range(folds), key=lambda i: (load[i], stable(seed, i, key)))
        load[fold] += len(key.split()) * len(groups[key])
        for r in groups[key]:
            out[str(r["id"])] = fold
    return out


def layout_ok(folder: Path) -> bool:
    for split in ("train", "dev"):
        path = folder / f"{split}.jsonl"
        if not path.is_file() or any(not r["audio"].startswith(f"{split}/") for r in read_rows(path)):
            return False
    return True


def make(export: Path, speaker: str, folds: int, seed: int = 17, prefix: str | None = None, root: Path = ROOT) -> dict:
    export = Path(export)
    out_dir = Path(root) / EXPORT if prefix is None else Path(prefix).parent
    name = f"cv-{speaker}" if prefix is None else Path(prefix).name
    labelled: list[dict] = []
    for split in ("train", "dev"):
        path = export / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing (python lab.py doctor --fix, or torgo_manifest.py --export)")
        labelled += read_rows(path)
    ids = [str(r["id"]) for r in labelled]
    if len(set(ids)) != len(ids):
        raise ValueError("a clip id appears twice in train and dev")
    missing = [r["audio"] for r in labelled if not (export / r["audio"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files of {export} are missing, e.g. {missing[0]}; python lab.py doctor --fix rebuilds them")
    fold_of = assign_folds(labelled, folds, seed)
    settings = {"speaker": speaker, "folds": folds, "seed": seed, "version": VERSION, "clips": len(labelled),
                "fingerprint": hashlib.sha256("\n".join(sorted(f"{i}:{fold_of[i]}" for i in ids)).encode()).hexdigest()[:16]}
    result = {"settings": settings, "folders": []}
    names = [f"{name}-all"] + [f"{name}-f{k}" for k in range(folds)]
    marker = out_dir / names[0] / "cv.json"
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("settings") == settings and all((out_dir / n / "test.jsonl").is_file() for n in names):
            if all(layout_ok(out_dir / n) for n in names if not n.endswith("-all")):
                print(f"{name}: already built with these settings; reusing")
                return old
            print(f"{name}: rebuilding folds whose audio is not under train/ and dev/")
            for n in names:
                shutil.rmtree(out_dir / n, ignore_errors=True)
        else:
            raise FileExistsError(f"{out_dir / names[0]} exists with other settings; remove or rename it first")
    for n in names:
        if (out_dir / n).exists():
            raise FileExistsError(f"{out_dir / n} exists but is not a finished build of this tool")
    scratch = out_dir / (name + ".partial")
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)

    def clip(r: dict, folder: Path, role: str) -> dict:
        new = f"{role}/{str(r['id']).replace('/', '_')}.wav"
        link_or_copy(export / r["audio"], folder / new)
        return {**r, "audio": new}

    for n in names:
        folder = scratch / n
        folder.mkdir(parents=True)
        if n.endswith("-all"):
            rows = [clip(r, folder, "all") for r in labelled]
            write_rows(folder / "train.jsonl", rows)
            write_rows(folder / "dev.jsonl", rows)
            write_rows(folder / "test.jsonl", rows)
            info = {"decode_only": True, "clips": len(rows)}
        else:
            k = int(n.rsplit("-f", 1)[1])
            held = [r for r in labelled if fold_of[str(r["id"])] == k]
            rest = [r for r in labelled if fold_of[str(r["id"])] != k]
            write_rows(folder / "train.jsonl", [clip(r, folder, "train") for r in rest])
            write_rows(folder / "dev.jsonl", [clip(r, folder, "dev") for r in held])
            write_rows(folder / "test.jsonl", [clip(r, folder, "test") for r in held])
            info = {"fold": k, "train_clips": len(rest), "heldout_clips": len(held), "heldout_words": sum(len(prompt_key(r).split()) for r in held)}
        (folder / "cv.json").write_text(json.dumps({"settings": settings, "folder": n, **info,
                                                    "folds": {str(i): fold_of[i] for i in ids} if n.endswith("-all") else None,
                                                    "note": "Training reads train and dev only. dev and test are both the held-out prompts."},
                                                   indent=1), encoding="utf-8")
        result["folders"].append({"folder": n, **info})
    for n in names:
        (out_dir / n).parent.mkdir(parents=True, exist_ok=True)
        (scratch / n).rename(out_dir / n)
    shutil.rmtree(scratch, ignore_errors=True)
    check(speaker, folds, Path(root), name=name, base=out_dir)
    words = [f["heldout_words"] for f in result["folders"] if "heldout_words" in f]
    print(f"{name}: {len(labelled)} labelled clips in {folds} prompt-grouped folds; held-out words per fold {words}")
    return result


def check(speaker: str, folds: int, root: Path = ROOT, name: str | None = None, base: Path | None = None) -> None:
    base = Path(base) if base else Path(root) / EXPORT
    name = name or f"cv-{speaker}"
    all_ids = [str(r["id"]) for r in read_rows(base / f"{name}-all" / "test.jsonl")]
    seen: list[str] = []
    for k in range(folds):
        folder = base / f"{name}-f{k}"
        train = read_rows(folder / "train.jsonl")
        held = read_rows(folder / "test.jsonl")
        shared = {prompt_key(r) for r in train} & {prompt_key(r) for r in held}
        if shared:
            raise ValueError(f"fold {k}: {len(shared)} prompts are on both sides, e.g. {sorted(shared)[:2]}")
        if {str(r["id"]) for r in train} & {str(r["id"]) for r in held}:
            raise ValueError(f"fold {k}: a clip is in both train and held-out")
        for r in train + held:
            if not (folder / r["audio"]).is_file():
                raise FileNotFoundError(f"fold {k}: audio {r['audio']} is missing")
        seen += [str(r["id"]) for r in held]
    if sorted(seen) != sorted(all_ids):
        raise ValueError("the held-out folds do not cover each labelled clip exactly once")


def fold_ids(speaker: str, folds: int, root: Path = ROOT, name: str | None = None, base: Path | None = None) -> dict[int, list[str]]:
    base = Path(base) if base else Path(root) / EXPORT
    name = name or f"cv-{speaker}"
    return {k: [str(r["id"]) for r in read_rows(base / f"{name}-f{k}" / "test.jsonl")] for k in range(folds)}


def merge(speaker: str, folds: int, pattern: str, out: Path, root: Path = ROOT, base: Path | None = None, force: bool = False) -> Path:
    base = Path(base) if base else Path(root) / EXPORT
    ids = fold_ids(speaker, folds, root, base=base)
    merged: dict[str, str] = {}
    for k in range(folds):
        path = Path(pattern.format(fold=k, speaker=speaker))
        path = path if path.is_absolute() else Path(root) / path
        hyps = lab.read_hyps(path)
        absent = [i for i in ids[k] if i not in hyps]
        if absent:
            raise ValueError(f"{path} has no answer for {len(absent)} held-out clips of fold {k}, e.g. {absent[:2]}")
        merged.update({i: hyps[i] for i in ids[k]})
    out = Path(out)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps({"id": i, "hyp": merged[i]}) + "\n" for i in sorted(merged)), encoding="utf-8")
    return out


def group_counts(speaker: str, hyps: dict[str, str], labelled: dict[str, dict]) -> dict[tuple[str, str], tuple[int, int]]:
    per: dict[tuple[str, str], list[int]] = {}
    counts = lab.clip_counts(labelled, hyps)
    for cid, (e, n) in counts.items():
        key = (speaker, prompt_key(labelled[cid]))
        cur = per.setdefault(key, [0, 0])
        cur[0] += e
        cur[1] += n
    return {k: (v[0], v[1]) for k, v in per.items()}


def bootstrap(errors: np.ndarray, words: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(errors), size=(draws, len(errors)))


def select(arms: dict[str, dict[str, dict[str, str]]], labelled: dict[str, dict[str, dict]], control: str, min_gain_points: float = 1.0,
           draws: int = 4000, seed: int = 17) -> dict:
    if control not in arms:
        raise ValueError(f"control {control!r} is not among the arms {sorted(arms)}")
    speakers = sorted(labelled)
    table: dict[str, dict] = {}
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    keys: list[tuple[str, str]] | None = None
    for arm, per in arms.items():
        grouped: dict[tuple[str, str], tuple[int, int]] = {}
        for s in speakers:
            if s not in per:
                raise ValueError(f"arm {arm!r} has no held-out decode for {s}")
            grouped.update(group_counts(s, per[s], labelled[s]))
        arm_keys = sorted(grouped)
        if keys is None:
            keys = arm_keys
        elif arm_keys != keys:
            raise ValueError(f"arm {arm!r} covers different prompts than the others")
        series[arm] = (np.array([grouped[k][0] for k in keys]), np.array([grouped[k][1] for k in keys]))
    assert keys is not None
    idx = bootstrap(*series[control], draws=draws, seed=seed)
    ctrl_e, ctrl_n = series[control]
    for arm, (e, n) in series.items():
        pooled = float(e.sum() / n.sum())
        boot = e[idx].sum(axis=1) / n[idx].sum(axis=1)
        entry = {"wrong": int(e.sum()), "words": int(n.sum()), "rate": pooled,
                 "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        per_speaker = {}
        for s in speakers:
            m = np.array([k[0] == s for k in keys])
            per_speaker[s] = {"wrong": int(e[m].sum()), "words": int(n[m].sum()), "rate": float(e[m].sum() / n[m].sum())}
        entry["per_speaker"] = per_speaker
        if arm != control:
            diff = boot - (ctrl_e[idx].sum(axis=1) / ctrl_n[idx].sum(axis=1))
            entry["minus_control"] = {"points": 100 * (pooled - float(ctrl_e.sum() / ctrl_n.sum())),
                                      "ci95_points": [100 * float(np.percentile(diff, 2.5)), 100 * float(np.percentile(diff, 97.5))],
                                      "speakers_better": sum(per_speaker[s]["rate"] < table[control]["per_speaker"][s]["rate"]
                                                             for s in speakers) if control in table else None}
        table[arm] = entry
    for arm, entry in table.items():
        if arm != control:
            entry["minus_control"]["speakers_better"] = sum(entry["per_speaker"][s]["rate"] < table[control]["per_speaker"][s]["rate"] for s in speakers)
    best = min(table, key=lambda a: (table[a]["rate"], a != control))
    chosen, reason = control, "no recipe beat the control by more than the noise"
    if best != control:
        m = table[best]["minus_control"]
        if m["points"] <= -min_gain_points and m["ci95_points"][1] < 0 and m["speakers_better"] >= max(2, (len(speakers) + 1) // 2 + (len(speakers) % 2 == 0)):
            chosen, reason = best, (f"{best} has {-m['points']:.1f} points fewer held-out errors than the control, the paired 95% interval "
                                    f"({m['ci95_points'][0]:.1f} to {m['ci95_points'][1]:.1f}) stays below zero, and it is better for "
                                    f"{m['speakers_better']} of {len(speakers)} speakers")
        else:
            reason = (f"best was {best} ({m['points']:+.1f} points, interval {m['ci95_points'][0]:.1f} to {m['ci95_points'][1]:.1f}, "
                      f"better for {m['speakers_better']} of {len(speakers)} speakers) but that is not clearly better than the control")
    return {"format": "willspeak-cv-selection-v1", "control": control, "selected": chosen, "reason": reason, "best_by_rate": best,
            "min_gain_points": min_gain_points, "groups": len(keys), "draws": draws, "arms": table,
            "note": "Held-out prompts of the train and dev clips only; the test clips were not read."}


def parse_arm(text: str) -> tuple[str, str]:
    name, sep, pattern = text.partition("=")
    if not sep or not name or not pattern:
        raise argparse.ArgumentTypeError(f"--arm wants NAME=PATTERN, got {text!r}")
    return name, pattern


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--export", required=True)
    p.add_argument("--speaker", required=True)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--root", default=str(ROOT))
    p = sub.add_parser("check")
    p.add_argument("--speaker", required=True)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--root", default=str(ROOT))
    p = sub.add_parser("merge")
    p.add_argument("--speaker", required=True)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--arm", type=parse_arm, required=True, help="NAME=PATTERN with {fold}")
    p.add_argument("--out", required=True, help="folder; writes <arm>-<speaker>.jsonl")
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("select")
    p.add_argument("--speaker", nargs="+", required=True)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--control", required=True)
    p.add_argument("--arm", type=parse_arm, action="append", required=True, help="NAME=PATTERN with {fold} and {speaker}")
    p.add_argument("--min-gain-points", type=float, default=1.0)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--out", required=True)
    p.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "make":
        make(root / a.export if not Path(a.export).is_absolute() else Path(a.export), a.speaker, a.folds, a.seed, root=root)
    elif a.cmd == "check":
        check(a.speaker, a.folds, root)
        print(f"{a.speaker}: {a.folds} folds are consistent")
    elif a.cmd == "merge":
        name, pattern = a.arm
        out = merge(a.speaker, a.folds, pattern, Path(a.out) / f"{name}-{a.speaker}.jsonl", root, force=a.force)
        print(f"wrote {out}")
    else:
        arms: dict[str, dict[str, dict[str, str]]] = {}
        labelled = {s: lab.read_manifest(root / EXPORT / f"cv-{s}-all" / "test.jsonl") for s in a.speaker}
        for name, pattern in a.arm:
            arms[name] = {}
            for s in a.speaker:
                ids = fold_ids(s, a.folds, root)
                merged: dict[str, str] = {}
                for k in range(a.folds):
                    path = Path(pattern.format(fold=k, speaker=s))
                    path = path if path.is_absolute() else root / path
                    hyps = lab.read_hyps(path)
                    for i in ids[k]:
                        if i not in hyps:
                            raise ValueError(f"{path} has no answer for held-out clip {i}")
                        merged[i] = hyps[i]
                arms[name][s] = merged
        result = select(arms, labelled, a.control, a.min_gain_points, a.draws)
        out = Path(a.out)
        if out.exists():
            raise FileExistsError(f"{out} exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"{'recipe':<22}{'wrong':>7}{'of':>6}{'rate':>8}   95% interval      vs control (points, interval)")
        for arm, e in result["arms"].items():
            m = e.get("minus_control")
            extra = f"{m['points']:+6.1f} ({m['ci95_points'][0]:+.1f} to {m['ci95_points'][1]:+.1f}), better for {m['speakers_better']} speakers" if m else "control"
            print(f"{arm:<22}{e['wrong']:>7}{e['words']:>6}{100 * e['rate']:>7.1f}%   {100 * e['ci95'][0]:.1f}-{100 * e['ci95'][1]:.1f}%   {extra}")
        print(f"selected: {result['selected']}. {result['reason']}")
    return 0


if __name__ == "__main__":
    try:
        import _cli
    except ImportError:
        _cli = None
    sys.exit(_cli.run(main, "cv_folds") if _cli else main())
