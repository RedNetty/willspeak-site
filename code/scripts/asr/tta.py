#!/usr/bin/env python3
"""Test-time adaptation on his own recordings, with no transcripts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import hyp_selector as hs  # noqa: E402
import labkit as lab  # noqa: E402
import pseudo_label  # noqa: E402
from willspeak.normalize import normalize  # noqa: E402

VERSION = 1


def stable(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def natural(text: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def chunk_ids(ids: list[str], n: int, order: str = "hash", seed: int = 17) -> list[list[str]]:
    if n < 2:
        raise ValueError("adaptation needs at least 2 chunks: one to learn from and one to be decoded after learning")
    if len(set(ids)) != len(ids):
        raise ValueError("a clip id appears twice")
    if len(ids) < 2 * n:
        raise ValueError(f"{len(ids)} clips cannot fill {n} chunks")
    if order == "hash":
        ranked = sorted(ids, key=lambda i: stable(seed, i))
    elif order == "id":
        ranked = sorted(ids, key=natural)
    else:
        raise ValueError("--order is hash or id")
    size, extra = divmod(len(ranked), n)
    out, at = [], 0
    for k in range(n):
        take = size + (1 if k < extra else 0)
        out.append(ranked[at: at + take])
        at += take
    return out


def read_rows(path: str | Path) -> list[dict]:
    rows = []
    with lab.here(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def parse_chunk_hyps(items: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for item in items:
        head, _, path = item.partition("=")
        if not head.isdigit() or not path:
            raise ValueError(f"--chunk-hyps wants INDEX=PATH, got {item!r}")
        if int(head) in out:
            raise ValueError(f"--chunk-hyps {head} is given twice")
        out[int(head)] = path
    return out


def answers_for(ids: list[str], path: str, what: str) -> dict[str, str]:
    hyps = lab.read_hyps(path)
    missing = [i for i in ids if i not in hyps]
    if missing:
        raise ValueError(f"{what}: {path} has no answer for {len(missing)} clips, first {missing[:3]}")
    return {i: hyps[i] for i in ids}


def gate(hyp: str, others: list[str], minimum: float) -> tuple[bool, str, float | None]:
    if not normalize(hyp).split():
        return False, "empty answer", None
    if not others:
        return True, "ok", None
    mean = sum(hs.similarity(hs.words(hyp), hs.words(o)) for o in others) / len(others)
    return (mean >= minimum), ("ok" if mean >= minimum else "the other systems disagree"), mean


def build(train_export: Path, test_manifest: Path, n: int, round_: int, chunk_hyps: dict[int, str], agree: list[str], out: Path,
          agree_min: float = 0.5, max_pseudo_share: float = 1.0, order: str = "hash", seed: int = 17,
          test_audio_root: Path | None = None) -> dict:
    train_export, out = Path(train_export), Path(out)
    if not 1 <= round_ < n:
        raise ValueError(f"--round is 1 to {n - 1}: round 0 is the base system and trains nothing")
    if max_pseudo_share <= 0:
        raise ValueError("--max-pseudo-share must be above 0")
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new name")
    for need in range(round_):
        if need not in chunk_hyps:
            raise ValueError(f"round {round_} needs --chunk-hyps for chunks 0 to {round_ - 1}; chunk {need} is missing")
    audio_rows = pseudo_label.read_audio_rows(lab.here(test_manifest))
    chunks = chunk_ids([r["id"] for r in audio_rows], n, order, seed)
    by_id = {r["id"]: r for r in audio_rows}
    test_root = Path(test_audio_root) if test_audio_root else lab.here(test_manifest).parent
    train_rows = read_rows(train_export / "train.jsonl")
    dev_rows = read_rows(train_export / "dev.jsonl")
    if not train_rows or not dev_rows:
        raise ValueError("his train and dev splits must both hold clips")
    dev_prompts = {normalize(str(r.get("prompt_text") or r["text"]), us_spelling=False) for r in dev_rows}
    agree_hyps = [lab.read_hyps(a) for a in agree]

    candidates, reasons = [], {}
    for k in range(round_):
        ids = chunks[k]
        own = answers_for(ids, chunk_hyps[k], f"chunk {k}")
        for cid in ids:
            others = [h[cid] for h in agree_hyps if cid in h]
            if normalize(own[cid], us_spelling=False) in dev_prompts:
                ok, why, score = False, "equals a dev prompt", None
            else:
                ok, why, score = gate(own[cid], others, agree_min)
            reasons[why] = reasons.get(why, 0) + 1
            if ok:
                candidates.append({"id": cid, "chunk": k, "text": own[cid], "agreement": score})
    cap = max(1, int(max_pseudo_share * len(train_rows)))
    capped = sorted(candidates, key=lambda c: stable(seed, "cap", round_, c["id"]))[:cap]
    kept_ids = {c["id"] for c in capped}
    kept = [c for c in candidates if c["id"] in kept_ids]

    scratch = out.with_name(out.name + ".partial")
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    try:
        for r in train_rows + dev_rows:
            src = train_export / r["audio"]
            if not src.is_file():
                raise FileNotFoundError(f"{src} is missing (python lab.py doctor --fix rebuilds exports)")
            link_or_copy(src, scratch / r["audio"])
        pseudo_rows = []
        for c in kept:
            src_row = by_id[c["id"]]
            src = test_root / src_row["audio"]
            if not src.is_file():
                raise FileNotFoundError(f"{src} is missing")
            name = f"train/pseudo-{c['id']}{Path(src_row['audio']).suffix}"
            link_or_copy(src, scratch / name)
            pseudo_rows.append({"id": f"tta-{c['id']}", "audio": name, "text": c["text"], "duration_s": src_row["duration_s"],
                                "speaker": src_row["speaker"], "kind": "sentence" if len(c["text"].split()) >= 3 else "word",
                                "condition": "unknown", "prompt_set": "test-time"})
        splits = {"train": train_rows + pseudo_rows, "dev": dev_rows, "test": dev_rows}
        for name, rows in splits.items():
            (scratch / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        info = {"format": "willspeak-tta-v1", "version": VERSION, "round": round_, "chunks": n, "order": order, "seed": seed,
                "labelled_train_clips": len(train_rows), "pseudo_clips": len(pseudo_rows), "pseudo_candidates": len(candidates),
                "pseudo_cap": cap, "agree_min": agree_min, "agree_systems": len(agree), "gate": reasons,
                "from_chunks": list(range(round_)), "test_text_read": False,
                "note": "pseudo labels are the system's own answers; the test manifest's text column was never read"}
        (scratch / "tta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        scratch.rename(out)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return info


def write_manifest(test_manifest: Path, n: int, index: int, out: Path, order: str = "hash", seed: int = 17) -> int:
    out = lab.here(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new name")
    rows = read_rows(test_manifest)
    chunks = chunk_ids([str(r["id"]) for r in rows], n, order, seed)
    if not 0 <= index < n:
        raise ValueError(f"--index is 0 to {n - 1}")
    wanted = set(chunks[index])
    out.parent.mkdir(parents=True, exist_ok=True)
    picked = [r for r in rows if str(r["id"]) in wanted]
    out.write_text("".join(json.dumps(r) + "\n" for r in picked), encoding="utf-8")
    return len(picked)


def merge(test_manifest: Path, n: int, chunk_hyps: dict[int, str], out: Path, order: str = "hash", seed: int = 17) -> Path:
    out = lab.here(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new name")
    ids = [str(r["id"]) for r in read_rows(test_manifest)]
    chunks = chunk_ids(ids, n, order, seed)
    lines = []
    for k in range(n):
        if k not in chunk_hyps:
            raise ValueError(f"--chunk-hyps for chunk {k} is missing")
        own = answers_for(chunks[k], chunk_hyps[k], f"chunk {k}")
        lines += [json.dumps({"id": i, "hyp": own[i]}) for i in chunks[k]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def score(test_manifest: Path, n: int, systems: dict[str, str], order: str = "hash", seed: int = 17) -> dict:
    rows = {str(r["id"]): r for r in read_rows(test_manifest)}
    chunks = chunk_ids(list(rows), n, order, seed)
    out: dict = {"chunks": [len(c) for c in chunks], "systems": {}}
    for name, path in systems.items():
        hyps = answers_for(list(rows), path, name)
        per = []
        for ids in chunks:
            ref_words = [hs.words(str(rows[i]["text"])) for i in ids]
            wrong = sum(hs.distance(r, hs.words(hyps[i])) for r, i in zip(ref_words, ids))
            per.append({"wrong": wrong, "words": sum(len(r) for r in ref_words)})
        out["systems"][name] = {"by_chunk": per, "wrong": sum(c["wrong"] for c in per), "words": sum(c["words"] for c in per)}
    return out


def show_score(result: dict) -> None:
    names = list(result["systems"])
    print(f"{'wrong words / words':<22}" + "".join(f"chunk {k:<8}" for k in range(len(result['chunks']))) + "all")
    for name in names:
        e = result["systems"][name]
        cells = "".join(f"{c['wrong']:>3}/{c['words']:<9}" for c in e["by_chunk"])
        print(f"{name:<22}{cells}{e['wrong']}/{e['words']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, test=True):
        if test:
            p.add_argument("--test-manifest", required=True)
        p.add_argument("--n", type=int, default=4, help="number of chunks")
        p.add_argument("--order", default="hash", choices=["hash", "id"])
        p.add_argument("--seed", type=int, default=17)

    p = sub.add_parser("chunks", help="print the chunk sizes")
    p.add_argument("--manifest", required=True)
    common(p, test=False)
    p = sub.add_parser("manifest", help="write one chunk's test rows for a decode")
    p.add_argument("--manifest", required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--out", required=True)
    common(p, test=False)
    p = sub.add_parser("build", help="write the training export for one round")
    p.add_argument("--train-export", required=True)
    p.add_argument("--round", type=int, required=True, dest="round_")
    p.add_argument("--chunk-hyps", action="append", default=[], metavar="INDEX=PATH")
    p.add_argument("--agree", nargs="*", default=[], help="other systems' answers on the test clips, for the agreement gate")
    p.add_argument("--agree-min", type=float, default=0.5)
    p.add_argument("--max-pseudo-share", type=float, default=1.0)
    p.add_argument("--test-audio-root")
    p.add_argument("--out", required=True)
    common(p)
    p = sub.add_parser("score", help="wrong words per chunk for several systems, to see whether later chunks improve")
    p.add_argument("--system", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--out")
    common(p)
    p = sub.add_parser("merge", help="join the chunks' answers into one hypotheses file")
    p.add_argument("--chunk-hyps", action="append", default=[], metavar="INDEX=PATH")
    p.add_argument("--out", required=True)
    common(p)
    a = ap.parse_args(argv)

    if a.cmd == "chunks":
        ids = [str(r["id"]) for r in read_rows(a.manifest)]
        sizes = [len(c) for c in chunk_ids(ids, a.n, a.order, a.seed)]
        print(f"{len(ids)} clips in {a.n} chunks ({a.order} order): {sizes}")
    elif a.cmd == "manifest":
        count = write_manifest(Path(a.manifest), a.n, a.index, Path(a.out), a.order, a.seed)
        print(f"chunk {a.index}: {count} clips -> {a.out}")
    elif a.cmd == "build":
        info = build(lab.here(a.train_export), Path(a.test_manifest), a.n, a.round_, parse_chunk_hyps(a.chunk_hyps), a.agree,
                     lab.here(a.out), a.agree_min, a.max_pseudo_share, a.order, a.seed,
                     lab.here(a.test_audio_root) if a.test_audio_root else None)
        print(f"round {a.round_}: {info['labelled_train_clips']} labelled clips + {info['pseudo_clips']} pseudo-labelled "
              f"(of {info['pseudo_candidates']} that passed the gate; cap {info['pseudo_cap']}); gate: {info['gate']}")
    elif a.cmd == "score":
        systems = {}
        for item in a.system:
            name, _, path = item.partition("=")
            if not name or not path:
                raise ValueError(f"--system wants NAME=PATH, got {item!r}")
            systems[name] = path
        if not systems:
            raise ValueError("give at least one --system NAME=PATH")
        result = score(Path(a.test_manifest), a.n, systems, a.order, a.seed)
        show_score(result)
        if a.out:
            target = lab.here(a.out)
            if target.exists():
                raise FileExistsError(f"{target} exists; choose a new name")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=1), encoding="utf-8")
    else:
        out = merge(Path(a.test_manifest), a.n, parse_chunk_hyps(a.chunk_hyps), Path(a.out), a.order, a.seed)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
