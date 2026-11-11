#!/usr/bin/env python3
"""Choose the fixed clip list for the expo comparison page and copy the clips."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak.normalize import normalize  # noqa: E402
from willspeak.torgo import CITATION, kind_of  # noqa: E402

SPEAKERS = ("M01", "M02", "M04")


def project_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose(rows: list[dict], n_sentences: int, n_words: int, rng: random.Random) -> list[dict]:
    by_text: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: str(r["id"])):
        by_text.setdefault(normalize(r["text"]), []).append(r)
    picked = []
    for kind, n in (("sentence", n_sentences), ("word", n_words)):
        texts = sorted(t for t, rs in by_text.items() if kind_of(rs[0]["text"]) == kind)
        if len(texts) < n:
            raise SystemExit(f"only {len(texts)} distinct {kind} prompts, asked for {n}")
        for t in rng.sample(texts, n):
            picked.append(rng.choice(by_text[t]) | {"kind": kind})
    return picked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-root", default="data/public/torgo/export", help="folder holding <SPEAKER>/test.jsonl")
    ap.add_argument("--speakers", nargs="+", default=list(SPEAKERS))
    ap.add_argument("--sentences", type=int, default=8, help="sentence clips per speaker")
    ap.add_argument("--words", type=int, default=2, help="single-word clips per speaker")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--force", action="store_true", help="replace an existing list")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.sentences < 0 or a.words < 0 or a.sentences + a.words == 0:
        ap.error("ask for at least one clip")

    export, out = project_path(a.export_root), project_path(a.out)
    if (out / "clips.json").exists() and not (a.force or a.dry_run):
        raise SystemExit(f"{out / 'clips.json'} exists; the list is fixed. Use --force to replace it.")

    clips = []
    for spk in a.speakers:
        manifest = export / spk / "test.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"missing {manifest}: run scripts/corpus/torgo_manifest.py --export first")
        rng = random.Random(f"{a.seed}-{spk}")
        for r in choose(read_rows(manifest), a.sentences, a.words, rng):
            src = (export / spk / r["audio"]).resolve()
            if not src.is_file():
                raise SystemExit(f"missing audio {src}")
            clips.append({"id": str(r["id"]), "speaker": spk, "kind": r["kind"], "text": r["text"],
                          "duration_s": r.get("duration_s"), "audio": f"{spk}/{r['id']}.wav", "_src": src})

    for c in clips:
        print(f"{c['speaker']}  {c['kind']:<8}  {c['id']:<24}  {c['text']}")
    if a.dry_run:
        print(f"{len(clips)} clips (dry run, nothing written)")
        return 0

    for spk in a.speakers:
        (out / spk).mkdir(parents=True, exist_ok=True)
    for c in clips:
        shutil.copyfile(c.pop("_src"), out / c["audio"])
    meta = {
        "made": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": (f"{a.sentences} sentences and {a.words} single words per speaker, drawn at random from the "
                 f"held-out test set (one recording per distinct prompt), seed {a.seed}; no model answers were read"),
        "seed": a.seed, "sentences": a.sentences, "words": a.words,
        "source": str(export.relative_to(ROOT)) if export.is_relative_to(ROOT) else str(export),
        "citation": CITATION,
        "clips": clips,
    }
    (out / "clips.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(clips)} clips to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
