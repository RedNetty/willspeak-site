#!/usr/bin/env python3
"""Build a personal recognition model for someone outside TORGO, with no scoring of their speech.

This is the device side of the project, not a research run. It reuses the best recorded recipe
(stage P: a base trained on severe dysarthric speech, then a block-8 adapter at 6e-4 for 15 epochs,
the last epoch kept) and puts it on the Pi. Nothing here decodes, scores or reports anything about
the person's recordings:

  * every accepted clip goes to training; there is no test split and none of their clips is in dev;
  * dev holds a few public TORGO clips only because train.py needs one. With --select last it
    chooses nothing, and the dev WER it prints is about those TORGO clips, not the person;
  * logs and the training export stay in the person's private folder (data/people/<name>/), and
    experiment tracking is off;
  * the models go to models/personal/, which the lab bundler skips.

Steps (from app/, on the training machine, in the training environment):

  1. base     train the shared base on every severe TORGO speaker (once; reused for anyone)
  2. export   turn the person's accepted recordings into a training folder
  3. train    train their adapter on top of the base
  4. deploy   export base + adapter to int8 ONNX for sherpa-onnx (models/personal/<name>)

`all` runs 2 to 4, and 1 first if the base is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _cli

TORGO_EXPORT = Path("data/public/torgo/export")
BASE_EXPORT = TORGO_EXPORT / "severe-base"
PERSONAL_MODELS = Path("models/personal")
BASE_DIR = PERSONAL_MODELS / "severe-base"
NAME = re.compile(r"^[a-z][a-z0-9-]{0,31}$")

# train.py flags shared with lab.py's Lab.train (the recorded runs used these).
COMMON = ["--batch", "2", "--grad-accum", "8", "--freeze-batchnorm", "--normalize-targets", "--dev-scoring", "evaluate"]
# Phase 1 of stage P: F-1's recipe (every encoder part train.py trains by default, stock kept if nothing beats it).
BASE_RECIPE = ["--keep-stock"]
# Stage P's adapter: lab.py's "ad8" recipe at stage P's learning rate.
ADAPTER_RECIPE = ["--adapter-after-blocks", "8", "--lr", "6e-4", "--epochs", "15", "--select", "last",
                  "--no-keep-stock", "--warmup-steps", "10"]


class Paths:
    def __init__(self, name: str, data: Path | None = None) -> None:
        if not NAME.match(name):
            raise ValueError("--name must be lowercase letters, digits or dashes, starting with a letter "
                             "(it becomes a folder name)")
        if name == "severe-base":
            raise ValueError("--name severe-base is the shared base's folder; choose another name")
        self.name = name
        self.data = data or Path("data/people") / name
        self.export = self.data / "model-export"
        self.logs = self.data / "logs"
        self.adapter_dir = PERSONAL_MODELS / f"{name}-adapter"
        self.model = PERSONAL_MODELS / name

    def full(self, p: Path) -> Path:
        return p if p.is_absolute() else ROOT / p


def base_checkpoint() -> Path:
    return ROOT / BASE_DIR / "parakeet-ft.nemo"


def run(cmd: list[str], env: dict | None = None, dry: bool = False) -> bool:
    print("  $ " + shlex.join(cmd), flush=True)
    if dry:
        return True
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode == 0


def private_env(p: Paths) -> dict:
    """Environment for anything that touches the person's audio: logs land in their own folder."""
    env = dict(os.environ)
    logs = p.full(p.logs)
    logs.mkdir(parents=True, exist_ok=True)
    env["WILLSPEAK_LOG_DIR"] = str(logs)
    env["WANDB_MODE"] = "disabled"
    return env


def set_aside(folder: Path) -> None:
    """Keep exactly one earlier version (<folder>.previous) so a bad retrain can be undone."""
    if not folder.exists():
        return
    aside = folder.with_name(folder.name + ".previous")
    if aside.exists():
        shutil.rmtree(aside)
    folder.rename(aside)
    print(f"moved the earlier {folder.name} to {aside}")


# ---------------------------------------------------------------------------- base

def cmd_base(a) -> int:
    if base_checkpoint().is_file():
        print(f"The base is already trained: {BASE_DIR}/parakeet-ft.nemo")
        return 0
    export = ROOT / BASE_EXPORT
    if not (export / "train.jsonl").is_file():
        print(f"No {BASE_EXPORT}/. Build it first (about 1 GB of TORGO audio must already be fetched):\n"
              f"  ./willspeak.sh torgo manifest --export --severe-base --only severe-base")
        return 1
    if (ROOT / BASE_DIR).exists() and not a.dry_run:
        print(f"{BASE_DIR}/ holds no parakeet-ft.nemo (an unfinished run); keeping it as {BASE_DIR}.previous")
        set_aside(ROOT / BASE_DIR)
    print("Training the shared base: every severe TORGO speaker (F01, M01, M02, M04), F-1's recipe.")
    print("This is TORGO only; nobody else's recordings are involved. On the 4060 expect a few hours.")
    cmd = [sys.executable, "scripts/asr/train.py", "--export", str(BASE_EXPORT), "--out", str(BASE_DIR),
           *COMMON, "--seed", str(a.seed), *BASE_RECIPE, *low_memory(a)]
    return 0 if run(cmd, dry=a.dry_run) else 1


def low_memory(a) -> list[str]:
    return ["--batch", "1", "--grad-accum", "16"] if getattr(a, "low_memory", False) else []


# ---------------------------------------------------------------------------- export

def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build_personal_export(data_dir: Path, out: Path, torgo_dev: Path, dev_clips: int = 24,
                          trim: bool = True, min_duration: float = 0.3) -> dict:
    """Write out/{train,dev}.jsonl: every accepted clip in train, a few public TORGO clips as dev.

    The person's sticky split file (split_assignments.json) is read but never written, so a later
    approved study can still split their corpus by prompt.
    """
    from willspeak.db import Store
    from willspeak.export import _build_export
    from willspeak.normalize import normalize

    data_dir, out = data_dir.resolve(), out.resolve()
    if not (data_dir / "corpus.db").is_file():
        raise FileNotFoundError(f"No recorder corpus in {data_dir}. Record with WILLSPEAK_DATA={data_dir} first.")
    dev_source = read_rows(torgo_dev / "dev.jsonl")
    if not dev_source:
        raise FileNotFoundError(f"No TORGO dev clips in {torgo_dev}/dev.jsonl (build export/severe-base first)")
    store = Store(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with store._lock, tempfile.TemporaryDirectory(prefix=".personal-export-", dir=out.parent) as temporary:
        stage = Path(temporary) / "export"
        summary, _ = _build_export(store, data_dir, stage, 0.0, 0.0, trim, min_duration, 0, 0)
        train: list[dict] = []
        for split in ("train", "dev", "test"):
            for row in read_rows(stage / f"{split}.jsonl"):
                if split != "train":
                    name = Path(row["audio"]).name
                    (stage / split / name).rename(stage / "train" / name)
                    row["audio"] = f"train/{name}"
                train.append(row)
            if split != "train":
                (stage / f"{split}.jsonl").unlink()
                shutil.rmtree(stage / split)
        if not train:
            raise ValueError("No accepted clips with a transcript yet; review recordings on the Review page first")
        write_rows(stage / "train.jsonl", train)

        theirs = {normalize(t, us_spelling=False) for r in train for t in (r["text"], r.get("prompt_text") or r["text"])}
        dev: list[dict] = []
        (stage / "dev").mkdir()
        for row in sorted(dev_source, key=lambda r: str(r["id"])):
            if len(dev) >= dev_clips:
                break
            if normalize(row["text"], us_spelling=False) in theirs:
                continue  # a prompt in both splits would be refused by train.py
            name = f"torgo-{Path(row['audio']).name}"
            shutil.copy2(torgo_dev / row["audio"], stage / "dev" / name)
            dev.append({**row, "id": f"torgo-{row['id']}", "audio": f"dev/{name}"})
        if not dev:
            raise ValueError("Every TORGO dev prompt also appears in the person's recordings; none left for dev")
        write_rows(stage / "dev.jsonl", dev)
        write_rows(stage / "test.jsonl", [])
        record = {
            "purpose": "personal model for the device; no test split, nothing about this person is scored",
            "train_clips": len(train),
            "train_minutes": round(sum(float(r["duration_s"]) for r in train) / 60, 1),
            "dev": f"{len(dev)} public TORGO clips from {torgo_dev.name} (required by train.py; chooses nothing "
                   "with --select last)",
            "skipped": summary["skipped"],
        }
        (stage / "summary.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        stage.rename(out)
    return record


def cmd_export(a) -> int:
    p = Paths(a.name, a.data)
    record = build_personal_export(p.full(p.data), p.full(p.export), ROOT / BASE_EXPORT, a.dev_clips)
    print(f"Training folder: {p.export}/  ({record['train_clips']} clips, {record['train_minutes']} min)")
    if record["skipped"]:
        print(f"  {len(record['skipped'])} clips skipped (no transcript, too short or missing audio); "
              f"see {p.export}/summary.json")
    return 0


# ---------------------------------------------------------------------------- train

def cmd_train(a) -> int:
    p = Paths(a.name, a.data)
    if not base_checkpoint().is_file():
        print(f"No base at {BASE_DIR}/parakeet-ft.nemo. Run: ./willspeak.sh personal base")
        return 1
    if not (p.full(p.export) / "train.jsonl").is_file():
        print(f"No training folder at {p.export}/. Run: ./willspeak.sh personal export --name {p.name}")
        return 1
    out = p.full(p.adapter_dir)
    if (out / "adapter.pt").is_file() and not a.retrain:
        print(f"{p.adapter_dir}/adapter.pt already exists; add --retrain to replace it")
        return 0
    if not a.dry_run:
        set_aside(out)
    print("Training the personal adapter: stage P's recipe (block 8, 6e-4, 15 epochs, last epoch kept).")
    print("The dev WER lines below are TORGO clips, kept only because train.py needs a dev set.")
    cmd = [sys.executable, "scripts/asr/train.py", "--export", str(p.export), "--out", str(p.adapter_dir),
           "--base", str(BASE_DIR / "parakeet-ft.nemo"), *COMMON, "--seed", str(a.seed), *ADAPTER_RECIPE,
           "--wandb", "off", *low_memory(a)]
    return 0 if run(cmd, env=private_env(p), dry=a.dry_run) else 1


# ---------------------------------------------------------------------------- deploy

def cmd_deploy(a) -> int:
    p = Paths(a.name, a.data)
    adapter = p.adapter_dir / "adapter.pt"
    if not (ROOT / adapter).is_file():
        print(f"No adapter at {adapter}. Run: ./willspeak.sh personal train --name {p.name}")
        return 1
    if not a.dry_run:
        set_aside(p.full(p.model))
    # matmul-only int8: lab test 19 and 28 found it costs nothing measurable; the export's own check decodes
    # the stock sample recording, not the person.
    cmd = [sys.executable, "scripts/asr/export.py", "--nemo", str(BASE_DIR / "parakeet-ft.nemo"),
           "--adapter", str(adapter), "--int8-ops", "matmul", "--out", str(p.model)]
    if not run(cmd, env=private_env(p), dry=a.dry_run):
        return 1
    print(f"\nModel ready: {p.model}/\n"
          f"Copy it to the Pi (from app/ on this machine):\n"
          f"  scp -r {p.model} jdev@<pi-address>:~/Desktop/willspeak/app/{PERSONAL_MODELS}/\n"
          f"On the Pi, pick it in the Compare page's Model list, or make it the Speak page's model:\n"
          f"  deploy/pi5/setup.sh --service --skip-apt --skip-tests --model-dir {p.model}")
    return 0


def cmd_all(a) -> int:
    if not base_checkpoint().is_file() and cmd_base(a) != 0:
        return 1
    for step in (cmd_export, cmd_train, cmd_deploy):
        if step(a) != 0:
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def person(sp, train: bool = False):
        sp.add_argument("--name", required=True, help="short folder name for the person, e.g. alex")
        sp.add_argument("--data", type=Path, default=None,
                        help="their recorder corpus (default data/people/<name>; record with WILLSPEAK_DATA set to it)")
        sp.add_argument("--dry-run", action="store_true", help="print the commands without running them")
        if train:
            sp.add_argument("--seed", type=int, default=17)
            sp.add_argument("--low-memory", action="store_true", help="batch 1 x 16 accumulation if batch 2 runs out of GPU memory")
            sp.add_argument("--retrain", action="store_true", help="replace an existing adapter")
            sp.add_argument("--dev-clips", type=int, default=24, help="public TORGO clips used as the required dev set")

    base = sub.add_parser("base", help="train the shared base on every severe TORGO speaker (once)")
    base.add_argument("--seed", type=int, default=17)
    base.add_argument("--low-memory", action="store_true")
    base.add_argument("--dry-run", action="store_true")
    base.set_defaults(func=cmd_base)
    exp = sub.add_parser("export", help="turn the person's accepted recordings into a training folder")
    person(exp)
    exp.add_argument("--dev-clips", type=int, default=24)
    exp.set_defaults(func=cmd_export)
    tr = sub.add_parser("train", help="train the person's adapter on the base")
    person(tr, train=True)
    tr.set_defaults(func=cmd_train)
    dep = sub.add_parser("deploy", help="export base + adapter to int8 ONNX in models/personal/<name>")
    person(dep)
    dep.set_defaults(func=cmd_deploy)
    every = sub.add_parser("all", help="export, train and deploy (and the base first if it is missing)")
    person(every, train=True)
    every.set_defaults(func=cmd_all)
    a = ap.parse_args()
    return a.func(a)


def private_log_dir(argv: list[str]) -> Path:
    """This script's own log names the person and counts their clips, so it stays in their folder."""
    name = data = None
    for i, arg in enumerate(argv):
        if arg in ("--name", "--data") and i + 1 < len(argv):
            name, data = (argv[i + 1], data) if arg == "--name" else (name, argv[i + 1])
        elif arg.startswith("--name="):
            name = arg.split("=", 1)[1]
        elif arg.startswith("--data="):
            data = arg.split("=", 1)[1]
    if data:
        folder = Path(data)
    elif name and NAME.match(name):
        folder = Path("data/people") / name
    else:
        folder = Path("data/people")
    return (folder if folder.is_absolute() else ROOT / folder) / "logs"


if __name__ == "__main__":
    os.environ.setdefault("WILLSPEAK_LOG_DIR", str(private_log_dir(sys.argv[1:])))
    sys.exit(_cli.run(main, "personal"))
