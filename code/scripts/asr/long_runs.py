#!/usr/bin/env python3
"""Long lab experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP / "src"))

from willspeak.artifacts import contained_file, file_record, new_directory, validate_export, write_json
from willspeak.normalize import normalize

SPEAKERS = ("M01", "M02", "M04")
BASE = "nvidia/parakeet-tdt-0.6b-v2"
REPORTS = {"adapter": "f2-long-adapter", "acoustic": "f2-acoustic-only", "replay": "f2-speaker-replay"}


def load_manifest(path: Path) -> dict:
    from willspeak.evaluate import load_manifest as read
    return read(path)


@dataclass(frozen=True)
class Recipe:
    name: str
    epochs: int
    lr: str
    form: str = "stock"
    adapter: bool = False
    blocks: int = 8
    frozen_head: bool = False
    replay: str = "personal"
    repeat: int = 1

    def flags(self, speaker: str) -> list[str]:
        args = ["--lr", self.lr, "--target-form", self.form]
        args += (["--adapter-after-blocks", "8"] if self.adapter else
                 ["--train-parts", "all", "--train-blocks", str(self.blocks)])
        if self.frozen_head:
            args.append("--freeze-transducer-head")
        if self.repeat > 1:
            args += ["--repeat-speakers", speaker, "--repeat", str(self.repeat)]
        return args


def recipes(study: str) -> list[Recipe]:
    control = Recipe("control", 15, "3e-4", form="scoring", adapter=True)
    if study == "adapter":
        return [control,
                Recipe("scoring-75", 75, "3e-4", form="scoring", adapter=True),
                Recipe("scoring-200", 200, "3e-4", form="scoring", adapter=True),
                Recipe("stock-75", 75, "3e-4", adapter=True),
                Recipe("stock-200", 200, "3e-4", adapter=True),
                Recipe("stock-200-low-lr", 200, "1e-4", adapter=True),
                Recipe("stock-200-high-lr", 200, "1e-3", adapter=True)]
    if study == "acoustic":
        return [control] + [Recipe(f"b{blocks}-{head}-{tag}", 100, lr,
                                  blocks=blocks, frozen_head=head == "frozen")
                            for blocks in (8, 24) for head in ("open", "frozen")
                            for tag, lr in (("low-lr", "3e-6"), ("high-lr", "1e-5"))]
    common = dict(epochs=100, lr="1e-5", frozen_head=True)
    return [Recipe("control", repeat=4, **common),
            Recipe("personal75", replay="all75", repeat=3, **common),
            Recipe("personal50", replay="all50", repeat=2, **common),
            Recipe("severe50", replay="severe50", repeat=2, **common)]


MIN_CLIP_SECONDS = 0.1
MAX_CLIP_SECONDS = 25


def usable_clip(row: dict) -> bool:
    return MIN_CLIP_SECONDS <= row["duration_s"] <= MAX_CLIP_SECONDS


def replay_budget_dropped(budget: dict, usable: int, batch: int) -> int:
    target = 4 * usable
    slack = max(2, target // 200)
    loaded = budget["clips"]
    if loaded > target or target - loaded > slack:
        raise RuntimeError(f"NeMo loaded {loaded} replay clip uses; the matched budget is {target} (up to {slack} may be dropped). Inspect run.json")
    if not math.ceil((target - slack) / batch) <= budget["batches_per_epoch"] <= math.ceil(target / batch):
        raise RuntimeError(f"NeMo's {budget['batches_per_epoch']} batches per epoch do not match the {math.ceil(target / batch)} expected. Inspect run.json")
    return target - loaded


def balanced_others(rows: list[dict], count: int, seed: int = 17) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["speaker"]].append(row)
    queues = {speaker: deque(sorted(group, key=lambda r: hashlib.sha256(
        f"{seed}|{r['id']}".encode()).hexdigest())) for speaker, group in sorted(groups.items())}
    if len(rows) < count:
        raise ValueError(f"Replay needs {count} distinct other-speaker clips; only {len(rows)} eligible")
    chosen = []
    while len(chosen) < count:
        for queue in queues.values():
            if queue and len(chosen) < count:
                chosen.append(queue.popleft())
    return chosen


def prepare_replay(personal: Path, others: Path, speaker: str, mode: str, out: Path) -> dict:
    source_paths = {f"personal-{split}": personal / f"{split}.jsonl" for split in ("train", "dev", "test")}
    source_paths["others-train"] = others / "train.jsonl"
    sources = {key: file_record(path) for key, path in source_paths.items()}
    if out.exists():
        record = json.loads((out / "replay.json").read_text())
        if record["sources"] != sources or record["mode"] != mode:
            raise ValueError(f"Replay inputs changed: {out}; use a new stamp")
        _, records = validate_export(out, ("train", "dev"))
        if records != record["export_inputs"]:
            raise ValueError(f"Prepared replay audio/manifests changed: {out}; use a new stamp")
        return record
    train = [r for r in load_manifest(personal / "train.jsonl").values() if usable_clip(r)]
    dev = list(load_manifest(personal / "dev.jsonl").values())
    test = list(load_manifest(personal / "test.jsonl").values())
    if not train or any(r.get("speaker") != speaker for r in train):
        raise ValueError(f"Personal export must contain training speech from {speaker}: {personal}")
    blocked = {normalize(r.get(key) or r["text"], us_spelling=False)
               for r in dev + test for key in ("text", "prompt_text")}
    heldout_audio = {file_record(contained_file(personal, r["audio"]))["sha256"] for r in dev + test}
    pool = []
    for row in load_manifest(others / "train.jsonl").values():
        other = row.get("speaker")
        if not other or other == speaker:
            raise ValueError(f"Other-speaker export includes a missing/target speaker: {row['id']}")
        if mode == "severe50" and other not in {"F01", "M01", "M02", "M04"}:
            continue
        if not usable_clip(row) or any(normalize(row.get(key) or row["text"], us_spelling=False) in blocked
                                          for key in ("text", "prompt_text")):
            continue
        if file_record(contained_file(others, row["audio"]))["sha256"] not in heldout_audio:
            pool.append(row)
    count = len(train) * (1 if mode == "all75" else 2)
    chosen = balanced_others(pool, count)
    record = {"mode": mode, "sources": sources, "personal_clips": len(train), "other_clips": len(chosen),
              "eligible_other_clips": len(pool), "other_ids": [r["id"] for r in chosen], "subset_seed": 17,
              "other_speakers": {s: sum(r["speaker"] == s for r in chosen) for s in sorted({r["speaker"] for r in chosen})}}
    with new_directory(out) as stage:
        for split, entries in (("train", [(personal, r) for r in train] + [(others, r) for r in chosen]),
                               ("dev", [(personal, r) for r in dev])):
            (stage / split).mkdir()
            rows = []
            for root, row in entries:
                name = hashlib.sha256(str(row["id"]).encode()).hexdigest() + ".wav"
                dest = f"{split}/{name}"
                shutil.copyfile(contained_file(root, row["audio"]), stage / dest)
                rows.append(dict(row, audio=dest))
            (stage / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        _, record["export_inputs"] = validate_export(stage, ("train", "dev"))
        write_json(stage / "replay.json", record)
        if sources != {key: file_record(path) for key, path in source_paths.items()}:
            raise ValueError("Source manifests changed while building replay")
    return record


def pooled_scores(files: list[Path]) -> dict:
    if len(files) != len(SPEAKERS):
        raise ValueError("Pooled scoring requires every planned speaker")
    per = [json.loads(path.read_text()) for path in files]
    errors = sum(r["word"]["errors"] for r in per)
    words = sum(r["word"]["n_ref"] for r in per)
    if words < 1 or any(not math.isfinite(r["wer"]) for r in per):
        raise ValueError("Cannot select on an empty or non-finite dev score")
    return {"errors": errors, "words": words, "wer": errors / words,
            "speakers": {s: r["wer"] for s, r in zip(SPEAKERS, per)}}


def choose_dev(scores: dict[str, dict]) -> str:
    if not scores or any(not math.isfinite(r["wer"]) for r in scores.values()):
        raise ValueError("All planned recipes need finite pooled dev scores")
    if len({r["words"] for r in scores.values()}) != 1:
        raise ValueError("Recipes were scored on different dev sets")
    return min(scores, key=lambda name: (scores[name]["errors"], name != "control", name))


@contextmanager
def gpu_lock():
    import fcntl
    (APP / "logs").mkdir(exist_ok=True)
    with (APP / "logs/.long-runs.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another long experiment is running in this checkout") from exc
        busy = subprocess.run(["pgrep", "-af", r"scripts/asr/train\.py|lab_runs\.sh|lab_rerun\.sh"],
                              capture_output=True, text=True, check=False)
        if busy.returncode == 0:
            raise RuntimeError("Wait for the current lab run to finish:\n" + busy.stdout)
        yield


class DiskFull(Exception):
    pass


class Experiment:
    def __init__(self, args):
        self.a = args
        self.items = recipes(args.study)
        self.report = APP / "reports" / REPORTS[args.study] / args.stamp
        self.models = APP / "models" / f"parakeet-ft-long-{args.study}-{args.stamp}"
        self.export = APP / "data/public/torgo/export"
        self.log = APP / "logs" / f"lab-long-{args.study}-{args.stamp}.txt"

    def verify_sources(self, recipe: Recipe, speaker: str):
        if not hasattr(self, "configuration"):
            return
        config = self.configuration
        for name, recorded in config["code"].items():
            if file_record(APP / name) != recorded:
                raise ValueError(f"Code changed during the experiment: {name}; use a new stamp")
        for name, recorded in config["inputs"].items():
            if file_record(APP / name) != recorded:
                raise ValueError(f"Manifest changed during the experiment: {name}; use a new stamp")
        prefix = str((self.export / speaker).relative_to(APP)) + "/"
        for name, recorded in config["audio_inputs"].items():
            if name.startswith(prefix) and file_record(APP / name) != recorded:
                raise ValueError(f"Audio changed during the experiment: {name}; use a new stamp")
        if recipe.replay != "personal":
            root = self.training_export(recipe, speaker)
            record = json.loads((root / "replay.json").read_text())
            _, current = validate_export(root, ("train", "dev"))
            if current != record["export_inputs"]:
                raise ValueError(f"Replay inputs changed during the experiment: {root}")

    def run(self, python: str, script: str, *args: str):
        command = [python, f"scripts/asr/{script}.py", *map(str, args)]
        print("+ " + shlex.join(command), flush=True)
        if self.a.dry_run:
            return
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write("\n+ " + shlex.join(command) + "\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=APP, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)

    def model_dir(self, recipe: Recipe, speaker: str, seed: int) -> Path:
        root = self.models / "final" if getattr(self, "final_phase", False) else self.models
        return root / recipe.name / f"{speaker}-seed{seed}"

    def model_file(self, recipe: Recipe, speaker: str, seed: int) -> Path:
        return self.model_dir(recipe, speaker, seed) / ("adapter.pt" if recipe.adapter else "parakeet-ft.nemo")

    def model_flags(self, recipe: Recipe, speaker: str, seed: int) -> list[str]:
        path = str(self.model_file(recipe, speaker, seed))
        return ["--stock", "--adapter", path] if recipe.adapter else ["--nemo", path]

    def training_export(self, recipe: Recipe, speaker: str) -> Path:
        if recipe.replay == "personal":
            return self.export / speaker
        return self.export / f"long-replay-{self.a.stamp}" / speaker / recipe.replay

    def score_file(self, recipe: Recipe, speaker: str, seed: int, split: str) -> Path:
        if split == "dev" and getattr(self, "final_phase", False):
            split = "confirmation-dev"
        return self.report / recipe.name / f"{speaker}-seed{seed}-{split}" / "results.json"

    def train_command(self, recipe: Recipe, speaker: str, seed: int, smoke=False) -> list[str]:
        args = ["--export", str(self.training_export(recipe, speaker)),
                "--out", str(self.model_dir(recipe, speaker, seed)) + ("-smoke" if smoke else ""),
                "--base", BASE, "--batch", str(self.a.batch), "--grad-accum", str(self.a.grad_accum),
                "--epochs", str(1 if smoke else recipe.epochs), "--warmup-steps", "1" if smoke else "10",
                "--seed", str(seed), "--freeze-batchnorm", "--dev-scoring", "evaluate", "--no-keep-stock",
                "--average-top", "1" if smoke else "4", *recipe.flags(speaker)]
        args += ["--max-steps", "2", "--select", "dev"] if smoke else ["--select", "last"]
        if self.a.offload_optimizer and not recipe.adapter:
            args.append("--offload-optimizer")
        return args

    def decode_score(self, flags: list[str], manifest: Path, audio: Path, out: Path):
        if not (out / "results.json").is_file():
            hypotheses = out.with_suffix(".jsonl")
            if not hypotheses.is_file():
                self.run(self.a.train_python, "nemo_hyp", *flags, "--manifest", str(manifest),
                         "--audio-root", str(audio), "--out", str(hypotheses), "--batch", str(self.a.batch))
            self.run(self.a.cpu_python, "evaluate", "--manifest", str(manifest), "--audio-root", str(audio),
                     "--hyp", str(hypotheses), "--group-by", "kind", "--out", str(out))

    def train_score(self, recipe: Recipe, speaker: str, seed: int):
        self.verify_sources(recipe, speaker)
        folder = self.model_dir(recipe, speaker, seed)
        dev = self.score_file(recipe, speaker, seed, "dev")
        if dev.is_file() and (not getattr(self, "final_phase", False) or self.model_file(recipe, speaker, seed).is_file()):
            return
        if folder.exists():
            record_path = folder / "run.json"
            record = json.loads(record_path.read_text()) if record_path.is_file() else {}
            if record.get("status") != "complete" or not self.model_file(recipe, speaker, seed).is_file():
                self.release_weights(recipe, speaker, seed)
                saved = folder.with_name(folder.name + "-failed")
                index = 2
                while saved.exists():
                    saved = folder.with_name(folder.name + f"-failed-{index}")
                    index += 1
                folder.rename(saved)
        if not folder.exists():
            if not self.a.dry_run and shutil.disk_usage(APP).free < self.a.min_free_gb * 2**30:
                self.make_room(folder)
            self.run(self.a.train_python, "train", *self.train_command(recipe, speaker, seed))
        if not self.a.dry_run and self.a.study == "replay":
            progress = json.loads((folder / "run.json").read_text())
            n = sum(usable_clip(r) for r in load_manifest(self.export / speaker / "train.jsonl").values())
            dropped = replay_budget_dropped(progress["phase_budgets"]["train_all"], n, self.a.batch)
            if dropped:
                print(f"[long] {folder.name}: NeMo dropped {dropped} of {4 * n} clip uses per epoch; within the allowed slack", flush=True)
        personal = self.export / speaker
        self.decode_score(self.model_flags(recipe, speaker, seed), personal / "dev.jsonl", personal, dev.parent)
        if not self.a.dry_run:
            archive = dev.parent / "training"
            archive.mkdir(exist_ok=True)
            for name in ("run.json", "curve.csv"):
                shutil.copyfile(folder / name, archive / name)
            if not self.a.keep_all_models:
                shutil.rmtree(folder / "ckpt", ignore_errors=True)

    def make_room(self, folder: Path):
        print(f"Less than {self.a.min_free_gb} GiB free before {folder}; freeing the weights of finished runs", flush=True)
        script = APP / "scripts/asr/lab_space.py"
        if script.is_file():
            subprocess.run([self.a.cpu_python, str(script), "clean", "--apply", "--lab-models", "--long-models"], cwd=APP, check=False)
        free = shutil.disk_usage(APP).free
        if free < self.a.min_free_gb * 2**30:
            raise DiskFull(f"Less than {self.a.min_free_gb} GiB free before {folder} ({free / 2**30:.0f} GiB after cleaning). "
                           "Free space (python lab.py export-models export --apply --free-after-verify), then run the same command again with --resume")

    def release_weights(self, recipe: Recipe, speaker: str, seed: int):
        if self.a.keep_all_models:
            return
        folder = self.model_dir(recipe, speaker, seed)
        shutil.rmtree(folder / "ckpt", ignore_errors=True)
        path = self.model_file(recipe, speaker, seed)
        if path.exists():
            print(f"Release model weights: {path} (scores and training records kept)", flush=True)
            path.unlink()

    def record_failure(self, phase: str, recipe: Recipe, speaker: str, seed: int, error: Exception):
        self.report.mkdir(parents=True, exist_ok=True)
        record = {"phase": phase, "recipe": recipe.name, "speaker": speaker, "seed": seed,
                  "error": f"{type(error).__name__}: {error}"}
        with (self.report / "failures.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"FAILED {phase} {recipe.name} {speaker} seed {seed}: {error}; continuing", flush=True)
        self.release_weights(recipe, speaker, seed)

    def prepare(self):
        source_paths = [self.export / s / f"{split}.jsonl" for s in SPEAKERS for split in ("train", "dev", "test")]
        if self.a.study == "replay":
            source_paths += [self.export / f"loso-{s}/train.jsonl" for s in SPEAKERS]
        audio = {}
        for speaker in SPEAKERS:
            root = self.export / speaker
            _, records = validate_export(root)
            audio.update({str(root.relative_to(APP) / name): record for name, record in records.items()})
            if self.a.study == "replay":
                root = self.export / f"loso-{speaker}"
                _, records = validate_export(root, ("train",))
                audio.update({str(root.relative_to(APP) / name): record for name, record in records.items()})
        probe = ("import importlib.metadata as m,json,platform; "
                 "names=('torch','nemo_toolkit','lightning','numpy','soundfile','transformers'); "
                 "installed={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}; "
                 "print(json.dumps({'python':platform.python_version(), 'packages':"
                 "{n:installed.get(n.replace('_','-')) for n in names}}))")
        environments = {python: json.loads(subprocess.run([python, "-c", probe], check=True,
                        capture_output=True, text=True).stdout) for python in (self.a.train_python, self.a.cpu_python)}
        config = {"study": self.a.study, "recipes": [asdict(r) for r in self.items],
                  "batch": self.a.batch, "grad_accum": self.a.grad_accum, "offload": self.a.offload_optimizer,
                  "train_python": self.a.train_python, "cpu_python": self.a.cpu_python,
                  "environments": environments, "audio_inputs": audio,
                  "inputs": {str(p.relative_to(APP)): file_record(p) for p in source_paths},
                  "code": {name: file_record(APP / name) for name in
                           [f"scripts/asr/{name}.py" for name in ("long_runs", "train", "block_adapter", "nemo_hyp",
                                                                  "evaluate", "pool_table", "battery", "ci")] +
                           [f"src/willspeak/{name}" for name in ("artifacts.py", "evaluate.py", "normalize.py", "us_spelling.json",
                                                              "g2p.py", "db.py")]}}
        marker = self.report / "experiment.json"
        if marker.exists():
            if not self.a.resume:
                raise FileExistsError(f"Already exists: {self.report}. Use --resume or a new stamp")
            if json.loads(marker.read_text()) != config:
                if not getattr(self.a, "restart_changed", False):
                    raise ValueError("Run configuration, code or manifests changed; use a new stamp")
                for path in (self.report, self.models):
                    if path.exists():
                        saved = path.with_name(path.name + "-previous")
                        index = 2
                        while saved.exists():
                            saved = path.with_name(path.name + f"-previous-{index}")
                            index += 1
                        print(f"Inputs changed: preserving {path} as {saved}; restarting this study", flush=True)
                        path.rename(saved)
                        if path == self.models and not self.a.keep_all_models:
                            for checkpoint_dir in saved.rglob("ckpt"):
                                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                            for name in ("parakeet-ft.nemo", "adapter.pt"):
                                for weights in saved.rglob(name):
                                    weights.unlink()
                self.report.mkdir(parents=True)
                write_json(marker, config)
        else:
            if self.report.exists() or self.models.exists():
                raise FileExistsError("Unrecorded output directory already exists; use a new stamp")
            self.report.mkdir(parents=True)
            write_json(marker, config)
        self.configuration = config
        if self.a.study == "replay":
            for speaker in SPEAKERS:
                for recipe in self.items[1:]:
                    prepare_replay(self.export / speaker, self.export / f"loso-{speaker}", speaker,
                                   recipe.replay, self.training_export(recipe, speaker))

    def smoke(self):
        marker = self.report / "smoke-passed.json"
        if marker.exists():
            return
        items = self.items if self.a.study != "adapter" else [self.items[0], self.items[-2], self.items[-1]]
        failures = []
        for recipe in items:
            folder = Path(self.train_command(recipe, "M02", 17, smoke=True)[3])
            try:
                self.verify_sources(recipe, "M02")
                if folder.exists():
                    shutil.rmtree(folder / "ckpt", ignore_errors=True)
                    for name in ("adapter.pt", "parakeet-ft.nemo"):
                        (folder / name).unlink(missing_ok=True)
                    saved = folder.with_name(folder.name + "-previous")
                    index = 2
                    while saved.exists():
                        saved = folder.with_name(folder.name + f"-previous-{index}")
                        index += 1
                    folder.rename(saved)
                self.run(self.a.train_python, "train", *self.train_command(recipe, "M02", 17, smoke=True))
                if not self.a.dry_run:
                    progress = json.loads((folder / "run.json").read_text())
                    if progress.get("status") != "complete" or sum(r["epoch_optimizer_steps"] for r in progress["curve"]) < 2:
                        raise RuntimeError(f"Smoke did not finish two optimizer steps: {folder}")
                    dest = self.report / "smoke" / recipe.name
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(folder / "run.json", dest / "run.json")
                    shutil.rmtree(folder)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                self.record_failure("smoke", recipe, "M02", 17, error)
                failures.append(recipe.name)
                shutil.rmtree(folder / "ckpt", ignore_errors=True)
                for name in ("adapter.pt", "parakeet-ft.nemo"):
                    (folder / name).unlink(missing_ok=True)
        if failures:
            raise RuntimeError("Smoke trials failed after remaining trials were attempted: " + ", ".join(failures))
        if not self.a.dry_run:
            write_json(marker, {"recipes": [r.name for r in items], "optimizer_steps_each": 2})

    def search(self) -> str:
        lock = self.report / "selection.json"
        if lock.exists():
            record = json.loads(lock.read_text())
            if (set(record["scores"]) != {r.name for r in self.items}
                    or record["recipe"] != choose_dev(record["scores"])
                    or record["selection_split"] != "dev"
                    or record["dev_files"] != {path: file_record(Path(path)) for path in record["dev_files"]}):
                raise ValueError("Locked dev selection or scores changed; use a new stamp")
            return record["recipe"]
        scores, failures = {}, []
        for recipe in self.items:
            complete = True
            for speaker in SPEAKERS:
                try:
                    self.train_score(recipe, speaker, 17)
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                    self.record_failure("search", recipe, speaker, 17, error)
                    failures.append(f"{recipe.name}:{speaker}")
                    complete = False
                finally:
                    self.release_weights(recipe, speaker, 17)
            if complete:
                files = [self.score_file(recipe, s, 17, "dev") for s in SPEAKERS]
                scores[recipe.name] = pooled_scores(files)
                print(f"DEV {recipe.name}: {scores[recipe.name]['errors']}/{scores[recipe.name]['words']} "
                      f"({100 * scores[recipe.name]['wer']:.2f}%); current leader: {choose_dev(scores)}", flush=True)
        if failures:
            raise RuntimeError("Search trials failed after remaining trials were attempted: " + ", ".join(failures))
        record = {"recipe": choose_dev(scores), "selection_split": "dev", "search_seed": 17,
                  "scores": scores, "dev_files": {str(self.score_file(r, s, 17, "dev")): file_record(
                      self.score_file(r, s, 17, "dev")) for r in self.items for s in SPEAKERS}}
        write_json(lock, record)
        print(f"LOCKED recipe {record['recipe']} before test decoding: {lock}", flush=True)
        return record["recipe"]

    def finish(self, chosen: str):
        finalists = [self.items[0]] + [r for r in self.items if r.name == chosen and r.name != "control"]
        self.final_phase = True
        failures, summaries = [], {}
        for speaker in SPEAKERS:
            self.verify_sources(self.items[0], speaker)
            personal = self.export / speaker
            try:
                self.decode_score(["--stock"], personal / "test.jsonl", personal, self.report / "stock" / speaker)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                self.record_failure("stock-test", self.items[0], speaker, 0, error)
                failures.append(f"stock:{speaker}")
        for seed in (17, 18, 19):
            args = ["--system", "stock=" + ",".join(f"{s}:{self.report / 'stock' / s}" for s in SPEAKERS)]
            for recipe in finalists:
                for speaker in SPEAKERS:
                    try:
                        personal = self.export / speaker
                        out = self.score_file(recipe, speaker, seed, "test").parent
                        label = f"{recipe.name}-{speaker}-seed{seed}"
                        battery_out = self.report / "battery-hypotheses" / f"{label}.jsonl"
                        if not (out / "results.json").is_file() or not battery_out.is_file():
                            self.train_score(recipe, speaker, seed)
                            self.decode_score(self.model_flags(recipe, speaker, seed), personal / "test.jsonl", personal, out)
                            self.battery_trial(label, self.model_flags(recipe, speaker, seed))
                    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                        self.record_failure("final", recipe, speaker, seed, error)
                        failures.append(f"{recipe.name}:{speaker}:{seed}")
                    finally:
                        self.release_weights(recipe, speaker, seed)
                files = [self.score_file(recipe, s, seed, "test") for s in SPEAKERS]
                if all(p.is_file() for p in files):
                    summaries.setdefault(recipe.name, []).append(pooled_scores(files))
                    args += ["--system", recipe.name + "=" + ",".join(
                        f"{s}:{self.score_file(recipe, s, seed, 'test').parent}" for s in SPEAKERS)]
            table = self.report / f"test-seed-{seed}.json"
            if not failures and not table.exists():
                try:
                    self.run(self.a.cpu_python, "pool_table", *args, "--baseline", "control", "--by", "prompt",
                             "--export", str(self.export), "--out", str(table))
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                    self.record_failure("table", self.items[0], "pooled", seed, error)
                    failures.append(f"table:{seed}")
        if failures:
            raise RuntimeError("Final trials failed after remaining trials were attempted: " + ", ".join(failures))
        confirmation = {recipe.name: {str(seed): pooled_scores([
            self.score_file(recipe, s, seed, "dev") for s in SPEAKERS]) for seed in (17, 18, 19)}
            for recipe in finalists}
        write_json(self.report / "confirmation-dev.json", confirmation)
        summary = {name: {"seed_scores": group, "mean_wer": sum(r["wer"] for r in group) / len(group),
                          "wer_range": [min(r["wer"] for r in group), max(r["wer"] for r in group)]}
                   for name, group in summaries.items()}
        write_json(self.report / "summary.json", {"selection": chosen, "test": summary,
                                                  "warning": "Exploratory TORGO set previously inspected; no Pi/int8 result"})
        self.battery(finalists)
        for name, result in summary.items():
            print(f"TEST {name}: three-seed mean {100 * result['mean_wer']:.2f}%, "
                  f"range {100 * result['wer_range'][0]:.2f}–{100 * result['wer_range'][1]:.2f}%", flush=True)
        print(f"Results: {self.report}; full live log: {self.log}", flush=True)

    def battery_trial(self, label: str, flags: list[str]):
        root = self.report / "synthetic-battery"
        manifest = root / "manifest.jsonl"
        if not manifest.exists():
            self.run(self.a.cpu_python, "battery", "build", "--out", str(root), "--seed", "3000", "--per-kind", "20")
        out = self.report / "battery-hypotheses" / f"{label}.jsonl"
        if not out.exists():
            self.run(self.a.train_python, "nemo_hyp", *flags, "--manifest", str(manifest),
                     "--audio-root", str(root), "--out", str(out), "--batch", str(self.a.batch))

    def battery(self, finalists: list[Recipe]):
        self.battery_trial("stock", ["--stock"])
        labels = ["stock"] + [f"{r.name}-{s}-seed{seed}" for r in finalists for seed in (17, 18, 19) for s in SPEAKERS]
        systems = [item for label in labels for item in ("--system", f"{label}={self.report / 'battery-hypotheses' / (label + '.jsonl')}")]
        table = self.report / "battery.json"
        if not table.exists():
            self.run(self.a.cpu_python, "battery", "score", "--manifest", str(self.report / "synthetic-battery/manifest.jsonl"),
                     *systems, "--out", str(table))

    def dry_plan(self):
        print(f"Study: {self.a.study}; {len(self.items)} recipes; speakers: {' '.join(SPEAKERS)}")
        print("Search: seed 17, pooled dev. Confirm locked winner + control: seeds 18/19. Test: locked finalists only.")
        for recipe in self.items:
            print(f"\n{recipe.name}: {recipe.epochs} epochs; lr={recipe.lr}; form={recipe.form}; "
                  f"replay={recipe.replay}; personal repeat={recipe.repeat}")
            for speaker in SPEAKERS:
                self.run(self.a.train_python, "train", *self.train_command(recipe, speaker, 17))
        print(f"No commands executed or files written. Expected output: {self.report}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=REPORTS)
    parser.add_argument("stamp", help="new run name, e.g. 1008-overnight")
    parser.add_argument("--phase", choices=("smoke", "search", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse only finished work, with identical code/config/inputs")
    parser.add_argument("--restart-changed", action="store_true", help="preserve incompatible prior outputs and start this study again")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--offload-optimizer", action="store_true", help="use system RAM for non-adapter AdamW state")
    parser.add_argument("--keep-all-models", action="store_true", help="retain losing models and intermediate checkpoints")
    parser.add_argument("--min-free-gb", type=int, default=14)
    parser.add_argument("--train-python", default=os.environ.get("TRAIN_PY", str(Path.home() / "venvs/willspeak-train/bin/python")))
    parser.add_argument("--cpu-python", default=os.environ.get("SC", str(APP / ".venv/bin/python")))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.stamp) or min(args.batch, args.grad_accum, args.min_free_gb) < 1:
        parser.error("use a simple new stamp and positive batch/accumulation/free-space values")
    exp = Experiment(args)
    if args.dry_run:
        exp.dry_plan()
        return 0
    for python in (args.train_python, args.cpu_python):
        if not Path(python).is_file():
            parser.error(f"Python environment missing: {python}; run ./willspeak.sh lab-setup")
    with gpu_lock():
        exp.prepare()
        exp.smoke()
        if args.phase == "smoke":
            return 0
        chosen = exp.search()
        if args.phase == "all":
            exp.finish(chosen)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiskFull as error:
        print(f"STOPPED, disk full: {error}", file=sys.stderr, flush=True)
        raise SystemExit(75)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAILED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
