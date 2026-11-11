#!/usr/bin/env python3
"""Check the lab computer's inputs and outputs before a long run, and repair what can be repaired safely."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EX = "data/public/torgo/export"
FT = "reports/finetune-torgo"
SEVERE = ("M01", "M02", "M04")
PINS = ".lab/manifest-pins.json"
CPU_MODULES = ("numpy", "scipy", "soundfile", "rapidfuzz", "cmudict")
TRAIN_MODULES = ("torch", "nemo", "numba", "resampy", "onnx", "onnxruntime")
OPTIONAL_TRAIN = ("wandb", "resampy")
MIN_PYTHON = (3, 11)
REQUIRED_KEYS = ("id", "audio", "text", "duration_s")


@dataclass
class Finding:
    level: str
    area: str
    message: str
    advice: str = ""
    fix: tuple | None = None
    detail: list[str] = field(default_factory=list)

    def line(self) -> str:
        tag = {"ok": "PASS", "info": "INFO", "warn": "WARN", "fail": "FAIL"}[self.level]
        text = f"  {tag:<4} [{self.area}] {self.message}"
        if self.advice:
            text += f"\n         {self.advice}"
        if self.fix:
            text += f"\n         --fix: {self.fix[0]}" + (f" {self.fix[1]}" if self.fix[1] not in (None, "") else "")
        for extra in self.detail[:4]:
            text += f"\n         - {extra}"
        if len(self.detail) > 4:
            text += f"\n         - ... and {len(self.detail) - 4} more"
        return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                problems.append(f"line {number} is not valid JSON")
                continue
            if not isinstance(row, dict):
                problems.append(f"line {number} is not an object")
                continue
            rows.append(row)
    return rows, problems


def prompt_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))


def quarantine(path: Path) -> Path:
    target = path.with_name(path.name + "-corrupt")
    n = 2
    while target.exists():
        target = path.with_name(f"{path.name}-corrupt-{n}")
        n += 1
    path.rename(target)
    return target


def run(cmd: list[str], timeout: int = 120, cwd: Path | None = None) -> tuple[int, str]:
    try:
        done = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s"
    except OSError as exc:
        return 127, str(exc)


def probe_python(python: str, modules: tuple[str, ...]) -> dict | None:
    code = ("import sys,json,importlib.util as u;"
            f"print(json.dumps({{'v':list(sys.version_info[:3]),'missing':[m for m in {list(modules)!r} if u.find_spec(m) is None]}}))")
    status, out = run([python, "-c", code], timeout=60)
    if status != 0:
        return None
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def check_runtime(cpu_python: str, train_python: str | Path | None, need_gpu: bool = True) -> list[Finding]:
    out: list[Finding] = []
    info = probe_python(cpu_python, CPU_MODULES)
    if info is None:
        out.append(Finding("fail", "runtime", f"cannot run the CPU Python {cpu_python}", "set the cpupy variable or run ./willspeak.sh setup"))
    else:
        version = tuple(info["v"])
        if version < MIN_PYTHON:
            out.append(Finding("fail", "runtime", f"CPU Python is {'.'.join(map(str, version))}; the scripts need 3.11 or newer "
                               "(scripts/_cli.py imports datetime.UTC)", "run ./willspeak.sh setup, which makes .venv with a newer Python"))
        else:
            out.append(Finding("ok", "runtime", f"CPU Python {'.'.join(map(str, version))}"))
        if info["missing"]:
            out.append(Finding("fail", "runtime", "CPU Python is missing " + ", ".join(info["missing"]),
                               fix=("pip", " ".join(info["missing"]))))
        else:
            out.append(Finding("ok", "runtime", "CPU packages present"))
    if train_python is None or not Path(str(train_python)).is_file():
        out.append(Finding("fail", "runtime", f"training environment not found: {train_python}", "run ./willspeak.sh lab-setup"))
        return out
    info = probe_python(str(train_python), TRAIN_MODULES)
    if info is None:
        out.append(Finding("fail", "runtime", f"the training Python does not start: {train_python}"))
        return out
    hard = [m for m in info["missing"] if m not in OPTIONAL_TRAIN]
    soft = [m for m in info["missing"] if m in OPTIONAL_TRAIN]
    if hard:
        out.append(Finding("fail", "runtime", "training environment is missing " + ", ".join(hard), "./willspeak.sh lab-setup rebuilds it"))
    else:
        out.append(Finding("ok", "runtime", "training packages present"))
    if soft:
        out.append(Finding("warn", "runtime", "training environment lacks " + ", ".join(soft) + " (speed perturbation silently skips without resampy)",
                           fix=("pip-train", " ".join(soft))))
    if need_gpu:
        status, out_text = run([str(train_python), "-c",
                                "import torch;ok=torch.cuda.is_available();"
                                "print(ok, torch.cuda.get_device_name(0) if ok else '', "
                                "torch.cuda.mem_get_info()[0]/2**30 if ok else 0, torch.cuda.mem_get_info()[1]/2**30 if ok else 0)"], timeout=240)
        parts = out_text.strip().splitlines()[-1].split() if status == 0 and out_text.strip() else []
        if status == 124:
            out.append(Finding("warn", "runtime", "importing torch took over 4 minutes; the GPU was not checked"))
        elif not parts or parts[0] != "True":
            out.append(Finding("fail", "runtime", "torch cannot see a GPU", "check the NVIDIA driver; nvidia-smi should list the card"))
        else:
            free, total = float(parts[-2]), float(parts[-1])
            level = "ok" if free >= 0.85 * total - 1 else "warn"
            out.append(Finding(level, "runtime", f"GPU {' '.join(parts[1:-2])}: {free:.1f} of {total:.1f} GiB free",
                               "" if level == "ok" else "another program is using GPU memory; training that spills into RAM is several times slower. "
                                                         "Close it, or train with --offload-optimizer."))
    return out


def free_gib(root: Path) -> float:
    return shutil.disk_usage(root).free / 2 ** 30


def check_disk(root: Path, need_gb: float = 40.0) -> list[Finding]:
    free = free_gib(root)
    if free < 8:
        return [Finding("fail", "disk", f"{free:.0f} GiB free; a training run needs about 8", fix=("clean-space", None))]
    if free < need_gb:
        return [Finding("warn", "disk", f"{free:.0f} GiB free (the plan wants about {need_gb:.0f})",
                        "finished runs' weights can be freed", fix=("clean-space", None))]
    return [Finding("ok", "disk", f"{free:.0f} GiB free")]


def export_dir(root: Path, name: str) -> Path:
    return root / EX / name


def wav_problem(path: Path, row: dict, deep: bool) -> str | None:
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size < 100:
        return f"only {size} bytes"
    if not deep:
        return None
    try:
        with wave.open(str(path), "rb") as w:
            rate, channels, frames = w.getframerate(), w.getnchannels(), w.getnframes()
    except (wave.Error, EOFError):
        return "not a readable PCM WAV"
    if rate != 16000 or channels != 1:
        return f"{rate} Hz, {channels} channels (expected 16000 Hz mono)"
    seconds = frames / rate
    want = float(row.get("duration_s") or 0)
    if want and abs(seconds - want) > max(0.06, 0.03 * want):
        return f"{seconds:.2f} s long, manifest says {want:.2f} s"
    return None


def check_export(root: Path, name: str, deep: bool = False, splits=("train", "dev", "test")) -> list[Finding]:
    folder = export_dir(root, name)
    if not folder.is_dir():
        return [Finding("fail", "data", f"export folder {name} is missing", fix=("make-exports", name))]
    out: list[Finding] = []
    seen_prompts: dict[str, set[str]] = {}
    for split in splits:
        manifest = folder / f"{split}.jsonl"
        if not manifest.is_file():
            out.append(Finding("fail", "data", f"{name}/{split}.jsonl is missing", fix=("make-exports", name)))
            continue
        rows, bad = read_jsonl(manifest)
        if bad:
            out.append(Finding("fail", "data", f"{name}/{split}.jsonl has unreadable lines", detail=bad))
        absent = [k for k in REQUIRED_KEYS if rows and any(k not in r for r in rows)]
        if absent:
            out.append(Finding("fail", "data", f"{name}/{split}.jsonl: some rows lack {', '.join(absent)}"))
        ids = [r.get("id") for r in rows]
        if len(ids) != len(set(ids)):
            out.append(Finding("fail", "data", f"{name}/{split}.jsonl has duplicate ids"))
        problems = []
        for row in rows:
            audio = row.get("audio")
            if not audio:
                continue
            issue = wav_problem(folder / audio, row, deep)
            if issue:
                problems.append(f"{audio}: {issue}")
        if problems:
            gone = [p for p in problems if p.endswith(": missing")]
            out.append(Finding("fail", "data", f"{name}/{split}: {len(problems)} of {len(rows)} audio files have problems",
                               fix=("repair-audio", name) if gone else None, detail=problems))
        seen_prompts[split] = {prompt_key(r.get("prompt_text") or r.get("text") or "") for r in rows}
        if split == "test" and not rows:
            out.append(Finding("fail", "data", f"{name}/test.jsonl is empty"))
    if "train" in seen_prompts and "test" in seen_prompts:
        shared = (seen_prompts["train"] & seen_prompts["test"]) - {""}
        if shared:
            out.append(Finding("fail", "data", f"{name}: {len(shared)} prompts are in both train and test (leak)", detail=sorted(shared)))
    if not any(f.level == "fail" for f in out):
        out.append(Finding("ok", "data", f"export {name}: manifests and audio consistent"
                           + (" (lengths and formats checked)" if deep else "")))
    return out


def check_torgo(root: Path, need_controls: bool) -> list[Finding]:
    base = root / "data/public/torgo/TORGO"
    out = []
    missing = [s for s in SEVERE if not (base / s).is_dir()]
    out.append(Finding("warn" if missing else "ok", "data", "TORGO audio for " + (", ".join(missing) + " is missing" if missing else "M01, M02 and M04 is present"),
                       "needed only to rebuild exports" if missing else ""))
    if need_controls:
        controls = [s for s in ("FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04") if not (base / s).is_dir()]
        if controls and len(controls) == 7:
            out.append(Finding("warn", "data", "TORGO control speakers (FC, MC) are not downloaded; lab tests 16-17 train on their recordings",
                               "about 2.5 GB; python scripts/corpus/fetch_torgo.py --only FC MC", fix=("fetch-controls", None)))
        elif controls:
            out.append(Finding("warn", "data", "some control speakers are missing: " + ", ".join(controls)))
        else:
            out.append(Finding("ok", "data", "TORGO control speakers present"))
    return out


def pin_targets(root: Path) -> list[Path]:
    return [export_dir(root, s) / f"{split}.jsonl" for s in SEVERE for split in ("train", "dev", "test")]


def load_pins(root: Path) -> dict:
    path = root / PINS
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def write_pins(root: Path, repin: bool = False) -> dict:
    pins = load_pins(root)
    for path in pin_targets(root):
        if path.is_file():
            key = str(path.relative_to(root)).replace("\\", "/")
            if repin or key not in pins:
                pins[key] = sha256_file(path)
    (root / PINS).parent.mkdir(parents=True, exist_ok=True)
    (root / PINS).write_text(json.dumps(pins, indent=2, sort_keys=True), encoding="utf-8")
    return pins


def check_pins(root: Path) -> list[Finding]:
    pins = load_pins(root)
    if not pins:
        return [Finding("info", "pins", "the speakers' manifests are not pinned yet", "python lab.py doctor --pin remembers their checksums",
                        fix=("pin", None))]
    changed, new = [], []
    for path in pin_targets(root):
        key = str(path.relative_to(root)).replace("\\", "/")
        if not path.is_file():
            continue
        if key not in pins:
            new.append(key)
        elif sha256_file(path) != pins[key]:
            changed.append(key)
    out = []
    if changed:
        out.append(Finding("fail", "pins", f"{len(changed)} manifests changed since they were pinned: every earlier number was measured on the old ones",
                           "restore them, or after a deliberate change: python lab.py doctor --pin --repin", detail=changed))
    if new:
        out.append(Finding("info", "pins", f"{len(new)} manifests are not pinned yet", fix=("pin", None), detail=new))
    if not changed and not new:
        out.append(Finding("ok", "pins", "manifests match their pins"))
    return out


def speaker_split_of(name: str) -> tuple[str | None, str | None]:
    speakers = re.findall(r"(?<![A-Za-z0-9])(M0[124])(?![0-9])", name)
    found = re.findall(r"-(test|dev)(?=-)", name)
    split = found[-1] if found else None
    return (speakers[0] if len(set(speakers)) == 1 else None), split


def check_outputs(root: Path) -> list[Finding]:
    folder = root / FT
    if not folder.is_dir():
        return [Finding("info", "outputs", "no reports/finetune-torgo yet")]
    out: list[Finding] = []
    orphan_hyps, orphan_prov, unscored, bad_results = [], [], [], []
    for hyp in sorted(folder.rglob("*-nemo.jsonl")):
        prov = hyp.with_suffix(".provenance.json")
        results = hyp.with_suffix("") / "results.json"
        rel = str(hyp.relative_to(root))
        if not prov.is_file():
            orphan_hyps.append(rel)
            continue
        if not results.is_file():
            unscored.append(rel)
        else:
            try:
                data = json.loads(results.read_text(encoding="utf-8"))
                if "wer" not in data:
                    bad_results.append(str(results.relative_to(root)))
            except (OSError, ValueError):
                bad_results.append(str(results.relative_to(root)))
    for prov in sorted(folder.rglob("*-nemo.provenance.json")):
        if not prov.with_name(prov.name[: -len(".provenance.json")] + ".jsonl").is_file():
            orphan_prov.append(str(prov.relative_to(root)))
    if orphan_hyps:
        out.append(Finding("fail", "outputs", f"{len(orphan_hyps)} decodes have no provenance file (a decode stopped between its two writes); "
                           "the decoder refuses to overwrite them", fix=("quarantine", orphan_hyps), detail=orphan_hyps))
    if orphan_prov:
        out.append(Finding("fail", "outputs", f"{len(orphan_prov)} provenance files have no decode", fix=("quarantine", orphan_prov), detail=orphan_prov))
    if bad_results:
        out.append(Finding("fail", "outputs", f"{len(bad_results)} results.json files are unreadable or lack 'wer'",
                           fix=("quarantine", bad_results), detail=bad_results))
    if unscored:
        out.append(Finding("warn", "outputs", f"{len(unscored)} finished decodes were never scored",
                           "the stage that made them stopped before scoring; --fix scores the ones whose manifest can be identified",
                           fix=("rescore", unscored), detail=unscored))
    models = root / "models"
    damaged = []
    if models.is_dir():
        for run_dir in sorted(models.glob("parakeet-ft-*")):
            info = run_dir / "run.json"
            nemo, adapter = run_dir / "parakeet-ft.nemo", run_dir / "adapter.pt"
            for weights in (nemo, adapter):
                if weights.is_file():
                    if weights.stat().st_size < 1024:
                        damaged.append(str(weights.relative_to(root)) + " (empty)")
                    elif weights.suffix == ".nemo" and not (tarfile.is_tarfile(weights) or zipfile.is_zipfile(weights)):
                        damaged.append(str(weights.relative_to(root)) + " (not an archive)")
            if info.is_file():
                try:
                    json.loads(info.read_text(encoding="utf-8"))
                except ValueError:
                    damaged.append(str(info.relative_to(root)) + " (run.json unreadable)")
    if damaged:
        out.append(Finding("fail", "outputs", f"{len(damaged)} model files are damaged", "move them aside; the stage trains them again",
                           fix=("quarantine", [d.split(" (")[0] for d in damaged]), detail=damaged))
    leftovers = [str(p.relative_to(root)) for pattern in ("models/smoke-*", "models/*.partial", "data/public/torgo/export/*.partial")
                 for p in sorted(root.glob(pattern))]
    if leftovers:
        out.append(Finding("info", "outputs", f"{len(leftovers)} scratch folders from interrupted checks or builds", detail=leftovers))
    if not out:
        out.append(Finding("ok", "outputs", "decodes, scores and model files are consistent"))
    return out


def fix_pip(python: str, packages: str) -> str:
    status, text = run([python, "-m", "pip", "install", "-q", *packages.split()], timeout=900)
    return f"pip install {packages}: " + ("ok" if status == 0 else "failed: " + text.strip().splitlines()[-1][:200])


def fix_quarantine(root: Path, paths: list[str]) -> str:
    moved = []
    for rel in paths:
        path = root / rel
        if path.exists():
            moved.append(f"{rel} -> {quarantine(path).name}")
            sibling = path.with_suffix(".provenance.json") if path.suffix == ".jsonl" else None
            if sibling is not None and sibling.exists():
                quarantine(sibling)
    return "moved aside: " + "; ".join(moved) if moved else "nothing to move"


def fix_rescore(root: Path, paths: list[str], cpu_python: str) -> str:
    done, skipped = [], []
    for rel in paths:
        hyp = root / rel
        speaker, split = speaker_split_of(hyp.name)
        manifest = export_dir(root, speaker) / f"{split}.jsonl" if speaker and split else None
        try:
            prov = json.loads(hyp.with_suffix(".provenance.json").read_text(encoding="utf-8"))
            wanted = prov["inputs"]["manifest"]["sha256"]
        except (OSError, ValueError, KeyError):
            wanted = None
        if manifest is None or not manifest.is_file() or wanted != sha256_file(manifest):
            skipped.append(hyp.name)
            continue
        status, text = run([cpu_python, "scripts/asr/evaluate.py", "--manifest", str(manifest), "--audio-root", str(manifest.parent),
                            "--hyp", str(hyp), "--group-by", "kind", "--out", str(hyp.with_suffix(""))], timeout=900, cwd=root)
        (done if status == 0 else skipped).append(hyp.name)
    return f"scored {len(done)}; could not identify or score {len(skipped)}" + (": " + ", ".join(skipped[:5]) if skipped else "")


def fix_exports(root: Path, name: str, cpu_python: str) -> str:
    if not (root / "data/public/torgo/TORGO").is_dir():
        return f"cannot make export {name}: the TORGO download is not here (python scripts/corpus/fetch_torgo.py)"
    cmd = [cpu_python, "scripts/corpus/torgo_manifest.py", "--export", "--missing-only"]
    if name == "controls":
        cmd += ["--only", "controls"]
    else:
        base = name.replace("loso-severe-", "").replace("loso-", "")
        cmd += ["--speakers", *((base,) if base in SEVERE else SEVERE), "--severe-loso", "--only", name]
    status, text = run(cmd, timeout=3600, cwd=root)
    return f"export {name}: " + ("made" if status == 0 else "failed: " + text.strip().splitlines()[-1][:200])


def repair_audio(root: Path, name: str) -> str:
    sys.path.insert(0, str(root / "src"))
    import soundfile as sf

    from willspeak import audio as A
    from willspeak.torgo import scan

    torgo = root / "data/public/torgo/TORGO"
    if not torgo.is_dir():
        return "the TORGO download is not here"
    by_id = {u.id: u for u in scan(torgo)}
    folder = export_dir(root, name)
    made, lost = 0, []
    for split in ("train", "dev", "test"):
        manifest = folder / f"{split}.jsonl"
        if not manifest.is_file():
            continue
        rows, _ = read_jsonl(manifest)
        for row in rows:
            dst = folder / row["audio"]
            if dst.is_file() and dst.stat().st_size >= 100:
                continue
            u = by_id.get(row["id"])
            if u is None or not (torgo / u.audio).is_file():
                lost.append(row["id"])
                continue
            x, sr = sf.read(str(torgo / u.audio), dtype="float32")
            dst.parent.mkdir(parents=True, exist_ok=True)
            A.write_wav(str(dst), A.to_mono_16k(x, sr), 16000, subtype="PCM_16")
            made += 1
    return f"{name}: re-created {made} files; {len(lost)} could not be found in the download" + (f" ({', '.join(lost[:3])})" if lost else "")


def apply_fixes(findings: list[Finding], root: Path, cpu_python: str, train_python: str | None = None, allow_download: bool = False) -> list[str]:
    done, seen = [], set()
    for f in findings:
        if not f.fix or f.level == "ok":
            continue
        action, arg = f.fix
        key = (action, json.dumps(arg, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        if action == "pip":
            done.append(fix_pip(cpu_python, arg))
        elif action == "pip-train" and train_python:
            done.append(fix_pip(train_python, arg))
        elif action == "clean-space":
            status, text = run([cpu_python, "scripts/asr/lab_space.py", "clean", "--apply", "--lab-models", "--long-models"], timeout=1800, cwd=root)
            done.append("freed space: " + ("ok" if status == 0 else text.strip().splitlines()[-1][:160]))
        elif action == "quarantine":
            done.append(fix_quarantine(root, arg))
        elif action == "rescore":
            done.append(fix_rescore(root, arg, cpu_python))
        elif action == "make-exports":
            done.append(fix_exports(root, arg, cpu_python))
        elif action == "repair-audio":
            status, text = run([cpu_python, str(Path(__file__).resolve()), "repair-audio", "--export", arg], timeout=3600, cwd=root)
            done.append(text.strip().splitlines()[-1] if text.strip() else f"repair-audio {arg}: no output")
        elif action == "pin":
            write_pins(root)
            done.append("pinned the manifests' checksums")
        elif action == "fetch-controls":
            if allow_download:
                status, text = run([cpu_python, "scripts/corpus/fetch_torgo.py", "--only", "FC", "MC"], timeout=7200, cwd=root)
                done.append("fetched the control speakers: " + ("ok" if status == 0 else text.strip().splitlines()[-1][:160]))
            else:
                done.append("control speakers not fetched: about 2.5 GB, so it needs --allow-download")
    return done


@dataclass
class Failure:
    category: str
    retryable: bool
    remedy: str


FAILURE_PATTERNS = [
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|CUDA error: out of memory", re.I),
     True, "retry with a smaller batch (batch 1 with 16 accumulation is the same update)"),
    ("disk", re.compile(r"No space left on device|Errno 28|Less than \d+ GiB free|disk quota|not enough space|[\d.]+ GB free where .* needs about", re.I),
     True, "free the weights of finished runs, then retry"),
    ("network", re.compile(r"ConnectionError|MaxRetryError|Temporary failure in name resolution|Name or service not known|Read timed out|"
                           r"HTTP Error 5\d\d|SSLError|RemoteDisconnected|Connection reset|IncompleteRead|ReadTimeout|LocalEntryNotFoundError", re.I),
     True, "wait and retry (downloads resume)"),
    ("killed", re.compile(r"\bKilled\b|exit code -9|SIGKILL|MemoryError|std::bad_alloc", re.I),
     True, "the machine ran out of RAM: retry with the optimizer offloaded and the batch halved"),
    ("missing", re.compile(r"\bMISSING\b|FileNotFoundError|No such file or directory", re.I),
     True, "run the doctor with --fix (it rebuilds exports and baselines), then retry"),
    ("exists", re.compile(r"ALREADY EXISTS|FileExistsError|Already exists", re.I),
     False, "an earlier result exists: nothing to repair; use --rerun to move it aside"),
    ("changed", re.compile(r"configuration, code or manifests changed", re.I),
     False, "a long study's saved configuration no longer matches: it needs --restart-changed or a new stamp"),
    ("interrupted", re.compile(r"KeyboardInterrupt|interrupted during stage", re.I),
     False, "stopped by hand"),
    ("cuda", re.compile(r"CUDA error|CUDA driver|no CUDA-capable device|NCCL error|cudnn", re.I),
     False, "a driver or card problem: check nvidia-smi; not fixable from here"),
]


def classify_failure(text: str) -> Failure:
    tail = "\n".join(str(text).splitlines()[-200:])
    for category, pattern, retryable, remedy in FAILURE_PATTERNS:
        if pattern.search(tail):
            return Failure(category, retryable, remedy)
    return Failure("unknown", False, "read the stage's log")


def run_checks(root: Path, cpu_python: str, train_python: str | Path | None, speakers=SEVERE, need_controls: bool = False,
               deep: bool = False, need_gb: float = 40.0, need_gpu: bool = True, loso: bool = True) -> list[Finding]:
    root = Path(root)
    findings = check_runtime(cpu_python, train_python, need_gpu)
    findings += check_disk(root, need_gb)
    findings += check_torgo(root, need_controls)
    names = list(speakers)
    if loso:
        names += [f"loso-{s}" for s in speakers]
    if need_controls:
        names.append("controls")
    for name in names:
        findings += check_export(root, name, deep)
    findings += check_pins(root)
    findings += check_outputs(root)
    return findings


def report(findings: list[Finding], quiet_ok: bool = False) -> tuple[int, int]:
    for f in findings:
        if quiet_ok and f.level == "ok":
            continue
        print(f.line(), flush=True)
    fails = sum(f.level == "fail" for f in findings)
    warns = sum(f.level == "warn" for f in findings)
    print(f"\n== doctor: {sum(f.level == 'ok' for f in findings)} pass, {warns} warn, {fails} FAIL")
    return fails, warns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("repair-audio", help="re-create missing WAVs of one export from the TORGO download")
    r.add_argument("--export", required=True)
    c = sub.add_parser("check", help="the checks (default)")
    for p in (ap, c):
        p.add_argument("--root", default=str(ROOT))
        p.add_argument("--cpu-python", default=sys.executable)
        p.add_argument("--train-python", default=str(Path.home() / "venvs/willspeak-train/bin/python"))
        p.add_argument("--speakers", nargs="*", default=list(SEVERE))
        p.add_argument("--controls", action="store_true", help="also check the control speakers (lab tests 16-17)")
        p.add_argument("--deep", action="store_true")
        p.add_argument("--no-gpu", action="store_true")
        p.add_argument("--fix", action="store_true")
        p.add_argument("--allow-download", action="store_true", help="let --fix download the control speakers (about 2.5 GB)")
        p.add_argument("--pin", action="store_true")
        p.add_argument("--repin", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "repair-audio":
        print(repair_audio(root, a.export))
        return 0
    if a.pin:
        pins = write_pins(root, a.repin)
        print(f"pinned {len(pins)} manifests in {PINS}")
    findings = run_checks(root, a.cpu_python, a.train_python, tuple(a.speakers), a.controls, a.deep, need_gpu=not a.no_gpu)
    fails, _ = report(findings)
    if a.fix and any(f.fix for f in findings if f.level != "ok"):
        print("\n== fixing")
        for line in apply_fixes(findings, root, a.cpu_python, a.train_python, a.allow_download):
            print("  " + line)
        print("\n== checking again")
        findings = run_checks(root, a.cpu_python, a.train_python, tuple(a.speakers), a.controls, a.deep, need_gpu=not a.no_gpu)
        fails, _ = report(findings)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
