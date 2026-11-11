#!/usr/bin/env python3
"""Text-to-speech as a source of training audio."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from willspeak.normalize import normalize  # noqa: E402

SR = 16_000
VERSION = 1
ESPEAK_VOICES = ["en-us", "en-gb", "en-us+f3", "en-us+m3", "en-us+f5", "en-us+m7", "en-gb+f2", "en-us+m1"]


def stable(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def key_of(text: str) -> str:
    return normalize(text, us_spelling=False)


def to_16k_mono(x: np.ndarray, rate: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != SR:
        g = gcd(SR, int(rate))
        x = resample_poly(x, SR // g, int(rate) // g)
    return x


def tidy(x: np.ndarray, lead_tail_ms: int = 120) -> np.ndarray:
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return x
    loud = np.flatnonzero(np.abs(x) > 0.02 * peak)
    pad = int(SR * lead_tail_ms / 1000)
    start, end = max(0, int(loud[0]) - pad), min(x.size, int(loud[-1]) + pad)
    y = x[start:end]
    return (0.5 / max(float(np.max(np.abs(y))), 1e-9)) * y


def write_wav(path: Path, x: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0).astype(np.float32), SR, subtype="PCM_16")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, rate = sf.read(str(path), dtype="float32", always_2d=False)
    return x, rate


class Backend:
    name = "base"

    def variants(self) -> list[dict]:
        raise NotImplementedError

    def synth(self, text: str, variant: dict, out: Path) -> None:
        raise NotImplementedError

    def describe(self, variant: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in variant.items())


class Espeak(Backend):
    name = "espeak"

    def __init__(self, program: str | None = None):
        self.program = program or shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.program:
            raise RuntimeError("espeak-ng is not installed (sudo apt install espeak-ng)")

    def variants(self) -> list[dict]:
        out = []
        for i, voice in enumerate(ESPEAK_VOICES):
            out.append({"voice": voice, "speed": 120 + 12 * (i % 4), "pitch": 35 + 8 * (i % 5)})
        return out

    def synth(self, text: str, variant: dict, out: Path) -> None:
        done = subprocess.run([self.program, "-v", variant["voice"], "-s", str(variant["speed"]), "-p", str(variant["pitch"]), "-w", str(out), text],
                              capture_output=True, text=True, check=False, timeout=60)
        if done.returncode != 0 or not out.is_file():
            raise RuntimeError(f"espeak failed ({done.returncode}): {done.stderr.strip()[:200]}")


class Piper(Backend):
    name = "piper"

    def __init__(self, voice_dir: str | None = None, voices: list[str] | None = None, program: str | None = None):
        self.program = program or shutil.which("piper")
        self.voice_dir = Path(voice_dir or os.environ.get("PIPER_VOICES", "")) if (voice_dir or os.environ.get("PIPER_VOICES")) else None
        self.voices = voices or self.find_voices()
        if not (self.program or self.module()):
            raise RuntimeError("piper is not installed (pip install piper-tts)")
        if not self.voices:
            raise RuntimeError("no piper voice found: python -m piper.download_voices en_US-lessac-medium, then --voice-dir or PIPER_VOICES")

    @staticmethod
    def module() -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("piper") is not None
        except (ImportError, ValueError):
            return False

    def find_voices(self) -> list[str]:
        folders = [self.voice_dir] if self.voice_dir else []
        folders += [Path.cwd(), Path.home() / ".local/share/piper", Path.home() / "piper-voices"]
        found: list[str] = []
        for folder in folders:
            if folder and folder.is_dir():
                found += [str(p) for p in sorted(folder.glob("*.onnx"))]
        return found

    def variants(self) -> list[dict]:
        return [{"model": v, "length_scale": s} for v in self.voices for s in (1.0, 1.15, 0.9, 1.3)]

    def synth(self, text: str, variant: dict, out: Path) -> None:
        cmd = [self.program or sys.executable, *([] if self.program else ["-m", "piper"]), "--model", variant["model"],
               "--length_scale", str(variant["length_scale"]), "--output_file", str(out)]
        done = subprocess.run(cmd, input=text, capture_output=True, text=True, check=False, timeout=120)
        if done.returncode != 0 or not out.is_file():
            raise RuntimeError(f"piper failed ({done.returncode}): {done.stderr.strip()[:200]}")

    def describe(self, variant: dict) -> str:
        return f"{Path(variant['model']).stem},length={variant['length_scale']}"


class Fake(Backend):
    name = "fake"

    def variants(self) -> list[dict]:
        return [{"f0": 110 + 25 * i, "rate": 4.0 + 0.5 * i} for i in range(4)]

    def synth(self, text: str, variant: dict, out: Path) -> None:
        import soundfile as sf

        n_syll = max(1, sum(1 for ch in text.lower() if ch in "aeiouy"))
        seconds = 0.25 + n_syll / variant["rate"]
        t = np.arange(int(seconds * 22050)) / 22050
        carrier = sum(np.sin(2 * np.pi * variant["f0"] * k * t) / k for k in range(1, 6))
        env = 0.5 * (1 - np.cos(2 * np.pi * variant["rate"] * t)) * (t > 0.1) * (t < seconds - 0.1)
        sf.write(str(out), (0.3 * carrier * env).astype(np.float32), 22050, subtype="PCM_16")


def available(voice_dir: str | None = None) -> dict[str, tuple[bool, str]]:
    result: dict[str, tuple[bool, str]] = {}
    try:
        e = Espeak()
        result["espeak"] = (True, e.program)
    except RuntimeError as exc:
        result["espeak"] = (False, str(exc))
    try:
        p = Piper(voice_dir)
        result["piper"] = (True, f"{len(p.voices)} voice(s), e.g. {Path(p.voices[0]).name}")
    except RuntimeError as exc:
        result["piper"] = (False, str(exc))
    result["fake"] = (True, "not speech; tests only")
    return result


def get_backend(name: str, voice_dir: str | None = None, allow_fake: bool = False) -> Backend:
    if name == "auto":
        for candidate in ("piper", "espeak"):
            try:
                return get_backend(candidate, voice_dir)
            except RuntimeError:
                continue
        raise RuntimeError("no text-to-speech backend found: install espeak-ng (sudo apt install espeak-ng) or piper-tts; see `tts_synth.py backends`")
    if name == "espeak":
        return Espeak()
    if name == "piper":
        return Piper(voice_dir)
    if name == "fake":
        if not allow_fake:
            raise RuntimeError("the fake backend is not speech; pass --allow-fake for tests and dry runs")
        return Fake()
    raise ValueError(f"unknown backend {name!r}: auto, espeak, piper or fake")


def read_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_texts(manifests: list[Path], extra_files: list[Path], exclude: list[Path]) -> tuple[list[str], dict]:
    blocked = {key_of(str(r.get("prompt_text") or r.get("text") or "")) for m in exclude for r in read_rows(m)}
    seen: dict[str, str] = {}
    dropped = 0
    sources = 0
    candidates: list[str] = []
    for m in manifests:
        for r in read_rows(m):
            text = str(r.get("prompt_text") or r.get("text") or "")
            if "(live)" in text:
                continue
            candidates.append(text)
    for f in extra_files:
        candidates += [ln.strip() for ln in Path(f).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    for text in candidates:
        k = key_of(text)
        if not k:
            continue
        sources += 1
        if k in blocked:
            dropped += 1
            continue
        seen.setdefault(k, " ".join(text.split()))
    return list(seen.values()), {"candidates": sources, "dropped_as_excluded_prompts": dropped, "distinct_kept": len(seen), "excluded_prompts": len(blocked)}


def settings_fingerprint(**kw) -> dict:
    return {**kw, "version": VERSION}


def make(out: Path, backend: Backend, texts: list[str], variants: int, max_clips: int, seed: int = 17, workers: int = 4, force: bool = False,
         info_extra: dict | None = None) -> dict:
    out = Path(out)
    if not texts:
        raise ValueError("no texts left to synthesise (all were excluded prompts, or the lists are empty)")
    options = backend.variants()
    settings = settings_fingerprint(backend=backend.name, variants=variants, max_clips=max_clips, seed=seed,
                                    texts=hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16],
                                    voices=hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:12])
    marker = out / "tts.json"
    if marker.is_file() and not force:
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("settings") == settings and old.get("complete"):
            print(f"{out}: already built with these settings ({old['clips']} clips); reusing")
            return old
        raise FileExistsError(f"{out} exists with other settings; use a new --out or --force (the old folder is moved aside)")
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} exists and is not a finished build of this tool")
    jobs = []
    for text in texts:
        for v in range(variants):
            jobs.append((text, v, options[stable(seed, text, v) % len(options)]))
    jobs.sort(key=lambda j: stable(seed, "order", j[0], j[1]))
    jobs = jobs[:max_clips]
    scratch = out.with_name(out.name + ".partial")
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "train").mkdir(parents=True)
    (scratch / "dev").mkdir(parents=True)

    def one(job):
        text, v, variant = job
        cid = f"tts-{backend.name}-{stable(seed, text, v):012x}"
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.wav"
            try:
                backend.synth(text, variant, raw)
                x, rate = read_wav(raw)
            except Exception as exc:  # noqa: BLE001
                return None, str(exc)
        y = tidy(to_16k_mono(x, rate))
        if y.size < SR * 0.15:
            return None, "clip is shorter than 0.15 s"
        split = "dev" if stable(seed, "split", key_of(text)) % 10 == 0 else "train"
        write_wav(scratch / split / f"{cid}.wav", y)
        n_words = len(key_of(text).split())
        return split, {"id": cid, "audio": f"{split}/{cid}.wav", "text": text, "prompt_text": text, "duration_s": round(y.size / SR, 3),
                       "speaker": f"TTS-{backend.name}", "kind": "word" if n_words == 1 else "sentence", "condition": "tts",
                       "voice": backend.describe(variant), "variant": v}

    rows: dict[str, list[dict]] = {"train": [], "dev": []}
    failures: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for split, result in pool.map(one, jobs):
            if split is None:
                failures.append(str(result))
            else:
                rows[split].append(result)
    if jobs and len(failures) > 0.05 * len(jobs):
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError(f"{len(failures)} of {len(jobs)} clips failed, first: {failures[0]}; not writing a partial export")
    if not rows["dev"]:
        moved = rows["train"].pop()
        new = f"dev/{moved['id']}.wav"
        (scratch / new).parent.mkdir(exist_ok=True)
        (scratch / moved["audio"]).rename(scratch / new)
        rows["dev"].append({**moved, "audio": new})
    for split, items in rows.items():
        (scratch / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(items, key=lambda r: r["id"])), encoding="utf-8")
    info = {"settings": settings, "complete": True, "clips": len(rows["train"]) + len(rows["dev"]), "train": len(rows["train"]), "dev": len(rows["dev"]),
            "failed": len(failures), "first_failure": failures[0] if failures else None, "texts": len(texts),
            "minutes": round(sum(r["duration_s"] for s in rows.values() for r in s) / 60.0, 2), "seconds_to_build": round(time.time() - t0, 1),
            "backend": backend.name, "voices": [backend.describe(v) for v in options], **(info_extra or {}),
            "note": "These are clean synthetic voices; degrade them with synth_dysarthria.py build before training."}
    (scratch / "tts.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if out.exists():
        aside = out.with_name(out.name + "-previous")
        k = 2
        while aside.exists():
            aside = out.with_name(f"{out.name}-previous-{k}")
            k += 1
        out.rename(aside)
    scratch.rename(out)
    print(f"{out}: {info['clips']} clips ({info['minutes']} min) from {len(texts)} texts; {len(failures)} failed")
    return info


def merge(sources: list[Path], out: Path) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists")
    scratch = out.with_name(out.name + ".partial")
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    joined: dict[str, list[dict]] = {"train": [], "dev": []}
    seen: set[str] = set()
    for i, src in enumerate(sources):
        for split in ("train", "dev"):
            for r in read_rows(Path(src) / f"{split}.jsonl"):
                if str(r["id"]) in seen:
                    shutil.rmtree(scratch, ignore_errors=True)
                    raise ValueError(f"clip id {r['id']} is in more than one source")
                seen.add(str(r["id"]))
                new = f"{split}/s{i}-{Path(r['audio']).name}"
                dst = scratch / new
                dst.parent.mkdir(parents=True, exist_ok=True)
                s = Path(src) / r["audio"]
                if not s.is_file():
                    shutil.rmtree(scratch, ignore_errors=True)
                    raise FileNotFoundError(f"{s} is missing")
                try:
                    os.link(s, dst)
                except OSError:
                    shutil.copyfile(s, dst)
                joined[split].append({**r, "audio": new})
    for split, items in joined.items():
        (scratch / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in items), encoding="utf-8")
    info = {"complete": True, "sources": [str(s) for s in sources], "train": len(joined["train"]), "dev": len(joined["dev"])}
    (scratch / "tts.json").write_text(json.dumps({"settings": {"merge": [str(s) for s in sources], "version": VERSION}, **info}, indent=1), encoding="utf-8")
    scratch.rename(out)
    print(f"{out}: {info['train']} train and {info['dev']} dev clips from {len(sources)} sources")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backends").add_argument("--voice-dir")
    p = sub.add_parser("speak")
    p.add_argument("--backend", default="auto")
    p.add_argument("--voice-dir")
    p.add_argument("--allow-fake", action="store_true")
    p.add_argument("--text", required=True)
    p.add_argument("--variant", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("make")
    p.add_argument("--backend", default="auto")
    p.add_argument("--voice-dir")
    p.add_argument("--allow-fake", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--texts-from", nargs="*", default=[], metavar="MANIFEST")
    p.add_argument("--extra-text", nargs="*", default=[], metavar="FILE")
    p.add_argument("--exclude-from", nargs="*", default=[], metavar="MANIFEST", help="dev and test manifests whose prompts must never be spoken")
    p.add_argument("--variants", type=int, default=3)
    p.add_argument("--max-clips", type=int, default=1200)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("merge")
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "backends":
        for name, (ok, detail) in available(a.voice_dir).items():
            print(f"{'yes' if ok else 'no ':<4}{name:<8}{detail}")
        return 0
    if a.cmd == "merge":
        merge([Path(s) for s in a.sources], Path(a.out))
        return 0
    backend = get_backend(a.backend, a.voice_dir, a.allow_fake)
    if a.cmd == "speak":
        opts = backend.variants()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.wav"
            backend.synth(a.text, opts[a.variant % len(opts)], raw)
            x, rate = read_wav(raw)
        write_wav(Path(a.out), tidy(to_16k_mono(x, rate)))
        print(f"wrote {a.out} ({backend.describe(opts[a.variant % len(opts)])})")
        return 0
    if not a.texts_from and not a.extra_text:
        ap.error("give texts: --texts-from MANIFEST... and/or --extra-text FILE...")
    if not a.exclude_from:
        print("note: no --exclude-from, so test prompts reaching training through the synthesiser is not checked", file=sys.stderr)
    texts, stats = collect_texts([Path(m) for m in a.texts_from], [Path(f) for f in a.extra_text], [Path(m) for m in a.exclude_from])
    print(f"{stats['distinct_kept']} distinct texts kept; {stats['dropped_as_excluded_prompts']} dropped because they are dev/test prompts")
    make(Path(a.out), backend, texts, a.variants, a.max_clips, a.seed, a.workers, a.force, {"text_stats": stats})
    return 0


if __name__ == "__main__":
    try:
        import _cli
    except ImportError:
        _cli = None
    sys.exit(_cli.run(main, "tts_synth") if _cli else main())
