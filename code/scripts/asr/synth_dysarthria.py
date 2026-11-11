#!/usr/bin/env python3
"""Synthetic dysarthric speech."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SR = 16_000
HOP = 160
WIN = 400
EFFECTS = ("timing", "pauses", "consonants", "flatten", "nasal", "breath", "tremor", "level")
MIN_SECONDS = 0.25
VERSION = 3


def frame_features(x: np.ndarray) -> dict[str, np.ndarray]:
    n = 1 + max(0, len(x) - WIN) // HOP
    padded = np.pad(x, (0, max(0, WIN + (n - 1) * HOP - len(x))))
    frames = np.lib.stride_tricks.sliding_window_view(padded, WIN)[::HOP][:n]
    energy = (frames ** 2).mean(1)
    diff = np.diff(frames, axis=1)
    centred = frames - frames.mean(1, keepdims=True)
    spec = np.fft.rfft(centred, 1024, axis=1)
    ac = np.fft.irfft(spec.real ** 2 + spec.imag ** 2, 1024, axis=1)
    voicing = ac[:, 40:201].max(1) / (ac[:, 0] + 1e-12)
    return {
        "db": 10.0 * np.log10(energy + 1e-12),
        "zcr": (np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=1)).sum(1) / (WIN - 1)),
        "hf": (diff ** 2).sum(1) / ((frames ** 2).sum(1) + 1e-12),
        "voicing": voicing,
    }


def speech_threshold(db: np.ndarray) -> float:
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    return float(floor + 0.2 * (peak - floor))


def clip_stats(x: np.ndarray) -> dict:
    if len(x) < WIN:
        return {"seconds": len(x) / SR, "level_db": None, "pause_frac": None, "centroid_hz": None}
    f = frame_features(x)
    db = f["db"]
    thr = speech_threshold(db)
    speech = db > thr
    out = {"seconds": round(len(x) / SR, 3), "level_db": None, "pause_frac": None, "centroid_hz": None}
    if speech.sum() < 3:
        return out
    idx = np.flatnonzero(speech)
    inner = slice(idx[0], idx[-1] + 1)
    out["level_db"] = round(float(np.median(db[speech])), 2)
    out["pause_frac"] = round(float(1.0 - speech[inner].mean()), 4)
    freqs, _, z = signal.stft(x, SR, nperseg=512, noverlap=384)
    mag = np.abs(z)
    e = 10 * np.log10((mag ** 2).sum(0) + 1e-12)
    keep = e > (np.percentile(e, 10) + 0.2 * (np.percentile(e, 95) - np.percentile(e, 10)))
    if keep.any():
        cen = (freqs[:, None] * mag[:, keep]).sum(0) / (mag[:, keep].sum(0) + 1e-9)
        out["centroid_hz"] = round(float(np.median(cen)), 1)
    return out


@dataclass
class Unit:
    start: int
    end: int
    label: str


def segment_units(x: np.ndarray, min_len: int = 40 * 16) -> list[Unit]:
    n = len(x)
    if n < 2 * WIN:
        return [Unit(0, n, "vow")]
    f = frame_features(x)
    db = gaussian_filter1d(f["db"], 1.0)
    thr = speech_threshold(db)
    dips, _ = signal.find_peaks(-db, prominence=4.0, distance=7)
    state = db > thr
    changes = np.flatnonzero(np.diff(state.astype(np.int8)) != 0) + 1
    frames = sorted(set(int(i) for i in dips) | set(int(i) for i in changes))
    bounds = [0] + [min(n, i * HOP + WIN // 2) for i in frames] + [n]
    bounds = sorted(set(bounds))
    pieces: list[tuple[int, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if pieces and b - a < min_len:
            pieces[-1] = (pieces[-1][0], b)
        else:
            pieces.append((a, b))
    if len(pieces) > 1 and pieces[0][1] - pieces[0][0] < min_len:
        pieces[1] = (pieces[0][0], pieces[1][1])
        pieces.pop(0)
    units = []
    for a, b in pieces:
        fa, fb = max(0, a // HOP), min(len(db), max(a // HOP + 1, b // HOP))
        seg = slice(fa, fb)
        speechy = float(state[seg].mean()) if fb > fa else 0.0
        if speechy < 0.5:
            label = "sil"
        else:
            voiced = f["voicing"][seg] > 0.45
            label = "vow" if voiced.mean() >= 0.45 else "con"
        units.append(Unit(a, b, label))
    return units


def wsola(x: np.ndarray, stretch: float, frame: int = WIN, tol: int = 80) -> np.ndarray:
    n = len(x)
    if n == 0 or abs(stretch - 1.0) < 0.03:
        return x.astype(np.float32, copy=True)
    n_out = max(1, int(round(n * stretch)))
    if n < 3 * frame:
        return np.interp(np.linspace(0, n - 1, n_out), np.arange(n), x).astype(np.float32)
    hs = frame // 2
    ha = hs / stretch
    xp = np.pad(x.astype(np.float64), (tol, frame + tol + 1))
    out = np.zeros(n_out + 2 * frame)
    norm = np.zeros_like(out)
    win = np.hanning(frame)
    prev = None
    k = 0
    pos = 0
    while pos < n_out:
        nominal = int(round(k * ha)) + tol
        if prev is None:
            start = nominal
        else:
            target = xp[prev + hs: prev + hs + frame]
            lo = max(nominal - tol, 0)
            hi = min(nominal + tol, len(xp) - frame)
            if hi <= lo or len(target) < frame:
                start = min(max(nominal, 0), len(xp) - frame)
            else:
                cand = np.lib.stride_tricks.sliding_window_view(xp[lo: hi + frame], frame)
                score = (cand @ target) / (np.sqrt((cand ** 2).sum(1)) + 1e-9)
                start = lo + int(np.argmax(score))
        seg = xp[start: start + frame]
        if len(seg) < frame:
            break
        w = win.copy()
        if k == 0:
            w[:hs] = 1.0
        out[pos: pos + frame] += seg * w
        norm[pos: pos + frame] += w
        prev = start
        pos += hs
        k += 1
    y = out[:n_out] / np.maximum(norm[:n_out], 0.3)
    return y.astype(np.float32)


def join(pieces: list[np.ndarray], fade: int = 64) -> np.ndarray:
    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return np.zeros(0, np.float32)
    out = pieces[0].astype(np.float32, copy=True)
    for p in pieces[1:]:
        m = min(fade, len(out), len(p))
        if m <= 1:
            out = np.concatenate([out, p])
            continue
        ramp = np.linspace(0.0, 1.0, m, dtype=np.float32)
        mixed = out[-m:] * (1.0 - ramp) + p[:m] * ramp
        out = np.concatenate([out[:-m], mixed, p[m:]])
    return out


@dataclass
class Params:
    stretch_mean: float = 1.0
    stretch_sigma: float = 0.0
    vowel_bias: float = 1.0
    pause_prob: float = 0.0
    pause_min_s: float = 0.0
    pause_max_s: float = 0.0
    con_att_db: float = 0.0
    con_smear: float = 0.0
    con_drop_prob: float = 0.0
    flatten: float = 0.0
    nasal_db: float = 0.0
    breath_snr_db: float = 99.0
    tremor_amp: float = 0.0
    tremor_pitch: float = 0.0
    tremor_hz: float = 5.0
    shimmer_db: float = 0.0
    level_sigma_db: float = 0.0
    gain_db: float = 0.0


def virtual_style(seed: int, name: str) -> dict[str, float]:
    rng = np.random.default_rng(int(hashlib.sha256(f"style:{seed}:{name}".encode()).hexdigest()[:12], 16))
    return {e: float(np.clip(rng.normal(1.0, 0.22), 0.6, 1.5)) for e in EFFECTS}


def draw_params(severity: float, rng: np.random.Generator, effects=EFFECTS, style: dict | None = None) -> Params:
    s = max(0.0, float(severity))
    st = {e: 1.0 for e in EFFECTS}
    st.update(style or {})
    on = {e: (e in effects) and s > 0 for e in EFFECTS}
    p = Params()
    if on["timing"]:
        p.stretch_mean = 1.0 + 1.3 * s * st["timing"]
        p.stretch_sigma = 0.35 * s
        p.vowel_bias = 1.0 + 0.35 * s
    if on["pauses"]:
        p.pause_prob = min(0.85, 0.65 * s * st["pauses"])
        p.pause_min_s = 0.05 + 0.03 * s
        p.pause_max_s = 0.10 + 0.55 * s
    if on["consonants"]:
        p.con_att_db = 11.0 * s * st["consonants"]
        p.con_smear = min(1.0, 0.9 * s)
        p.con_drop_prob = min(0.6, 0.30 * s) * st["consonants"]
    if on["flatten"]:
        p.flatten = min(0.85, 0.55 * s * st["flatten"])
    if on["nasal"]:
        p.nasal_db = 6.5 * s * st["nasal"]
    if on["breath"]:
        p.breath_snr_db = 40.0 - 32.0 * min(s, 1.2) * st["breath"]
    if on["tremor"]:
        p.tremor_amp = 0.30 * s * st["tremor"] * rng.uniform(0.6, 1.2)
        p.tremor_pitch = 0.030 * s * st["tremor"] * rng.uniform(0.6, 1.2)
        p.tremor_hz = float(rng.uniform(3.5, 7.0))
        p.shimmer_db = 3.0 * s
    if on["level"]:
        p.level_sigma_db = 5.0 * s
        p.gain_db = -6.0 * s * rng.uniform(0.5, 1.2)
    return p


def smooth_gain_db(n: int, sigma_db: float, rng: np.random.Generator, hop: int = 480, corr_hops: float = 3.0) -> np.ndarray:
    steps = n // hop + 3
    g = gaussian_filter1d(rng.normal(0.0, 1.0, steps), corr_hops)
    g = g / (g.std() + 1e-9) * sigma_db
    return 10.0 ** (np.interp(np.arange(n), np.arange(steps) * hop, g) / 20.0).astype(np.float32)


def cepstral_envelope(logmag: np.ndarray, keep: int) -> np.ndarray:
    cep = np.fft.irfft(logmag, axis=1)
    n = cep.shape[1]
    cep[:, keep: n - keep + 1] = 0.0
    return np.fft.rfft(cep, axis=1).real


def bump(freqs: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((freqs - centre) / width) ** 2)


def spectral_stage(y: np.ndarray, con_weight: np.ndarray, p: Params) -> np.ndarray:
    if p.flatten <= 0 and p.con_att_db <= 0 and p.nasal_db <= 0:
        return y
    freqs, _, z = signal.stft(y, SR, nperseg=512, noverlap=384, boundary="zeros", padded=True)
    mag = np.abs(z) + 1e-7
    phase = np.angle(z)
    logm = np.log(mag).T
    frame_db = 10 * np.log10((mag ** 2).sum(0) + 1e-12)
    thr = np.percentile(frame_db, 10) + 0.2 * (np.percentile(frame_db, 95) - np.percentile(frame_db, 10))
    speech = frame_db > thr
    out = logm.copy()
    if (p.flatten > 0 or p.nasal_db > 0) and speech.any():
        env = cepstral_envelope(logm, 30)
        if p.flatten > 0 and speech.sum() > 3:
            out[speech] += p.flatten * (env[speech].mean(0) - env[speech])
        if p.nasal_db > 0:
            curve = (bump(freqs, 280.0, 110.0) - 0.7 * bump(freqs, 1150.0, 380.0)
                     - 0.8 * np.clip((freqs - 3500.0) / 4500.0, 0, 1)) * p.nasal_db * math.log(10) / 20.0
            out[speech] += curve[None, :]
        before = 0.5 * np.log((np.exp(2 * logm[speech])).sum(1))
        after = 0.5 * np.log((np.exp(2 * out[speech])).sum(1))
        out[speech] += (before - after)[:, None]
    if p.con_att_db > 0:
        w = np.interp(np.arange(logm.shape[0]) * 128 / SR, np.arange(len(con_weight)) * HOP / SR, con_weight)[:, None]
        smooth = cepstral_envelope(out, 8)
        att = -p.con_att_db * math.log(10) / 20.0
        ramp = -np.clip((freqs - 3000.0) / 5000.0, 0, 1) * (p.con_att_db * 0.6) * math.log(10) / 20.0
        target = (1 - p.con_smear) * out + p.con_smear * smooth + att + ramp[None, :]
        out = (1 - w) * out + w * target
    y2 = signal.istft(np.exp(out).T * np.exp(1j * phase), SR, nperseg=512, noverlap=384, boundary=True)[1]
    y2 = np.asarray(y2, dtype=np.float32)
    if len(y2) < len(y):
        y2 = np.pad(y2, (0, len(y) - len(y2)))
    return y2[: len(y)]


def breath_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if snr_db > 40:
        return y
    sos = signal.butter(2, [400.0, 5500.0], btype="band", fs=SR, output="sos")
    noise = signal.sosfilt(sos, rng.standard_normal(len(y))).astype(np.float32)
    env = np.convolve(np.abs(y), np.ones(640) / 640.0, mode="same").astype(np.float32)
    noise = noise / (noise.std() + 1e-9) * env * 10.0 ** (-snr_db / 20.0)
    return y + noise


def tremor_warp(y: np.ndarray, p: Params, rng: np.random.Generator) -> np.ndarray:
    if p.tremor_amp <= 0 and p.tremor_pitch <= 0:
        return y
    n = len(y)
    t = np.arange(n) / SR
    phase = rng.uniform(0, 2 * math.pi)
    w = 2 * math.pi * p.tremor_hz
    amp = 1.0 + p.tremor_amp * np.sin(w * t + phase)
    warped = t + (p.tremor_pitch / w) * np.sin(w * t + phase + 1.0)
    z = np.interp(warped * SR, np.arange(n), y)
    z = z * amp
    if p.shimmer_db > 0:
        z = z * smooth_gain_db(n, p.shimmer_db, rng, hop=480, corr_hops=0.7)
    return z.astype(np.float32)


def noise_floor_rms(x: np.ndarray) -> float:
    db = frame_features(x)["db"] if len(x) >= WIN else np.array([-60.0])
    return float(10 ** (np.percentile(db, 10) / 20.0))


def degrade(x: np.ndarray, severity: float, seed: int, effects=EFFECTS, style: dict | None = None,
            target_seconds: float | None = None) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float32)
    meta: dict = {"severity": float(severity), "seed": int(seed), "version": VERSION, "in_seconds": round(len(x) / SR, 3)}
    if len(x) < MIN_SECONDS * SR or not np.isfinite(x).all() or float(np.abs(x).max()) < 1e-4:
        meta["skipped"] = "too short, silent or not finite"
        return x.copy(), meta
    if severity <= 0:
        meta["skipped"] = "severity 0 changes nothing"
        return x.copy(), meta
    rng = np.random.default_rng(seed)
    p = draw_params(severity, rng, effects, style)
    units = segment_units(x)
    meta["units"] = len(units)
    floor = noise_floor_rms(x)
    audio: list[np.ndarray] = []
    marks: list[np.ndarray] = []
    for u in units:
        seg = x[u.start: u.end]
        if u.label == "sil":
            audio.append(seg)
            marks.append(np.zeros(len(seg), np.float32))
            continue
        if u.label == "con" and p.con_drop_prob > 0 and rng.random() < p.con_drop_prob and (u.end - u.start) < 0.12 * SR:
            seg = seg * 10 ** (-rng.uniform(14, 26) / 20.0)
        stretch = p.stretch_mean * (p.vowel_bias if u.label == "vow" else 1.0 / (0.5 * (1 + p.vowel_bias)))
        stretch *= float(np.exp(rng.normal(0.0, p.stretch_sigma)))
        stretch = float(np.clip(stretch, 1.0 if "timing" in effects else 0.97, 4.0))
        piece = wsola(seg, stretch)
        audio.append(piece)
        marks.append(np.full(len(piece), 1.0 if u.label == "con" else 0.0, np.float32))
        if p.pause_prob > 0 and rng.random() < p.pause_prob:
            m = int(rng.uniform(p.pause_min_s, p.pause_max_s) * SR)
            audio.append((rng.standard_normal(m) * floor).astype(np.float32))
            marks.append(np.zeros(m, np.float32))
    y = join(audio)
    mask = join(marks)
    y = spectral_stage(y, frame_mask(mask), p)
    y = breath_noise(y, p.breath_snr_db, rng)
    y = tremor_warp(y, p, rng)
    if p.level_sigma_db > 0:
        y = y * smooth_gain_db(len(y), p.level_sigma_db, rng)
    if target_seconds and target_seconds > 0:
        ratio = float(np.clip(target_seconds * SR / max(1, len(y)), 0.6, 2.5))
        y = wsola(y, ratio)
        meta["rate_ratio"] = round(ratio, 3)
    y = y * (10 ** (p.gain_db / 20.0))
    peak = float(np.abs(y).max())
    if peak > 0.99:
        y = y * (0.99 / peak)
    y = np.nan_to_num(y).astype(np.float32)
    meta.update({"out_seconds": round(len(y) / SR, 3), "stretch_mean": round(p.stretch_mean, 3), "gain_db": round(p.gain_db, 2)})
    return y, meta


def frame_mask(mask: np.ndarray) -> np.ndarray:
    if len(mask) < HOP:
        return np.zeros(1, np.float32)
    frames = max(1, len(mask) // HOP)
    pos = np.linspace(0, len(mask) - 1, frames)
    return gaussian_filter1d(np.interp(pos, np.arange(len(mask)), mask).astype(np.float32), 1.5)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}, line {number}: not valid JSON ({exc})") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    from willspeak import audio as A

    data, sr = sf.read(str(path), dtype="float32")
    return A.to_mono_16k(data, sr).astype(np.float32)


def write_audio(path: Path, x: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prompt_key(text: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))


def letters(text: str) -> int:
    return sum(c.isalpha() for c in str(text))


def stable_int(*parts) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def profile_export(export: Path, split: str = "train", audio_limit: int = 120) -> dict:
    rows = read_jsonl(export / f"{split}.jsonl")
    per_kind: dict[str, list[float]] = {}
    for r in rows:
        n = letters(r.get("text", ""))
        if n and float(r.get("duration_s", 0)) > 0:
            per_kind.setdefault(r.get("kind", "unknown"), []).append(float(r["duration_s"]) / n)
    out = {"export": str(export), "split": split, "clips": len(rows),
           "sec_per_char": {k: round(float(np.median(v)), 4) for k, v in per_kind.items() if v},
           "counts": {k: len(v) for k, v in per_kind.items()}, "audio": None}
    stats = []
    for r in sorted(rows, key=lambda r: stable_int(r.get("id")))[:audio_limit]:
        path = export / r["audio"]
        if path.is_file():
            try:
                stats.append(clip_stats(read_audio(path)))
            except Exception:  # noqa: BLE001
                continue
    if stats:
        def med(key):
            vals = [s[key] for s in stats if s.get(key) is not None]
            return round(float(np.median(vals)), 3) if vals else None
        out["audio"] = {"clips": len(stats), "level_db": med("level_db"), "pause_frac": med("pause_frac"),
                        "centroid_hz": med("centroid_hz")}
    return out


def pooled_profile(exports: list[Path], split: str = "train") -> dict:
    per_kind: dict[str, list[float]] = {}
    for export in exports:
        for r in read_jsonl(Path(export) / f"{split}.jsonl"):
            n = letters(r.get("text", ""))
            if n and float(r.get("duration_s", 0)) > 0:
                per_kind.setdefault(r.get("kind", "unknown"), []).append(float(r["duration_s"]) / n)
    return {"exports": [str(e) for e in exports], "split": split,
            "sec_per_char": {k: round(float(np.median(v)), 4) for k, v in per_kind.items() if v},
            "counts": {k: len(v) for k, v in per_kind.items()}}


def rate_ratio(target: dict, controls: dict) -> dict[str, float]:
    out = {}
    for kind, value in target.get("sec_per_char", {}).items():
        base = controls.get("sec_per_char", {}).get(kind)
        if base:
            out[kind] = round(value / base, 3)
    return out


def _job(task: dict):
    try:
        x = read_audio(Path(task["src"]))
        y, meta = degrade(x, task["severity"], task["seed"], tuple(task["effects"]), task["style"], task["target_seconds"])
        if "skipped" in meta:
            return task, None, meta
        if len(y) / SR > task["max_seconds"]:
            meta["skipped"] = "longer than the training limit"
            return task, None, meta
        write_audio(Path(task["dst"]), y)
        meta["duration_s"] = round(len(y) / SR, 3)
        return task, meta, meta
    except Exception as exc:  # noqa: BLE001
        return task, None, {"skipped": f"{type(exc).__name__}: {exc}"}


def _run_jobs(tasks: list[dict], workers: int):
    if workers > 1 and len(tasks) > 8:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(_job, tasks, chunksize=8)
    else:
        for t in tasks:
            yield _job(t)


def build_synthetic(controls: Path, target: Path, out: Path, severity: float, scale: float, seed: int = 17,
                    splits=("train", "dev"), variants: int = 3, effects=EFFECTS, only_synthetic: bool = False,
                    match_rate: bool = True, workers: int = 0, max_seconds: float = 24.0, force: bool = False) -> dict:
    controls, target, out = Path(controls), Path(target), Path(out)
    settings = {"severity": severity, "scale": scale, "seed": seed, "splits": list(splits), "variants": variants,
                "effects": list(effects), "only_synthetic": only_synthetic, "match_rate": match_rate,
                "controls": str(controls), "target": str(target), "version": VERSION, "code": code_fingerprint()}
    marker = out / "synthetic.json"
    if marker.is_file() and not force:
        try:
            old = json.loads(marker.read_text(encoding="utf-8"))
        except ValueError:
            old = {}
        if old.get("settings") == settings and old.get("complete"):
            print(f"{out}: already built with these settings ({old.get('synthetic_clips')} synthetic clips); reusing")
            return old
        raise FileExistsError(f"{out} exists with other settings; use a new --out or --force (the old folder is moved aside)")
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} exists and is not a finished synthetic build; use a new --out or --force")
    for need in (controls, target):
        for split in ("train", "dev", "test"):
            if need is controls and split == "test":
                continue
            if not (need / f"{split}.jsonl").is_file():
                raise FileNotFoundError(f"{need / (split + '.jsonl')} is missing (run torgo_manifest.py --export first)")
    target_rows = {s: read_jsonl(target / f"{s}.jsonl") for s in ("train", "dev", "test")}
    missing = [r["audio"] for s, rows in target_rows.items() for r in rows if not (target / r["audio"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files of {target} are missing, e.g. {missing[0]}; "
                                "python lab.py doctor --fix can rebuild them from the TORGO download")
    pool = [r for s in splits for r in read_jsonl(controls / f"{s}.jsonl")]
    held_out = {prompt_key(r.get("prompt_text") or r.get("text")) for s in ("dev", "test") for r in target_rows[s]}
    leak = [r for r in pool if prompt_key(r.get("prompt_text") or r.get("text")) in held_out]
    if leak and "dev" not in splits:
        pass
    pool = [r for r in pool if prompt_key(r.get("prompt_text") or r.get("text")) not in held_out]
    pool = [r for r in pool if (controls / r["audio"]).is_file()]
    if not pool:
        raise ValueError("no usable control recordings (all missing, or all prompts shared with the target's dev/test sets)")
    prof_target = profile_export(target, "train", audio_limit=0)
    prof_ctrl = profile_export(controls, "train", audio_limit=0)
    ratios = rate_ratio(prof_target, prof_ctrl) if match_rate else {}
    want = max(1, int(round(scale * len(target_rows["train"]))))
    share: dict[str, float] = {}
    for r in target_rows["train"]:
        share[r.get("kind", "unknown")] = share.get(r.get("kind", "unknown"), 0) + 1
    chosen: list[tuple[dict, int]] = []
    by_kind: dict[str, list[dict]] = {}
    for r in pool:
        by_kind.setdefault(r.get("kind", "unknown"), []).append(r)
    for kind, count in share.items():
        n = int(round(want * count / len(target_rows["train"])))
        rows = by_kind.get(kind) or pool
        order = sorted(((r, v) for r in rows for v in range(variants)), key=lambda rv: stable_int(seed, rv[0]["id"], rv[1]))
        chosen += order[:n]
    scratch = out.with_name(out.name + ".partial")
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    tasks = []
    for row, variant in chosen:
        kind = row.get("kind", "unknown")
        ctrl_seconds = float(row.get("duration_s") or 0)
        cid = f"synth-v{variant}-{row['id']}"
        rng = np.random.default_rng(stable_int(seed, "len", cid))
        target_seconds = None
        if kind in ratios and ctrl_seconds > 0:
            target_seconds = ctrl_seconds * ratios[kind] * float(np.exp(rng.normal(0, 0.12)))
        tasks.append({"src": str(controls / row["audio"]), "dst": str(scratch / "train" / f"{cid}.wav"), "id": cid,
                      "row": row, "variant": variant, "severity": severity, "seed": stable_int(seed, cid),
                      "effects": list(effects), "style": virtual_style(seed, f"{row.get('speaker')}-v{variant}"),
                      "target_seconds": target_seconds, "max_seconds": max_seconds})
    workers = workers or min(4, os.cpu_count() or 1)
    made, skipped, durations, ratios_seen = [], 0, [], []
    t0 = time.time()
    for task, ok, meta in _run_jobs(tasks, workers):
        if ok is None:
            skipped += 1
            continue
        src = task["row"]
        made.append({**src, "id": task["id"], "audio": f"train/{task['id']}.wav", "duration_s": meta["duration_s"],
                     "speaker": f"SYN-{src.get('speaker')}-v{task['variant']}", "severity": "synthetic",
                     "session": "synthetic", "condition": "synthetic"})
        durations.append(meta["duration_s"])
        if "rate_ratio" in meta:
            ratios_seen.append(meta["rate_ratio"])
    if tasks and skipped > 0.05 * len(tasks):
        shutil.rmtree(scratch, ignore_errors=True)
        raise RuntimeError(f"{skipped} of {len(tasks)} synthetic clips failed; not writing a partial export")
    for split, rows in target_rows.items():
        for r in rows:
            link_or_copy(target / r["audio"], scratch / r["audio"])
        write_jsonl(scratch / f"{split}.jsonl", ([] if (only_synthetic and split == "train") else rows) + (made if split == "train" else []))
    info = {"settings": settings, "complete": True, "synthetic_clips": len(made), "skipped": skipped,
            "wanted": want, "leaked_prompts_dropped": len(leak), "rate_ratio_by_kind": ratios,
            "target_profile": prof_target["sec_per_char"], "controls_profile": prof_ctrl["sec_per_char"],
            "synthetic_minutes": round(sum(durations) / 60.0, 2), "median_rate_adjust": (round(float(np.median(ratios_seen)), 3) if ratios_seen else None),
            "own_training_clips": len(target_rows["train"]), "built_seconds": round(time.time() - t0, 1),
            "note": "Synthetic speech transformations; severity and use in training are experimental settings."}
    (scratch / "synthetic.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if out.exists():
        aside = out.with_name(out.name + "-previous")
        k = 2
        while aside.exists():
            aside = out.with_name(f"{out.name}-previous-{k}")
            k += 1
        out.rename(aside)
    scratch.rename(out)
    print(f"{out}: {len(made)} synthetic clips ({info['synthetic_minutes']} min) plus the speaker's own {len(target_rows['train'])} "
          f"train clips; {skipped} skipped; {info['built_seconds']} s")
    return info


def build_validation(controls: Path, out: Path, severities: list[float], seed: int = 17, limit: int = 150,
                     split: str = "test", effects=EFFECTS, target: list[Path] | Path | None = None, workers: int = 0) -> dict:
    controls, out = Path(controls), Path(out)
    rows = read_jsonl(controls / f"{split}.jsonl")
    rows = [r for r in rows if (controls / r["audio"]).is_file()]
    if not rows:
        raise FileNotFoundError(f"no audio found for {controls}/{split}.jsonl")
    rows = sorted(rows, key=lambda r: stable_int(seed, r["id"]))[:limit]
    ratios = {}
    targets = [] if target is None else [Path(t) for t in (target if isinstance(target, (list, tuple)) else [target])]
    targets = [t for t in targets if (t / "train.jsonl").is_file()]
    if targets:
        ratios = rate_ratio(pooled_profile(targets), profile_export(controls, "train", 0))
    index = {"severities": [], "limit": limit, "split": split, "seed": seed, "rate_ratio_by_kind": ratios, "version": VERSION}
    clean = out / "clean"
    if not (clean / "done.json").is_file():
        if clean.exists():
            shutil.rmtree(clean, ignore_errors=True)
        kept = []
        for r in rows:
            link_or_copy(controls / r["audio"], clean / "test" / Path(r["audio"]).name)
            kept.append({**r, "audio": f"test/{Path(r['audio']).name}"})
        write_jsonl(clean / "test.jsonl", kept)
        (clean / "done.json").write_text(json.dumps({"clips": len(kept), "severity": 0, "version": VERSION}), encoding="utf-8")
    index["clean"] = {"folder": "clean", "clips": len(read_jsonl(clean / "test.jsonl"))}
    workers = workers or min(4, os.cpu_count() or 1)
    for sev in severities:
        folder = out / f"s{sev:.2f}"
        done = folder / "done.json"
        if done.is_file():
            index["severities"].append({"severity": sev, "folder": folder.name, "clips": json.loads(done.read_text())["clips"]})
            continue
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        tasks, made = [], []
        for r in rows:
            cid = f"v{sev:.2f}-{r['id']}"
            seconds = None
            if r.get("kind") in ratios and r.get("duration_s"):
                seconds = float(r["duration_s"]) * ratios[r["kind"]]
            tasks.append({"src": str(controls / r["audio"]), "dst": str(folder / "test" / f"{cid}.wav"), "id": cid, "row": r,
                          "variant": 0, "severity": sev, "seed": stable_int(seed, cid), "effects": list(effects),
                          "style": virtual_style(seed, f"{r.get('speaker')}-v0"), "target_seconds": seconds, "max_seconds": 60.0})
        for task, ok, meta in _run_jobs(tasks, workers):
            if ok is None:
                continue
            r = task["row"]
            made.append({**r, "id": task["id"], "audio": f"test/{task['id']}.wav", "duration_s": meta["duration_s"]})
        write_jsonl(folder / "test.jsonl", made)
        done.write_text(json.dumps({"clips": len(made), "severity": sev, "version": VERSION}), encoding="utf-8")
        index["severities"].append({"severity": sev, "folder": folder.name, "clips": len(made)})
        print(f"  severity {sev:.2f}: {len(made)} clips -> {folder}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def calibrate(table: list[tuple[float, float]], target_wer: float) -> dict:
    pts = sorted(table)
    if not pts:
        raise ValueError("empty table")
    xs = [p[0] for p in pts]
    ys = list(np.maximum.accumulate([p[1] for p in pts]))
    note = "interpolated"
    if target_wer <= ys[0]:
        sev, note = xs[0], "target at or below the mildest severity tried"
    elif target_wer >= ys[-1]:
        sev, note = xs[-1], "target above the harshest severity tried: raise --severities"
    else:
        sev = float(np.interp(target_wer, ys, xs))
    return {"severity": round(float(sev), 3), "target_wer": target_wer, "note": note,
            "curve": [{"severity": x, "wer": round(float(y), 4)} for x, y in zip(xs, ys)]}


def summarize(results: Path, real: dict[str, Path], out: Path) -> dict:
    def wer_of(path: Path):
        data = json.loads((path / "results.json").read_text(encoding="utf-8"))
        return float(data["wer"]), int(data["word"]["n_ref"])
    rows = []
    for folder in sorted(results.glob("s[0-9]*")):
        if (folder / "results.json").is_file():
            wer, n = wer_of(folder)
            rows.append({"severity": float(folder.name[1:]), "wer": round(wer, 4), "n_ref": n})
    clean = None
    if (results / "clean" / "results.json").is_file():
        wer, n = wer_of(results / "clean")
        clean = {"wer": round(wer, 4), "n_ref": n}
    real_rows = {}
    for name, path in real.items():
        if (Path(path) / "results.json").is_file():
            wer, n = wer_of(Path(path))
            real_rows[name] = {"wer": round(wer, 4), "n_ref": n}
    summary = {"synthetic": rows, "clean_controls": clean, "real_stock": real_rows, "calibration": {}}
    if rows:
        for name, info in real_rows.items():
            summary["calibration"][name] = calibrate([(r["severity"], r["wer"]) for r in rows], info["wer"])
        wers = [v["wer"] for v in real_rows.values()]
        if wers:
            summary["calibration"]["pooled"] = calibrate([(r["severity"], r["wer"]) for r in rows], float(np.mean(wers)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def toy_speech(seconds: float = 2.4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    t_v = np.arange(int(0.22 * SR)) / SR
    for i in range(int(seconds / 0.38)):
        f0 = 115.0 + 10 * rng.standard_normal()
        v = sum(np.sin(2 * np.pi * f0 * k * t_v) / k for k in range(1, 30))
        sos = signal.butter(2, [500 + 150 * (i % 3), 900 + 200 * (i % 3)], "band", fs=SR, output="sos")
        v = signal.sosfilt(sos, v)
        v = v / (np.abs(v).max() + 1e-9) * 0.4 * np.hanning(len(v))
        c = signal.sosfilt(signal.butter(2, [3500, 7000], "band", fs=SR, output="sos"), rng.standard_normal(int(0.07 * SR)))
        c = c / (np.abs(c).max() + 1e-9) * 0.12 * np.hanning(len(c))
        parts += [c, v, np.zeros(int(0.06 * SR))]
    x = np.concatenate(parts)
    return (x + 0.0015 * rng.standard_normal(len(x))).astype(np.float32)


def selftest(verbose: bool = True) -> list[str]:
    problems: list[str] = []
    x = toy_speech()
    base = clip_stats(x)
    for sev in (0.0, 0.3, 0.7, 1.0):
        y, meta = degrade(x, sev, 5)
        y2, _ = degrade(x, sev, 5)
        if not np.isfinite(y).all():
            problems.append(f"severity {sev}: not finite")
        if float(np.abs(y).max()) > 1.0:
            problems.append(f"severity {sev}: clipped")
        if not np.array_equal(y, y2):
            problems.append(f"severity {sev}: not repeatable with the same seed")
        if sev >= 0.7 and len(y) < 1.4 * len(x):
            problems.append(f"severity {sev}: only {len(y) / len(x):.2f}x as long")
        if verbose:
            print(f"  severity {sev:.1f}: {len(x) / SR:.2f} s -> {len(y) / SR:.2f} s, {clip_stats(y)}")
    for effect in EFFECTS:
        y, _ = degrade(x, 0.8, 7, effects=(effect,))
        if not np.isfinite(y).all() or len(y) == 0:
            problems.append(f"effect {effect} alone failed")
    short, meta = degrade(x[:1000], 0.8, 1)
    if "skipped" not in meta:
        problems.append("a 60 ms clip should be returned unchanged")
    silent, meta = degrade(np.zeros(16000, np.float32), 0.8, 1)
    if "skipped" not in meta or float(np.abs(silent).max()) != 0:
        problems.append("silence should be returned unchanged")
    z, _ = degrade(x, 0.8, 3, target_seconds=7.0)
    if abs(len(z) / SR - 7.0) > 0.6:
        problems.append(f"target_seconds 7.0 gave {len(z) / SR:.2f}")
    if verbose:
        print(f"  original: {base}")
    return problems


def parse_effects(text: str | None) -> tuple[str, ...]:
    if not text or text == "all":
        return EFFECTS
    names = tuple(t.strip() for t in text.replace(",", " ").split() if t.strip())
    unknown = [n for n in names if n not in EFFECTS]
    if unknown:
        raise SystemExit(f"unknown effect {unknown}; choose from {', '.join(EFFECTS)}")
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="seconds per letter by kind (and clip statistics when the audio is there)")
    p.add_argument("--export", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out")

    p = sub.add_parser("preview", help="degrade one recording")
    p.add_argument("--wav", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--severity", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--effects")

    p = sub.add_parser("build", help="an export folder for train.py: the speaker's own data plus synthetic clips")
    p.add_argument("--controls", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--severity", type=float, default=0.7)
    p.add_argument("--calibration", help="a summarize/calibrate json; its severity for the target speaker is used instead of --severity")
    p.add_argument("--speaker", help="with --calibration: which speaker's severity (default: pooled)")
    p.add_argument("--scale", type=float, default=2.0, help="synthetic clips per own training clip")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--variants", type=int, default=3, help="differently degraded copies of each control clip allowed")
    p.add_argument("--effects")
    p.add_argument("--only-synthetic", action="store_true", help="train.jsonl holds only synthetic clips (for pre-training)")
    p.add_argument("--no-rate-match", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("validate-set", help="degraded copies of the controls' test recordings, one folder per severity")
    p.add_argument("--controls", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--severities", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9, 1.1])
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--target", nargs="+", help="speakers' exports; their pooled pace sets each clip's length")
    p.add_argument("--effects")
    p.add_argument("--workers", type=int, default=0)

    p = sub.add_parser("summarize", help="WER per severity from scored folders, and the severity that matches each real speaker")
    p.add_argument("--results", required=True)
    p.add_argument("--real", nargs="*", default=[], metavar="NAME=FOLDER")
    p.add_argument("--out", required=True)

    p = sub.add_parser("selftest", help="property checks on a toy signal")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        problems = selftest()
        print("selftest: " + ("ok" if not problems else "FAILED\n  " + "\n  ".join(problems)))
        return 1 if problems else 0
    if a.cmd == "profile":
        prof = profile_export(Path(a.export), a.split)
        print(json.dumps(prof, indent=2))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(prof, indent=2), encoding="utf-8")
        return 0
    if a.cmd == "preview":
        x = read_audio(Path(a.wav))
        y, meta = degrade(x, a.severity, a.seed, parse_effects(a.effects))
        write_audio(Path(a.out), y)
        print(json.dumps({"before": clip_stats(x), "after": clip_stats(y), "meta": meta}, indent=2))
        return 0
    if a.cmd == "build":
        severity = a.severity
        if a.calibration:
            cal = json.loads(Path(a.calibration).read_text(encoding="utf-8"))["calibration"]
            key = a.speaker if a.speaker in cal else "pooled"
            if key in cal:
                severity = float(cal[key]["severity"])
                print(f"severity {severity} from {a.calibration} ({key})")
        build_synthetic(Path(a.controls), Path(a.target), Path(a.out), severity, a.scale, a.seed, variants=a.variants,
                        effects=parse_effects(a.effects), only_synthetic=a.only_synthetic, match_rate=not a.no_rate_match,
                        workers=a.workers, force=a.force)
        return 0
    if a.cmd == "validate-set":
        build_validation(Path(a.controls), Path(a.out), a.severities, a.seed, a.limit, effects=parse_effects(a.effects),
                         target=[Path(t) for t in a.target] if a.target else None, workers=a.workers)
        return 0
    if a.cmd == "summarize":
        real = {}
        for item in a.real:
            name, _, folder = item.partition("=")
            real[name] = Path(folder)
        summary = summarize(Path(a.results), real, Path(a.out))
        print(json.dumps({k: summary[k] for k in ("synthetic", "clean_controls", "real_stock")}, indent=2))
        for name, cal in summary["calibration"].items():
            print(f"{name}: severity {cal['severity']} for stock WER {cal['target_wer']:.3f} ({cal['note']})")
        return 0
    return 2


if __name__ == "__main__":
    try:
        import _cli
    except ImportError:
        _cli = None
    sys.exit(_cli.run(main, "synth_dysarthria") if _cli else main())
