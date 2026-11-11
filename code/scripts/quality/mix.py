"""Make labelled training clips by damaging clean speech by known amounts.

Each clip starts as a clean recording: dysarthric speech from TORGO or typical
speech from LibriSpeech, in equal shares. The mixer then may

  1. put it in a room (convolve with a room impulse response)  -> label: that room's C50
  2. add another person talking                                -> label: second voice = 1
  3. add background noise at a chosen SNR                      -> label: that SNR
  4. change the gain, sometimes muffle or overdrive it          -> no label

so the label is exactly the damage applied and nobody rates clips by hand.
Because severe speech appears equally often under every label, recognising the
voice cannot help the network predict a label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from net import C50_CAP, SNR_CAP
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 16_000
MAX_LEN = 4 * SR
BASE_SNR_MIN = 25.0  # base recordings noisier than this would be mislabelled as clean


def read(path: str, start: int | None = None, frames: int = -1) -> np.ndarray:
    if start is None:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    else:
        x, sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    x = x[:, 0]
    if sr != SR:
        x = resample_poly(x, SR // 100, sr // 100).astype(np.float32)
    return x


def active_power(x: np.ndarray, frame: int = 320, hop: int = 160, rel_db: float = -35.0) -> float:
    """Mean power over the frames within 35 dB of the loudest, i.e. where there is sound."""
    if len(x) < frame:
        return float(np.mean(x ** 2) + 1e-12)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    p = np.mean(x[idx] ** 2, axis=1) + 1e-12
    pdb = 10 * np.log10(p)
    return float(p[pdb >= pdb.max() + rel_db].mean())


def rir_c50(h: np.ndarray) -> float:
    """C50: energy in the first 50 ms after the direct sound against everything later, in dB."""
    t0 = int(np.argmax(np.abs(h)))
    start, split = max(0, t0 - 16), t0 + int(0.050 * SR)
    early = np.sum(h[start:split] ** 2)
    late = np.sum(h[split:] ** 2) + 1e-12
    return float(np.clip(10 * np.log10(early / late + 1e-12), -10.0, C50_CAP))


class Mixer:
    def __init__(self, manifest_dir: Path, split: str, p_reverb: float = 0.45, p_spk2: float = 0.3,
                 p_noise: float = 0.65, rir_source: str | None = None):
        def load(name):
            return [r for r in json.loads((manifest_dir / f"{name}.json").read_text()) if r["split"] == split]

        speech = [s for s in load("speech") if s.get("base_snr", 99) >= BASE_SNR_MIN]
        self.dys = [s for s in speech if s["corpus"] == "torgo"]
        self.typ = [s for s in speech if s["corpus"] == "libri"]
        self.noise = load("noise")
        self.rirs = [r for r in load("rirs") if rir_source is None or r["source"] == rir_source]
        self.p_reverb, self.p_spk2, self.p_noise = p_reverb, p_spk2, p_noise
        self._rir_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._noise_info: dict[str, tuple[int, int]] = {}

    def speech_clip(self, rng, exclude_speaker: str | None = None):
        pool = self.typ if rng.random() < 0.5 else self.dys
        if exclude_speaker is not None and not any(p["speaker"] != exclude_speaker for p in pool):
            pool = self.typ  # a split can hold a single dysarthric speaker
        while True:
            item = pool[rng.integers(len(pool))]
            if item["speaker"] != exclude_speaker:
                break
        x = read(item["path"])
        if len(x) > MAX_LEN:
            s = rng.integers(0, len(x) - MAX_LEN)
            x = x[s:s + MAX_LEN]
        return x - x.mean(), item

    def rir(self, rng):
        p = self.rirs[rng.integers(len(self.rirs))]["path"]
        if p not in self._rir_cache:
            if len(self._rir_cache) >= 2000:  # the full set of 60,000 rooms does not fit in memory
                self._rir_cache.pop(next(iter(self._rir_cache)))
            h = read(p)[: int(1.2 * SR)]
            h = h / (np.max(np.abs(h)) + 1e-9)
            self._rir_cache[p] = (h, rir_c50(h))
        return self._rir_cache[p]

    def noise_seg(self, rng, n: int):
        item = self.noise[rng.integers(len(self.noise))]
        p = item["path"]
        if p not in self._noise_info:
            info = sf.info(p)
            self._noise_info[p] = (info.frames, info.samplerate)
        total, sr = self._noise_info[p]
        need = int(np.ceil(n * sr / SR)) + 10
        seg = read(p, start=int(rng.integers(0, total - need)), frames=need) if total > need else read(p)
        if len(seg) < n:
            seg = np.tile(seg, int(np.ceil(n / max(len(seg), 1))) + 1)
        seg = seg[:n]
        return seg - seg.mean(), item

    @staticmethod
    def reverb(x: np.ndarray, h: np.ndarray) -> np.ndarray:
        t0 = int(np.argmax(np.abs(h)))  # keep the direct sound at the original time
        y = fftconvolve(x, h)[t0: t0 + len(x)]
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        return y.astype(np.float32)

    def __call__(self, rng):
        x, item = self.speech_clip(rng)
        lab = {"snr": SNR_CAP, "c50": C50_CAP, "spk2": 0.0, "group": item["group"], "speaker": item["speaker"]}

        if self.rirs and rng.random() < self.p_reverb:
            h, c50 = self.rir(rng)
            x = self.reverb(x, h)
            lab["c50"] = c50
        p_speech = active_power(x)
        mix = x.copy()

        if rng.random() < self.p_spk2:
            y, _ = self.speech_clip(rng, exclude_speaker=item["speaker"])
            if self.rirs and rng.random() < 0.6:
                y = self.reverb(y, self.rir(rng)[0])
            sir = rng.uniform(-5.0, 15.0)  # the other voice from 5 dB louder to 15 dB quieter
            y = y * np.sqrt(p_speech / (active_power(y) * 10 ** (sir / 10)))
            off = int(rng.integers(-len(y) // 2, len(mix) // 2 + 1))
            a, b = max(0, off), min(len(mix), off + len(y))
            if b - a > SR // 4:
                mix[a:b] += y[a - off: b - off]
                lab["spk2"], lab["sir"] = 1.0, float(sir)

        if self.noise and rng.random() < self.p_noise:
            nz, nitem = self.noise_seg(rng, len(mix))
            snr = rng.uniform(-5.0, 35.0)
            mix = mix + nz * np.sqrt(p_speech / ((np.mean(nz ** 2) + 1e-12) * 10 ** (snr / 10)))
            lab["snr"], lab["noise_kind"] = float(snr), nitem["kind"]

        if rng.random() < 0.10:  # a muffled or cheap microphone
            mix = sosfilt(butter(6, rng.uniform(3000, 6500), btype="low", fs=SR, output="sos"), mix)
        peak = np.max(np.abs(mix)) + 1e-9
        if rng.random() < 0.05:
            mix = np.clip(mix / peak * 10 ** (rng.uniform(1, 12) / 20), -1.0, 1.0)
        else:
            mix = mix / peak * 10 ** (rng.uniform(-30, -1) / 20)
        return mix.astype(np.float32), lab


def collate(batch):
    import torch

    n = max(400, max(len(w) for w, _ in batch))
    wav = np.zeros((len(batch), n), dtype=np.float32)
    lens = np.zeros(len(batch), dtype=np.int64)
    for i, (w, _) in enumerate(batch):
        wav[i, : len(w)] = w
        lens[i] = len(w)
    labs = [lab for _, lab in batch]

    def col(k):
        return torch.tensor([lab[k] for lab in labs], dtype=torch.float32)

    return torch.from_numpy(wav), torch.from_numpy(lens), col("snr"), col("c50"), col("spk2"), labs
