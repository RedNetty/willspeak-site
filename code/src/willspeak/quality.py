"""Recording-quality gate for the Record page.

A small convolutional network (about 54,000 weights) estimates three things
about a clip that the level checks in audio.py cannot measure directly:

  snr_db              background noise: how far the speech is above it (higher is cleaner)
  c50_db              echo: early direct sound against late reflections (higher is drier)
  second_voice_prob   chance that another person is talking in the clip

It judges the recording, not the speech. It never decodes words, and it was
trained with dysarthric and typical speech in equal shares under every label,
so the voice itself carries no information about the verdict.

Inference is plain NumPy, so the server needs no extra runtime. Weights and
the decision thresholds live in models/quality_gate.npz; training and the
tests behind the thresholds are in scripts/quality/ and docs/design.md,
**Recording quality gate**.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SR = 16_000
N_FFT = 400   # 25 ms window
HOP = 160     # 10 ms step
MIN_SAMPLES = N_FFT

MODEL_PATH = Path(__file__).resolve().parent / "models" / "quality_gate.npz"

MESSAGES = {
    "noisy": "Too much background noise. Turn off the TV or fan, or move somewhere quieter.",
    "echo": "Too much echo. Move the microphone closer to the speaker.",
    "second_voice": "Another voice is in the recording. Record when no one else is talking.",
}


@dataclass
class Verdict:
    snr_db: float
    c50_db: float
    second_voice_prob: float
    keep: bool
    reasons: list[str] = field(default_factory=list)

    def dict(self) -> dict:
        d = asdict(self)
        d["messages"] = [MESSAGES[r] for r in self.reasons]
        return d


def _hann_dft() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(N_FFT)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * t / N_FFT)
    k = np.arange(N_FFT // 2 + 1)[:, None]
    cos = (np.cos(2 * np.pi * k * t / N_FFT) * win).astype(np.float32)
    sin = (-np.sin(2 * np.pi * k * t / N_FFT) * win).astype(np.float32)
    return cos, sin


class QualityGate:
    """Loads the weights once and scores one clip at a time."""

    def __init__(self, path: Path | str = MODEL_PATH):
        z = np.load(path)
        self.fb = z["mel_fb"].astype(np.float32)                      # (64, 201)
        self.convs = []
        i = 0
        while f"conv{i}.weight" in z:
            bn_scale = z[f"norm{i}.weight"] / np.sqrt(z[f"norm{i}.running_var"] + 1e-5)
            bn_shift = z[f"norm{i}.bias"] - z[f"norm{i}.running_mean"] * bn_scale
            self.convs.append((z[f"conv{i}.weight"].astype(np.float32), z[f"conv{i}.bias"].astype(np.float32),
                               int(z[f"conv{i}.dilation"]), bn_scale.astype(np.float32),
                               bn_shift.astype(np.float32)))
            i += 1
        self.h1w, self.h1b = z["head1.weight"], z["head1.bias"]
        self.h2w, self.h2b = z["head2.weight"], z["head2.bias"]
        self.thresholds = {
            "snr_min_db": float(z["snr_min_db"]),
            "c50_min_db": float(z["c50_min_db"]),
            "spk2_max": float(z["spk2_max"]),
        }
        self.cos, self.sin = _hann_dft()

    def _log_mel(self, x: np.ndarray) -> np.ndarray:
        n = 1 + (len(x) - N_FFT) // HOP
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
        frames = x[idx]                                               # (frames, 400)
        power = (frames @ self.cos.T) ** 2 + (frames @ self.sin.T) ** 2  # (frames, 201)
        return np.log(power @ self.fb.T + 1e-8).T                     # (64, frames)

    def estimate(self, x: np.ndarray) -> tuple[float, float, float]:
        x = np.asarray(x, dtype=np.float32).ravel()
        if len(x) < MIN_SAMPLES:
            x = np.pad(x, (0, MIN_SAMPLES - len(x)))
        h = self._log_mel(x)
        h = h - h.mean()  # remove overall loudness; the model sees spectral shape only
        for w, b, dil, scale, shift in self.convs:
            k = w.shape[2]
            pad = (k // 2) * dil
            hp = np.pad(h, ((0, 0), (pad, pad)))
            t = h.shape[1]
            out = np.zeros((w.shape[0], t), dtype=np.float32)
            for j in range(k):
                out += w[:, :, j] @ hp[:, j * dil: j * dil + t]
            out += b[:, None]
            h = np.maximum(out * scale[:, None] + shift[:, None], 0.0)
        pooled = np.concatenate([h.mean(axis=1), np.sqrt(h.var(axis=1) + 1e-6)])
        z = np.maximum(self.h1w @ pooled + self.h1b, 0.0)
        o = self.h2w @ z + self.h2b
        return float(o[0] * 10 + 20), float(o[1] * 10 + 20), float(1 / (1 + np.exp(-o[2])))

    def __call__(self, x: np.ndarray) -> Verdict:
        snr, c50, spk2 = self.estimate(x)
        th = self.thresholds
        reasons = []
        if snr < th["snr_min_db"]:
            reasons.append("noisy")
        if c50 < th["c50_min_db"]:
            reasons.append("echo")
        if spk2 > th["spk2_max"]:
            reasons.append("second_voice")
        return Verdict(round(snr, 1), round(c50, 1), round(spk2, 3), not reasons, reasons)


_gate: QualityGate | None = None
_gate_failed = False
_lock = threading.Lock()


def gate() -> QualityGate | None:
    """The shared gate, or None when it is switched off or its weights are missing.

    WILLSPEAK_QUALITY_GATE=0 switches it off; the Record page then relies on the
    level checks alone.
    """
    global _gate, _gate_failed
    if os.environ.get("WILLSPEAK_QUALITY_GATE", "1") == "0":
        return None
    with _lock:
        if _gate is None and not _gate_failed:
            try:
                _gate = QualityGate()
            except (OSError, KeyError, ValueError) as e:
                _gate_failed = True
                print(f"[willspeak] quality gate off: could not load {MODEL_PATH.name}: {e}")
    return _gate
