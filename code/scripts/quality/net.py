"""The recording-quality network, in PyTorch for training.

src/willspeak/quality.py runs the same network in NumPy on the device; export
in gate.py copies these weights into src/willspeak/models/quality_gate.npz and
checks both versions agree.

Input: raw 16 kHz audio (batch, samples) and each clip's length in samples.
Output per clip: estimated SNR (dB), estimated C50 (dB), second-voice logit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR = 16_000
N_FFT = 400   # 25 ms window
HOP = 160     # 10 ms step
N_MELS = 64

# Labels are capped here: a clip with no added noise is labelled 40 dB SNR,
# a clip with no room added is labelled 35 dB C50.
SNR_CAP = 40.0
C50_CAP = 35.0


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR,
                   fmin: float = 40.0, fmax: float = 7800.0) -> np.ndarray:
    """Triangular filters on the HTK mel scale, shape (n_mels, n_fft // 2 + 1)."""
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.maximum(0, np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)))
    return fb


class LogMel(nn.Module):
    """Short-time Fourier transform as a fixed convolution, then mel bands and log. Nothing here is learned."""

    def __init__(self):
        super().__init__()
        t = np.arange(N_FFT)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * t / N_FFT)
        k = np.arange(N_FFT // 2 + 1)[:, None]
        cos = np.cos(2 * np.pi * k * t / N_FFT) * win
        sin = -np.sin(2 * np.pi * k * t / N_FFT) * win
        self.register_buffer("dft", torch.from_numpy(np.concatenate([cos, sin])[:, None, :].astype(np.float32)))
        self.register_buffer("fb", torch.from_numpy(mel_filterbank()))

    def forward(self, wav):
        spec = F.conv1d(wav.unsqueeze(1), self.dft, stride=HOP)
        n = spec.shape[1] // 2
        power = spec[:, :n] ** 2 + spec[:, n:] ** 2
        return torch.log(torch.matmul(self.fb, power) + 1e-8)


class QualityNet(nn.Module):
    def __init__(self, ch: int = 48):
        super().__init__()
        self.front = LogMel()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_ch = N_MELS
        for dil in (1, 2, 4, 8):  # about 0.6 s of context by the last layer
            self.convs.append(nn.Conv1d(in_ch, ch, kernel_size=5, dilation=dil, padding=2 * dil))
            self.norms.append(nn.BatchNorm1d(ch))
            in_ch = ch
        self.head = nn.Sequential(nn.Linear(2 * ch, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, wav, lengths):
        x = self.front(wav)
        n_valid = torch.clamp((lengths - N_FFT) // HOP + 1, min=1)
        mask = (torch.arange(x.shape[2], device=x.device)[None, :] < n_valid[:, None]).float()[:, None, :]
        # Subtracting the clip's mean log value removes the input gain, so loudness
        # cannot drive the estimates; the level checks in audio.py cover loudness.
        mean = (x * mask).sum(dim=(1, 2), keepdim=True) / (mask.sum(dim=2, keepdim=True) * x.shape[1])
        x = (x - mean) * mask
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x))) * mask
        cnt = mask.sum(dim=2)
        mu = x.sum(dim=2) / cnt
        var = ((x - mu[:, :, None]) ** 2 * mask).sum(dim=2) / cnt
        out = self.head(torch.cat([mu, torch.sqrt(var + 1e-6)], dim=1))
        return out[:, 0] * 10.0 + 20.0, out[:, 1] * 10.0 + 20.0, out[:, 2]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
