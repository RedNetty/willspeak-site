from __future__ import annotations

import io
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly

TARGET_SR = 16_000

RMS_TARGET_LO = -23.0
RMS_TARGET_HI = -18.0
RMS_HARD_LO = -35.0
PEAK_CLIP_THRESH = 0.99
CLIP_RATIO_WARN = 0.0005
MIN_DURATION_S = 0.35
MAX_DURATION_S = 25.0
MIN_SNR_DB = 12.0
LEAD_TRAIL_SILENCE_WARN_S = 2.5


@dataclass
class QCResult:
    duration_s: float
    sample_rate: int
    channels: int
    rms_dbfs: float
    peak_dbfs: float
    clip_ratio: float
    est_snr_db: float
    noise_dbfs: float
    speech_dbfs: float
    lead_silence_s: float
    trail_silence_s: float
    speech_ratio: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gate: dict | None = None  # recording-quality gate verdict (quality.py); None when it is off

    @property
    def ok(self) -> bool:
        return not self.errors

    def dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _dbfs(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-12))


def to_mono_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64, copy=False)
    if sr != TARGET_SR:
        g = math.gcd(int(sr), TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, int(sr) // g)
    return data.astype(np.float32)


def _frame_rms(x: np.ndarray, sr: int, win_ms: float = 25.0, hop_ms: float = 10.0):
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < win:
        return np.array([np.sqrt(np.mean(x ** 2) + 1e-20)])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-20)



def _levels(fr_db: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    if fr_db.size == 0:
        return -90.0, -90.0, 0.0, np.zeros(0, dtype=bool)

    p5 = float(np.percentile(fr_db, 5))
    p95 = float(np.percentile(fr_db, 95))

    seed = max(p5 + 3.0, p95 - 25.0)
    quiet = fr_db <= seed
    noise_db = float(np.median(fr_db[quiet])) if quiet.sum() >= 3 else p5

    thresh = max(noise_db + 6.0, p95 - 25.0)
    active = fr_db > thresh
    speech_db = float(np.median(fr_db[active])) if active.any() else p95
    return noise_db, speech_db, speech_db - noise_db, active


def analyze(x: np.ndarray, sr: int, orig_sr: int, channels: int) -> QCResult:
    dur = len(x) / sr
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-20))
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    clip_ratio = float(np.mean(np.abs(x) >= PEAK_CLIP_THRESH)) if len(x) else 0.0

    fr = _frame_rms(x, sr)
    fr_db = 20 * np.log10(fr + 1e-12)
    noise_db, speech_db, snr, active = _levels(fr_db)

    hop_s = 0.010
    speech_ratio = float(np.mean(active)) if len(active) else 0.0
    if active.any():
        first = int(np.argmax(active))
        last = len(active) - 1 - int(np.argmax(active[::-1]))
        lead = first * hop_s
        trail = max(0.0, dur - (last + 1) * hop_s)
    else:
        lead = trail = dur

    r = QCResult(
        duration_s=round(dur, 3),
        sample_rate=sr,
        channels=channels,
        rms_dbfs=round(_dbfs(rms), 2),
        peak_dbfs=round(_dbfs(peak), 2),
        clip_ratio=round(clip_ratio, 6),
        est_snr_db=round(snr, 1),
        noise_dbfs=round(float(noise_db), 1),
        speech_dbfs=round(float(speech_db), 1),
        lead_silence_s=round(lead, 2),
        trail_silence_s=round(trail, 2),
        speech_ratio=round(speech_ratio, 3),
    )

    if dur < MIN_DURATION_S:
        r.errors.append(f"Too short ({dur:.2f}s) -likely a mis-trigger.")
    if dur > MAX_DURATION_S:
        r.errors.append(f"Too long ({dur:.1f}s). Keep clips under {MAX_DURATION_S:.0f}s.")
    if r.rms_dbfs < RMS_HARD_LO:
        r.errors.append(
            f"Far too quiet ({r.rms_dbfs:.1f} dBFS). Check the mic is selected "
            f"and positioned; target {RMS_TARGET_LO:.0f} to {RMS_TARGET_HI:.0f} dBFS."
        )
    if speech_ratio < 0.02 or snr < 4.0:
        r.errors.append(
            "No speech detected -the signal has almost no dynamic range "
            "(silence, a dead mic, or constant noise)."
        )

    if clip_ratio > CLIP_RATIO_WARN:
        r.warnings.append(
            f"Clipping on {clip_ratio*100:.2f}% of samples. Lower the input gain -"
            "clipped audio cannot be repaired."
        )
    if RMS_HARD_LO <= r.rms_dbfs < RMS_TARGET_LO:
        r.warnings.append(
            f"Quiet ({r.rms_dbfs:.1f} dBFS). Target {RMS_TARGET_LO:.0f} to "
            f"{RMS_TARGET_HI:.0f}. Move the mic closer or raise input gain."
        )
    if r.rms_dbfs > RMS_TARGET_HI + 4:
        r.warnings.append(f"Hot ({r.rms_dbfs:.1f} dBFS). Back off the gain a little.")
    if snr < MIN_SNR_DB:
        r.warnings.append(
            f"Low SNR ({snr:.0f} dB). Background noise is close to speech level -"
            "quieter room, or closer mic."
        )
    if lead > LEAD_TRAIL_SILENCE_WARN_S or trail > LEAD_TRAIL_SILENCE_WARN_S:
        r.warnings.append(
            f"Long silence at the edges ({lead:.1f}s lead, {trail:.1f}s trail). "
            "Trim the clip before training."
        )
    if channels > 1:
        r.warnings.append(f"Recorded {channels} channels; downmixed to mono.")
    if orig_sr < TARGET_SR:
        r.warnings.append(
            f"Source sample rate {orig_sr} Hz is below 16 kHz -upsampling cannot "
            "recover the missing high frequencies. Check the capture device."
        )
    return r


def ingest(raw: bytes) -> tuple[np.ndarray, int, QCResult, np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if not data.size or not np.all(np.isfinite(data)):
        raise ValueError("audio must contain finite samples")
    channels = 1 if data.ndim == 1 else data.shape[1]
    canon = to_mono_16k(data, sr)
    qc = analyze(canon, TARGET_SR, sr, channels)
    return canon, TARGET_SR, qc, data, sr


def write_wav(path: str, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    sf.write(path, x, sr, subtype=subtype)


def trim_silence(x: np.ndarray, sr: int, pad_s: float = 0.20) -> np.ndarray:
    fr = _frame_rms(x, sr)
    fr_db = 20 * np.log10(fr + 1e-12)
    _, _, _, active = _levels(fr_db)
    if not active.any():
        return x
    hop = int(sr * 0.010)
    i0 = max(0, (int(np.argmax(active)) * hop) - int(pad_s * sr))
    i1 = min(len(x), ((len(active) - int(np.argmax(active[::-1]))) * hop) + int(pad_s * sr))
    return x[i0:i1]


def same_audio(x: np.ndarray, y: np.ndarray, sr: int, max_shift_s: float = 0.05,
               min_corr: float = 0.98) -> bool:
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    if len(x) == 0 or len(y) == 0:
        return False
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    if len(long_) - len(short) > int(max_shift_s * sr) * 2 + 1:
        return False
    short = short.astype(np.float64)
    long_ = long_.astype(np.float64)
    short -= short.mean()
    long_ -= long_.mean()
    energy = float(short @ short)
    if energy < 1e-12:
        return False
    n = len(short)
    sums = np.concatenate(([0.0], np.cumsum(long_)))
    squares = np.concatenate(([0.0], np.cumsum(long_ * long_)))
    window_sums = sums[n:] - sums[:-n]
    window_energy = np.maximum(0.0, squares[n:] - squares[:-n] - window_sums ** 2 / n)
    valid = window_energy >= 1e-12
    if not valid.any():
        return False
    dots = correlate(long_, short, mode="valid", method="auto")
    return bool(np.any(dots[valid] / np.sqrt(energy * window_energy[valid]) >= min_corr))
