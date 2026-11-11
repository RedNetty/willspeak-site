from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from ..paths import MODELS_DIR

DEFAULT_DIR_NAME = "parakeet-tdt-0.6b-v2-int8"


def default_model_dir() -> Path:
    return Path(os.environ.get("WILLSPEAK_PARAKEET_DIR") or (MODELS_DIR / DEFAULT_DIR_NAME))


def model_files(d: Path) -> dict[str, Path] | None:
    out = {}
    for part in ("encoder", "decoder", "joiner"):
        for cand in (d / f"{part}.int8.onnx", d / f"{part}.onnx"):
            if cand.is_file() and cand.stat().st_size:
                out[part] = cand
                break
        else:
            return None
    tok = d / "tokens.txt"
    if not tok.is_file() or not tok.stat().st_size:
        return None
    out["tokens"] = tok
    return out


class ParakeetASR:
    name = "parakeet"

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else default_model_dir()
        name = self.model_dir.name
        if self.model_dir.parent.resolve() != MODELS_DIR.resolve() and self.model_dir.parent.name:
            name = f"{self.model_dir.parent.name}/{name}"
        self.model_name = f"parakeet:{name}"
        self.device = "cpu"
        self.actual_device: str | None = None
        self.threads = int(os.environ.get("WILLSPEAK_ASR_THREADS", "0") or 0) or max(1, min(4, os.cpu_count() or 2))
        self._model = None
        self._lock = threading.Lock()
        self.load_s: float | None = None
        self.warm = False

    @staticmethod
    def installed() -> bool:
        try:
            import sherpa_onnx  # noqa: F401
            return True
        except ImportError:
            return False

    def model_present(self) -> bool:
        return model_files(self.model_dir) is not None

    def available(self) -> bool:
        return self.installed() and self.model_present()

    def _ensure(self):
        with self._lock:
            if self._model is not None:
                return self._model
            import sherpa_onnx
            files = model_files(self.model_dir)
            if files is None:
                raise FileNotFoundError(
                    f"no Parakeet model in {self.model_dir} -run: python scripts/fetch_models.py")
            kwargs = {
                "encoder": str(files["encoder"]), "decoder": str(files["decoder"]),
                "joiner": str(files["joiner"]), "tokens": str(files["tokens"]),
                "num_threads": self.threads, "sample_rate": 16000, "feature_dim": 128,
                "model_type": "nemo_transducer", "provider": "cpu",
                "decoding_method": "greedy_search",
            }
            t0 = time.time()
            self._model = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
            self.actual_device = "cpu"
            self.load_s = time.time() - t0
            print(f"[willspeak] asr: {self.model_name} loaded in {self.load_s:.1f}s "
                  f"({kwargs['decoding_method']}, {self.threads} threads)")
            return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warm_up(self) -> None:
        import numpy as np
        import soundfile as sf

        m = self._ensure()
        if self.warm:
            return
        x = None
        wavs = sorted((self.model_dir / "test_wavs").glob("*.wav"))
        if wavs:
            try:
                from .. import audio as A
                y, sr = sf.read(str(wavs[0]), dtype="float32")
                x = A.to_mono_16k(y, sr)[: 16000 * 3]
            except Exception:  # noqa: BLE001
                x = None
        if x is None:
            x = np.zeros(16000, dtype="float32")
        t0 = time.time()
        stream = m.create_stream()
        stream.accept_waveform(16000, x)
        m.decode_stream(stream)
        self.warm = True
        print(f"[willspeak] asr: warm-up decode of {len(x) / 16000:.1f}s took {time.time() - t0:.2f}s")

    def transcribe(self, wav_path: str) -> dict:
        import soundfile as sf

        from .. import audio as A
        x, sr = sf.read(wav_path, dtype="float32")
        return self.transcribe_array(A.to_mono_16k(x, sr), 16000)

    def transcribe_array(self, x, sr: int = 16000) -> dict:
        from .. import audio as A
        if sr != 16000 or getattr(x, "ndim", 1) != 1:
            x = A.to_mono_16k(x, sr)
        cold = not self.loaded
        m = self._ensure()
        t0 = time.time()
        stream = m.create_stream()
        stream.accept_waveform(16000, x)
        m.decode_stream(stream)
        r = stream.result
        text = (r.text or "").strip()
        self.warm = True
        out = {
            "text": text,
            "latency_ms": int((time.time() - t0) * 1000),
            "model": self.model_name,
            "device": self.actual_device,
        }
        logprobs = getattr(r, "ys_log_probs", None)
        if logprobs:
            try:
                values = [float(v) for v in logprobs]
                out["tokens"] = len(values)
                out["mean_logprob"] = round(sum(values) / len(values), 3)
                out["min_logprob"] = round(min(values), 3)
            except (TypeError, ValueError):
                pass
        if cold:
            out["load_ms"] = int((self.load_s or 0) * 1000)
        return out
