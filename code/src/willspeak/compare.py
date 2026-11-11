from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

from .asr.parakeet import DEFAULT_DIR_NAME, ParakeetASR, model_files
from .evaluate import align
from .normalize import normalize
from .paths import DATA_DIR, MODELS_DIR, ROOT

SPEAKERS = ("M01", "M02", "M04")
DEMO_DIR = DATA_DIR / "demo"

RECORDED = {
    "rows": [
        {"name": "Stock model", "pooled": 76.0, "sentences": 62.0,
         "source": "docs/measurements.md, stock on the common lab set (NeMo)"},
        {"name": "Speaker models, mean of three training runs", "pooled": 41.9, "sentences": 28.3,
         "source": "docs/measurements.md, stage P pretrained adapter at 6e-4 (NeMo, seeds 17-19)"},
        {"name": "Speaker models, int8 file as the Pi runs it (one training run)", "pooled": 41.6, "sentences": None,
         "source": "app/reports/lab-28-deployable-export/notes.md, int8 --int8-ops matmul, 177 of 425 (sherpa-onnx)"},
    ],
}


def model_dir_for(speaker: str) -> Path:
    override = os.environ.get(f"WILLSPEAK_DEMO_MODEL_{speaker}")
    if override:
        p = Path(override)
        return p if p.is_absolute() else (ROOT / p)
    return MODELS_DIR / f"lab-28-export-{speaker}-matmul"


def stock_dir() -> Path:
    return MODELS_DIR / DEFAULT_DIR_NAME


def model_choices(models_dir: Path | None = None, depth: int = 3) -> list[str]:
    """Every complete model folder under models/ (up to `depth` levels down), as paths relative to it."""
    base = (models_dir or MODELS_DIR).resolve()
    stock = stock_dir().resolve()
    found: list[str] = []

    def walk(d: Path, level: int) -> None:
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        for p in children:
            if model_files(p) is not None:
                if p.resolve() != stock:
                    found.append(p.relative_to(base).as_posix())
            elif level < depth:
                walk(p, level + 1)

    walk(base, 1)
    return found


def resolve_model(rel: str, models_dir: Path | None = None) -> Path:
    base = (models_dir or MODELS_DIR).resolve()
    path = (base / rel).resolve()
    if base not in path.parents:
        raise ValueError("the model must be a folder inside models/")
    if model_files(path) is None:
        raise FileNotFoundError(f"no complete model in models/{rel}")
    return path


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def recorded_speaker_score(speaker: str, variant: str = "matmul") -> dict | None:
    p = ROOT / "reports" / "lab-28-deployable-export" / f"{speaker}-{variant}" / "results.json"
    try:
        word = json.loads(p.read_text(encoding="utf-8"))["word"]
        return {"errors": int(word["errors"]), "n_ref": int(word["n_ref"]), "source": str(p.relative_to(ROOT))}
    except (OSError, KeyError, TypeError, ValueError):
        return None


def load_clips(demo_dir: Path = DEMO_DIR) -> dict:
    meta = json.loads((demo_dir / "clips.json").read_text(encoding="utf-8"))
    clips = meta.get("clips")
    if not isinstance(clips, list):
        raise ValueError(f"{demo_dir / 'clips.json'} has no clip list")
    seen = set()
    for c in clips:
        if not {"id", "speaker", "text", "audio", "kind"} <= set(c):
            raise ValueError(f"clip entry is missing fields: {c}")
        if c["id"] in seen:
            raise ValueError(f"duplicate clip id {c['id']}")
        seen.add(c["id"])
    return meta


def score(ref: str, hyp: str) -> dict:
    ops = align(normalize(ref).split(), normalize(hyp or "").split())
    errors = sum(op != "ok" for op, _, _ in ops)
    n_ref = sum(op != "ins" for op, _, _ in ops)
    return {"errors": errors, "n_ref": n_ref,
            "ops": [{"op": op, "ref": r, "hyp": h} for op, r, h in ops]}


def rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def _return_freed_memory() -> None:
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


class Comparison:
    def __init__(self, stock: ParakeetASR, stock_lock: threading.Lock | None = None,
                 demo_dir: Path = DEMO_DIR, factory=ParakeetASR) -> None:
        same = stock.model_dir.resolve() == stock_dir().resolve()
        self.stock = stock if same else factory(stock_dir())
        self._stock_lock = stock_lock if same and stock_lock is not None else threading.Lock()
        self.demo_dir = demo_dir
        self._factory = factory
        self._lock = threading.Lock()
        self.speaker: str | None = None
        self.model: ParakeetASR | None = None
        self.model_path: Path | None = None
        self.custom = False
        self.last_swap: dict | None = None

    def clips(self) -> list[dict]:
        try:
            return load_clips(self.demo_dir)["clips"]
        except FileNotFoundError:
            return []

    def clip(self, clip_id: str) -> dict:
        for c in self.clips():
            if c["id"] == clip_id:
                return c
        raise KeyError(clip_id)

    def state(self) -> dict:
        try:
            meta = load_clips(self.demo_dir)
            error = None
        except FileNotFoundError:
            meta, error = {"clips": []}, "no clip list: run scripts/corpus/demo_clips.py"
        except ValueError as e:
            meta, error = {"clips": []}, str(e)
        public = [{k: c[k] for k in ("id", "speaker", "kind", "text") if k in c} | {"duration_s": c.get("duration_s")}
                  for c in meta["clips"]]
        return {
            "speakers": [{"id": s, "model_dir": str(model_dir_for(s)), "present": model_files(model_dir_for(s)) is not None,
                          "recorded": recorded_speaker_score(s),
                          "clips": sum(c["speaker"] == s for c in meta["clips"])} for s in SPEAKERS],
            "loaded": self.speaker,
            "loaded_model": _rel(self.model_path) if self.model is not None else None,
            "custom": self.custom and self.model is not None,
            "models": model_choices(),
            "stock": {"model": self.stock.model_name, "present": self.stock.model_present(), "loaded": self.stock.loaded},
            "clips": public,
            "rule": meta.get("rule"),
            "clip_error": error,
            "last_swap": self.last_swap,
            "rss_mb": rss_mb(),
            "recorded": RECORDED["rows"],
        }

    def select(self, speaker: str, model: str | None = None) -> dict:
        """Load the model for the trained side. `model` (a folder inside models/) replaces the speaker's own."""
        if speaker not in SPEAKERS:
            raise KeyError(speaker)
        target = resolve_model(model) if model else model_dir_for(speaker)
        with self._lock:
            if self.model is not None and self.model_path is not None and self.model_path.resolve() == target.resolve():
                # Same files: only the recordings change, so skip the reload.
                self.speaker, self.custom = speaker, bool(model)
                return {"speaker": speaker, "model": _rel(target), "swapped": False, "rss_mb": rss_mb()}
            new = self._factory(target)
            if not new.model_present():
                raise FileNotFoundError(f"no model for {speaker} in {target}")
            t0 = time.perf_counter()
            before = rss_mb()
            previous = self.speaker
            self.model, self.speaker, self.model_path = None, None, None
            _return_freed_memory()
            after_unload = rss_mb()
            new.warm_up()
            self.model, self.speaker, self.model_path, self.custom = new, speaker, target, bool(model)
            after_load = rss_mb()
            self.last_swap = {
                "from": previous, "to": speaker, "model": new.model_name, "folder": _rel(target),
                "swap_ms": int((time.perf_counter() - t0) * 1000),
                "load_ms": int((new.load_s or 0) * 1000),
                "rss_before_mb": before, "rss_after_unload_mb": after_unload, "rss_after_load_mb": after_load,
            }
            print(f"[willspeak] compare: {previous or 'none'} -> {speaker} in {self.last_swap['swap_ms']} ms "
                  f"(load {self.last_swap['load_ms']} ms), rss {before} -> {after_unload} -> {after_load} MB")
            return {"speaker": speaker, "swapped": True, **self.last_swap}

    def release(self) -> dict:
        with self._lock:
            self.model, self.speaker, self.model_path, self.custom = None, None, None, False
            _return_freed_memory()
            return {"loaded": None, "rss_mb": rss_mb()}

    def warm_stock(self) -> None:
        with self._stock_lock:
            self.stock.warm_up()

    def audio_path(self, clip_id: str) -> Path:
        clip = self.clip(clip_id)
        path = (self.demo_dir / clip["audio"]).resolve()
        if self.demo_dir.resolve() not in path.parents:
            raise ValueError("clip audio must be inside the demo folder")
        if not path.is_file():
            raise FileNotFoundError(f"no recording for clip {clip_id}")
        return path

    def _both(self, x, speaker: str | None) -> tuple[dict, dict, str]:
        with self._lock:
            if self.model is None or (speaker is not None and self.speaker != speaker):
                raise RuntimeError(f"load the {speaker or 'a speaker'} model first")
            with self._stock_lock:
                stock = self.stock.transcribe_array(x, 16000)
            return stock, self.model.transcribe_array(x, 16000), self.speaker

    def decode(self, clip_id: str) -> dict:
        import soundfile as sf

        from . import audio as A
        clip = self.clip(clip_id)
        path = self.audio_path(clip_id)
        x, sr = sf.read(str(path), dtype="float32")
        stock, mine, _ = self._both(A.to_mono_16k(x, sr), clip["speaker"])
        out = {"clip": {k: clip.get(k) for k in ("id", "speaker", "kind", "text", "duration_s")}}
        for key, r in (("stock", stock), ("speaker", mine)):
            out[key] = {"text": r["text"], "latency_ms": r["latency_ms"], "model": r["model"], **score(clip["text"], r["text"])}
        print(f"[willspeak] compare clip {clip_id}: stock {out['stock']['errors']}/{out['stock']['n_ref']}, "
              f"{clip['speaker']} {out['speaker']['errors']}/{out['speaker']['n_ref']}")
        return out

    def try_voice(self, x, duration_s: float) -> dict:
        # A visitor's own speech: nothing is written to disk and there is no reference text to score against.
        stock, mine, speaker = self._both(x, None)
        print(f"[willspeak] compare try: {duration_s:.1f} s against stock and {speaker} "
              f"({stock['latency_ms']} ms, {mine['latency_ms']} ms)")
        return {"speaker": speaker, "duration_s": round(duration_s, 2),
                "stock": {k: stock[k] for k in ("text", "latency_ms", "model")},
                "mine": {k: mine[k] for k in ("text", "latency_ms", "model")}}
