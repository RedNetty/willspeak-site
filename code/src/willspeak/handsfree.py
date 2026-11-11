from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path

from rapidfuzz import fuzz

DEFAULTS = {
    "stop_phrase": "purple elephant",
    "stop_text_sim": 0.85,
    "stop_match_thr": 0.50,
    "stop_match_margin": 0.08,
    "retract_window_s": 6.0,
    "retract_text": "Scratch that.",
    "sleep_phrase": "voice go to sleep",
    "wake_phrase": "voice wake up",
    "sleep_text": "Going quiet.",
    "wake_text": "I'm listening.",
    "special_text_sim": 0.70,
    "speak_delay_s": 0.0,
    "vocab_override": False,
    "vocab_override_thr": 0.85,
    "vocab_override_margin": 0.10,
}


def norm_text(s: str) -> str:
    s = (s or "").lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", s).split())


def similarity(a: str, b: str) -> float:
    return fuzz.ratio(a, b) * 0.01


WORD_FLOOR = 0.4
COMPACT_FLOOR = 0.9


def _window_matches(phrase_words: list[str], window: list[str], sim: float) -> bool:
    if similarity(" ".join(phrase_words), " ".join(window)) < sim:
        return False
    return all(similarity(p, w) >= WORD_FLOOR for p, w in zip(phrase_words, window))


def _compact_matches(phrase_words: list[str], window: list[str], sim: float) -> bool:
    return similarity("".join(phrase_words), "".join(window)) >= max(sim, COMPACT_FLOOR)


def contains_phrase(text: str, phrase: str, sim: float) -> bool:
    normalized_text = norm_text(text)
    normalized_phrase = norm_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    words = normalized_text.split()
    phrase_words = normalized_phrase.split()
    width = len(phrase_words)
    if len(words) < width:
        return _compact_matches(phrase_words, words, sim)
    for start in range(len(words) - width + 1):
        if _window_matches(phrase_words, words[start:start + width], sim):
            return True
    for other in (width - 1, width + 1):
        if other < 1 or other > len(words):
            continue
        for start in range(len(words) - other + 1):
            if _compact_matches(phrase_words, words[start:start + other], sim):
                return True
    return False


def detect_phrase(text: str, match: dict | None, phrase: str,
                  settings: dict, text_sim: float) -> str | None:
    if not phrase:
        return None
    if contains_phrase(text, phrase, text_sim):
        return "text"
    if match and match.get("ok") and match.get("candidates"):
        top = match["candidates"][0]
        margin_threshold = float(settings["stop_match_margin"])
        margin_passes = margin_threshold > 0 and float(match.get("margin", 0)) >= margin_threshold
        if (top["text"] == norm_text(phrase) and top.get("own_takes", 0) > 0
                and (top["score"] >= float(settings["stop_match_thr"]) or margin_passes)):
            return "matcher"
    return None


class Settings:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "live_settings.json"
        self._lock = threading.Lock()
        self.values = dict(DEFAULTS)
        if self.path.exists():
            try:
                self.values = self._validated(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, TypeError, ValueError) as exc:
                print(f"[willspeak] invalid live settings; using defaults: {exc}")

    def _validated(self, changes: dict) -> dict:
        if not isinstance(changes, dict):
            raise TypeError("settings must be a JSON object")
        values = dict(self.values)
        for key, value in changes.items():
            if key not in DEFAULTS:
                continue
            default = DEFAULTS[key]
            if isinstance(default, bool):
                text = str(value).lower()
                if text not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
                    raise ValueError(f"{key} must be true or false")
                value = text in {"1", "true", "yes", "on"}
            elif isinstance(default, (int, float)):
                value = float(value)
                upper = {"speak_delay_s": 5, "retract_window_s": 15}.get(key, 1)
                if not math.isfinite(value) or not 0 <= value <= upper:
                    raise ValueError(f"{key} must be between 0 and {upper}")
            else:
                if not isinstance(value, str):
                    raise TypeError(f"{key} must be text")
                value = value.strip()
                if key.endswith("_phrase") and not norm_text(value):
                    raise ValueError(f"{key} must contain words")
            values[key] = value
        phrases = [norm_text(values[key]) for key in ("stop_phrase", "sleep_phrase", "wake_phrase")]
        if len(set(phrases)) != len(phrases):
            raise ValueError("stop, sleep and wake phrases must be different")
        return values

    def get(self) -> dict:
        with self._lock:
            return dict(self.values)

    def update(self, changes: dict) -> dict:
        with self._lock:
            values = self._validated(changes)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, indent=1, allow_nan=False), encoding="utf-8")
            temporary.replace(self.path)
            self.values = values
            return dict(self.values)


def decide(text: str, match: dict | None, settings: dict, last_spoken_at: float | None,
           awake: bool = True, *, now: float | None = None) -> dict:
    special_sim = float(settings.get("special_text_sim", 0.7))

    if not awake:
        by = detect_phrase(text, match, settings.get("wake_phrase", ""), settings, special_sim)
        if by:
            return {"action": "wake", "by": by, "say": settings.get("wake_text", "")}
        return {"action": "ignore", "reason": "asleep", "asleep": True}

    by = detect_phrase(text, match, settings.get("sleep_phrase", ""), settings, special_sim)
    if by:
        return {"action": "sleep", "by": by, "say": settings.get("sleep_text", "")}

    stop_by = detect_phrase(text, match, settings["stop_phrase"], settings, float(settings["stop_text_sim"]))
    if stop_by:
        window = float(settings["retract_window_s"])
        elapsed = (time.time() if now is None else now) - last_spoken_at if last_spoken_at is not None else None
        recent = window > 0 and elapsed is not None and 0 <= elapsed <= window
        return {
            "action": "stop",
            "by": stop_by,
            "retract": recent,
            "retract_text": settings["retract_text"] if recent else "",
        }

    say = text.strip()
    override = _vocab_override(match, settings)
    if override is not None:
        say = override

    if not say:
        return {"action": "ignore", "reason": "nothing heard"}
    return {"action": "speak", "text": say, "override": override, "delay_s": float(settings["speak_delay_s"])}


def _vocab_override(match: dict | None, settings: dict) -> str | None:
    if not settings.get("vocab_override") or not match or not match.get("ok") or not match.get("candidates"):
        return None
    top = match["candidates"][0]
    if (top["score"] < float(settings["vocab_override_thr"])
            or match.get("margin", 0) < float(settings["vocab_override_margin"])
            or top.get("own_takes", 0) <= 0
            or top["text"] == norm_text(settings["stop_phrase"])):
        return None
    return top["text"]
