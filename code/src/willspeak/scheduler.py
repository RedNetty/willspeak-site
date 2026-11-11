from __future__ import annotations

import json
import math
import random

from .g2p import ARPABET, Coverage, text_diphones, text_phones

SET_ORDER = ["critical", "functional", "phonetic", "custom"]


def difficulty_weights(payload: dict) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise TypeError("Difficulty weights must be a JSON object")
    if payload.get("format") == "willspeak-difficulty-v1":
        if payload.get("source_split") != "dev":
            raise ValueError("Use development-set difficulty for recording; test errors must not guide training data selection")
        payload = payload.get("weights")
        if not isinstance(payload, dict):
            raise ValueError("Difficulty record needs a weights object")
    weights = {}
    for phone, value in payload.items():
        if phone not in ARPABET:
            raise ValueError(f"Unknown phoneme: {phone!r}")
        if isinstance(value, bool):
            raise TypeError(f"Weight for {phone} must be a finite positive number")
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0 or weight > 5:
            raise ValueError(f"Weight for {phone} must be greater than zero and at most 5")
        weights[phone] = weight
    return weights


class Scheduler:
    def __init__(self, store, difficulty: dict[str, float] | None = None,
                 redundancy: int = 2):
        self.store = store
        self.difficulty = difficulty or {}
        self.redundancy = redundancy
        self._cov: Coverage | None = None

    def coverage(self, refresh: bool = False) -> Coverage:
        if self._cov is None or refresh:
            c = Coverage()
            c.add_many(self.store.accepted_texts())
            self._cov = c
        return self._cov

    def invalidate(self) -> None:
        self._cov = None

    def _weighted_gain(self, text: str, cov: Coverage) -> float:
        gain = 0.0
        for a, b in text_diphones(text):
            if cov.diphone_counts.get((a, b), 0) < self.redundancy:
                w = 1.0
                w *= self.difficulty.get(a, 1.0)
                w *= self.difficulty.get(b, 1.0)
                gain += w
        return gain

    def next_batch(self, n: int = 25, prompt_set: str | None = None) -> list[dict]:
        taken = self.store.prompt_take_counts()
        cov = self.coverage(refresh=True)
        out: list[dict] = []

        sets = [prompt_set] if prompt_set else SET_ORDER
        for ps in sets:
            if len(out) >= n:
                break
            rows = [r for r in self.store.prompts(ps)
                    if taken.get(r["id"], 0) < r["target_takes"]]
            if not rows:
                continue

            if ps == "phonetic":
                local = Coverage()
                local.phone_counts = dict(cov.phone_counts)
                local.diphone_counts = dict(cov.diphone_counts)
                pool = list(rows)
                while pool and len(out) < n:
                    best, bg = None, -1.0
                    for r in pool:
                        g = self._weighted_gain(r["text"], local)
                        if g > bg:
                            best, bg = r, g
                    if best is None:
                        break
                    if bg <= 0:
                        random.shuffle(pool)
                        for r in pool[: n - len(out)]:
                            out.append(self._fmt(r, taken, 0.0, cov))
                        break
                    pool.remove(best)
                    local.add(best["text"])
                    out.append(self._fmt(best, taken, bg, cov))
            else:
                rows.sort(key=lambda r: (-r["priority"], taken.get(r["id"], 0), r["id"]))
                for r in rows[: n - len(out)]:
                    out.append(self._fmt(r, taken, self._weighted_gain(r["text"], cov), cov))
        return out

    def _fmt(self, r, taken: dict, gain: float, cov: Coverage) -> dict:
        try:
            phones = json.loads(r["phones"] or "[]")
        except (TypeError, json.JSONDecodeError):
            phones = []
        if not phones:
            phones, _ = text_phones(r["text"])
        return {
            "prompt_id": r["id"],
            "text": r["text"],
            "prompt_set": r["prompt_set"],
            "kind": r["kind"],
            "takes_done": taken.get(r["id"], 0),
            "target_takes": r["target_takes"],
            "phones": phones,
            "diphone_gain": round(gain, 1),
        }

    def coverage_report(self) -> dict:
        cov = self.coverage(refresh=True)
        st = cov.stats(min_count=1)

        legal: set[tuple[str, str]] = set()
        for r in self.store.prompts(active_only=False):
            legal |= text_diphones(r["text"])

        have = {d for d, c in cov.diphone_counts.items() if c >= 1}
        have_r = {d for d, c in cov.diphone_counts.items() if c >= self.redundancy}
        missing = sorted(legal - have)

        return {
            **st,
            "redundancy_target": self.redundancy,
            "diphones_in_bank": len(legal),
            "diphones_covered": len(have & legal),
            "diphones_covered_pct": round(100 * len(have & legal) / max(1, len(legal)), 1),
            "diphones_redundant": len(have_r & legal),
            "diphones_redundant_pct": round(100 * len(have_r & legal) / max(1, len(legal)), 1),
            "phones_missing": cov.missing_phones(),
            "phones_thin": cov.thin_phones(min_count=5),
            "missing_diphones_sample": ["".join(d) if d else "" for d in
                                        [f"{a}-{b}" for a, b in missing[:40]]],
            "phone_counts": {p: cov.phone_counts.get(p, 0) for p in ARPABET},
            "recommendations": self.recommend(n=12),
            "phone_examples": self.phone_examples(),
        }

    def recommend(self, n: int = 12) -> list[dict]:
        taken = self.store.prompt_take_counts()
        cov = self.coverage()
        rows = [r for r in self.store.prompts()
                if taken.get(r["id"], 0) < r["target_takes"]]
        scored = [(self._weighted_gain(r["text"], cov), r) for r in rows]
        scored.sort(key=lambda x: -x[0])
        return [{"text": r["text"], "prompt_set": r["prompt_set"],
                 "gain": round(g, 1)} for g, r in scored[:n] if g > 0]

    def phone_examples(self, max_per_phone: int = 3) -> dict[str, list[str]]:
        taken = self.store.prompt_take_counts()
        cov = self.coverage()
        weak = {p for p, _ in cov.thin_phones(min_count=5)}
        out: dict[str, list[str]] = {p: [] for p in weak}
        for r in self.store.prompts():
            if taken.get(r["id"], 0) >= r["target_takes"]:
                continue
            ph = set(text_phones(r["text"])[0])
            for p in ph & weak:
                if len(out[p]) < max_per_phone:
                    out[p].append(r["text"])
        return {k: v for k, v in out.items() if v}


def load_prompt_file(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def classify_kind(text: str) -> str:
    n = len(text.split())
    if n == 1:
        return "word"
    return "phrase" if n <= 4 else "sentence"


def seed_prompts(store, prompt_dir: str, redundancy_targets: dict | None = None) -> dict:
    from pathlib import Path
    targets = redundancy_targets or {
        "critical": 15,
        "functional": 4,
        "phonetic": 1,
        "custom": 2,
        "probe": 1000,
    }
    files = {
        "critical": "critical.txt",
        "functional": "functional.txt",
        "phonetic": "harvard_seed.txt",
        "probe": "probe.txt",
    }
    added = {}
    for ps, fn in files.items():
        p = Path(prompt_dir) / fn
        if not p.exists():
            continue
        n = 0
        for i, text in enumerate(load_prompt_file(str(p))):
            phones, oov = text_phones(text)
            pid = store.add_prompt(
                text, ps, classify_kind(text),
                target_takes=targets.get(ps, 1),
                priority=(1000 - i) if ps == "critical" else 0,
                phones=phones, oov=oov,
            )
            if pid:
                n += 1
        added[ps] = n
    for p in sorted(Path(prompt_dir).glob("phonetic_*.txt")):
        n = 0
        for text in load_prompt_file(str(p)):
            phones, oov = text_phones(text)
            if store.add_prompt(text, "phonetic", classify_kind(text),
                                target_takes=targets.get("phonetic", 1),
                                phones=phones, oov=oov):
                n += 1
        added[p.stem] = n
    return added
