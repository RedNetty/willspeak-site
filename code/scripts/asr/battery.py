#!/usr/bin/env python3
"""Clips with no speech in them, and how often a model writes words for them."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import labkit as lab
from willspeak.normalize import normalize

SR = 16000
KINDS = ("silence", "white", "pink", "hum", "room", "breath", "taps")
LEVEL_DB = (-55.0, -30.0)


def _scale(x: np.ndarray, level_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x**2))) or 1.0
    return np.clip(x * (10 ** (level_db / 20) / rms), -0.99, 0.99).astype(np.float32)


def _shape(noise: np.ndarray, gain) -> np.ndarray:
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / SR)
    return np.fft.irfft(spectrum * gain(freqs), n=len(noise))


def synth(kind: str, seconds: float, level_db: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    noise = rng.normal(size=n)
    if kind == "silence":
        return (rng.normal(size=n) * 3e-5).astype(np.float32)
    if kind == "white":
        return _scale(noise, level_db)
    if kind == "pink":
        return _scale(_shape(noise, lambda f: 1 / np.sqrt(np.maximum(f, 20.0))), level_db)
    if kind == "hum":
        base = rng.choice([50.0, 60.0])
        x = sum(np.sin(2 * np.pi * base * k * t + rng.uniform(0, 2 * np.pi)) / k for k in (1, 2, 3, 5))
        return _scale(x + 0.05 * noise, level_db)
    if kind == "room":
        x = _shape(noise, lambda f: 1 / (1 + (f / 400.0) ** 2))
        swell = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 2 * np.pi))
        return _scale(x * swell, level_db)
    if kind == "breath":
        x = _shape(noise, lambda f: ((f > 200) & (f < 3000)).astype(float))
        envelope = np.zeros(n)
        for _ in range(2):
            length = int(rng.uniform(0.3, 0.9) * SR)
            start = int(rng.uniform(0, max(n - length, 1)))
            envelope[start : start + length] += np.hanning(min(length, n - start))
        return _scale(x * (0.05 + envelope), level_db)
    if kind == "taps":
        x = 0.05 * noise
        for _ in range(int(rng.integers(3, 7))):
            start = int(rng.uniform(0, max(n - 0.1 * SR, 1)))
            burst = _shape(rng.normal(size=int(0.1 * SR)), lambda f: (f > 300).astype(float))
            burst = burst * np.exp(-np.arange(len(burst)) / (0.03 * SR))
            x[start : start + len(burst)] += burst[: n - start] * rng.uniform(2.0, 6.0)
        return _scale(x, level_db)
    raise ValueError(f"Unknown kind {kind!r}; choose from {KINDS}")


def edge_segments(rows: list[dict], audio_root: Path, seconds: float = 0.4, limit: int = 0) -> list[tuple[str, np.ndarray]]:
    out = []
    size = int(seconds * SR)
    for i, row in enumerate(rows):
        audio = lab.read_audio(audio_root / row["audio"])
        if len(audio) < 3 * size:
            continue
        whole = float(np.sqrt(np.mean(audio**2)))
        if whole == 0:
            continue
        for part, piece in (("head", audio[:size]), ("tail", audio[-size:])):
            if float(np.sqrt(np.mean(piece**2))) <= 0.1 * whole:
                out.append((f"{part}-{i}", piece.copy()))
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def build(out: Path, per_kind: int, seed: int, kinds: tuple[str, ...] = KINDS, seconds: tuple[float, float] = (1.0, 3.0),
          edge_rows: list[dict] | None = None, edge_root: Path | None = None, edge_limit: int = 0) -> list[dict]:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    (out / "audio").mkdir(parents=True)
    rows = []

    def add(kind: str, name: str, samples: np.ndarray) -> None:
        rel = f"audio/{name}.wav"
        sf.write(out / rel, samples, SR, subtype="PCM_16")
        rows.append({"id": name, "audio": rel, "text": "", "duration_s": round(len(samples) / SR, 3), "condition": kind})

    for kind in kinds:
        for i in range(per_kind):
            level = float(rng.uniform(*LEVEL_DB))
            add(kind, f"{kind}-{i:03d}", synth(kind, float(rng.uniform(*seconds)), level, rng))
    if edge_rows:
        for name, samples in edge_segments(edge_rows, edge_root, limit=edge_limit):
            add("edge", f"edge-{name}", samples)
    with (out / "manifest.jsonl").open("x", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def score(rows: list[dict], hyps: dict[str, str]) -> dict:
    kinds: dict[str, dict] = {}
    for row in rows:
        words = normalize(hyps[row["id"]]).split() if row["id"] in hyps else None
        if words is None:
            raise KeyError(f"No hypothesis for clip {row['id']}")
        k = kinds.setdefault(row["condition"], {"clips": 0, "with_words": 0, "words": 0})
        k["clips"] += 1
        k["with_words"] += bool(words)
        k["words"] += len(words)
    total = {"clips": sum(k["clips"] for k in kinds.values()), "with_words": sum(k["with_words"] for k in kinds.values()),
             "words": sum(k["words"] for k in kinds.values())}
    total["false_insert_rate"] = total["with_words"] / total["clips"] if total["clips"] else None
    return {"kinds": kinds, "total": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write a battery of non-speech clips")
    b.add_argument("--out", required=True, help="new folder (refuses to overwrite)")
    b.add_argument("--seed", type=int, default=2000)
    b.add_argument("--per-kind", type=int, default=20)
    b.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    b.add_argument("--edge-from", help="a manifest.jsonl of real clips to cut quiet start and end stretches from")
    b.add_argument("--edge-audio-root", help="its audio root (default: the manifest's folder)")
    b.add_argument("--edge-limit", type=int, default=60)
    s = sub.add_parser("score", help="count the words models write for a battery")
    s.add_argument("--manifest", required=True)
    s.add_argument("--system", action="append", required=True, metavar="LABEL=HYPOTHESES")
    s.add_argument("--out", help="also write the table as .json (refuses to overwrite)")
    a = ap.parse_args()

    if a.command == "build":
        lab.refuse_existing(a.out)
        edge_rows = edge_root = None
        if a.edge_from:
            manifest = lab.here(a.edge_from)
            edge_rows = list(lab.read_manifest(manifest).values())
            edge_root = lab.here(a.edge_audio_root) if a.edge_audio_root else manifest.parent
        rows = build(lab.here(a.out), a.per_kind, a.seed, tuple(a.kinds), edge_rows=edge_rows, edge_root=edge_root,
                     edge_limit=a.edge_limit)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["condition"]] = counts.get(r["condition"], 0) + 1
        print(f"wrote {len(rows)} clips to {lab.here(a.out)}: {counts}")
        return 0

    if a.out:
        lab.refuse_existing(a.out)
    rows = list(lab.read_manifest(a.manifest).values())
    results, sources = {}, {"manifest": a.manifest}
    for text in a.system:
        label, _, path = text.partition("=")
        if not path:
            ap.error(f"--system wants LABEL=HYPOTHESES, got {text!r}")
        results[label] = score(rows, lab.read_hyps(path))
        sources[label] = path
    kinds = sorted({r["condition"] for r in rows})
    print(f"{'system':<24}" + "".join(f"{k:>9}" for k in kinds) + "    total")
    for label, result in results.items():
        cells = "".join(f"{result['kinds'][k]['with_words']:>4}/{result['kinds'][k]['clips']:<4}" for k in kinds)
        t = result["total"]
        print(f"{label:<24}{cells}  {t['with_words']:>3}/{t['clips']:<3} ({100 * t['false_insert_rate']:.1f}% of clips got words, {t['words']} words)")
    if a.out:
        lab.save_json(a.out, {"systems": results, "kinds": kinds, "provenance": lab.provenance(__file__, sources)})
        print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "battery"))
