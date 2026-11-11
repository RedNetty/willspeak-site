#!/usr/bin/env python3
"""How loud the training clips are, to choose white-noise levels for ``train.py --white-noise``."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _cli

FLOOR_DB = -120.0


def clip_level_db(samples) -> float:
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0:
        return FLOOR_DB
    rms = float(np.sqrt(np.mean(x * x)))
    return max(FLOOR_DB, 20.0 * math.log10(rms)) if rms > 0 else FLOOR_DB


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def levels_of(folder: Path) -> list[float]:
    import soundfile as sf

    manifest = folder / "train.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"no train.jsonl in {folder}")
    out = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        audio = folder / json.loads(line)["audio"]
        samples, _ = sf.read(audio, dtype="float32")
        out.append(clip_level_db(samples))
    if not out:
        raise ValueError(f"{manifest} has no clips")
    return sorted(out)


def report(name: str, levels: list[float], noise_db: tuple[int, int] | None) -> list[str]:
    p5, p25, p50, p75, p95 = (percentile(levels, q) for q in (5, 25, 50, 75, 95))
    lines = [(f"{name}: {len(levels)} training clips, level in dBFS: quietest {levels[0]:.1f}, "
              f"5th percentile {p5:.1f}, 25th {p25:.1f}, median {p50:.1f}, 75th {p75:.1f}, 95th {p95:.1f}, "
              f"loudest {levels[-1]:.1f}")]
    if noise_db:
        lo, hi = noise_db
        top = hi - 1
        for label, value in (("quietest clip", levels[0]), ("median clip", p50), ("loudest clip", levels[-1])):
            lines.append(f"    {label} ({value:.1f} dBFS): signal-to-noise {value - top:.0f} dB at the loudest noise "
                         f"({top} dBFS), {value - lo:.0f} dB at the quietest ({lo} dBFS)")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="export folders, each with a train.jsonl and its audio")
    ap.add_argument("--noise-db", type=int, nargs=2, metavar=("MIN_DB", "MAX_DB"), default=None,
                    help="the --white-noise range to compare against")
    a = ap.parse_args()
    if a.noise_db and not a.noise_db[0] < a.noise_db[1]:
        ap.error("--noise-db needs MIN_DB lower than MAX_DB")
    for folder in a.folders:
        path = Path(folder) if Path(folder).is_absolute() else ROOT / folder
        print("\n".join(report(path.name, levels_of(path), tuple(a.noise_db) if a.noise_db else None)))
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "clip_levels"))
