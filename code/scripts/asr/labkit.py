from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ci

from willspeak.evaluate import OK, align, load_manifest
from willspeak.normalize import normalize

SEVERE_TEST = ("M01", "M02", "M04")
EXPORT = "data/public/torgo/export"
FT = "reports/finetune-torgo"
BAR = 0.35


def here(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def refuse_existing(*paths: str | Path) -> None:
    for path in paths:
        if here(path).exists():
            raise FileExistsError(f"Already exists; choose a new name: {here(path)}")


def bar_words(total_words: int, bar: float = BAR) -> int:
    return math.floor(bar * total_words + 1e-9)


def read_hyps(path: str | Path) -> dict[str, str]:
    path = here(path)
    if path.is_dir():
        path = path / "hypotheses.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No hypotheses at {path}")
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = str(row["id"])
                if key in out or not isinstance(row.get("hyp"), str):
                    raise ValueError(f"Duplicate id or non-text hypothesis {key!r} in {path}")
                out[key] = row["hyp"]
    return out


def read_manifest(path: str | Path) -> dict[str, dict]:
    return load_manifest(here(path))


def test_manifest(speaker: str, export: str = EXPORT) -> Path:
    return here(f"{export}/{speaker}/test.jsonl")


KINDS = ("sentence", "word")


def clip_kind(row: dict) -> str:
    return str(row.get("kind") or str(row.get("prompt_set", "")).replace("torgo-", "") or "unknown")


def only_kind(manifest: dict[str, dict], kind: str) -> dict[str, dict]:
    if kind == "all":
        return dict(manifest)
    if kind not in KINDS:
        raise ValueError(f"kind must be 'all' or one of {KINDS}, not {kind!r}")
    return {clip_id: row for clip_id, row in manifest.items() if clip_kind(row) == kind}


def parse_system(text: str) -> tuple[str, dict[str, str]]:
    if "=" not in text:
        raise ValueError(f"A system is label=SPK:path,SPK:path ... not {text!r}")
    label, rest = text.split("=", 1)
    per: dict[str, str] = {}
    for part in rest.split(","):
        if ":" not in part:
            raise ValueError(f"System {label!r}: expected SPK:path, got {part!r}")
        speaker, path = part.split(":", 1)
        if speaker in per:
            raise ValueError(f"System {label!r} lists {speaker} twice")
        per[speaker] = path
    return label.strip(), per


def read_audio(path: str | Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != 16000:
        raise ValueError(f"{path} is {rate} Hz; the lab tests decode 16 kHz exports only")
    if audio.ndim != 1:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected mono")
    return audio


def clip_counts(manifest: dict[str, dict], hyps: dict[str, str]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for clip_id, row in manifest.items():
        if clip_id not in hyps:
            raise KeyError(f"No hypothesis for clip {clip_id}")
        ref = normalize(row["text"]).split()
        hyp = normalize(hyps[clip_id]).split()
        out[clip_id] = (sum(op != OK for op, _, _ in align(ref, hyp)), len(ref))
    return out


def totals(counts: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return sum(e for e, _ in counts.values()), sum(n for _, n in counts.values())


def empty_share(manifest: dict[str, dict], hyps: dict[str, str]) -> tuple[int, int]:
    empty = sum(1 for clip_id in manifest if not normalize(hyps.get(clip_id, "")).split())
    return empty, len(manifest)


class Pooled:
    def __init__(self, label: str, manifests: dict[str, dict[str, dict]], hyps: dict[str, dict[str, str]]):
        self.label = label
        self.speakers = [s for s in manifests if s in hyps]
        if len(self.speakers) != len(manifests):
            missing = sorted(set(manifests) - set(hyps))
            raise ValueError(f"System {label!r} has no hypotheses for {missing}")
        self.counts = {s: clip_counts(manifests[s], hyps[s]) for s in self.speakers}
        self.empty = {s: empty_share(manifests[s], hyps[s]) for s in self.speakers}
        self.prompts = {s: {clip_id: normalize(row["text"]) for clip_id, row in manifests[s].items()} for s in self.speakers}

    def wrong(self, speaker: str) -> int:
        return totals(self.counts[speaker])[0]

    def words(self, speaker: str) -> int:
        return totals(self.counts[speaker])[1]

    def pooled_wrong(self) -> int:
        return sum(self.wrong(s) for s in self.speakers)

    def pooled_words(self) -> int:
        return sum(self.words(s) for s in self.speakers)

    def pooled_empty(self) -> tuple[int, int]:
        return sum(e for e, _ in self.empty.values()), sum(n for _, n in self.empty.values())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        errors = [e for s in self.speakers for e, _ in self.counts[s].values()]
        lengths = [n for s in self.speakers for _, n in self.counts[s].values()]
        return np.array(errors), np.array(lengths)


def load_pooled(label: str, folders: dict[str, str], manifests: dict[str, dict[str, dict]]) -> Pooled:
    return Pooled(label, manifests, {s: read_hyps(folders[s]) for s in manifests if s in folders})


def pooled_interval(system: Pooled, other: Pooled | None = None, by: str = "clip") -> dict:
    if by not in ("clip", "prompt"):
        raise ValueError(f"by must be 'clip' or 'prompt', not {by!r}")
    errors, lengths = system.arrays()
    clusters = [text for s in system.speakers for text in system.prompts[s].values()] if by == "prompt" else None
    if other is None:
        draws = ci.bootstrap(errors, lengths, clusters=clusters)
        low, high = ci.interval(draws["stock"])
        return {"wer": float(errors.sum() / lengths.sum()), "ci": [float(low), float(high)], "unit": by}
    other_errors, other_lengths = other.arrays()
    if not np.array_equal(lengths, other_lengths):
        raise ValueError("The two systems were scored on different clips")
    draws = ci.bootstrap(other_errors, lengths, errors, clusters=clusters)
    low, high = ci.interval(draws["delta"])
    return {"wer": float(errors.sum() / lengths.sum()),
            "gain_over_other": float((other_errors.sum() - errors.sum()) / lengths.sum()),
            "gain_ci": [float(low), float(high)], "unit": by}


def provenance(script: str | Path, inputs: dict[str, str | Path]) -> dict:
    from willspeak.artifacts import file_record, timestamp, versions

    files = {}
    for name, path in inputs.items():
        path = here(path)
        if path.is_dir():
            path = path / "hypotheses.jsonl"
        files[name] = file_record(path.resolve())
    return {"format": "willspeak-lab-v1", "created_at": timestamp(), "script": {Path(script).name: file_record(Path(script))},
            "inputs": files, "packages": {k: v for k, v in versions().items() if k in ("numpy", "torch", "nemo_toolkit")}}


def save_json(path: str | Path, value: dict) -> None:
    from willspeak.artifacts import write_json

    path = here(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, value)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Spearman needs two lists of the same length, at least 2")
    ra, rb = np.array(ranks(a)), np.array(ranks(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
