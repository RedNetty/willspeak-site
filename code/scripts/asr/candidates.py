#!/usr/bin/env python3
"""The speech models the lab tests compare, with the names to load them and what each one can be used for."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    tag: str
    family: str
    kind: str
    name: str
    size: str
    loss: str
    finetune: bool
    role: str
    note: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("v2", "nemo", "tdt", "nvidia/parakeet-tdt-0.6b-v2", "0.6B", "tdt", True, "screen",
              "the model the project uses; English only; Open ASR average WER about 6.1%"),
    Candidate("v3", "nemo", "tdt", "nvidia/parakeet-tdt-0.6b-v3", "0.6B", "tdt", True, "screen",
              "25 European languages; Open ASR average 6.34% per its card; same architecture as v2"),
    Candidate("1p1b", "nemo", "tdt", "nvidia/parakeet-tdt-1.1b", "1.1B", "tdt", True, "screen",
              "larger Parakeet-TDT; also in lab.py stage B"),
    Candidate("110m", "nemo", "tdt-ctc hybrid", "nvidia/parakeet-tdt_ctc-110m", "110M", "tdt", True, "screen",
              "small FastConformer TDT-CTC hybrid, the small-device test in lab.py stage B; decoded here with its TDT head"),
    Candidate("nemotron", "nemo", "cache-aware rnnt", "nvidia/nemotron-speech-streaming-en-0.6b", "0.6B", "rnnt", True,
              "screen", "streaming FastConformer-RNNT, 24 encoder layers; needs NeMo 25.11 or newer; decoded here "
                        "with its default context"),
    Candidate("rnnt06", "nemo", "rnnt", "nvidia/parakeet-rnnt-0.6b", "0.6B", "rnnt", True, "screen",
              "the 2023 FastConformer-RNNT; Open ASR average 7.5%"),
    Candidate("ctc06", "nemo", "ctc", "nvidia/parakeet-ctc-0.6b", "0.6B", "-", False, "screen",
              "CTC: no prediction network, so train.py cannot train it; stock screen only"),
    Candidate("qwen3", "hf", "qwen3-asr", "Qwen/Qwen3-ASR-0.6B", "0.6B", "-", False, "screen",
              "LLM-style decoder; needs the qwen-asr package; Apache-2.0"),
    Candidate("cohere", "hf", "cohere", "CohereLabs/cohere-transcribe-03-2026", "2B", "-", False, "probe",
              "needs transformers 5.4 or newer; Apache-2.0; 5.42% average WER on its card"),
    Candidate("granite41", "hf", "granite", "ibm-granite/granite-speech-4.1-2b", "2B", "-", False, "probe",
              "speech-LLM; transformers 4.52.1 or newer; Apache-2.0"),
    Candidate("granite4", "hf", "granite", "ibm-granite/granite-4.0-1b-speech", "1B-2B", "-", False, "probe",
              "smaller Granite speech model; its card states 2B parameters; same loader as granite41"),
)

BY_TAG = {c.tag: c for c in CANDIDATES}
assert len(BY_TAG) == len(CANDIDATES), "duplicate tag"


def get(tag: str) -> Candidate:
    if tag not in BY_TAG:
        raise KeyError(f"Unknown candidate {tag!r}; known: {', '.join(BY_TAG)}")
    return BY_TAG[tag]


def select(family: str = "all", role: str = "all", tags: list[str] | None = None) -> list[Candidate]:
    chosen = [c for c in CANDIDATES if family in ("all", c.family) and role in ("all", c.role)]
    if tags:
        unknown = [t for t in tags if t not in BY_TAG]
        if unknown:
            raise KeyError(f"Unknown candidate(s) {unknown}")
        chosen = [c for c in chosen if c.tag in tags]
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="all", choices=["all", "nemo", "hf"])
    ap.add_argument("--role", default="all", choices=["all", "screen", "probe"])
    ap.add_argument("--tags", nargs="*", help="only these tags")
    ap.add_argument("--finetune", action="store_true", help="only models train.py can train")
    ap.add_argument("--lines", action="store_true", help="print tag|kind|name lines for a shell loop instead of a table")
    a = ap.parse_args()
    rows = select(a.family, a.role, a.tags)
    if a.finetune:
        rows = [c for c in rows if c.finetune]
    if a.lines:
        for c in rows:
            print(f"{c.tag}|{c.kind}|{c.name}")
        return 0
    for c in rows:
        print(f"{c.tag:<10} {c.family:<5} {c.size:<6} {c.role:<7} train.py {'yes' if c.finetune else 'no':<3} {c.name}\n"
              f"{'':<10} {c.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
