from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence

import cmudict

SIL = "_"

ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]

_STRESS = re.compile(r"\d")
_WORD = re.compile(r"[a-z']+")


@functools.lru_cache(maxsize=1)
def _dict() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def normalize_word(w: str) -> str:
    return w.lower().strip(".,!?;:\"()[]—-")


VOICELESS = {"P", "T", "K", "F", "TH"}
SIBILANT = {"S", "Z", "SH", "ZH", "CH", "JH"}


def word_phones(word: str) -> list[str] | None:
    pronunciations = _dict()
    norm = normalize_word(word)
    if not norm:
        return None
    if norm in pronunciations:
        return [_STRESS.sub("", p) for p in pronunciations[norm][0]]
    flat = norm.replace("'", "")
    if flat in pronunciations:
        return [_STRESS.sub("", p) for p in pronunciations[flat][0]]
    for suf in ("'s", "s'", "es", "s"):
        if norm.endswith(suf) and len(norm) > len(suf) + 1:
            stem = norm[: -len(suf)]
            if stem in pronunciations:
                ph = [_STRESS.sub("", p) for p in pronunciations[stem][0]]
                last = ph[-1]
                if last in SIBILANT:
                    ph += ["IH", "Z"]
                elif last in VOICELESS:
                    ph += ["S"]
                else:
                    ph += ["Z"]
                return ph
    return None


@functools.lru_cache(maxsize=8192)
def text_phones(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    phones: list[str] = [SIL]
    oov: list[str] = []
    for raw in _WORD.findall(text.lower()):
        p = word_phones(raw)
        if p is None:
            oov.append(raw)
            continue
        phones.extend(p)
    phones.append(SIL)
    return tuple(phones), tuple(oov)


def diphones(phones: Sequence[str]) -> set[tuple[str, str]]:
    return {(phones[i], phones[i + 1]) for i in range(len(phones) - 1)}


def text_diphones(text: str) -> set[tuple[str, str]]:
    ph, _ = text_phones(text)
    return diphones(ph)


class Coverage:
    def __init__(self) -> None:
        self.phone_counts: dict[str, int] = {}
        self.diphone_counts: dict[tuple[str, str], int] = {}
        self.n_utterances = 0

    def add(self, text: str, times: int = 1) -> None:
        ph, _ = text_phones(text)
        for p in ph:
            self.phone_counts[p] = self.phone_counts.get(p, 0) + times
        for d in diphones(ph):
            self.diphone_counts[d] = self.diphone_counts.get(d, 0) + times
        self.n_utterances += times

    def add_many(self, texts: Iterable[str]) -> None:
        for t in texts:
            self.add(t)

    def gain(self, text: str, min_count: int = 1) -> int:
        return sum(
            1
            for d in text_diphones(text)
            if self.diphone_counts.get(d, 0) < min_count
        )

    def missing_phones(self) -> list[str]:
        return sorted(p for p in ARPABET if self.phone_counts.get(p, 0) == 0)

    def thin_phones(self, min_count: int = 5) -> list[tuple[str, int]]:
        return sorted(
            ((p, self.phone_counts.get(p, 0)) for p in ARPABET
             if self.phone_counts.get(p, 0) < min_count),
            key=lambda x: x[1],
        )

    def stats(self, min_count: int = 1) -> dict:
        covered_ph = sum(1 for p in ARPABET if self.phone_counts.get(p, 0) >= min_count)
        n_di = sum(1 for c in self.diphone_counts.values() if c >= min_count)
        return {
            "utterances": self.n_utterances,
            "phones_covered": covered_ph,
            "phones_total": len(ARPABET),
            "phone_coverage_pct": round(100 * covered_ph / len(ARPABET), 1),
            "diphones_observed": n_di,
            "distinct_diphones_seen": len(self.diphone_counts),
        }
