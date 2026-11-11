from __future__ import annotations

import glob
from pathlib import Path

import phrasebook
from willspeak.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]


def build_pool(patterns: list[str], kinds: list[str]) -> list[phrasebook.Item]:
    files = sorted({p for g in patterns for p in glob.glob(g if Path(g).is_absolute() else str(ROOT / g))})
    if not files:
        raise ValueError("No manifest files match --pool-glob")
    seen: dict[str, phrasebook.Item] = {}
    for path in files:
        for row in phrasebook.read_rows(path):
            text = str(row.get("text", ""))
            if "(live)" in text or phrasebook.kind_of(row) not in kinds:
                continue
            norm = normalize(text)
            if norm and norm not in seen:
                seen[norm] = phrasebook.Item(text)
    return list(seen.values())


def rank_pool(hyp: phrasebook.Item, pool: list[phrasebook.Item]) -> list[tuple[int, float]]:
    return sorted(((i, phrasebook.distance(hyp, p)) for i, p in enumerate(pool)), key=lambda x: (x[1], x[0]))


def pick_candidates(ranking: list[tuple[int, float]], pool: list[phrasebook.Item], true_norm: str, hyp_norm: str,
                    shortlist: int) -> list[dict]:
    rank_of = {index: r for r, (index, _d) in enumerate(ranking)}
    dist_of = dict(ranking)
    out: list[dict] = []
    seen: set[str] = set()
    for index, d in ranking[: shortlist + 1]:
        text = pool[index].norm
        out.append({"text": text, "dist": d, "rank": rank_of[index], "true": text == true_norm, "hyp": text == hyp_norm, "in_pool": True})
        seen.add(text)
    for index, p in enumerate(pool):
        if p.norm == true_norm and true_norm not in seen:
            out.append({"text": true_norm, "dist": dist_of[index], "rank": rank_of[index], "true": True, "hyp": true_norm == hyp_norm, "in_pool": True})
            seen.add(true_norm)
    if hyp_norm and hyp_norm not in seen:
        out.append({"text": hyp_norm, "dist": 0.0, "rank": None, "true": False, "hyp": True, "in_pool": False})
    return out
