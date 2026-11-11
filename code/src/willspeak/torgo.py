from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .normalize import normalize

SEVERITY = {
    "F01": "severe", "M01": "severe", "M02": "severe", "M04": "severe",
    "M05": "moderate-severe",
    "F03": "moderate",
    "F04": "mild", "M03": "mild",
    "FC01": "control", "FC02": "control", "FC03": "control",
    "MC01": "control", "MC02": "control", "MC03": "control", "MC04": "control",
}
SEVERE = ("F01", "M01", "M02", "M04")
DYSARTHRIC = tuple(s for s, sev in SEVERITY.items() if sev != "control")
CONTROLS = tuple(s for s, sev in SEVERITY.items() if sev == "control")

BASE_URL = "http://www.cs.toronto.edu/~complingweb/data/TORGO/"
ARCHIVES = {"F": "F.tar.bz2", "M": "M.tar.bz2", "FC": "FC.tar.bz2", "MC": "MC.tar.bz2"}

CITATION = ("Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012). The TORGO database of "
            "acoustic and articulatory speech from speakers with dysarthria. "
            "Language Resources and Evaluation, 46(4), 523-541.")

SPLIT_FRACS = {"train": 0.70, "dev": 0.10, "test": 0.20}


@dataclass
class Utterance:
    id: str
    speaker: str
    severity: str
    session: str
    number: str
    mic: str
    audio: str
    text: str
    kind: str
    split: str

    def row(self) -> dict:
        return asdict(self)


def clean_prompt(raw: str) -> str | None:
    t = " ".join((raw or "").replace("﻿", "").split())
    if not t or t.startswith("[") or t.lower() == "xxx":
        return None
    low = t.lower()
    if ".jpg" in low or ".png" in low or "input/" in low:
        return None
    t = re.sub(r"\s*\[[^\]]*\]", "", t).strip()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r'^"(.*)"$', r"\1", t).strip()
    return t or None


def kind_of(text: str) -> str:
    return "word" if len(normalize(text).split()) == 1 else "sentence"


def split_key(text: str) -> str:
    return normalize(text, us_spelling=False)


def assign_split(text: str, fracs: dict[str, float] = SPLIT_FRACS) -> str:
    key = split_key(text)
    salt = kind_of(text)
    h = int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest(), 16) % 10_000
    edge = 0.0
    for name, frac in fracs.items():
        edge += frac * 10_000
        if h < edge:
            return name
    return list(fracs)[-1]


def scan(root: str | Path, speakers: tuple[str, ...] | None = None) -> list[Utterance]:
    root = Path(root)
    out: list[Utterance] = []
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        spk = spk_dir.name
        if spk not in SEVERITY or (speakers and spk not in speakers):
            continue
        for sess_dir in sorted(spk_dir.glob("Session*")):
            prompt_dir = sess_dir / "prompts"
            if not prompt_dir.is_dir():
                continue
            for pf in sorted(prompt_dir.glob("*.txt")):
                text = clean_prompt(pf.read_text(encoding="utf-8", errors="replace"))
                if text is None:
                    continue
                wav = None
                for mic in ("headMic", "arrayMic"):
                    cand = sess_dir / f"wav_{mic}" / f"{pf.stem}.wav"
                    if cand.is_file() and cand.stat().st_size > 44:
                        wav = cand
                        break
                if wav is None:
                    continue
                out.append(Utterance(
                    id=f"{spk}_{sess_dir.name}_{pf.stem}", speaker=spk, severity=SEVERITY[spk],
                    session=sess_dir.name, number=pf.stem, mic=mic,
                    audio=str(wav.relative_to(root)).replace("\\", "/"),
                    text=text, kind=kind_of(text), split=assign_split(text)))
    return out


def session_holdout(utts: list[Utterance]) -> tuple[list[Utterance], str]:
    from dataclasses import replace

    sessions = sorted({u.session for u in utts}, key=lambda s: (len(s), s))
    if len(sessions) < 2:
        return list(utts), f"one session ({sessions[0] if sessions else '-'}): prompt-disjoint split only"
    last = sessions[-1]
    out = []
    for u in utts:
        if u.session == last:
            if u.split == "test":
                out.append(u)
        elif u.split != "test":
            out.append(replace(u))
    return out, f"train/dev sessions {', '.join(sessions[:-1])} -> test session {last} (prompt-disjoint too)"


def summarize(utts: list[Utterance]) -> dict:
    table: dict = {}
    for u in utts:
        s = table.setdefault(u.speaker, {"severity": u.severity, "sessions": set(), "word": 0, "sentence": 0,
                                         "train": 0, "dev": 0, "test": 0, "headMic": 0, "arrayMic": 0})
        s["sessions"].add(u.session)
        s[u.kind] += 1
        s[u.split] += 1
        s[u.mic] += 1
    for s in table.values():
        s["sessions"] = len(s["sessions"])
    return table


def write_jsonl(rows: list[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return len(rows)


def check_disjoint(utts: list[Utterance]) -> None:
    seen: dict[str, str] = {}
    for u in utts:
        k = split_key(u.text)
        if k in seen and seen[k] != u.split:
            raise ValueError(f"prompt in two splits: {u.text!r} ({seen[k]} and {u.split})")
        seen[k] = u.split
