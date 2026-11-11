from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import soundfile as sf

from . import audio as A
from .db import LIVE_PROMPT, is_placeholder
from .normalize import normalize


def split_key(prompt_text: str, text: str) -> str:
    base = text if is_placeholder(prompt_text) else prompt_text
    return base.strip().lower()


def _bucket(text: str) -> float:
    h = hashlib.sha256(text.strip().lower().encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _assign_splits(rows, data_dir: Path, dev_frac: float, test_frac: float,
                   min_dev_prompts: int, min_test_prompts: int, persist: bool = True) -> dict[str, str]:
    assign_path = data_dir / "split_assignments.json"
    assigned: dict[str, str] = {}
    if assign_path.exists():
        assigned = json.loads(assign_path.read_text(encoding="utf-8"))
        if (not isinstance(assigned, dict)
                or any(not isinstance(k, str) or v not in ("train", "dev", "test") for k, v in assigned.items())):
            raise ValueError("Invalid split_assignments.json; restore its backup before exporting")

    def key(text: str) -> str:
        return text.strip().lower()

    assigned.pop(key(LIVE_PROMPT), None)

    fresh: list[tuple[float, str]] = []
    seen: set[str] = set()
    for r in rows:
        if r["prompt_set"] in ("critical", "probe"):
            continue
        k = split_key(r["prompt_text"], r["transcript"] or "")
        if not k or is_placeholder(k) or k in seen:
            continue
        seen.add(k)
        if k not in assigned:
            fresh.append((_bucket(k), k))

    for b, k in fresh:
        assigned[k] = "test" if b < test_frac else (
            "dev" if b < test_frac + dev_frac else "train")

    fresh_train = sorted(b_k for b_k in fresh if assigned[b_k[1]] == "train")
    for sp, want in (("test", min_test_prompts), ("dev", min_dev_prompts)):
        have = sum(1 for k in seen if assigned.get(k) == sp)
        while have < want and fresh_train:
            _, k = fresh_train.pop(0)
            assigned[k] = sp
            have += 1

    if persist:
        assign_path.write_text(json.dumps(assigned, indent=1, sort_keys=True), encoding="utf-8")
    return {k: v for k, v in assigned.items()}


def export_corpus(store, data_dir: Path, out_dir: Path | str | None = None,
                  dev_frac: float = 0.1, test_frac: float = 0.1,
                  trim: bool = True, min_duration: float = 0.3,
                  min_dev_prompts: int = 2, min_test_prompts: int = 2) -> dict:
    data_dir = Path(data_dir).resolve()
    destination = Path(out_dir) if out_dir else data_dir / "export"
    if destination.is_symlink():
        raise ValueError("Export destination must not be a symbolic link")
    out = destination.resolve()
    if (not all(math.isfinite(v) and 0 <= v < 1 for v in (dev_frac, test_frac))
            or dev_frac + test_frac >= 1):
        raise ValueError("dev/test fractions must be nonnegative and sum to less than 1")
    if data_dir.is_relative_to(out) or any(out.is_relative_to(data_dir / name) for name in ("clips", "raw", "backups")):
        raise ValueError("Export destination must not replace the corpus or its recordings/backups")
    if out.exists() and not out.is_dir():
        raise ValueError("Export destination must be a directory")
    if not math.isfinite(min_duration) or min_duration < 0 or min(min_dev_prompts, min_test_prompts) < 0:
        raise ValueError("Minimum duration and split sizes must be nonnegative")
    out.parent.mkdir(parents=True, exist_ok=True)
    with store._lock, tempfile.TemporaryDirectory(prefix=".willspeak-export-", dir=out.parent) as temporary:
        work = Path(temporary)
        stage, previous = work / "export", work / "previous"
        summary, assignments = _build_export(store, data_dir, stage, dev_frac, test_frac, trim, min_duration,
                                              min_dev_prompts if dev_frac else 0,
                                              min_test_prompts if test_frac else 0)
        summary["out_dir"] = str(out)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        assignment_file = work / "split_assignments.json"
        assignment_file.write_text(json.dumps(assignments, indent=1, sort_keys=True), encoding="utf-8")
        if out.exists():
            out.rename(previous)
        published = False
        try:
            stage.rename(out)
            published = True
            assignment_file.replace(data_dir / "split_assignments.json")
        except BaseException:
            if published:
                shutil.rmtree(out)
            if previous.exists():
                previous.rename(out)
            raise
    return summary


def _build_export(store, data_dir, out, dev_frac, test_frac, trim, min_duration,
                  min_dev_prompts, min_test_prompts):
    for sp in ("train", "dev", "test"):
        (out / sp).mkdir(parents=True, exist_ok=True)

    rows = store.con.execute(
        """SELECT c.*, p.prompt_set, s.condition, s.started_at AS session_started_at,
                  s.self_rating_start, s.fatigue_start, s.self_rating_end, s.fatigue_end,
                  s.rater_severity
           FROM clip c
           LEFT JOIN prompt p ON p.id = c.prompt_id
           LEFT JOIN session s ON s.id = c.session_id
           WHERE c.status = 'accepted'
           ORDER BY c.id"""
    ).fetchall()

    split_of = _assign_splits(rows, data_dir, dev_frac, test_frac,
                              min_dev_prompts, min_test_prompts, persist=False)

    manifest: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    counts = {"train": 0, "dev": 0, "test": 0}
    secs = {"train": 0.0, "dev": 0.0, "test": 0.0}
    skipped = []
    labels_seen = {}

    for r in rows:
        text = (r["transcript"] or r["prompt_text"]).strip()
        if not text or is_placeholder(text):
            skipped.append({"clip_id": r["id"], "why": "no verified transcript"})
            continue
        if r["duration_s"] < min_duration:
            skipped.append({"clip_id": r["id"], "why": "too short"})
            continue

        if r["prompt_set"] in ("critical", "probe"):
            split = "train"
        else:
            split = split_of.get(split_key(r["prompt_text"], text), "train")

        src = data_dir / r["path"]
        if not src.exists():
            skipped.append({"clip_id": r["id"], "why": "missing audio file"})
            continue

        x, sr = sf.read(src, dtype="float32")
        if trim:
            x = A.trim_silence(x, sr)
        dur = len(x) / sr
        if dur < min_duration:
            skipped.append({"clip_id": r["id"], "why": "too short after trimming"})
            continue
        for label in (text, r["prompt_text"]):
            if is_placeholder(label):
                continue
            normalized = normalize(label, us_spelling=False)
            if not normalized:
                raise ValueError(f"Clip {r['id']} has a label with no scorable words; review its transcript")
            if normalized in labels_seen and labels_seen[normalized] != split:
                raise ValueError(f"Transcript/prompt {label!r} would appear in two splits; "
                                 "review the conflicting labels before exporting")
            labels_seen[normalized] = split
        dst_name = f"{r['id']:06d}.wav"
        A.write_wav(str(out / split / dst_name), x, sr, subtype="PCM_16")

        manifest[split].append({
            "id": r["id"],
            "audio": f"{split}/{dst_name}",
            "text": text,
            "prompt_text": r["prompt_text"],
            "prompt_set": r["prompt_set"] or "custom",
            "duration_s": round(dur, 3),
            "session_id": r["session_id"],
            "condition": r["condition"] or "unknown",
            "minutes_into_session": (round((r["created_at"] - r["session_started_at"]) / 60.0, 1)
                                     if r["session_started_at"] else None),
            "self_rating_start": r["self_rating_start"],
            "fatigue_start": r["fatigue_start"],
            "self_rating_end": r["self_rating_end"],
            "fatigue_end": r["fatigue_end"],
            "rater_severity": r["rater_severity"] or None,
            "corrected": bool(r["transcript"] and r["transcript"].strip()
                              != r["prompt_text"].strip()),
        })
        counts[split] += 1
        secs[split] += dur

    for sp in ("train", "dev", "test"):
        with open(out / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest[sp])

    corr = store.con.execute(
        """SELECT c.*, cl.path FROM correction c
           JOIN clip cl ON cl.id = c.clip_id ORDER BY c.id"""
    ).fetchall()
    with open(out / "corrections.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({
                "clip_id": c["clip_id"], "before": c["before_text"],
                "after": c["after_text"], "source": c["source"],
            }, ensure_ascii=False) + "\n" for c in corr)

    empty = [sp for sp in ("dev", "test") if counts[sp] == 0]
    summary = {
        "warnings": [
            f"'{sp}' split is empty -the eval harness needs it. With a small "
            "corpus, raise its fraction or record more distinct prompts."
            for sp in empty
        ],
        "counts": counts,
        "minutes": {k: round(v / 60, 1) for k, v in secs.items()},
        "total_minutes": round(sum(secs.values()) / 60, 1),
        "skipped": skipped,
        "out_dir": str(out),
        "split_rule": "by prompt text, sticky across exports "
                      "(data/split_assignments.json); critical/probe sets forced to train; "
                      "normalized transcript/prompt overlap across splits refused",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, split_of
