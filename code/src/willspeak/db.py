from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS prompt (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    text          TEXT NOT NULL,
    prompt_set    TEXT NOT NULL,          -- critical | functional | phonetic | probe | custom
    kind          TEXT NOT NULL,          -- word | phrase | sentence
    target_takes  INTEGER NOT NULL DEFAULT 1,
    priority      INTEGER NOT NULL DEFAULT 0,
    phones        TEXT,                   -- JSON list, cached
    oov           TEXT,                   -- JSON list of out-of-vocabulary words
    active        INTEGER NOT NULL DEFAULT 1,
    UNIQUE(text, prompt_set)
);

CREATE TABLE IF NOT EXISTS session (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    label         TEXT DEFAULT '',
    condition     TEXT DEFAULT 'unknown',  -- good_day | tired | bad_day | unknown
    mic           TEXT DEFAULT '',
    room          TEXT DEFAULT '',
    notes         TEXT DEFAULT ''
    -- Additional ratings and rater fields are added by _migrate().
);

CREATE TABLE IF NOT EXISTS clip (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES session(id),
    prompt_id     INTEGER REFERENCES prompt(id),
    prompt_text   TEXT NOT NULL,
    transcript    TEXT,                   -- actual spoken text; null = unreviewed
    path          TEXT NOT NULL,          -- canonical 16k mono wav
    raw_path      TEXT,                   -- original upload, original sample rate
    duration_s    REAL NOT NULL,
    qc            TEXT NOT NULL,          -- JSON QCResult
    qc_ok         INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending|accepted|rejected|retake
    reviewed_at   REAL,
    created_at    REAL NOT NULL,
    split         TEXT                    -- unused; splits live in data/split_assignments.json (export.py)
);

CREATE INDEX IF NOT EXISTS idx_clip_session ON clip(session_id);
CREATE INDEX IF NOT EXISTS idx_clip_prompt  ON clip(prompt_id);
CREATE INDEX IF NOT EXISTS idx_clip_status  ON clip(status);

-- Every edit to a transcript is a training signal in its own right. The KIT
-- case study folded 8.8 hours of user corrections back into training.
CREATE TABLE IF NOT EXISTS correction (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id       INTEGER NOT NULL REFERENCES clip(id),
    before_text   TEXT,
    after_text    TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'review',  -- review | live
    created_at    REAL NOT NULL
);
"""

LIVE_PROMPT = "(live)"


def is_placeholder(text: str | None) -> bool:
    return (text or "").strip().lower() == LIVE_PROMPT


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "clips").mkdir(exist_ok=True)
        (self.root / "raw").mkdir(exist_ok=True)
        self.path = self.root / "corpus.db"
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.execute("PRAGMA busy_timeout=5000")
        self.con.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.con.executescript(SCHEMA)
        self._migrate()
        self.con.commit()

    SESSION_EXTRA_COLUMNS = (
        ("self_rating_start", "REAL"),
        ("fatigue_start", "REAL"),
        ("self_rating_end", "REAL"),
        ("fatigue_end", "REAL"),
        ("rater", "TEXT"),
        ("rater_severity", "TEXT"),
    )

    def _migrate(self) -> None:
        have = {r["name"] for r in self.con.execute("PRAGMA table_info(session)")}
        for col, typ in self.SESSION_EXTRA_COLUMNS:
            if col not in have:
                self.con.execute(f"ALTER TABLE session ADD COLUMN {col} {typ}")
        self.con.execute(
            """UPDATE clip SET transcript = NULL, status = 'pending', reviewed_at = NULL
               WHERE status = 'accepted' AND prompt_text = ?
                 AND (transcript IS NULL OR LOWER(TRIM(transcript)) = ?)""",
            (LIVE_PROMPT, LIVE_PROMPT),
        )
        self.con.commit()

    def add_prompt(self, text: str, prompt_set: str, kind: str,
                   target_takes: int = 1, priority: int = 0,
                   phones: list[str] | None = None,
                   oov: list[str] | None = None) -> int | None:
        with self._lock, self.con:
            cur = self.con.execute(
                """INSERT OR IGNORE INTO prompt
                   (text, prompt_set, kind, target_takes, priority, phones, oov)
                   VALUES (?,?,?,?,?,?,?)""",
                (text, prompt_set, kind, target_takes, priority,
                 json.dumps(phones or []), json.dumps(oov or [])),
            )
            return cur.lastrowid if cur.rowcount else None

    def prompts(self, prompt_set: str | None = None, active_only: bool = True) -> list[sqlite3.Row]:
        sql = "SELECT * FROM prompt WHERE 1=1"
        params: list[Any] = []
        if prompt_set:
            sql += " AND prompt_set = ?"
            params.append(prompt_set)
        if active_only:
            sql += " AND active = 1"
        return list(self.con.execute(sql + " ORDER BY priority DESC, id", params))

    def prompt_take_counts(self) -> dict[int, int]:
        rows = self.con.execute(
            """SELECT prompt_id, COUNT(*) n FROM clip
               WHERE prompt_id IS NOT NULL AND status IN ('accepted','pending')
               GROUP BY prompt_id"""
        )
        return {r["prompt_id"]: r["n"] for r in rows}

    def start_session(self, label: str = "", condition: str = "unknown",
                      mic: str = "", room: str = "",
                      self_rating: float | None = None, fatigue: float | None = None,
                      rater: str = "", rater_severity: str = "") -> int:
        with self._lock, self.con:
            cur = self.con.execute(
                """INSERT INTO session (started_at, label, condition, mic, room,
                                        self_rating_start, fatigue_start, rater, rater_severity)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (time.time(), label, condition, mic, room, self_rating, fatigue, rater, rater_severity),
            )
            return int(cur.lastrowid)

    def end_session(self, sid: int, notes: str = "",
                    self_rating: float | None = None, fatigue: float | None = None) -> None:
        with self._lock, self.con:
            self.con.execute(
                """UPDATE session SET ended_at = ?, notes = ?,
                          self_rating_end = COALESCE(?, self_rating_end),
                          fatigue_end = COALESCE(?, fatigue_end)
                   WHERE id = ?""",
                (time.time(), notes, self_rating, fatigue, sid),
            )

    def session(self, sid: int) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM session WHERE id = ?", (sid,)).fetchone()

    def open_session(self) -> sqlite3.Row | None:
        return self.con.execute(
            "SELECT * FROM session WHERE ended_at IS NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def add_clip(self, session_id: int, prompt_id: int | None, prompt_text: str,
                 path: str, raw_path: str | None, duration_s: float,
                 qc: dict) -> int:
        with self._lock, self.con:
            cur = self.con.execute(
                """INSERT INTO clip
                   (session_id, prompt_id, prompt_text, transcript, path, raw_path,
                    duration_s, qc, qc_ok, status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (session_id, prompt_id, prompt_text, None, path, raw_path,
                 duration_s, json.dumps(qc), 1 if qc.get("ok") else 0,
                 "pending" if qc.get("ok") else "retake", time.time()),
            )
            return int(cur.lastrowid)

    def clip(self, cid: int) -> sqlite3.Row | None:
        return self.con.execute("SELECT * FROM clip WHERE id = ?", (cid,)).fetchone()

    def clips(self, status: str | None = None, session_id: int | None = None,
              limit: int = 500, offset: int = 0) -> list[sqlite3.Row]:
        sql, params = "SELECT * FROM clip WHERE 1=1", []
        if status:
            sql += " AND status = ?"; params.append(status)
        if session_id:
            sql += " AND session_id = ?"; params.append(session_id)
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        return list(self.con.execute(sql, params))

    def suggest_transcript(self, cid: int, text: str) -> None:
        with self._lock, self.con:
            row = self.clip(cid)
            if row is None:
                raise KeyError(cid)
            if row["status"] in ("accepted", "rejected"):
                return
            self.con.execute(
                "UPDATE clip SET transcript = ?, status = 'pending' WHERE id = ?",
                (text.strip() or None, cid),
            )

    def review_clip(self, cid: int, transcript: str, status: str) -> None:
        with self._lock, self.con:
            row = self.clip(cid)
            if row is None:
                raise KeyError(cid)
            if status == "accepted":
                final = transcript.strip() or (row["prompt_text"] or "").strip()
                if is_placeholder(final) or not final:
                    raise ValueError(
                        "a live clip has no prompt to fall back on: type what was said, or reject it")
            before = row["transcript"] if row["transcript"] is not None else row["prompt_text"]
            if transcript.strip() and transcript.strip() != (before or "").strip():
                self.con.execute(
                    """INSERT INTO correction (clip_id, before_text, after_text, source, created_at)
                       VALUES (?,?,?,?,?)""",
                    (cid, before, transcript.strip(), "review", time.time()),
                )
            self.con.execute(
                "UPDATE clip SET transcript = ?, status = ?, reviewed_at = ? WHERE id = ?",
                (transcript.strip() or None, status, time.time(), cid),
            )

    def totals(self) -> dict:
        row = self.con.execute(
            """SELECT COUNT(*) n,
                      COALESCE(SUM(duration_s),0) secs,
                      SUM(status='accepted') accepted,
                      SUM(status='pending')  pending,
                      SUM(status='retake')   retake,
                      SUM(status='rejected') rejected
               FROM clip"""
        ).fetchone()
        usable = self.con.execute(
            "SELECT COALESCE(SUM(duration_s),0) s FROM clip WHERE status='accepted'"
        ).fetchone()["s"]
        ncorr = self.con.execute("SELECT COUNT(*) n FROM correction").fetchone()["n"]
        return {
            "clips": row["n"] or 0,
            "total_seconds": round(row["secs"] or 0, 1),
            "usable_seconds": round(usable or 0, 1),
            "accepted": row["accepted"] or 0,
            "pending": row["pending"] or 0,
            "retake": row["retake"] or 0,
            "rejected": row["rejected"] or 0,
            "corrections": ncorr,
        }

    def accepted_texts(self) -> list[str]:
        rows = self.con.execute(
            """SELECT COALESCE(transcript, prompt_text) t FROM clip
               WHERE status = 'accepted'"""
        )
        return [r["t"] for r in rows if r["t"]]
