from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from willspeak import audio as A
from willspeak.db import LIVE_PROMPT, Store, is_placeholder
from willspeak.evaluate import load_manifest
from willspeak.export import export_corpus, split_key


def _store(tmp_path):
    return Store(tmp_path)


def _live_clip(store, tmp_path, name="a.wav", seconds=1.0):
    sid = store.start_session(label="live", condition="unknown")
    (tmp_path / "clips").mkdir(exist_ok=True)
    x = np.random.default_rng(len(name)).standard_normal(int(16000 * seconds)).astype("float32") * 0.1
    sf.write(tmp_path / "clips" / name, x, 16000, subtype="PCM_16")
    return store.add_clip(sid, None, LIVE_PROMPT, f"clips/{name}", None, seconds, {"ok": True})


def test_placeholder_detection():
    assert is_placeholder("(live)") and is_placeholder(" (LIVE) ") and not is_placeholder("live music")
    assert split_key("(live)", "Hello there") == "hello there"
    assert split_key("The prompt.", "the prompt") == "the prompt."


def test_suggestion_keeps_clip_pending_and_is_not_a_correction(tmp_path):
    store = _store(tmp_path)
    cid = _live_clip(store, tmp_path)
    store.suggest_transcript(cid, "hello how are you")
    row = store.clip(cid)
    assert row["status"] == "pending" and row["transcript"] == "hello how are you"
    assert store.totals()["corrections"] == 0
    assert store.accepted_texts() == []
    store.review_clip(cid, "hello, how are you?", "accepted")
    assert store.clip(cid)["status"] == "accepted"
    corr = store.con.execute("SELECT before_text, after_text FROM correction").fetchone()
    assert (corr["before_text"], corr["after_text"]) == ("hello how are you", "hello, how are you?")
    store.suggest_transcript(cid, "something else")
    assert store.clip(cid)["transcript"] == "hello, how are you?"


def test_accepting_the_placeholder_is_refused(tmp_path):
    store = _store(tmp_path)
    cid = _live_clip(store, tmp_path)
    with pytest.raises(ValueError):
        store.review_clip(cid, "", "accepted")
    with pytest.raises(ValueError):
        store.review_clip(cid, "(live)", "accepted")
    assert store.clip(cid)["status"] == "pending"
    store.review_clip(cid, "", "rejected")
    assert store.clip(cid)["status"] == "rejected"


def test_old_placeholder_labels_are_healed_on_open(tmp_path):
    store = _store(tmp_path)
    cid = _live_clip(store, tmp_path)
    store.con.execute("UPDATE clip SET transcript = '(live)', status = 'accepted' WHERE id = ?", (cid,))
    store.con.commit()
    store.con.close()
    reopened = Store(tmp_path)
    row = reopened.clip(cid)
    assert row["status"] == "pending" and row["transcript"] is None


def test_export_skips_placeholders_and_splits_live_clips_by_transcript(tmp_path):
    store = _store(tmp_path)
    texts = ["I would like some water please.", "Can you open the window?", "Where is my brother today?",
             "The bus was late again this morning.", "Please turn the light off now.",
             "I want to watch the game tonight.", "My arm hurts a little bit.", "Thank you for helping me."]
    for i, t in enumerate(texts):
        cid = _live_clip(store, tmp_path, name=f"l{i}.wav")
        store.review_clip(cid, t, "accepted")
    pending = _live_clip(store, tmp_path, name="p.wav")
    store.suggest_transcript(pending, "guess")
    legacy = _live_clip(store, tmp_path, name="q.wav")
    store.con.execute("UPDATE clip SET transcript = '(live)', status = 'accepted' WHERE id = ?", (legacy,))
    store.con.commit()

    summary = export_corpus(store, tmp_path, dev_frac=0.25, test_frac=0.25, trim=False)
    skipped = {s["clip_id"]: s["why"] for s in summary["skipped"]}
    assert skipped == {legacy: "no verified transcript"}
    assigned = json.loads((tmp_path / "split_assignments.json").read_text())
    assert "(live)" not in assigned
    assert set(assigned) == {t.lower() for t in texts}
    counts = summary["counts"]
    assert counts["train"] + counts["dev"] + counts["test"] == len(texts)
    assert counts["dev"] >= 1 and counts["test"] >= 1, "live clips must not all land in one split"
    for split in ("train", "dev", "test"):
        for row in load_manifest(tmp_path / "export" / f"{split}.jsonl").values():
            assert not is_placeholder(row["text"]) and row["prompt_text"] == LIVE_PROMPT


def test_manifest_loader_refuses_placeholder_reference(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps({"id": 1, "audio": "test/000001.wav", "text": "(live)"}) + "\n")
    with pytest.raises(ValueError):
        load_manifest(p)


def test_same_audio_detects_repeats_but_not_retakes():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(16000).astype("float32") * 0.1
    assert A.same_audio(x, x, 16000)
    assert A.same_audio(x, x[200:], 16000)
    assert A.same_audio(x, (x * 0.7 + rng.standard_normal(16000).astype("float32") * 0.002), 16000)
    y = rng.standard_normal(16000).astype("float32") * 0.1
    assert not A.same_audio(x, y, 16000)
    assert not A.same_audio(x, x[:8000], 16000)
    assert not A.same_audio(np.zeros(1600, "float32"), np.zeros(1600, "float32"), 16000)


@pytest.mark.parametrize("padding", [0, 17, 201])
def test_duplicate_correlation_matches_direct_calculation(padding):
    rng = np.random.default_rng(11)
    x = rng.standard_normal(2000).astype("float32")
    for correlated in (True, False):
        y = np.pad(x * 0.7, (padding, 9)) if correlated else rng.standard_normal(len(x) + padding + 9)
        y = (y + 12).astype("float32")
        centered = x.astype(float) - x.mean(dtype=float)
        direct = max(np.corrcoef(centered, y[i:i + len(x)])[0, 1] for i in range(len(y) - len(x) + 1))
        assert A.same_audio(x, y, 16000) == (direct >= 0.98)


def test_review_rolls_back_correction_if_clip_update_fails(tmp_path):
    import sqlite3

    store = _store(tmp_path)
    cid = _live_clip(store, tmp_path)
    store.con.execute("""CREATE TRIGGER fail_review BEFORE UPDATE ON clip
                         BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END""")
    store.con.commit()
    with pytest.raises(sqlite3.IntegrityError, match="simulated"):
        store.review_clip(cid, "water", "accepted")
    assert store.clip(cid)["status"] == "pending"
    assert store.totals()["corrections"] == 0
    assert not store.con.in_transaction


@pytest.mark.parametrize("operation", ["prompt", "start", "end", "clip", "suggest", "review"])
def test_all_database_writes_wait_for_snapshot_lock(tmp_path, operation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    store = _store(tmp_path)
    cid = _live_clip(store, tmp_path)
    sid = store.clip(cid)["session_id"]
    actions = {
        "prompt": lambda: store.add_prompt("water", "custom", "word"),
        "start": lambda: store.start_session(), "end": lambda: store.end_session(sid),
        "clip": lambda: store.add_clip(sid, None, "water", "clips/another.wav", None, 1, {"ok": True}),
        "suggest": lambda: store.suggest_transcript(cid, "water"),
        "review": lambda: store.review_clip(cid, "water", "accepted"),
    }
    started = Event()

    def write():
        started.set()
        return actions[operation]()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store._lock:
            pending = pool.submit(write)
            assert started.wait(2)
            assert not pending.done()
        pending.result(timeout=2)
