from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tests.helpers import fake_wav


@pytest.fixture()
def client(monkeypatch, tmp_path):
    tmp = tmp_path
    monkeypatch.setenv("WILLSPEAK_DATA", str(tmp))
    monkeypatch.setenv("WILLSPEAK_MATCH_EAGER", "0")
    monkeypatch.setenv("WILLSPEAK_ASR_EAGER", "0")
    for m in list(sys.modules):
        if m.startswith("willspeak"):
            del sys.modules[m]
    from fastapi.testclient import TestClient

    from willspeak.server import app, store

    with TestClient(app) as c:
        yield c, store, tmp
    store.con.close()


def test_difficulty_feedback_requires_development_data(client):
    c, _, _ = client
    development = {"format": "willspeak-difficulty-v1", "source_split": "dev", "weights": {"NG": 4.0}}
    assert c.post("/api/difficulty", json=development).status_code == 200
    for split in ("test", "train", "unknown"):
        response = c.post("/api/difficulty", json={**development, "source_split": split})
        assert response.status_code == 400 and "development" in response.json()["detail"]
        assert c.get("/api/difficulty").json()["weights"] == {"NG": 4.0}
    assert c.post("/api/difficulty", json={}).status_code == 200


@pytest.mark.parametrize("weights", [{"NG": 0}, {"NG": -1}, {"NG": 6}, {"NG": True},
                                      {"NG": float("nan")}, {"NG": float("inf")}, {"UNKNOWN": 2}])
def test_invalid_difficulty_leaves_previous_weights(client, weights):
    c, _, _ = client
    assert c.post("/api/difficulty", json={"NG": 2}).status_code == 200
    response = c.post("/api/difficulty", content=json.dumps({"ZH": 3, **weights}),
                      headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert c.get("/api/difficulty").json()["weights"] == {"NG": 2}


def test_uploads_in_same_millisecond_keep_separate_audio(client, monkeypatch):
    from willspeak import server

    c, store, _ = client
    sid = store.start_session()
    monkeypatch.setattr(server.time, "time", lambda: 1234567890.0)
    for level in (0.1, 0.2):
        response = c.post("/api/clip", data={"session_id": sid, "prompt_text": "hello"},
                          files={"audio": ("clip.wav", fake_wav(level=level), "audio/wav")})
        assert response.status_code == 200
    clips = store.clips()
    assert len({row["path"] for row in clips}) == 2


def test_reject_missing_live_clip_returns_404(client):
    c, _, _ = client
    assert c.post("/api/live/fix", data={"clip_id": 999, "correct_text": ""}).status_code == 404


def test_upload_with_missing_session_keeps_no_audio(client):
    c, store, tmp = client
    response = c.post("/api/clip", data={"session_id": 999, "prompt_text": "hello"},
                      files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
    assert response.status_code == 404 and store.clips() == []
    assert not list((tmp / "clips").glob("*.wav"))


def test_pending_stop_cancellation_preserves_preceding_spoken_clip(client, monkeypatch):
    import time

    from willspeak import server

    c, store, _ = client
    sid = store.start_session()
    cid = store.add_clip(sid, None, "(live)", "clips/previous.wav", None, 1, {"ok": True})
    store.suggest_transcript(cid, "hello")
    server._last_spoken.update(at=time.time(), clip_id=cid, text="hello")
    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", lambda *_: {"text": "purple elephant"})
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    response = c.post("/api/live/transcribe", data={"pending_output": "true"},
                      files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
    assert response.status_code == 200 and response.json()["decision"]["action"] == "stop"
    assert not response.json()["decision"]["retract"]
    assert store.clip(cid)["status"] == "pending" and store.clip(cid)["transcript"] == "hello"


def test_retraction_window_counts_from_the_end_of_playback(client, monkeypatch):
    import time

    from willspeak import server

    c, store, _ = client
    sid = store.start_session()
    cid = store.add_clip(sid, None, "(live)", "clips/previous.wav", None, 1, {"ok": True})
    r = c.post("/api/live/confirm", data={"clip_id": cid, "spoken_text": "a long sentence"})
    assert r.status_code == 200
    server._last_spoken["at"] = time.time() - 10
    r = c.post("/api/live/confirm", data={"clip_id": cid, "spoken_text": "a long sentence", "stage": "end"})
    assert r.status_code == 200
    assert time.time() - server._last_spoken["at"] < 2
    stamp = server._last_spoken["at"]
    other = {"clip_id": cid + 99, "spoken_text": "x", "stage": "end"}
    assert c.post("/api/live/confirm", data=other).status_code == 200
    assert server._last_spoken["at"] == stamp
    assert c.post("/api/live/confirm", data={"clip_id": cid, "spoken_text": "x", "stage": "middle"}).status_code == 400

    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", lambda *_: {"text": "purple elephant"})
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    response = c.post("/api/live/transcribe", files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
    decision = response.json()["decision"]
    assert decision["action"] == "stop" and decision["retract"] is True
    assert store.clip(cid)["status"] == "rejected"


def test_recogniser_error_is_a_shown_decision_not_a_500(client, monkeypatch):
    from willspeak import server

    c, store, _ = client

    def broken(*_):
        raise RuntimeError("onnxruntime: out of memory")

    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", broken)
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    r = c.post("/api/live/transcribe", files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and "recogniser error" in body["reason"]
    assert body["decision"]["action"] == "ignore"
    assert store.clip(body["clip_id"]) is not None


def test_live_master_keeps_edges_and_original_rate(client, monkeypatch):
    import soundfile as sf

    from willspeak import server

    c, store, tmp = client
    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", lambda *_: {"text": "hello there"})
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    r = c.post("/api/live/transcribe", files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
    assert r.status_code == 200 and r.json()["ok"] is True
    row = store.clip(r.json()["clip_id"])
    master, sr = sf.read(tmp / row["path"], dtype="float32")
    assert sr == 16000
    assert row["raw_path"] and (tmp / row["raw_path"]).exists()
    assert abs(len(master) / sr - row["duration_s"]) < 0.01
    assert abs(len(master) / sr - 1.6) < 0.01


def test_first_sentence_of_a_warning_keeps_its_number():
    from willspeak.server import _first_sentence

    assert _first_sentence("Clipping on 6.36% of samples. Lower the gain.") == "Clipping on 6.36% of samples"
    assert _first_sentence("Quiet (-30.1 dBFS). Move the microphone closer.") == "Quiet (-30.1 dBFS)"
    assert _first_sentence("Recorded 2 channels; downmixed to mono.") == "Recorded 2 channels; downmixed to mono"


def test_review_removes_stale_matcher_exemplar(client, monkeypatch):
    from willspeak import server
    from willspeak.matcher.engine import DIM

    c, store, tmp = client
    sid = store.start_session()
    (tmp / "clips/x.wav").write_bytes(fake_wav())
    cid = store.add_clip(sid, None, "water", "clips/x.wav", None, 1, {"ok": True})
    store.review_clip(cid, "water", "accepted")
    matcher = server.matcher
    monkeypatch.setattr(matcher, "available", lambda: True)
    monkeypatch.setattr(matcher, "emb", lambda _: np.arange(DIM, dtype=np.float32))
    assert matcher.add_exemplar(cid, "water", np.zeros(16000), sid)
    assert matcher.take_counts() == {"water": 1}
    response = c.post("/api/live/fix", data={"clip_id": cid, "correct_text": "help"})
    assert response.json()["exemplar_added"]
    assert matcher.take_counts() == {"help": 1}
    assert c.post(f"/api/clip/{cid}/review", data={"status": "rejected"}).status_code == 200
    assert matcher.take_counts() == {} and matcher.bank_mean is None
    assert not matcher.cache_path.exists()
    assert not matcher.add_exemplar(cid, "this sentence has more than six words", np.zeros(16000), sid)


def test_empty_and_nonfinite_audio_returns_400(client):
    import io

    import soundfile as sf

    c, store, _ = client
    sid = store.start_session()
    for samples in (np.array([], np.float32), np.array([np.nan] * 16000, np.float32)):
        buffer = io.BytesIO()
        sf.write(buffer, samples, 16000, format="WAV", subtype="FLOAT")
        response = c.post("/api/clip", data={"session_id": sid, "prompt_text": "hello"},
                          files={"audio": ("bad.wav", buffer.getvalue(), "audio/wav")})
        assert response.status_code == 400
    assert store.clips() == []


@pytest.mark.parametrize("phrase", ["purple elephant", "voice go to sleep"])
def test_control_phrase_preview_does_not_change_live_state(client, monkeypatch, phrase):
    import time

    from willspeak import server

    c, store, _ = client
    sid = store.start_session()
    cid = store.add_clip(sid, None, "(live)", "clips/previous.wav", None, 1, {"ok": True})
    store.suggest_transcript(cid, "hello")
    server._last_spoken.update(at=time.time(), clip_id=cid, text="hello")
    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", lambda *_: {"text": phrase})
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    response = c.post("/api/live/transcribe", data={"preview_only": "true"},
                      files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
    assert response.status_code == 200
    assert response.json()["decision"]["action"] in {"stop", "sleep"}
    assert server._live_state["awake"]
    assert store.clip(cid)["status"] == "pending"
    assert server._last_spoken["clip_id"] == cid


def test_status_responds_while_recognition_is_running(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from willspeak import server

    c, _, _ = client
    started, release = Event(), Event()

    def transcribe(*_):
        started.set()
        assert release.wait(5)
        return {"text": "hello"}

    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", transcribe)
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        upload = pool.submit(c.post, "/api/live/transcribe",
                             files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
        try:
            assert started.wait(5)
            status = pool.submit(c.get, "/api/live/awake")
            assert status.result(timeout=2).status_code == 200
        finally:
            release.set()
        assert upload.result(timeout=5).status_code == 200


def test_parakeet_configuration(monkeypatch, tmp_path):
    for m in list(sys.modules):
        if m.startswith("willspeak"):
            del sys.modules[m]
    monkeypatch.setenv("WILLSPEAK_DATA", str(tmp_path))
    from willspeak.asr.parakeet import ParakeetASR, model_files

    monkeypatch.setenv("WILLSPEAK_PARAKEET_DIR", str(tmp_path / "nope"))
    assert model_files(tmp_path / "nope") is None
    d = tmp_path / "pk"
    d.mkdir()
    for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (d / f).write_bytes(b"x")
    monkeypatch.setenv("WILLSPEAK_PARAKEET_DIR", str(d))
    pk = ParakeetASR()
    assert pk.model_present() and pk.available() == pk.installed()


def test_session_ratings_probe_and_export(client):
    c, store, tmp = client
    probe = c.get("/api/probe").json()["prompts"]
    assert len(probe) == 10 and probe[0]["prompt_set"] == "probe"
    queued = c.get("/api/next?n=200").json()["prompts"]
    assert all(p["prompt_set"] != "probe" for p in queued)

    r = c.post(
        "/api/session/start",
        data={
            "condition": "bad_day",
            "self_rating": "3",
            "fatigue": "8",
            "rater": "Mom",
            "rater_severity": "Severe",
            "mic": "wheelchair",
        },
    ).json()
    sid = r["session_id"]
    s = store.session(sid)
    assert (s["self_rating_start"], s["fatigue_start"], s["rater"], s["rater_severity"]) == (3.0, 8.0, "Mom", "severe")
    assert s["self_rating_end"] is None
    for p in (probe[0], queued[0]):
        up = c.post(
            "/api/clip",
            files={"audio": ("t.wav", fake_wav(), "audio/wav")},
            data={"session_id": sid, "prompt_text": p["text"], "prompt_id": p["prompt_id"]},
        ).json()
        c.post(f"/api/clip/{up['clip_id']}/review", data={"transcript": p["text"], "status": "accepted"})
    c.post("/api/session/end", data={"session_id": sid, "self_rating": "2", "fatigue": "9.5"})
    s = store.session(sid)
    assert (s["self_rating_end"], s["fatigue_end"]) == (2.0, 9.5) and s["ended_at"]
    sid2 = c.post("/api/session/start", data={"self_rating": "", "fatigue": "14"}).json()["session_id"]
    s2 = store.session(sid2)
    assert s2["self_rating_start"] is None and s2["fatigue_start"] == 10.0

    from willspeak.export import export_corpus

    export_corpus(store, tmp, dev_frac=0.0, test_frac=0.0)
    rows = [json.loads(line) for line in (tmp / "export" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(r["prompt_set"] in ("probe", "functional", "critical") for r in rows)
    r0 = rows[0]
    assert r0["self_rating_start"] == 3.0 and r0["fatigue_end"] == 9.5 and r0["rater_severity"] == "severe"
    assert r0["minutes_into_session"] is not None and 0 <= r0["minutes_into_session"] < 1
    assert all(r["prompt_set"] != "probe" or r["audio"].startswith("train/") for r in rows)

    st = c.get("/api/live/status").json()
    assert st["asr_backend"] == "parakeet" and "projection" in st["matcher"]
    assert c.get("/api/live/vocab").status_code == 200


def test_matcher_status_without_models(client):
    _c, _store, _tmp = client
    from willspeak.matcher.engine import DIM
    from willspeak.server import matcher

    st = matcher.status()
    assert st["projection_dim"] == 128
    assert st["pool"] == "mean+std" and st["session_warmup"] == 20.0
    assert "session transform" in st["projection"]
    assert matcher.proj.T is None and matcher.proj.W.shape == (DIM, 128) and DIM == 1536
    rng = np.random.default_rng(0)
    matcher.ids = [f"clip:{i}" for i in range(6)]
    matcher.labels = ["water", "water", "help", "help", "water", "help"]
    matcher.sessions = ["s1", "s1", "s1", "s2", "s2", "s2"]
    matcher.raw = [rng.normal(size=DIM).astype(np.float32) for _ in range(6)]
    matcher.new_session()
    assert set(matcher.sess_vecs) == {"s1", "s2"} and matcher.vecs.shape == (6, 128)
    assert matcher.proto_labels == ["water", "help"] and matcher.protos.shape == (2, 128)
    assert matcher.take_counts() == {"water": 3, "help": 3}
    assert matcher.live_n == 0 and np.allclose(matcher.live_mean, np.stack(matcher.raw).mean(0), atol=1e-5)
    v = rng.normal(size=DIM).astype(np.float32)
    prior = matcher.live_mean.copy()
    matcher._live_norm(v, update=True)
    assert matcher.live_n == 1 and np.allclose(matcher.live_mean, (20 * prior + v) / 21, atol=1e-4)


def test_live_clips_stay_pending_until_a_person_confirms(client):
    c, store, tmp = client
    from willspeak.db import LIVE_PROMPT

    sid = store.start_session(label="live", condition="unknown")
    (tmp / "clips").mkdir(exist_ok=True)
    import soundfile as sf

    sf.write(tmp / "clips" / "x.wav", np.zeros(16000, "float32"), 16000, subtype="PCM_16")
    cid = store.add_clip(sid, None, LIVE_PROMPT, "clips/x.wav", None, 1.0, {"ok": True})

    r = c.post("/api/live/confirm", data={"clip_id": cid, "spoken_text": "machine guess", "was_edited": False})
    assert r.status_code == 200 and r.json()["status"] == "pending"
    assert store.clip(cid)["status"] == "pending" and store.clip(cid)["transcript"] == "machine guess"
    recent = c.get("/api/live/recent").json()["clips"]
    assert recent[0]["id"] == cid and recent[0]["status"] == "pending"

    r = c.post(f"/api/clip/{cid}/review", data={"transcript": "(live)", "status": "accepted"})
    assert r.status_code == 400
    r = c.post(f"/api/clip/{cid}/review", data={"transcript": "", "status": "accepted"})
    assert r.status_code == 400
    assert store.clip(cid)["status"] == "pending"

    r = c.post("/api/live/fix", data={"clip_id": cid, "correct_text": "what he said"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    row = store.clip(cid)
    assert row["status"] == "accepted" and row["transcript"] == "what he said"
    assert store.totals()["corrections"] == 1


def test_duplicate_live_upload_is_not_stored_twice(client, monkeypatch):
    from willspeak import server

    c, store, _tmp = client
    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", lambda *_: {"text": "hello"})
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    wav = fake_wav(seconds=2.0)
    before = store.totals()["clips"]
    r1 = c.post("/api/live/transcribe", files={"audio": ("u.wav", wav, "audio/wav")})
    r2 = c.post("/api/live/transcribe", files={"audio": ("u.wav", wav, "audio/wav")})
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["clip_id"] is not None
    assert r2.json()["ok"] is False and "duplicate" in r2.json()["reason"]
    assert store.totals()["clips"] == before + 1


def test_failed_first_attempt_is_not_a_duplicate(client, monkeypatch):
    from willspeak import server

    c, _store, _tmp = client
    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", lambda *_: {"text": "hello"})
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    wav = fake_wav(seconds=2.0)
    r1 = c.post("/api/live/transcribe", data={"session_id": 9999},
                files={"audio": ("u.wav", wav, "audio/wav")})
    assert r1.status_code == 404
    r2 = c.post("/api/live/transcribe", files={"audio": ("u.wav", wav, "audio/wav")})
    assert r2.status_code == 200 and r2.json()["ok"] is True


@pytest.mark.parametrize("sent, logged", [("-31.5", "gate=-32dBFS"), ("nan", None), ("5", None), (None, None)])
def test_live_clip_log_carries_the_page_gate_level(client, monkeypatch, capsys, sent, logged):
    from willspeak import server

    c, _, _ = client
    monkeypatch.setattr(server.asr, "available", lambda: True)
    monkeypatch.setattr(server.asr, "transcribe_array", lambda *_: {"text": "hello there"})
    monkeypatch.setattr(server.matcher, "available", lambda: False)
    data = {} if sent is None else {"gate_dbfs": sent}
    r = c.post("/api/live/transcribe", data=data, files={"audio": ("clip.wav", fake_wav(), "audio/wav")})
    assert r.status_code == 200 and r.json()["ok"] is True
    line = next(x for x in capsys.readouterr().out.splitlines() if "live clip" in x)
    assert (logged in line) if logged else ("gate=" not in line)
