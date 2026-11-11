from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak import quality
from willspeak.quality import MODEL_PATH, QualityGate, Verdict
from tests.helpers import fake_wav


@pytest.fixture(scope="module")
def gate():
    return QualityGate(MODEL_PATH)


def test_shipped_weights_load_with_sane_thresholds(gate):
    th = gate.thresholds
    assert 5.0 <= th["snr_min_db"] <= 20.0
    assert 0.0 <= th["c50_min_db"] <= 15.0
    assert 0.5 <= th["spk2_max"] < 1.0
    assert len(gate.convs) == 4


def test_estimates_are_deterministic_and_handle_short_clips(gate):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(32000) * 0.01).astype(np.float32)
    assert gate.estimate(x) == gate.estimate(x.copy())
    for n in (0, 10, 399, 400):
        snr, c50, p = gate.estimate(np.zeros(n, np.float32) + 1e-4)
        assert np.isfinite([snr, c50, p]).all() and 0.0 <= p <= 1.0


def test_loudness_does_not_change_the_estimates(gate):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(32000) * 0.01).astype(np.float32)
    a, b = gate.estimate(x), gate.estimate(x * 10)
    assert a == pytest.approx(b, abs=1e-3)


def test_white_noise_is_rejected_as_noisy(gate):
    rng = np.random.default_rng(2)
    v = gate((rng.standard_normal(48000) * 0.1).astype(np.float32))
    assert not v.keep and "noisy" in v.reasons
    assert v.dict()["messages"] == [quality.MESSAGES[r] for r in v.reasons]


def test_gate_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("WILLSPEAK_QUALITY_GATE", "0")
    assert quality.gate() is None


class _Stub:
    def __init__(self, keep):
        self.keep = keep

    def __call__(self, x):
        return Verdict(5.0, 30.0, 0.1, self.keep, [] if self.keep else ["noisy"])


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("WILLSPEAK_DATA", str(tmp_path))
    monkeypatch.setenv("WILLSPEAK_MATCH_EAGER", "0")
    monkeypatch.setenv("WILLSPEAK_ASR_EAGER", "0")
    for m in list(sys.modules):
        if m.startswith("willspeak"):
            del sys.modules[m]
    from fastapi.testclient import TestClient

    from willspeak.server import app, store

    with TestClient(app) as c:
        yield c, store
    store.con.close()


def _upload(c, sid, retry=None):
    data = {"session_id": sid, "prompt_text": "hello"}
    if retry is not None:
        data["quality_retry"] = retry
    return c.post("/api/clip", data=data, files={"audio": ("clip.wav", fake_wav(), "audio/wav")}).json()


def test_first_failure_sends_the_take_back_and_second_is_kept(client, monkeypatch):
    c, store = client
    import willspeak.quality as q

    monkeypatch.setattr(q, "gate", lambda: _Stub(keep=False))
    sid = store.start_session()

    first = _upload(c, sid)
    assert not first["qc"]["ok"] and q.MESSAGES["noisy"] in first["qc"]["errors"]
    assert first["qc"]["gate"]["keep"] is False
    assert store.clip(first["clip_id"])["status"] == "retake"

    second = _upload(c, sid, retry=1)
    assert second["qc"]["ok"] and q.MESSAGES["noisy"] in second["qc"]["warnings"]
    assert second["qc"]["gate"]["kept_after_retry"] is True
    assert store.clip(second["clip_id"])["status"] == "pending"


def test_passing_take_records_the_verdict(client, monkeypatch):
    c, store = client
    import willspeak.quality as q

    monkeypatch.setattr(q, "gate", lambda: _Stub(keep=True))
    res = _upload(c, store.start_session())
    assert res["qc"]["ok"] and res["qc"]["gate"]["keep"] is True and res["qc"]["gate"]["snr_db"] == 5.0


def test_gate_off_leaves_the_level_checks_alone(client):
    c, store = client
    res = _upload(c, store.start_session())
    assert res["qc"]["gate"] is None and res["qc"]["ok"]
