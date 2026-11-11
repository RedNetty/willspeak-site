from __future__ import annotations

import importlib.util
import json
import random
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from willspeak import compare as C

ROOT = Path(__file__).resolve().parents[2]


class FakeASR:
    events: list[str] = []

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.model_name = f"fake:{self.model_dir.name}"
        self.loaded = False
        self.load_s = 0.01
        self.answer = "hello there" if "M01" in self.model_dir.name else "yellow hair"

    def model_present(self):
        return "missing" not in self.model_dir.name

    def warm_up(self):
        FakeASR.events.append(f"load {self.model_dir.name}")
        self.loaded = True

    def transcribe_array(self, x, sr=16000):
        return {"text": self.answer, "latency_ms": 1, "model": self.model_name}

    def __del__(self):
        FakeASR.events.append(f"drop {self.model_dir.name}")


def _demo(tmp_path, clips):
    d = tmp_path / "demo"
    for c in clips:
        (d / c["audio"]).parent.mkdir(parents=True, exist_ok=True)
        sf.write(d / c["audio"], np.zeros(1600, "float32"), 16000)
    (d / "clips.json").write_text(json.dumps({"rule": "test", "clips": clips}))
    return d


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    for s in C.SPEAKERS:
        monkeypatch.setenv(f"WILLSPEAK_DEMO_MODEL_{s}", str(tmp_path / f"model-{s}"))
    FakeASR.events = []
    d = _demo(tmp_path, [
        {"id": "a", "speaker": "M01", "kind": "sentence", "text": "Hello there.", "audio": "M01/a.wav"},
        {"id": "b", "speaker": "M02", "kind": "word", "text": "yellow", "audio": "M02/b.wav"},
    ])
    stock = FakeASR(C.stock_dir())
    return C.Comparison(stock, threading.Lock(), demo_dir=d, factory=FakeASR)


def test_score_counts_like_evaluate():
    s = C.score("The quick brown fox.", "the quick round box jumps")
    assert (s["errors"], s["n_ref"]) == (3, 4)
    assert sorted(o["op"] for o in s["ops"]) == ["ins", "ok", "ok", "sub", "sub"]
    assert C.score("hello", "")["errors"] == 1


def test_stock_server_recogniser_is_reused(comparison):
    assert comparison.stock.model_dir == C.stock_dir()


def test_decode_needs_that_speakers_model(comparison):
    with pytest.raises(RuntimeError):
        comparison.decode("a")
    comparison.select("M01")
    r = comparison.decode("a")
    assert r["stock"]["errors"] == 2 and r["speaker"]["errors"] == 0
    with pytest.raises(RuntimeError):
        comparison.decode("b")
    with pytest.raises(KeyError):
        comparison.decode("zzz")


def test_swap_drops_previous_model_before_loading(comparison):
    comparison.select("M01")
    FakeASR.events.clear()
    out = comparison.select("M02")
    assert out["from"] == "M01" and out["swapped"]
    assert FakeASR.events.index("drop model-M01") < FakeASR.events.index("load model-M02")
    assert comparison.select("M02")["swapped"] is False
    comparison.release()
    assert comparison.speaker is None and comparison.model is None


def test_missing_model_and_unknown_speaker(comparison, tmp_path, monkeypatch):
    monkeypatch.setenv("WILLSPEAK_DEMO_MODEL_M04", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        comparison.select("M04")
    with pytest.raises(KeyError):
        comparison.select("F01")


def _model(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (folder / name).write_text("x")
    return folder


@pytest.fixture
def models(tmp_path, monkeypatch):
    base = tmp_path / "models"
    _model(base / C.DEFAULT_DIR_NAME)
    _model(base / "lab-28-export-M01-matmul")
    _model(base / "lab-28-export-M02-default" / "model")
    _model(base / "lab-baselines" / "deep" / "onnx-int8")
    (base / "half-copied").mkdir()
    (base / "half-copied" / "tokens.txt").write_text("x")
    monkeypatch.setattr(C, "MODELS_DIR", base)
    return base


def test_model_choices_lists_complete_folders_but_not_stock(models):
    assert C.model_choices() == [
        "lab-28-export-M01-matmul",
        "lab-28-export-M02-default/model",
        "lab-baselines/deep/onnx-int8",
    ]


def test_resolve_model_stays_inside_models(models):
    assert C.resolve_model("lab-28-export-M02-default/model") == (models / "lab-28-export-M02-default" / "model").resolve()
    with pytest.raises(ValueError):
        C.resolve_model("../outside")
    with pytest.raises(FileNotFoundError):
        C.resolve_model("half-copied")


def test_any_model_decodes_a_speakers_clips_and_is_marked_custom(comparison, models):
    out = comparison.select("M02", "lab-28-export-M01-matmul")
    assert out["swapped"] and out["folder"] == "lab-28-export-M01-matmul"
    st = comparison.state()
    assert st["loaded"] == "M02" and st["custom"] and st["loaded_model"] == "lab-28-export-M01-matmul"
    r = comparison.decode("b")
    assert r["speaker"]["model"] == "fake:lab-28-export-M01-matmul"
    # The same files for another speaker's recordings: no reload.
    FakeASR.events.clear()
    assert comparison.select("M01", "lab-28-export-M01-matmul")["swapped"] is False
    assert FakeASR.events == [] and comparison.decode("a")["speaker"]["errors"] == 0
    # Back to the speaker's own model clears the custom flag.
    comparison.select("M01")
    assert comparison.state()["custom"] is False


def test_speaker_route_passes_the_model_and_rejects_paths_outside(comparison, models, monkeypatch):
    from fastapi.testclient import TestClient

    from willspeak import server
    monkeypatch.setattr(server, "comparison", comparison)
    client = TestClient(server.app)
    assert client.post("/api/compare/speaker", data={"speaker": "M01", "model": "lab-28-export-M02-default/model"}).status_code == 200
    assert comparison.state()["loaded_model"] == "lab-28-export-M02-default/model"
    assert client.post("/api/compare/speaker", data={"speaker": "M01", "model": "../../etc"}).status_code == 400
    assert client.post("/api/compare/speaker", data={"speaker": "M01", "model": "half-copied"}).status_code == 503


def test_state_hides_audio_paths(comparison):
    st = comparison.state()
    assert st["clip_error"] is None and {c["id"] for c in st["clips"]} == {"a", "b"}
    assert all("audio" not in c for c in st["clips"])


def test_clip_list_validation(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "clips.json").write_text(json.dumps({"clips": [{"id": "a"}]}))
    with pytest.raises(ValueError):
        C.load_clips(d)
    comp = C.Comparison(FakeASR(C.stock_dir()), demo_dir=tmp_path / "none", factory=FakeASR)
    assert comp.state()["clip_error"].startswith("no clip list")


def _picker():
    spec = importlib.util.spec_from_file_location("demo_clips", ROOT / "scripts/corpus/demo_clips.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _export(tmp_path, spk="M01"):
    root = tmp_path / "export"
    rows = []
    texts = [f"sentence number {i} here" for i in range(10)] + [f"word{i}" for i in range(8)]
    for i, t in enumerate(texts + texts[:3]):
        rid = f"{spk}_s1_{i:04d}"
        p = root / spk / "test" / f"{rid}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, np.zeros(160, "float32"), 16000)
        rows.append({"id": rid, "audio": f"test/{rid}.wav", "text": t, "duration_s": 0.01})
    (root / spk / "test.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return root, rows


def test_choose_is_seeded_and_one_per_prompt(tmp_path):
    m = _picker()
    _, rows = _export(tmp_path)
    a = m.choose(rows, 4, 3, random.Random("2026-M01"))
    b = m.choose(list(reversed(rows)), 4, 3, random.Random("2026-M01"))
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert [r["kind"] for r in a] == ["sentence"] * 4 + ["word"] * 3
    assert len({r["text"] for r in a}) == 7


def test_picker_writes_list_and_refuses_to_replace(tmp_path):
    m = _picker()
    root, _ = _export(tmp_path)
    out = tmp_path / "demo"
    args = ["--export-root", str(root), "--speakers", "M01", "--sentences", "2", "--words", "2", "--out", str(out)]
    assert m.main(args) == 0
    meta = json.loads((out / "clips.json").read_text())
    assert len(meta["clips"]) == 4 and all((out / c["audio"]).is_file() for c in meta["clips"])
    assert "_src" not in meta["clips"][0] and "seed 2026" in meta["rule"]
    with pytest.raises(SystemExit):
        m.main(args)
    assert C.load_clips(out)["clips"] == meta["clips"]


def test_audio_path_serves_only_listed_clips(comparison, tmp_path):
    assert comparison.audio_path("a") == (comparison.demo_dir / "M01/a.wav").resolve()
    with pytest.raises(KeyError):
        comparison.audio_path("../../etc/passwd")
    (comparison.demo_dir / "M02/b.wav").unlink()
    with pytest.raises(FileNotFoundError):
        comparison.audio_path("b")


def test_audio_outside_demo_folder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("WILLSPEAK_DEMO_MODEL_M01", str(tmp_path / "model-M01"))
    d = _demo(tmp_path, [{"id": "a", "speaker": "M01", "kind": "word", "text": "hi", "audio": "M01/a.wav"}])
    (d / "clips.json").write_text(json.dumps({"clips": [
        {"id": "a", "speaker": "M01", "kind": "word", "text": "hi", "audio": "../outside.wav"}]}))
    comp = C.Comparison(FakeASR(C.stock_dir()), demo_dir=d, factory=FakeASR)
    with pytest.raises(ValueError):
        comp.audio_path("a")
    comp.select("M01")
    with pytest.raises(ValueError):
        comp.decode("a")


def test_try_voice_runs_both_models_and_keeps_nothing(comparison, tmp_path):
    with pytest.raises(RuntimeError):
        comparison.try_voice(np.zeros(16000, "float32"), 1.0)
    comparison.select("M01")
    before = sorted(p for p in tmp_path.rglob("*"))
    r = comparison.try_voice(np.zeros(16000, "float32"), 1.0)
    assert r["speaker"] == "M01" and r["duration_s"] == 1.0
    assert r["stock"]["text"] == "yellow hair" and r["mine"]["text"] == "hello there"
    assert "errors" not in r["stock"] and sorted(p for p in tmp_path.rglob("*")) == before


def test_try_route_checks_the_audio_then_decodes(comparison, tmp_path, monkeypatch):
    import io

    monkeypatch.setenv("WILLSPEAK_DATA", str(tmp_path / "data"))
    import sys
    for m in [m for m in sys.modules if m.startswith("willspeak.server")]:
        del sys.modules[m]
    from fastapi.testclient import TestClient

    from willspeak import server
    monkeypatch.setattr(server, "comparison", comparison)
    client = TestClient(server.app)

    def wav(x):
        buf = io.BytesIO()
        sf.write(buf, x, 16000, format="WAV")
        return {"audio": ("try.wav", buf.getvalue(), "audio/wav")}

    rng = np.random.default_rng(0)
    speech = np.concatenate([np.zeros(4000), rng.normal(0, 0.2, 16000), np.zeros(4000)]).astype("float32")
    assert client.post("/api/compare/try", files=wav(speech)).status_code == 409
    comparison.select("M02")
    r = client.post("/api/compare/try", files=wav(speech))
    assert r.status_code == 200 and r.json()["speaker"] == "M02"
    assert client.post("/api/compare/try", files=wav(np.zeros(16000, "float32"))).status_code == 422
    assert client.post("/api/compare/try", files={"audio": ("x.wav", b"", "audio/wav")}).status_code == 400
