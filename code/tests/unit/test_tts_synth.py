import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
sys.path.insert(0, str(ROOT / "src"))
import synth_dysarthria as S  # noqa: E402
import tts_synth as T  # noqa: E402

sf = pytest.importorskip("soundfile")

TEXTS = ["open the door", "water", "please turn on the light", "good morning", "I am cold", "call my brother", "yes", "thank you very much",
         "bring me a blanket", "what time is it", "my head hurts", "close the window"]


def manifest(path: Path, texts: list[str], prefix: str = "r") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": f"{prefix}{i}", "audio": f"a/{prefix}{i}.wav", "text": t, "duration_s": 1.0, "kind": "word" if len(t.split()) == 1 else "sentence"}) + "\n"
                            for i, t in enumerate(texts)))


def test_the_fake_backend_needs_permission_and_the_others_say_what_is_missing(monkeypatch):
    with pytest.raises(RuntimeError, match="allow-fake"):
        T.get_backend("fake")
    assert isinstance(T.get_backend("fake", allow_fake=True), T.Fake)
    monkeypatch.setattr(T.shutil, "which", lambda _n: None)
    with pytest.raises(RuntimeError, match="espeak-ng"):
        T.get_backend("espeak")
    monkeypatch.setattr(T.Piper, "module", staticmethod(lambda: False))
    with pytest.raises(RuntimeError, match="piper"):
        T.get_backend("piper")
    with pytest.raises(RuntimeError, match="no text-to-speech backend"):
        T.get_backend("auto")
    with pytest.raises(ValueError):
        T.get_backend("nonsense")
    ok = T.available()
    assert ok["fake"][0] and not ok["espeak"][0] and not ok["piper"][0]


def test_espeak_runs_the_program_with_voice_speed_and_pitch(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        sf.write(cmd[cmd.index("-w") + 1], np.sin(np.arange(22050) / 10).astype(np.float32) * 0.2, 22050)

        class Done:
            returncode = 0
            stderr = ""
        return Done()

    monkeypatch.setattr(T.shutil, "which", lambda n: "/usr/bin/espeak-ng" if n == "espeak-ng" else None)
    monkeypatch.setattr(T.subprocess, "run", fake_run)
    e = T.get_backend("espeak")
    variant = e.variants()[2]
    out = tmp_path / "x.wav"
    e.synth("hello there", variant, out)
    assert calls[0][:2] == ["/usr/bin/espeak-ng", "-v"] and variant["voice"] in calls[0] and calls[0][-1] == "hello there"
    assert len({v["voice"] for v in e.variants()}) >= 6

    def failing(cmd, **kw):
        class Done:
            returncode = 1
            stderr = "no such voice"
        return Done()

    monkeypatch.setattr(T.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="no such voice"):
        e.synth("x", variant, tmp_path / "y.wav")


def test_piper_needs_a_voice_and_finds_one_in_the_voice_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(T.shutil, "which", lambda n: "/usr/bin/piper" if n == "piper" else None)
    monkeypatch.delenv("PIPER_VOICES", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(T.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    with pytest.raises(RuntimeError, match="no piper voice"):
        T.Piper()
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices/en_US-test-medium.onnx").write_bytes(b"x")
    p = T.Piper(str(tmp_path / "voices"))
    assert p.voices and len(p.variants()) >= 4
    assert "en_US-test-medium" in p.describe(p.variants()[0])


def test_excluded_prompts_never_reach_the_synthesiser(tmp_path):
    manifest(tmp_path / "pool.jsonl", TEXTS)
    manifest(tmp_path / "test.jsonl", ["Open the door!", "yes"], "t")
    (tmp_path / "typed.txt").write_text("# caregiver list\nI want my glasses\nwater\nopen the door\n")
    texts, stats = T.collect_texts([tmp_path / "pool.jsonl"], [tmp_path / "typed.txt"], [tmp_path / "test.jsonl"])
    keys = {T.key_of(t) for t in texts}
    assert "open the door" not in keys and "yes" not in keys
    assert "i want my glasses" in keys and "water" in keys
    assert stats["dropped_as_excluded_prompts"] >= 3 and stats["excluded_prompts"] == 2
    assert len(keys) == len(texts)


def test_make_writes_a_controls_shaped_export_and_reuses_it(tmp_path):
    backend = T.get_backend("fake", allow_fake=True)
    out = tmp_path / "tts"
    info = T.make(out, backend, TEXTS, variants=2, max_clips=20, workers=2)
    assert info["clips"] == 20 and info["train"] > 0 and info["dev"] > 0 and info["failed"] == 0
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert all((out / r["audio"]).is_file() for r in rows)
    x, rate = sf.read(str(out / rows[0]["audio"]))
    assert rate == 16000 and x.ndim == 1 and abs(float(np.max(np.abs(x))) - 0.5) < 0.02
    assert {json.loads(l)["speaker"] for l in (out / "dev.jsonl").read_text().splitlines()} == {"TTS-fake"}
    again = T.make(out, backend, TEXTS, variants=2, max_clips=20, workers=2)
    assert again["settings"] == info["settings"]
    with pytest.raises(FileExistsError):
        T.make(out, backend, TEXTS[:5], variants=2, max_clips=20, workers=2)
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(ValueError, match="no texts"):
        T.make(tmp_path / "empty", backend, [], 1, 5)


def test_many_failures_leave_no_partial_export(tmp_path):
    class Broken(T.Fake):
        def synth(self, text, variant, out):
            raise RuntimeError("engine crashed")

    with pytest.raises(RuntimeError, match="engine crashed"):
        T.make(tmp_path / "tts", Broken(), TEXTS, 2, 10)
    assert not (tmp_path / "tts").exists() and not list(tmp_path.glob("*.partial"))


def test_merge_joins_sources_and_refuses_clashing_ids(tmp_path):
    backend = T.get_backend("fake", allow_fake=True)
    T.make(tmp_path / "a", backend, TEXTS[:6], 2, 8)
    T.make(tmp_path / "b", backend, TEXTS[6:], 2, 8, seed=5)
    info = T.merge([tmp_path / "a", tmp_path / "b"], tmp_path / "ab")
    assert info["train"] + info["dev"] == 16
    rows = [json.loads(l) for l in (tmp_path / "ab/train.jsonl").read_text().splitlines()]
    assert all((tmp_path / "ab" / r["audio"]).is_file() for r in rows)
    with pytest.raises(ValueError, match="more than one"):
        T.merge([tmp_path / "a", tmp_path / "a"], tmp_path / "aa")
    assert not (tmp_path / "aa").exists()


def test_the_degrader_accepts_the_export_as_its_controls(tmp_path):
    backend = T.get_backend("fake", allow_fake=True)
    T.make(tmp_path / "tts", backend, TEXTS, 3, 30)
    target = tmp_path / "M01"
    rng = np.random.default_rng(1)
    for split, texts in (("train", ["tomato", "open the window slowly", "good night"]), ("dev", ["my arm"]), ("test", ["a test prompt"])):
        (target / split).mkdir(parents=True)
        rows = []
        for i, t in enumerate(texts):
            sf.write(target / split / f"{i}.wav", rng.normal(0, 0.05, 24000), 16000)
            rows.append({"id": f"{split}{i}", "audio": f"{split}/{i}.wav", "text": t, "duration_s": 1.5, "speaker": "M01",
                         "kind": "word" if len(t.split()) == 1 else "sentence"})
        (target / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    info = S.build_synthetic(tmp_path / "tts", target, tmp_path / "mix", severity=0.6, scale=4, workers=1)
    assert info["complete"] and info["synthetic_clips"] > 0
    train = [json.loads(l) for l in (tmp_path / "mix/train.jsonl").read_text().splitlines()]
    assert any(r["id"].startswith("synth-v") for r in train) and any(r["id"] == "train0" for r in train)


def test_command_line(tmp_path, capsys):
    manifest(tmp_path / "pool.jsonl", TEXTS)
    manifest(tmp_path / "test.jsonl", ["yes"], "t")
    assert T.main(["backends"]) == 0 and "fake" in capsys.readouterr().out
    assert T.main(["speak", "--backend", "fake", "--allow-fake", "--text", "open the door", "--out", str(tmp_path / "s.wav")]) == 0
    x, rate = sf.read(str(tmp_path / "s.wav"))
    assert rate == 16000 and len(x) > 1600
    assert T.main(["make", "--backend", "fake", "--allow-fake", "--out", str(tmp_path / "tts"), "--texts-from", str(tmp_path / "pool.jsonl"),
                   "--exclude-from", str(tmp_path / "test.jsonl"), "--variants", "2", "--max-clips", "10", "--workers", "1"]) == 0
    info = json.loads((tmp_path / "tts/tts.json").read_text())
    assert info["text_stats"]["dropped_as_excluded_prompts"] == 1
    with pytest.raises(RuntimeError, match="allow-fake"):
        T.main(["make", "--backend", "fake", "--out", str(tmp_path / "x"), "--texts-from", str(tmp_path / "pool.jsonl")])
