import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "matcher"))
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import labkit as lab
import layer_screen
import route_score as rs

D = 32
WORDS = [f"word{i}" for i in range(12)]
ORPHAN = "orphan"
SENTENCES = ["turn on the light", "i want some water", "please call my mother"]


def tid(speaker: str, session: str, n: int) -> str:
    return f"{speaker}_{session}_{n:04d}"


def audio_path(speaker: str, session: str, n: int) -> str:
    return f"/data/TORGO/{speaker}/{session}/wav_headMic/{n:04d}.wav"


def build(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(3)
    pattern = {w: rng.normal(size=D) for w in [*WORDS, ORPHAN]}
    vectors, labels, sessions, speakers, paths = [], [], [], [], []
    for speaker in ("M01", "M02", "F03", "F04"):
        for session in ("Session1", "Session2"):
            offset = rng.normal(size=D) * 0.2
            vocabulary = WORDS if session == "Session1" else [*WORDS, ORPHAN]
            for n, word in enumerate(vocabulary, start=1):
                vectors.append(pattern[word] + offset + rng.normal(size=D) * 0.3)
                labels.append(word)
                sessions.append(f"{speaker}_{session}")
                speakers.append(speaker)
                paths.append(audio_path(speaker, session, n))
    np.savez(tmp_path / "words.npz", raw=np.array(vectors, np.float32), labels=np.array(labels), sessions=np.array(sessions),
             speakers=np.array(speakers), paths=np.array(paths), meta="{}")
    rows = [{"id": tid("M01", "Session2", n), "audio": f"test/{n}.wav", "text": w, "kind": "word", "session": "Session2"}
            for n, w in enumerate([*WORDS, ORPHAN], start=1)]
    rows += [{"id": tid("M01", "Session2", 900 + i), "audio": f"test/{900 + i}.wav", "text": t, "kind": "sentence", "session": "Session2"}
             for i, t in enumerate(SENTENCES)]
    export = tmp_path / "export"
    (export / "M01").mkdir(parents=True)
    (export / "M01" / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    hyps = tmp_path / "parakeet"
    hyps.mkdir()
    (hyps / "hypotheses.jsonl").write_text("".join(json.dumps({"id": r["id"], "hyp": r["text"] if r["kind"] == "sentence" else ""}) + "\n"
                                                    for r in rows), encoding="utf-8")
    return tmp_path / "words.npz", export, hyps


def test_path_id_and_session_order():
    assert rs.path_id("/x/manifests/../TORGO/M01/Session2/wav_arrayMic/0042.wav") == "M01_Session2_0042"
    with pytest.raises(ValueError):
        rs.path_id("/x/M01/clip.wav")
    assert sorted(["Session10", "Session2", "Session1"], key=rs.session_key) == ["Session1", "Session2", "Session10"]


def test_matcher_enrols_from_earlier_sessions_and_answers_the_test_session(tmp_path):
    npz, _, _ = build(tmp_path)
    d = layer_screen.load_archive(str(npz))
    answers = rs.matcher_answers(d, 0, "M01", "M01_Session2", dim=8)
    assert set(answers) == {tid("M01", "Session2", n) for n in range(1, 14)}
    assert all(answers[tid("M01", "Session2", n)][0] == w for n, w in enumerate(WORDS, start=1))
    assert answers[tid("M01", "Session2", 13)][0] in WORDS
    with pytest.raises(ValueError, match="before"):
        rs.matcher_answers(d, 0, "M01", "M01_Session1", dim=8)
    with pytest.raises(ValueError, match="no word takes"):
        rs.matcher_answers(d, 0, "M01", "M01_Session3", dim=8)


def test_route_sends_words_to_the_matcher_and_falls_back_below_the_score():
    rows = [{"id": "w", "kind": "word"}, {"id": "s", "kind": "sentence"}, {"id": "n", "kind": "word"}]
    parakeet = {"w": "pa", "s": "sentence answer", "n": "pb"}
    answers = {"w": ("yes", 0.9), "s": ("no", 0.99)}
    assert rs.route(rows, parakeet, answers, None) == {"w": "yes", "s": "sentence answer", "n": "pb"}
    assert rs.route(rows, parakeet, answers, 0.95)["w"] == "pa"
    assert rs.route(rows, parakeet, answers, 0.5)["w"] == "yes"


def test_command_line_writes_variants_that_pool_table_can_read(tmp_path):
    npz, export, hyps = build(tmp_path)
    out = tmp_path / "route"
    command = [sys.executable, str(ROOT / "scripts" / "matcher" / "route_score.py"), "--emb", str(npz), "--export", str(export),
               "--speaker", f"M01={hyps}", "--min-scores", "0.999", "--dim", "8", "--out", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    report = json.loads((out / "route.json").read_text())["speakers"]["M01"]
    assert report["word_clips"] == 13 and report["sentence_clips"] == 3 and report["matcher_right"] == 12
    assert report["wrong_words"]["parakeet"] == 13
    assert report["wrong_words"]["routed"] == 1
    assert report["wrong_words"]["routed-min0.999"] == 13
    assert report["enrolled_from"] == ["M01_Session1"] and report["test_session"] == "M01_Session2"
    routed = lab.read_hyps(out / "routed" / "M01")
    assert routed[tid("M01", "Session2", 900)] == SENTENCES[0]
    assert (out / "parakeet" / "M01" / "hypotheses.jsonl").is_file()
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0 and "Already exists" in (again.stderr + again.stdout)
