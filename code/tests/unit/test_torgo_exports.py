import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
ROOT = Path(__file__).resolve().parents[2]
SPEAKERS = ("F01", "M01", "M02", "M04", "F03")
PROMPTS = [f"please bring me item number {n} from the shelf" for n in range(60)]


def make_torgo(root: Path) -> None:
    rng = np.random.default_rng(0)
    for speaker in SPEAKERS:
        for session in ("Session1", "Session2"):
            base = root / speaker / session
            (base / "prompts").mkdir(parents=True)
            (base / "wav_headMic").mkdir()
            for n, text in enumerate(PROMPTS, start=1):
                (base / "prompts" / f"{n:04d}.txt").write_text(text, encoding="utf-8")
                sf.write(base / "wav_headMic" / f"{n:04d}.wav", rng.normal(size=1600).astype("float32") * 0.1, 16000)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_severe_loso_trains_on_the_other_severe_speakers_only(tmp_path):
    make_torgo(tmp_path / "TORGO")
    command = [sys.executable, str(ROOT / "scripts" / "corpus" / "torgo_manifest.py"), "--out", str(tmp_path), "--export",
               "--speakers", "M01", "F01", "--severe-loso"]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    export = tmp_path / "export"
    for held, others in (("M01", {"F01", "M02", "M04"}), ("F01", {"M01", "M02", "M04"})):
        folder = export / f"loso-severe-{held}"
        for split in ("train", "dev"):
            speakers = {r["speaker"] for r in rows(folder / f"{split}.jsonl")}
            assert speakers and speakers <= others
        test = rows(folder / "test.jsonl")
        assert test and {r["speaker"] for r in test} == {held}
        prompts = {r["text"] for split in ("train", "dev") for r in rows(folder / f"{split}.jsonl")}
        assert prompts.isdisjoint({r["text"] for r in test})
    everyone = {r["speaker"] for r in rows(export / "loso-M01" / "train.jsonl")}
    assert "F03" in everyone


def test_without_the_flag_no_severe_export_is_written(tmp_path):
    make_torgo(tmp_path / "TORGO")
    command = [sys.executable, str(ROOT / "scripts" / "corpus" / "torgo_manifest.py"), "--out", str(tmp_path), "--export", "--speakers", "M01"]
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode == 0
    assert (tmp_path / "export" / "loso-M01").is_dir() and not (tmp_path / "export" / "loso-severe-M01").exists()


def test_missing_only_adds_new_exports_without_replacing_existing_manifests_or_audio(tmp_path):
    make_torgo(tmp_path / "TORGO")
    command = [sys.executable, str(ROOT / "scripts/corpus/torgo_manifest.py"), "--out", str(tmp_path),
               "--export", "--speakers", "M01"]
    assert subprocess.run(command, capture_output=True, check=False).returncode == 0
    manifest = tmp_path / "export/M01/test.jsonl"
    audio = next((tmp_path / "export/M01/test").glob("*.wav"))
    manifest.write_text("preserve this existing research split\n")
    audio.write_bytes(b"preserve this existing audio")
    done = subprocess.run([*command, "--missing-only", "--severe-loso"], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    assert manifest.read_text() == "preserve this existing research split\n"
    assert audio.read_bytes() == b"preserve this existing audio"
    assert (tmp_path / "export/loso-severe-M01/test.jsonl").is_file()


def test_pooled_export_holds_out_each_target_session_and_every_test_prompt(tmp_path):
    make_torgo(tmp_path / "TORGO")
    command = [sys.executable, str(ROOT / "scripts/corpus/torgo_manifest.py"), "--out", str(tmp_path), "--export",
               "--speakers", "M01", "M02", "--pooled", "--only", "pooled-M01-M02"]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    export = tmp_path / "export"
    assert sorted(p.name for p in export.iterdir()) == ["pooled-M01-M02"]
    folder = export / "pooled-M01-M02"
    train, dev = rows(folder / "train.jsonl"), rows(folder / "dev.jsonl")
    assert {r["speaker"] for r in train} == set(SPEAKERS)
    assert {r["speaker"] for r in dev} == {"M01", "M02"}
    assert not rows(folder / "test.jsonl")
    for target in ("M01", "M02"):
        assert not [r for r in train + dev if r["speaker"] == target and r["session"] == "Session2"]
    assert any(r["speaker"] == "F03" and r["session"] == "Session2" for r in train)
    personal = subprocess.run([sys.executable, str(ROOT / "scripts/corpus/torgo_manifest.py"), "--out", str(tmp_path), "--export",
                               "--speakers", "M01", "--only", "M01"], capture_output=True, text=True, check=False)
    assert personal.returncode == 0, personal.stderr
    test_prompts = {r["text"] for r in rows(export / "M01" / "test.jsonl")}
    assert test_prompts and test_prompts.isdisjoint({r["text"] for r in train + dev})
    record = json.loads((folder / "pooled.json").read_text())
    assert record["targets"] == ["M01", "M02"] and record["test_prompt_overlap"] == 0
    assert record["train_clips"] == len(train) and record["dev_clips"] == len(dev)
    assert set(record["speakers"]) == set(SPEAKERS) and record["bytes"] > 0


def test_severe_base_holds_every_severe_speaker_and_no_test_clips(tmp_path):
    make_torgo(tmp_path / "TORGO")
    command = [sys.executable, str(ROOT / "scripts/corpus/torgo_manifest.py"), "--out", str(tmp_path), "--export",
               "--severe-base", "--only", "severe-base"]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    folder = tmp_path / "export" / "severe-base"
    assert {r["speaker"] for r in rows(folder / "train.jsonl")} == {"F01", "M01", "M02", "M04"}
    assert rows(folder / "dev.jsonl") and not rows(folder / "test.jsonl")
    assert sorted(p.name for p in (tmp_path / "export").iterdir()) == ["severe-base"]
