from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("willspeak_clip_levels_script", ROOT / "scripts/asr/clip_levels.py")
levels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(levels)


def square(amplitude, n=16000):
    return np.where(np.arange(n) % 2 == 0, amplitude, -amplitude).astype(np.float32)


def make_export(folder: Path, amplitudes):
    folder.mkdir(parents=True)
    rows = []
    for i, amplitude in enumerate(amplitudes):
        sf.write(folder / f"{i}.wav", square(amplitude), 16000, subtype="FLOAT")
        rows.append({"id": f"c{i}", "text": "x", "audio": f"{i}.wav"})
    (folder / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folder / "test.jsonl").write_text("must never be opened")
    return folder


def test_level_is_rms_in_dbfs_and_silence_hits_the_floor():
    assert levels.clip_level_db(square(0.1)) == pytest.approx(-20.0, abs=1e-4)
    assert levels.clip_level_db(square(1.0)) == pytest.approx(0.0, abs=1e-4)
    assert levels.clip_level_db(np.zeros(100)) == levels.FLOOR_DB
    assert levels.clip_level_db(np.array([])) == levels.FLOOR_DB
    stereo = np.stack([square(0.1), square(0.1)], axis=1)
    assert levels.clip_level_db(stereo) == pytest.approx(-20.0, abs=1e-4)


def test_percentile_interpolates_between_neighbours():
    assert levels.percentile([0.0, 10.0], 50) == 5.0
    assert levels.percentile([1.0, 2.0, 4.0], 0) == 1.0
    assert levels.percentile([1.0, 2.0, 4.0], 100) == 4.0
    assert levels.percentile([7.0], 50) == 7.0
    with pytest.raises(ValueError):
        levels.percentile([], 50)


def test_levels_come_from_train_only_and_are_sorted(tmp_path):
    folder = make_export(tmp_path / "M01", [0.1, 0.01, 1.0])
    assert levels.levels_of(folder) == pytest.approx([-40.0, -20.0, 0.0], abs=1e-3)
    with pytest.raises(FileNotFoundError):
        levels.levels_of(tmp_path / "nowhere")


def test_report_gives_signal_to_noise_at_both_ends_of_the_range():
    lines = levels.report("M01", [-40.0, -20.0, 0.0], (-70, -40))
    assert lines[0].startswith("M01: 3 training clips") and "median -20.0" in lines[0]
    median = next(line for line in lines if line.strip().startswith("median clip"))
    assert "signal-to-noise 21 dB at the loudest noise (-41 dBFS), 50 dB at the quietest (-70 dBFS)" in median
    assert len(levels.report("M01", [-40.0, -20.0, 0.0], None)) == 1


def test_main_prints_one_block_per_folder_and_rejects_a_backwards_range(tmp_path, monkeypatch, capsys):
    a = make_export(tmp_path / "A", [0.1])
    b = make_export(tmp_path / "B", [0.01, 0.01])
    monkeypatch.setattr(levels.sys, "argv", ["clip_levels.py", str(a), str(b), "--noise-db", "-70", "-40"])
    assert levels.main() == 0
    out = capsys.readouterr().out
    assert "A: 1 training clips" in out and "B: 2 training clips" in out
    monkeypatch.setattr(levels.sys, "argv", ["clip_levels.py", str(a), "--noise-db", "-40", "-70"])
    with pytest.raises(SystemExit) as exc:
        levels.main()
    assert exc.value.code == 2
