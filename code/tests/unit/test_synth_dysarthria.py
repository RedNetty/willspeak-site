import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
sys.path.insert(0, str(ROOT / "src"))
import synth_dysarthria as S  # noqa: E402

soundfile = pytest.importorskip("soundfile")


def test_selftest_passes():
    assert S.selftest(verbose=False) == []


def test_severity_zero_changes_nothing_and_same_seed_repeats():
    x = S.toy_speech(1.2)
    y, meta = S.degrade(x, 0.0, 1)
    assert np.array_equal(x, y) and "skipped" in meta
    a, _ = S.degrade(x, 0.8, 4)
    b, _ = S.degrade(x, 0.8, 4)
    c, _ = S.degrade(x, 0.8, 5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_slower_at_higher_severity_and_never_clips():
    x = S.toy_speech(1.5)
    lengths = [len(S.degrade(x, sev, 3)[0]) for sev in (0.2, 0.6, 1.0)]
    assert lengths[0] < lengths[1] < lengths[2]
    y, _ = S.degrade(x * 2.5, 1.0, 3)
    assert float(np.abs(y).max()) <= 1.0 and np.isfinite(y).all()


def test_each_effect_alone_and_bad_input():
    x = S.toy_speech(1.2)
    for effect in S.EFFECTS:
        y, _ = S.degrade(x, 0.9, 2, effects=(effect,))
        assert np.isfinite(y).all() and len(y) > 0
    for bad in (np.zeros(16000, np.float32), x[:800], np.full(16000, np.nan, np.float32)):
        y, meta = S.degrade(bad, 0.9, 2)
        assert "skipped" in meta and len(y) == len(bad)


def test_wsola_changes_length_and_keeps_pitch():
    t = np.arange(16000) / S.SR
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    y = S.wsola(tone, 1.8)
    assert abs(len(y) / len(tone) - 1.8) < 0.05
    spectrum = np.abs(np.fft.rfft(y[2000:12000] * np.hanning(10000)))
    peak_hz = np.argmax(spectrum) * S.SR / 10000
    assert abs(peak_hz - 200) < 12


def test_units_cover_the_clip_without_gaps():
    x = S.toy_speech(2.0)
    units = S.segment_units(x)
    assert units[0].start == 0 and units[-1].end == len(x)
    assert all(a.end == b.start for a, b in zip(units, units[1:]))
    assert {u.label for u in units} >= {"vow", "sil"}


def test_target_seconds_sets_the_final_length():
    x = S.toy_speech(1.5)
    y, meta = S.degrade(x, 0.7, 2, target_seconds=4.0)
    assert abs(len(y) / S.SR - 4.0) < 0.35 and "rate_ratio" in meta


def test_calibration_is_monotone_and_reports_range_limits():
    table = [(0.3, 0.30), (0.5, 0.52), (0.7, 0.50), (0.9, 0.80)]
    cal = S.calibrate(table, 0.65)
    assert 0.7 < cal["severity"] < 0.9
    assert S.calibrate(table, 0.99)["note"].startswith("target above")
    assert S.calibrate(table, 0.05)["severity"] == 0.3


def make_export(root: Path, speaker: str, texts: dict[str, list[tuple[str, str]]], seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    for split, items in texts.items():
        rows = []
        for i, (kind, text) in enumerate(items):
            audio = S.toy_speech(seconds, seed=int(rng.integers(1000)))
            rel = f"{split}/{speaker}-{split}-{i}.wav"
            S.write_audio(root / rel, audio)
            rows.append({"id": f"{speaker}-{split}-{i}", "audio": rel, "text": text, "prompt_text": text, "duration_s": seconds,
                         "kind": kind, "speaker": speaker, "severity": "control", "session": "1", "condition": "unknown"})
        S.write_jsonl(root / f"{split}.jsonl", rows)


@pytest.fixture()
def exports(tmp_path):
    controls = tmp_path / "controls"
    target = tmp_path / "M01"
    words = [("word", w) for w in ("tomato", "carrot", "window", "yellow", "monday", "pencil")]
    sents = [("sentence", f"this is test sentence number {n}") for n in range(6)]
    make_export(controls, "FC01", {"train": words[:4] + sents[:4], "dev": words[4:5] + sents[4:5],
                                   "test": words[5:] + sents[5:]})
    make_export(target, "M01", {"train": [("word", "alpha beta"), ("word", "gamma"), ("sentence", "a longer own sentence here")],
                                "dev": [("word", "delta")], "test": [("word", "pencil"), ("sentence", "this is test sentence number 5")]},
                seconds=1.6, seed=9)
    return controls, target, tmp_path


def build(exports, **kw):
    controls, target, tmp = exports
    args = dict(severity=0.7, scale=2.0, seed=3, variants=2, workers=1)
    args.update(kw)
    return S.build_synthetic(controls, target, tmp / "out", **args)


def test_build_adds_synthetic_to_train_only_and_leaves_the_rest(exports):
    controls, target, tmp = exports
    info = build(exports)
    out = tmp / "out"
    train = S.read_jsonl(out / "train.jsonl")
    own = S.read_jsonl(target / "train.jsonl")
    assert len(train) == len(own) + info["synthetic_clips"] and info["synthetic_clips"] >= 4
    assert S.read_jsonl(out / "dev.jsonl") == S.read_jsonl(target / "dev.jsonl")
    assert S.read_jsonl(out / "test.jsonl") == S.read_jsonl(target / "test.jsonl")
    synth = [r for r in train if r["id"].startswith("synth-")]
    assert all((out / r["audio"]).is_file() and r["speaker"].startswith("SYN-FC01") for r in synth)
    assert not (tmp / "out.partial").exists()
    assert json.loads((out / "synthetic.json").read_text())["complete"] is True


def test_prompts_of_the_targets_dev_and_test_never_become_training_clips(exports):
    controls, target, tmp = exports
    build(exports)
    held = {S.prompt_key(r["text"]) for s in ("dev", "test") for r in S.read_jsonl(target / f"{s}.jsonl")}
    synth = [r for r in S.read_jsonl(tmp / "out" / "train.jsonl") if r["id"].startswith("synth-")]
    assert not {S.prompt_key(r["text"]) for r in synth} & held
    rows = S.read_jsonl(controls / "train.jsonl")
    rows[0]["text"] = rows[0]["prompt_text"] = "pencil"
    S.write_jsonl(controls / "train.jsonl", rows)
    S.build_synthetic(controls, target, tmp / "out2", 0.7, 2.0, 3, variants=2, workers=1)
    assert json.loads((tmp / "out2" / "synthetic.json").read_text())["leaked_prompts_dropped"] == 1


def test_same_settings_reuse_other_settings_refuse(exports):
    first = build(exports)
    again = build(exports)
    assert again["synthetic_clips"] == first["synthetic_clips"]
    with pytest.raises(FileExistsError):
        build(exports, severity=0.9)
    third = build(exports, severity=0.9, force=True)
    assert third["settings"]["severity"] == 0.9
    assert list(exports[2].glob("out-previous*"))


def test_unrelated_folder_is_not_touched_and_partial_scratch_is_rebuilt(exports):
    controls, target, tmp = exports
    (tmp / "out").mkdir()
    (tmp / "out" / "mine.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        build(exports)
    assert (tmp / "out" / "mine.txt").read_text() == "keep"
    (tmp / "out2.partial").mkdir()
    (tmp / "out2.partial" / "junk").write_text("x")
    S.build_synthetic(controls, target, tmp / "out2", 0.5, 1.0, 1, variants=1, workers=1)
    assert (tmp / "out2" / "synthetic.json").is_file() and not (tmp / "out2.partial").exists()


def test_missing_audio_is_named_in_the_error(exports):
    controls, target, tmp = exports
    first = S.read_jsonl(target / "train.jsonl")[0]
    (target / first["audio"]).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        build(exports)


def test_only_synthetic_and_rate_match(exports):
    controls, target, tmp = exports
    info = build(exports, only_synthetic=True)
    train = S.read_jsonl(tmp / "out" / "train.jsonl")
    assert all(r["id"].startswith("synth-") for r in train)
    assert info["rate_ratio_by_kind"].get("word", 0) > 1.2


def test_validation_set_resumes_and_summary_calibrates(exports):
    controls, target, tmp = exports
    index = S.build_validation(controls, tmp / "val", [0.4, 0.8], limit=5, workers=1, target=[target])
    assert [r["clips"] for r in index["severities"]] == [2, 2]
    assert index["clean"]["clips"] == 2 and (tmp / "val/clean/test.jsonl").is_file()
    clean_rows = S.read_jsonl(tmp / "val/clean/test.jsonl")
    assert all((tmp / "val/clean" / r["audio"]).is_file() for r in clean_rows)
    assert index["rate_ratio_by_kind"]
    stamp = (tmp / "val" / "s0.40" / "done.json").stat().st_mtime_ns
    S.build_validation(controls, tmp / "val", [0.4, 0.8], limit=5, workers=1)
    assert (tmp / "val" / "s0.40" / "done.json").stat().st_mtime_ns == stamp
    res = tmp / "res"
    for sev, wer in ((0.4, 0.3), (0.8, 0.7)):
        (res / f"s{sev:.2f}").mkdir(parents=True)
        (res / f"s{sev:.2f}" / "results.json").write_text(json.dumps({"wer": wer, "word": {"n_ref": 20}}))
    real = tmp / "real"
    real.mkdir()
    (real / "results.json").write_text(json.dumps({"wer": 0.5, "word": {"n_ref": 95}}))
    summary = S.summarize(res, {"M02": real}, tmp / "summary.json")
    assert abs(summary["calibration"]["M02"]["severity"] - 0.6) < 0.01


def test_profile_reads_pace_from_the_manifest_alone(exports):
    controls, target, tmp = exports
    prof = S.profile_export(target, "train", audio_limit=0)
    assert prof["sec_per_char"]["word"] > 0 and prof["audio"] is None
    assert S.profile_export(target, "train", audio_limit=3)["audio"]["clips"] == 3


def test_bad_json_line_names_the_file_and_line(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{oops\n')
    with pytest.raises(ValueError, match="line 2"):
        S.read_jsonl(path)


def test_pooled_pace_is_the_median_of_all_clips(exports):
    controls, target, tmp = exports
    other = tmp / "M02"
    make_export(other, "M02", {"train": [("word", "alpha beta")], "dev": [("word", "d")], "test": [("word", "t")]}, seconds=3.2)
    pooled = S.pooled_profile([target, other])
    own = S.profile_export(target, "train", 0)
    assert pooled["counts"]["word"] == own["counts"]["word"] + 1
    rates = sorted([1.6 / 9, 1.6 / 5, 3.2 / 9])
    assert abs(pooled["sec_per_char"]["word"] - rates[1]) < 1e-3
