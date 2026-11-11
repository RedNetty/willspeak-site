import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
import lab_doctor as D  # noqa: E402


def write_wav(path: Path, seconds=0.5, rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate) * channels)


def make_export(root: Path, name="M01", texts=None, seconds=0.5):
    folder = root / D.EX / name
    texts = texts or {"train": ["alpha", "beta"], "dev": ["gamma"], "test": ["delta"]}
    for split, items in texts.items():
        rows = []
        for i, text in enumerate(items):
            rel = f"{split}/{name}-{i}.wav"
            write_wav(folder / rel, seconds)
            rows.append({"id": f"{name}-{split}-{i}", "audio": rel, "text": text, "duration_s": seconds})
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return folder


def levels(findings):
    return [(f.level, f.area) for f in findings]


def test_a_consistent_export_passes_even_deep(tmp_path):
    make_export(tmp_path)
    found = D.check_export(tmp_path, "M01", deep=True)
    assert [f.level for f in found] == ["ok"]


def test_missing_wav_is_flagged_and_offers_a_repair(tmp_path):
    folder = make_export(tmp_path)
    (folder / "train/M01-0.wav").unlink()
    found = D.check_export(tmp_path, "M01")
    bad = [f for f in found if f.level == "fail"][0]
    assert bad.fix == ("repair-audio", "M01") and "train/M01-0.wav: missing" in bad.detail[0]


def test_wrong_rate_and_wrong_length_found_only_by_the_deep_check(tmp_path):
    folder = make_export(tmp_path)
    write_wav(folder / "test/M01-0.wav", rate=44100)
    write_wav(folder / "dev/M01-0.wav", seconds=2.0)
    assert [f.level for f in D.check_export(tmp_path, "M01")] == ["ok"]
    deep = D.check_export(tmp_path, "M01", deep=True)
    messages = " ".join(d for f in deep for d in f.detail)
    assert "44100 Hz" in messages and "2.00 s long" in messages


def test_duplicate_ids_bad_json_and_prompt_leak(tmp_path):
    folder = make_export(tmp_path, texts={"train": ["alpha", "alpha"], "dev": ["g"], "test": ["Alpha!"]})
    rows = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
    rows[1]["id"] = rows[0]["id"]
    (folder / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "{broken\n")
    text = " ".join(f.message for f in D.check_export(tmp_path, "M01"))
    assert "duplicate ids" in text and "unreadable lines" in text and "both train and test" in text


def test_missing_export_folder_offers_to_make_it(tmp_path):
    found = D.check_export(tmp_path, "loso-M02")
    assert found[0].level == "fail" and found[0].fix == ("make-exports", "loso-M02")


def test_pins_catch_a_changed_test_set_and_repin_accepts_it(tmp_path):
    for s in D.SEVERE:
        make_export(tmp_path, s)
    assert D.check_pins(tmp_path)[0].level == "info"
    D.write_pins(tmp_path)
    assert D.check_pins(tmp_path)[0].level == "ok"
    test = tmp_path / D.EX / "M02/test.jsonl"
    test.write_text(test.read_text() + "\n" + json.dumps({"id": "x", "audio": "a", "text": "t", "duration_s": 1}) + "\n")
    changed = D.check_pins(tmp_path)
    assert changed[0].level == "fail" and "M02/test.jsonl" in changed[0].detail[0]
    D.write_pins(tmp_path)
    assert D.check_pins(tmp_path)[0].level == "fail"
    D.write_pins(tmp_path, repin=True)
    assert D.check_pins(tmp_path)[0].level == "ok"


def make_decode(root: Path, name: str, scored=True, provenance=True, manifest=None):
    folder = root / D.FT
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jsonl").write_text('{"id": "a", "hyp": "x"}\n')
    if provenance:
        sha = D.sha256_file(manifest) if manifest else "0" * 64
        (folder / f"{name}.provenance.json").write_text(json.dumps({"inputs": {"manifest": {"sha256": sha, "size": 1}}}))
    if scored:
        (folder / name).mkdir()
        (folder / name / "results.json").write_text(json.dumps({"wer": 0.5}))


def test_outputs_flags_orphans_unscored_and_bad_results(tmp_path):
    make_decode(tmp_path, "M01-test-stock-nemo")
    make_decode(tmp_path, "M01-test-orphan-nemo", provenance=False, scored=False)
    make_decode(tmp_path, "M01-test-unscored-nemo", scored=False)
    make_decode(tmp_path, "M01-test-badjson-nemo")
    (tmp_path / D.FT / "M01-test-badjson-nemo/results.json").write_text("{nope")
    (tmp_path / D.FT / "lonely-nemo.provenance.json").write_text("{}")
    kinds = {f.fix[0]: f for f in D.check_outputs(tmp_path) if f.fix}
    assert set(kinds) == {"quarantine", "rescore"}
    text = " ".join(f.message for f in D.check_outputs(tmp_path))
    assert "no provenance" in text and "never scored" in text and "unreadable" in text and "no decode" in text


def test_quarantine_renames_never_deletes_and_does_not_collide(tmp_path):
    victim = tmp_path / "file.jsonl"
    victim.write_text("one")
    first = D.quarantine(victim)
    victim.write_text("two")
    second = D.quarantine(victim)
    assert first.read_text() == "one" and second.read_text() == "two" and first != second
    assert not victim.exists()


def test_fix_quarantine_takes_the_provenance_file_with_the_decode(tmp_path):
    make_decode(tmp_path, "M01-test-x-nemo", scored=False)
    note = D.fix_quarantine(tmp_path, [f"{D.FT}/M01-test-x-nemo.jsonl"])
    assert "corrupt" in note
    left = sorted(p.name for p in (tmp_path / D.FT).iterdir())
    assert left == ["M01-test-x-nemo.jsonl-corrupt", "M01-test-x-nemo.provenance.json-corrupt"]


def test_rescore_only_when_the_manifest_is_identified_by_checksum(tmp_path, monkeypatch):
    make_export(tmp_path, "M01")
    manifest = tmp_path / D.EX / "M01/test.jsonl"
    make_decode(tmp_path, "M01-test-ok-nemo", scored=False, manifest=manifest)
    make_decode(tmp_path, "M01-test-stale-nemo", scored=False)
    make_decode(tmp_path, "mixed-M01-M02-test-nemo", scored=False, manifest=manifest)
    calls = []
    monkeypatch.setattr(D, "run", lambda cmd, timeout=120, cwd=None: (calls.append(cmd) or (0, "")))
    paths = [f"{D.FT}/{n}.jsonl" for n in ("M01-test-ok-nemo", "M01-test-stale-nemo", "mixed-M01-M02-test-nemo")]
    note = D.fix_rescore(tmp_path, paths, "python")
    assert len(calls) == 1 and calls[0][calls[0].index("--manifest") + 1] == str(manifest)
    assert "scored 1" in note and "2" in note


def test_apply_fixes_runs_each_action_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(D, "run", lambda cmd, timeout=120, cwd=None: (calls.append(cmd) or (0, "done")))
    found = [D.Finding("warn", "disk", "low", fix=("clean-space", None)), D.Finding("warn", "disk", "low again", fix=("clean-space", None)),
             D.Finding("fail", "data", "no controls", fix=("fetch-controls", None))]
    done = D.apply_fixes(found, tmp_path, "python")
    assert sum("lab_space.py" in " ".join(c) for c in calls) == 1
    assert any("--allow-download" in line for line in done) and not any("fetch_torgo" in " ".join(c) for c in calls)
    D.apply_fixes(found, tmp_path, "python", allow_download=True)
    assert any("fetch_torgo" in " ".join(c) for c in calls)


@pytest.mark.parametrize("text, category, retry", [
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate", "oom", True),
    ("OSError: [Errno 28] No space left on device", "disk", True),
    ("RuntimeError: Less than 14 GiB free before models/x; save/free results first", "disk", True),
    ("urllib.error.URLError ... Temporary failure in name resolution", "network", True),
    ("training environment is not ready:\n  - 3.9 GB free where /home/student/models/parakeet-ft-big-c-wb-default-M01-s17 goes; "
     "this run needs about 6 GB (checkpoints + final .nemo). Use --out under models/, not /tmp", "disk", True),
    ("Killed", "killed", True),
    ("MISSING reports/finetune-torgo/x/results.json, stopping this stage", "missing", True),
    ("ALREADY EXISTS reports/x, stopping this stage", "exists", False),
    ("ValueError: Run configuration, code or manifests changed; use a new stamp", "changed", False),
    ("something odd happened", "unknown", False),
])
def test_failure_classes(text, category, retry):
    f = D.classify_failure("lots\nof\nlines\n" + text)
    assert f.category == category and f.retryable is retry


def test_oom_wins_over_a_generic_cuda_error():
    assert D.classify_failure("CUDA error: out of memory").category == "oom"
    assert D.classify_failure("CUDA error: an illegal memory access").category == "cuda"


def test_runtime_check_on_this_python_and_missing_module_gets_a_pip_fix(monkeypatch):
    found = D.check_runtime(sys.executable, None, need_gpu=False)
    assert any(f.level == "ok" and "CPU Python" in f.message for f in found) or sys.version_info < D.MIN_PYTHON
    monkeypatch.setattr(D, "CPU_MODULES", ("numpy", "definitely_not_a_module_xyz"))
    found = D.check_runtime(sys.executable, None, need_gpu=False)
    pip = [f for f in found if f.fix and f.fix[0] == "pip"]
    assert pip and "definitely_not_a_module_xyz" in pip[0].fix[1]
    assert any(f.level == "fail" and "training environment" in f.message for f in found)


def test_disk_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "free_gib", lambda root: 5.0)
    assert D.check_disk(tmp_path)[0].level == "fail"
    monkeypatch.setattr(D, "free_gib", lambda root: 20.0)
    assert D.check_disk(tmp_path)[0].level == "warn"
    monkeypatch.setattr(D, "free_gib", lambda root: 200.0)
    assert D.check_disk(tmp_path)[0].level == "ok"


def test_the_report_counts_and_returns_failures(capsys):
    fails, warns = D.report([D.Finding("ok", "a", "fine"), D.Finding("warn", "b", "hmm"), D.Finding("fail", "c", "bad", fix=("pin", None))])
    out = capsys.readouterr().out
    assert (fails, warns) == (1, 1) and "--fix: pin" in out and "1 pass, 1 warn, 1 FAIL" in out
