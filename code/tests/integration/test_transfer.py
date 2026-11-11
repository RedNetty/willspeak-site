from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from willspeak.artifacts import file_record, new_directory, portable_name, validate_export, write_json
from willspeak.backup import restore_backup
from willspeak.bundles import inspect_bundle, pack_corpus, pack_model, unpack_bundle
from willspeak.db import Store
from willspeak.export import export_corpus

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recorded_store(tmp_path):
    store = Store(tmp_path / "recordings")
    sid = store.start_session()
    for i, text in enumerate(("alpha", "beta")):
        relative = f"clips/{i}.wav"
        sf.write(store.root / relative, np.sin(np.arange(16000) * 0.03) * 0.1, 16000)
        cid = store.add_clip(sid, None, text, relative, None, 1, {"ok": True})
        store.review_clip(cid, text, "accepted")
    (store.root / "split_assignments.json").write_text('{"alpha":"train","beta":"test"}')
    yield store
    store.con.close()


def test_failed_export_preserves_previous_files_and_split_map(recorded_store, monkeypatch):
    store = recorded_store
    out = store.root / "export"
    export_corpus(store, store.root, trim=False)
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assignment = (store.root / "split_assignments.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk read failed")

    monkeypatch.setattr(sf, "read", fail)
    with pytest.raises(OSError, match="disk read failed"):
        export_corpus(store, store.root)
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    assert (store.root / "split_assignments.json").read_bytes() == assignment
    assert not list(store.root.glob(".willspeak-export-*"))


def test_export_refuses_corrected_transcript_leakage(recorded_store):
    store = recorded_store
    export_corpus(store, store.root, trim=False)
    previous = (store.root / "export/test.jsonl").read_bytes()
    store.review_clip(store.clips()[0]["id"], "ALPHA!", "accepted")
    with pytest.raises(ValueError, match="two splits"):
        export_corpus(store, store.root, trim=False)
    assert (store.root / "export/test.jsonl").read_bytes() == previous


@pytest.mark.parametrize("destination", [".", "clips", "raw", "backups"])
def test_export_protects_recordings_and_backups(recorded_store, destination):
    store = recorded_store
    audio = (store.root / "clips/0.wav").read_bytes()
    with pytest.raises(ValueError, match="must not replace"):
        export_corpus(store, store.root, store.root / destination)
    assert (store.root / "clips/0.wav").read_bytes() == audio


def test_export_refuses_corrupt_split_map(recorded_store):
    path = recorded_store.root / "split_assignments.json"
    path.write_text('{"alpha":"unknown"}')
    with pytest.raises(ValueError, match="Invalid split_assignments"):
        export_corpus(recorded_store, recorded_store.root)
    assert not (recorded_store.root / "export").exists()


def test_backup_preserves_previous_archive_and_settings(tmp_path, monkeypatch):
    from willspeak import backup as backup_module
    from willspeak.db import Store

    store = Store(tmp_path / "corpus")
    settings = store.root / "live_settings.json"
    settings.write_text('{"stop_phrase": "green tractor"}')
    monkeypatch.setattr(backup_module.time, "strftime", lambda _: "20261001-120000")
    try:
        first = backup_module.make_backup(store, store.root)
        before = first.read_bytes()
        second = backup_module.make_backup(store, store.root)
        assert first != second and first.read_bytes() == before
        target = tmp_path / "restored"
        restore_backup(second, target)
        assert (target / settings.name).read_bytes() == settings.read_bytes()

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(zipfile.ZipFile, "write", fail)
        with pytest.raises(OSError, match="disk full"):
            backup_module.make_backup(store, store.root)
        assert backup_module.backup_stats(store.root)["count"] == 2
        assert not list((store.root / "backups").glob("*.partial"))
        assert first.read_bytes() == before
    finally:
        store.con.close()


def test_ci_rejects_mismatched_saved_score_and_duplicate_hypotheses(tmp_path):
    script = load_script("asr/ci")
    manifest = tmp_path / "test.jsonl"
    manifest.write_text('{"id": "one", "text": "hello"}\n')
    (tmp_path / "hypotheses.jsonl").write_text('{"id": "one", "hyp": "hello"}\n')
    (tmp_path / "results.json").write_text('{"wer": 1.0}')
    with pytest.raises(ValueError, match="disagrees"):
        script.main(["--manifest", str(manifest), "--stock", str(tmp_path)])
    assert not (tmp_path / "ci.json").exists()
    (tmp_path / "hypotheses.jsonl").write_text('{"id": "one", "hyp": "hello"}\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        script.load_hypotheses(tmp_path)


def test_ci_paired_bootstrap_and_invalid_inputs():
    script = load_script("asr/ci")
    errors, lengths = np.array([1, 3]), np.array([3, 1])
    assert script.corpus_wer(errors, lengths) == 1
    result = script.bootstrap(errors, lengths, errors, n_resamples=20)
    assert np.all(result["delta"] == 0)
    for bad in (np.array([]), np.array([0, 1]), np.array([np.nan, 1])):
        with pytest.raises(ValueError):
            script.bootstrap(errors, bad)


def test_ci_cluster_bootstrap_keeps_whole_groups_paired():
    script = load_script("asr/ci")
    errors, lengths, trained = np.array([0, 2, 4]), np.array([2, 2, 4]), np.array([1, 1, 0])
    clustered = script.bootstrap(errors, lengths, trained, n_resamples=100,
                                 clusters=[("a",), ("a",), ("b",)])
    expected = script.bootstrap(np.array([2, 4]), np.array([4, 4]), np.array([2, 0]), n_resamples=100)
    for key in expected:
        assert np.array_equal(clustered[key], expected[key])
    with pytest.raises(ValueError, match="at least two"):
        script.bootstrap(errors, lengths, clusters=[("one",)] * 3)
    with pytest.raises(ValueError, match="Each clip"):
        script.bootstrap(errors, lengths, clusters=[("a",)])
    rows = {"1": {"speaker": "M01", "session": "Session1"},
            "2": {"speaker": "M04", "session": "Session1"}}
    assert script.cluster_keys(rows, ["speaker", "session"]) == [("M01", "Session1"), ("M04", "Session1")]
    with pytest.raises(ValueError, match="nonempty"):
        script.cluster_keys(rows, ["kind"])


def test_ci_archives_distinct_baselines_with_input_records(tmp_path):
    script = load_script("asr/ci")
    manifest = tmp_path / "test.jsonl"
    manifest.write_text('{"id": "1", "text": "hello"}\n{"id": "2", "text": "water"}\n')
    for name, hyps, wer in (("stock", ["world", "water"], 0.5), ("other", ["", ""], 1),
                            ("trained", ["hello", "water"], 0)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "results.json").write_text(json.dumps({"wer": wer}))
        (folder / "hypotheses.jsonl").write_text("".join(
            json.dumps({"id": str(i + 1), "hyp": hyp}) + "\n" for i, hyp in enumerate(hyps)))
    for stock in ("stock", "other"):
        assert script.main(["--manifest", str(manifest), "--stock", str(tmp_path / stock),
                            "--finetuned", str(tmp_path / "trained")]) == 0
    saved = list((tmp_path / "trained/ci").glob("*.json"))
    assert len(saved) == 2
    latest = json.loads((tmp_path / "trained/ci.json").read_text())
    assert latest["stock_report"]["path"] == str((tmp_path / "other").resolve())
    assert latest["manifest"]["sha256"] == file_record(manifest)["sha256"]
    assert latest["scoring"]["us_spelling.json"] == file_record(ROOT / "src/willspeak/us_spelling.json")
    assert latest["method"] == "paired-percentile-bootstrap" and latest["n_units"] == 2


def load_script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name.replace("/", "_"), ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "laptop" / "export"
    assignments = {}
    for i, (split, text) in enumerate(zip(("train", "dev", "test"), ("bring water", "hello friend", "please help"))):
        (root / split).mkdir(parents=True)
        samples = np.sin(np.arange(8000) * (i + 1) * 0.05) * 0.1
        sf.write(root / split / f"{i}.wav", samples, 16000)
        row = {
            "id": i,
            "audio": f"{split}/{i}.wav",
            "text": text,
            "prompt_text": text,
            "duration_s": 0.5,
            "prompt_set": "functional",
            "condition": "good_day",
        }
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        assignments[text] = split
    path = root.parent / "split_assignments.json"
    path.write_text(json.dumps(assignments))
    (root / "unreferenced.wav").write_bytes(b"must not transfer")
    (root.parent / "corpus.db").write_bytes(b"must not transfer")
    return root, path


@pytest.fixture
def model(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (directory / name).write_bytes(name.encode())
    return directory


def change_row(corpus, split, **values):
    path = corpus[0] / f"{split}.jsonl"
    row = json.loads(path.read_text())
    row.update(values)
    path.write_text(json.dumps(row) + "\n")


def test_training_preflight_distinguishes_unique_audio_and_repeats(corpus, tmp_path, monkeypatch, capsys):
    script = load_script("asr/train")
    change_row(corpus, "train", speaker="M01")
    (corpus[0] / "test.jsonl").unlink()
    out = tmp_path / "new model"
    monkeypatch.setattr(sys, "argv", ["train.py", "--export", str(corpus[0]), "--out", str(out),
                                    "--repeat-speakers", "M01", "--repeat", "3", "--check"])
    assert script.main() == 0
    text = capsys.readouterr().out
    assert "1 unique clips" in text and "3 clip uses / epoch" in text
    assert "test split was not opened" in text
    assert not out.exists()


def test_corpus_roundtrip_only_transfers_referenced_accepted_audio(corpus, tmp_path):
    before = {p: file_record(p) for p in corpus[0].parent.rglob("*") if p.is_file()}
    bundle = tmp_path / "snapshot.zip"
    packed = pack_corpus(*corpus, bundle)
    assert packed == inspect_bundle(bundle)
    assert len(packed["files"]) == 7
    target = tmp_path / "desktop snapshot"
    assert unpack_bundle(bundle, target) == packed
    assert not (target / "corpus.db").exists()
    assert not (target / "export/unreferenced.wav").exists()
    assert (target / "split_assignments.json").read_bytes() == corpus[1].read_bytes()
    assert {p: file_record(p) for p in before} == before
    with pytest.raises(FileExistsError):
        unpack_bundle(bundle, target)
    with pytest.raises(FileExistsError):
        pack_corpus(*corpus, bundle)


@pytest.mark.parametrize(
    "values,match",
    [
        ({"text": "bring water"}, "Prompt leakage"),
        ({"id": 0}, "appears in both"),
        ({"duration_s": -1}, "Invalid duration"),
        ({"duration_s": float("nan")}, "Invalid duration"),
        ({"audio": "../private.wav"}, "must be under"),
        ({"prompt_set": "critical"}, "Critical/probe clips belong in train"),
    ],
)
def test_rejects_invalid_or_leaking_corpus(corpus, tmp_path, values, match):
    change_row(corpus, "test", **values)
    with pytest.raises(ValueError, match=match):
        pack_corpus(*corpus, tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_identical_audio_in_other_split_rejected(corpus):
    (corpus[0] / "test/2.wav").write_bytes((corpus[0] / "train/0.wav").read_bytes())
    with pytest.raises(ValueError, match="Identical audio"):
        validate_export(corpus[0])


def test_train_preflight_never_opens_test(corpus, tmp_path):
    (corpus[0] / "test.jsonl").write_text("not json; must never be opened by training")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/asr/train.py"),
            "--check",
            "--export",
            str(corpus[0]),
            "--out",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "preflight passed" in result.stdout
    assert not (tmp_path / "run").exists()


def test_sticky_assignments_required(corpus, tmp_path):
    corpus[1].write_text("{}")
    with pytest.raises(ValueError, match="Sticky assignment"):
        pack_corpus(*corpus, tmp_path / "bad.zip")


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/escape",
        "a/../../escape",
        "a\\escape",
        "C:/escape",
        "a//b",
        "a/./b",
        "NUL.txt",
        "a/CON",
        "a/b.",
        "a/b ",
        "a:x",
        "",
    ],
)
def test_portable_paths_reject_windows_and_posix_traps(name):
    with pytest.raises(ValueError):
        portable_name(name)


def rewrite_zip(source, target, transform):
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(target, "w") as dst:
        for info in src.infolist():
            name, value = transform(info.filename, src.read(info))
            dst.writestr(name, value)


def test_corruption_leaves_no_installed_directory(corpus, tmp_path):
    original, corrupt = tmp_path / "good.zip", tmp_path / "bad.zip"
    pack_corpus(*corpus, original)
    rewrite_zip(
        original,
        corrupt,
        lambda name, content: (name, b"x" * len(content)) if name.endswith(".wav") else (name, content),
    )
    with pytest.raises(ValueError, match="Checksum"):
        unpack_bundle(corrupt, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


@pytest.mark.parametrize("extra", ["../escape", "export/TEST.jsonl", "extra.txt"])
def test_archive_traversal_collision_unexpected_files(corpus, tmp_path, extra):
    bundle = tmp_path / "bad.zip"
    pack_corpus(*corpus, bundle)
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr(extra, "bad")
    with pytest.raises(ValueError):
        unpack_bundle(bundle, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


def test_symlink_archive_rejected(tmp_path):
    bundle = tmp_path / "link.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "outside")
    with pytest.raises(ValueError, match="regular files"):
        inspect_bundle(bundle)


def evaluation_args(corpus, model, output, **kwargs):
    return argparse.Namespace(
        manifest=str(corpus[0] / "test.jsonl"),
        audio_root=str(corpus[0]),
        model_dir=str(model),
        out=str(output),
        group_by="condition",
        label="fixture, not real ASR",
        hyp=kwargs.get("hyp"),
    )


def test_json_record_survives_failed_replacement(tmp_path, monkeypatch):
    from willspeak import artifacts

    path = tmp_path / "run.json"
    write_json(path, {"status": "running", "curve": [1]})
    original = path.read_bytes()

    def fail(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failure"):
        write_json(path, {"status": "complete"})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        write_json(path, {"loss": float("nan")})
    assert path.read_bytes() == original


def test_empty_nemo_hypothesis_does_not_become_the_word_none():
    from types import SimpleNamespace

    script = load_script("asr/nemo_hyp")
    assert script.text_of(None) == ""
    assert script.text_of(SimpleNamespace(text=None)) == ""
    assert script.text_of(SimpleNamespace(text="hello")) == "hello"
    with pytest.raises(TypeError, match="non-text"):
        script.text_of(SimpleNamespace(text=[1, 2]))


@pytest.mark.parametrize("changed", [None, "manifest", "audio", "hypotheses"])
def test_imported_nemo_hypotheses_verify_decode_inputs(corpus, model, tmp_path, changed):
    script = load_script("asr/evaluate")
    hyp = tmp_path / "hyp.jsonl"
    hyp.write_text(json.dumps({"id": 2, "hyp": "please help"}) + "\n")
    decode = {
        "format": "willspeak-nemo-hypotheses-v1", "hypotheses": file_record(hyp),
        "inputs": {"manifest": file_record(corpus[0] / "test.jsonl"),
                   "audio/test/2.wav": file_record(corpus[0] / "test/2.wav")},
        "base": "fixture", "model_files": {},
    }
    write_json(hyp.with_suffix(".provenance.json"), decode)
    targets = {"manifest": corpus[0] / "test.jsonl", "audio": corpus[0] / "test/2.wav", "hypotheses": hyp}
    if changed:
        with targets[changed].open("ab") as handle:
            handle.write(b"\n")
    output = tmp_path / "report"
    args = evaluation_args(corpus, model, output, hyp=str(hyp))
    if changed:
        with pytest.raises(ValueError, match="decode provenance"):
            script.run_evaluation(args)
        assert not output.exists()
    else:
        assert script.run_evaluation(args) == 0
        saved = json.loads((output / "provenance.json").read_text())
        assert saved["decode"] == decode
        assert "decode_provenance" in saved["inputs"]


@pytest.mark.parametrize("changed", [None, "audio", "model"])
def test_nemo_decode_records_inputs_and_refuses_changes(corpus, tmp_path, monkeypatch, changed):
    from types import ModuleType, SimpleNamespace

    script = load_script("asr/nemo_hyp")
    out = tmp_path / "hyp.jsonl"
    base = tmp_path / "base.nemo"
    base.write_bytes(b"fixture model")

    class Model:
        def cpu(self):
            return self

        def eval(self):
            pass

        def transcribe(self, paths, **kwargs):
            assert paths == [str(corpus[0] / "test/2.wav")]
            if changed:
                target = base if changed == "model" else Path(paths[0])
                with target.open("ab") as handle:
                    handle.write(b"changed")
            return ["please help"]

    modules = {name: ModuleType(name) for name in ("nemo", "nemo.collections", "nemo.collections.asr", "torch")}
    modules["nemo"].collections = modules["nemo.collections"]
    modules["nemo.collections"].asr = modules["nemo.collections.asr"]
    modules["nemo.collections.asr"].models = SimpleNamespace(ASRModel=SimpleNamespace(restore_from=lambda _: Model()))
    from contextlib import nullcontext

    modules["torch"].cuda = SimpleNamespace(is_available=lambda: False)
    modules["torch"].no_grad = nullcontext
    for key, value in modules.items():
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.setattr(sys, "argv", ["nemo_hyp.py", "--nemo", str(base), "--manifest", str(corpus[0] / "test.jsonl"),
                                     "--audio-root", str(corpus[0]), "--out", str(out)])
    if changed:
        with pytest.raises(ValueError, match="changed during decoding"):
            script.main()
        assert not out.exists() and not out.with_suffix(".provenance.json").exists()
    else:
        assert script.main() == 0
        record = json.loads(out.with_suffix(".provenance.json").read_text())
        assert record["hypotheses"] == file_record(out)
        assert record["model_files"]["nemo"] == file_record(base)
        assert record["inputs"]["audio/test/2.wav"] == file_record(corpus[0] / "test/2.wav")


def test_evaluate_and_release_bind_exact_model_and_report(corpus, model, tmp_path, monkeypatch):
    script = load_script("asr/evaluate")
    monkeypatch.setattr(script, "transcribe_parakeet", lambda *a, **k: {"2": "please help"})
    report = tmp_path / "report"
    args = evaluation_args(corpus, model, report)
    assert script.run_evaluation(args) == 0
    provenance = json.loads((report / "provenance.json").read_text())
    assert provenance["model_files"]["tokens.txt"] == file_record(model / "tokens.txt")
    assert provenance["recognition"] == "parakeet"
    assert provenance["code"]["us_spelling.json"] == file_record(ROOT / "src/willspeak/us_spelling.json")
    difficulty = json.loads((report / "difficulty.json").read_text())
    assert difficulty["source_split"] == "test" and difficulty["format"] == "willspeak-difficulty-v1"
    bundle = tmp_path / "release.zip"
    pack_model(model, report, bundle)
    unpack_bundle(bundle, tmp_path / "installed")
    assert (tmp_path / "installed/model/tokens.txt").read_bytes() == (model / "tokens.txt").read_bytes()
    with pytest.raises(FileExistsError):
        script.run_evaluation(args)
    (model / "tokens.txt").write_text("different tokenizer")
    with pytest.raises(ValueError, match="exact model"):
        pack_model(model, report, tmp_path / "mismatch.zip")


def test_ci_rejects_changed_manifest_report_and_audio(corpus, model, tmp_path, monkeypatch):
    evaluator, ci = load_script("asr/evaluate"), load_script("asr/ci")
    monkeypatch.setattr(evaluator, "transcribe_parakeet", lambda *a, **k: {"2": "please help"})
    first, second = tmp_path / "first", tmp_path / "second"
    evaluator.run_evaluation(evaluation_args(corpus, model, first))
    manifest = corpus[0] / "test.jsonl"
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n")
    args = ["--manifest", str(manifest), "--stock", str(first)]
    with pytest.raises(ValueError, match="Manifest bytes"):
        ci.main(args)
    manifest.write_bytes(original)
    sf.write(corpus[0] / "test/2.wav", np.zeros(8000), 16000)
    evaluator.run_evaluation(evaluation_args(corpus, model, second))
    with pytest.raises(ValueError, match="different audio"):
        ci.main(args + ["--finetuned", str(second)])
    (first / "hypotheses.jsonl").write_text('{"id": "2", "hyp": "PLEASE HELP"}\n')
    with pytest.raises(ValueError, match="Report output changed"):
        ci.main(args)
    assert not (first / "ci.json").exists()


def test_modified_report_and_imported_predictions_cannot_be_released(corpus, model, tmp_path, monkeypatch):
    script = load_script("asr/evaluate")
    monkeypatch.setattr(script, "transcribe_parakeet", lambda *a, **k: {"2": "please help"})
    report = tmp_path / "report"
    script.run_evaluation(evaluation_args(corpus, model, report))
    (report / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="output changed"):
        pack_model(model, report, tmp_path / "bad.zip")
    hyp = tmp_path / "hyp.jsonl"
    hyp.write_text(json.dumps({"id": 2, "hyp": "please help"}) + "\n")
    imported = tmp_path / "imported"
    script.run_evaluation(evaluation_args(corpus, model, imported, hyp=str(hyp)))
    with pytest.raises(ValueError, match="imported hypotheses"):
        pack_model(model, imported, tmp_path / "imported.zip")


def test_missing_hypotheses_fail_without_report(corpus, model, tmp_path, monkeypatch):
    script = load_script("asr/evaluate")
    monkeypatch.setattr(script, "transcribe_parakeet", lambda *a, **k: {})
    with pytest.raises(ValueError, match="IDs must match"):
        script.run_evaluation(evaluation_args(corpus, model, tmp_path / "report"))
    assert not (tmp_path / "report").exists()


def test_failed_staging_leaves_existing_data_and_no_partial_destination(tmp_path):
    target = tmp_path / "new"
    with pytest.raises(RuntimeError), new_directory(target) as stage:
        (stage / "model").write_text("incomplete")
        raise RuntimeError("decode failed")
    assert not target.exists()


def test_cli_errors_have_nonzero_exit(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/corpus/transfer.py"), "inspect", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Transfer failed" in result.stderr


@pytest.fixture
def backup(tmp_path):
    database_path = tmp_path / "source.db"
    con = sqlite3.connect(database_path)
    con.executescript("CREATE TABLE clip(id); CREATE TABLE prompt(id); CREATE TABLE session(id);")
    con.close()
    archive_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("clips/test.wav", b"audio fixture")
        archive.write(database_path, "corpus.db")
    return archive_path


def test_recovery_dry_run_and_new_directory_only(backup, tmp_path):
    target = tmp_path / "recovered"
    assert restore_backup(backup, target, dry_run=True)["dry_run"]
    assert not target.exists()
    assert restore_backup(backup, target)["files"] == 2
    before = (target / "corpus.db").read_bytes()
    with pytest.raises(FileExistsError):
        restore_backup(backup, target)
    assert (target / "corpus.db").read_bytes() == before


@pytest.mark.parametrize("extra", ["../recovered-other/escape", "corpus.db-wal", "clips/TEST.wav"])
def test_recovery_validates_whole_archive_before_writing(backup, tmp_path, extra):
    with zipfile.ZipFile(backup, "a") as archive:
        archive.writestr(extra, "bad")
    with pytest.raises(ValueError):
        restore_backup(backup, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()


def test_recovery_invalid_database_leaves_no_partial_data(tmp_path):
    archive_path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("corpus.db", b"not sqlite")
    with pytest.raises(sqlite3.DatabaseError):
        restore_backup(archive_path, tmp_path / "recovered")
    assert not (tmp_path / "recovered").exists()



def test_failed_export_is_not_published(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace

    from willspeak.asr.parakeet import ParakeetASR

    script = load_script("asr/export")
    nemo = ModuleType("nemo")
    collections = ModuleType("nemo.collections")
    asr = ModuleType("nemo.collections.asr")
    model = SimpleNamespace(cpu=lambda: SimpleNamespace(eval=lambda: None))
    asr.models = SimpleNamespace(ASRModel=SimpleNamespace(from_pretrained=lambda name: model))
    nemo.collections = collections
    collections.asr = asr
    for name, module in (("nemo", nemo), ("nemo.collections", collections), ("nemo.collections.asr", asr)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ParakeetASR, "installed", staticmethod(lambda: True))

    def fail_export(model, stage, int8_parts, **options):
        (stage / "encoder.int8.onnx").write_bytes(b"incomplete")
        raise RuntimeError("smoke decode failed")

    monkeypatch.setattr(script, "export_model", fail_export)
    monkeypatch.setattr(sys, "argv", ["export_parakeet.py", "--stock", "--out", str(tmp_path / "export")])
    with pytest.raises(RuntimeError, match="smoke decode failed"):
        script.main()
    assert not (tmp_path / "export").exists()


def test_int8_weight_check_finds_channels_rounded_to_zero(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic

    script = load_script("asr/export")

    def depthwise_model(weights, path):
        channels = weights.shape[0]
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, channels, 16])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, channels, 16])
        w = numpy_helper.from_array(weights.astype(np.float32), "w")
        node = helper.make_node("Conv", ["x", "w"], ["y"], name="depthwise", group=channels,
                                kernel_shape=[3], pads=[1, 1])
        graph = helper.make_graph([node], "g", [x], [y], initializer=[w])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        onnx.save(model, str(path))

    rng = np.random.default_rng(0)
    small = rng.normal(0, 0.05, size=(64, 1, 3))
    damaged = small.copy()
    damaged[0] *= 2000
    for name, weights in (("healthy", small), ("damaged", damaged)):
        depthwise_model(weights, tmp_path / f"{name}.onnx")
        quantize_dynamic(str(tmp_path / f"{name}.onnx"), str(tmp_path / f"{name}.int8.onnx"),
                         weight_type=QuantType.QUInt8)
    healthy = script.int8_weight_check(tmp_path / "healthy.int8.onnx")
    broken = script.int8_weight_check(tmp_path / "damaged.int8.onnx")
    assert healthy["tensors"] == 1 and healthy["mostly_zero"] == 0
    assert broken["tensors"] == 1 and broken["mostly_zero"] == 1
    assert broken["worst"][0]["op"] == "ConvInteger" and broken["worst"][0]["at_zero_point"] > 0.9


def test_launcher_selects_model_before_shared_recognizer_is_created(model):
    code = (
        "import runpy, sys, types\n"
        "def start(*args, **kwargs):\n"
        "    from willspeak.asr import asr\n"
        "    print('selected=' + str(asr.model_dir))\n"
        "sys.modules['uvicorn'] = types.SimpleNamespace(run=start)\n"
        f"runpy.run_path({str(ROOT / 'run.py')!r}, run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "--model-dir", str(model)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert f"selected={model.resolve()}" in result.stdout
