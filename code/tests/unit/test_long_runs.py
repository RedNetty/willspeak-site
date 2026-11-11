import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
import long_runs as runs


def arguments(study="adapter", **kw):
    return SimpleNamespace(study=study, stamp="TEST", batch=2, grad_accum=8, offload_optimizer=False,
                           keep_all_models=False, train_python=sys.executable, cpu_python=sys.executable,
                           min_free_gb=1, dry_run=False, resume=False, **kw)


def export(root, speaker, train=2, others=False):
    root.mkdir(parents=True)
    rng = np.random.default_rng(19 if others else int(speaker[-1]))
    for split, count in (("train", train), ("dev", 1), ("test", 1)):
        (root / split).mkdir()
        rows = []
        for i in range(count):
            spk = ("M02" if i % 2 else "F03") if others else speaker
            audio = f"{split}/{i}.wav"
            sf.write(root / audio, rng.normal(0, 0.1, 8000), 16000)
            rows.append({"id": f"{'other' if others else speaker}-{split}-{i}", "text": f"{spk} {split} prompt {i}",
                         "audio": audio, "speaker": spk, "duration_s": 0.5, "kind": "sentence"})
        (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("study", ["adapter", "acoustic", "replay"])
def test_entry_points_and_flags_are_valid_without_writing_outputs(tmp_path, monkeypatch, study):
    monkeypatch.setattr(runs, "APP", tmp_path)
    exp = runs.Experiment(arguments(study))
    exp.a.dry_run = True
    exp.dry_plan()
    assert not list(tmp_path.iterdir())
    text = subprocess.run([sys.executable, str(ROOT / "scripts/asr/train.py"), "--help"],
                          capture_output=True, text=True, check=True).stdout
    for recipe in exp.items:
        cmd = exp.train_command(recipe, "M01", 17)
        assert all(flag in text for flag in cmd if flag.startswith("--"))
        assert cmd[cmd.index("--select") + 1] == "last"
        smoke = exp.train_command(recipe, "M02", 17, smoke=True)
        assert smoke[smoke.index("--max-steps") + 1] == "2"
        assert smoke[smoke.index("--select") + 1] == "dev"
    name = {"adapter": "long_adapter", "acoustic": "acoustic_only", "replay": "speaker_replay"}[study]
    wrapper = ROOT / f"scripts/asr/{name}.sh"
    assert subprocess.run(["bash", "-n", str(wrapper)], check=False).returncode == 0
    assert f"long_runs.py {study}" in wrapper.read_text()
    result = subprocess.run(["bash", str(wrapper), "TEST", "--dry-run"], capture_output=True, text=True,
                            check=True, env={**os.environ, "SC": sys.executable})
    assert "No commands executed or files written" in result.stdout


def test_selection_uses_pooled_error_counts_and_prefers_control_on_ties(tmp_path):
    paths = []
    for i, (errors, words) in enumerate(((1, 2), (1, 20), (8, 80))):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps({"wer": errors / words, "word": {"errors": errors, "n_ref": words}}))
        paths.append(path)
    got = runs.pooled_scores(paths)
    assert got["wer"] == pytest.approx(10 / 102)
    scores = {"long": got, "control": got}
    assert runs.choose_dev(scores) == "control"
    assert runs.choose_dev({"control": got, "long": dict(got, errors=9, wer=9 / 102)}) == "long"
    with pytest.raises(ValueError, match="different dev"):
        runs.choose_dev({"control": got, "long": dict(got, words=100)})
    with pytest.raises(ValueError, match="every planned speaker"):
        runs.pooled_scores(paths[:2])


def test_replay_excludes_heldout_prompts_and_duplicate_audio_and_matches_budget(tmp_path):
    personal, others = tmp_path / "personal", tmp_path / "others"
    export(personal, "M01")
    export(others, "M01", train=12, others=True)
    rows = list(runs.load_manifest(others / "train.jsonl").values())
    dev = next(iter(runs.load_manifest(personal / "dev.jsonl").values()))
    rows[0]["text"] = dev["text"].upper() + "."
    (others / rows[1]["audio"]).write_bytes((personal / dev["audio"]).read_bytes())
    (others / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    original = (personal / "train.jsonl").read_bytes()
    selected = {}
    for recipe in runs.recipes("replay")[1:]:
        out = tmp_path / recipe.replay
        record = runs.prepare_replay(personal, others, "M01", recipe.replay, out)
        assert "other-train-0" not in record["other_ids"] and "other-train-1" not in record["other_ids"]
        assert 2 * recipe.repeat + record["other_clips"] == 8
        assert not (out / "test.jsonl").exists()
        assert record == runs.prepare_replay(personal, others, "M01", recipe.replay, out)
        selected[recipe.replay] = record["other_ids"]
    assert set(selected["all75"]) <= set(selected["all50"])
    assert (personal / "train.jsonl").read_bytes() == original
    assert all(r["speaker"] == "M02" for r in runs.load_manifest(tmp_path / "severe50/train.jsonl").values()
               if r["speaker"] != "M01")
    with pytest.raises(ValueError, match="distinct other-speaker"):
        runs.balanced_others(rows[:1], 2)


def test_replay_pool_skips_clips_too_short_for_nemo_to_load(tmp_path):
    personal, others = tmp_path / "personal", tmp_path / "others"
    export(personal, "M01")
    export(others, "M01", train=12, others=True)
    rows = list(runs.load_manifest(others / "train.jsonl").values())
    rows[2]["duration_s"] = 0.05
    (others / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = runs.prepare_replay(personal, others, "M01", "all50", tmp_path / "out")
    assert rows[2]["id"] not in record["other_ids"] and record["eligible_other_clips"] == len(rows) - 1
    assert runs.usable_clip({"duration_s": 0.1}) and runs.usable_clip({"duration_s": 25}) and not runs.usable_clip({"duration_s": 25.01})


@pytest.mark.parametrize("loaded, batches, dropped", [(624, 312, 0), (623, 312, 1), (622, 311, 2), (327, 164, 1)])
def test_replay_budget_tolerates_a_clip_or_two_dropped_by_nemo(loaded, batches, dropped):
    target = 624 if loaded > 400 else 328
    assert runs.replay_budget_dropped({"clips": loaded, "batches_per_epoch": batches}, target // 4, 2) == dropped


@pytest.mark.parametrize("loaded, batches", [(625, 313), (620, 310), (600, 300), (624, 300)])
def test_replay_budget_still_fails_for_real_mismatches(loaded, batches):
    with pytest.raises(RuntimeError):
        runs.replay_budget_dropped({"clips": loaded, "batches_per_epoch": batches}, 156, 2)


def test_freezing_head_preserves_encoder_gradients():
    torch = pytest.importorskip("torch")
    import train
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(3, 3)
    model.decoder = torch.nn.Linear(3, 3)
    model.joint = torch.nn.Linear(3, 1)
    frozen = train.freeze_transducer_head(model)
    assert frozen == 16
    model.joint(model.decoder(model.encoder(torch.ones(2, 3)))).sum().backward()
    assert model.encoder.weight.grad is not None
    assert torch.isfinite(model.encoder.weight.grad).all()
    assert model.decoder.weight.grad is None and model.joint.weight.grad is None
    with pytest.raises(ValueError, match="decoder and joint"):
        train.freeze_transducer_head(SimpleNamespace(decoder=None, joint=None))


def test_search_locks_dev_winner_then_confirms_and_tests_only_finalists(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "APP", tmp_path)
    exp = runs.Experiment(arguments())
    exp.report.mkdir(parents=True)
    exp.items = [runs.Recipe("control", 15, "3e-4", adapter=True), runs.Recipe("better", 75, "3e-4", adapter=True),
                 runs.Recipe("loser", 75, "1e-4", adapter=True)]
    events = []

    def fake_train(recipe, speaker, seed):
        if not getattr(exp, "final_phase", False):
            assert not any(event[0] == "test" for event in events)
        else:
            assert (exp.report / "selection.json").is_file()
        assert not list(exp.models.rglob("adapter.pt"))
        events.append(("train", recipe.name, speaker, seed))
        model = exp.model_file(recipe, speaker, seed)
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text("fake model")
        score = exp.score_file(recipe, speaker, seed, "dev")
        score.parent.mkdir(parents=True, exist_ok=True)
        score.write_text(json.dumps({"wer": 0.2 if recipe.name == "better" else 0.5,
                                    "word": {"errors": 2 if recipe.name == "better" else 5, "n_ref": 10}}))

    def fake_decode(flags, manifest, audio, out):
        assert (exp.report / "selection.json").is_file()
        events.append(("test", str(out)))
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps({"wer": 0.2, "word": {"errors": 2, "n_ref": 10}}))

    monkeypatch.setattr(exp, "train_score", fake_train)
    monkeypatch.setattr(exp, "decode_score", fake_decode)
    monkeypatch.setattr(exp, "run", lambda *a: None)
    monkeypatch.setattr(exp, "battery", lambda *a: None)
    monkeypatch.setattr(exp, "battery_trial", lambda *a: None)
    chosen = exp.search()
    assert chosen == "better"
    assert not exp.model_file(exp.items[2], "M01", 17).exists()
    assert not exp.model_file(exp.items[0], "M01", 17).exists()
    exp.finish(chosen)
    assert not any(e[1] == "loser" and e[3] != 17 for e in events if e[0] == "train")
    assert all("loser" not in e[1] for e in events if e[0] == "test")
    assert json.loads((exp.report / "summary.json").read_text())["selection"] == "better"


def test_resume_rejects_changed_audio_not_just_changed_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "APP", tmp_path)
    exp = runs.Experiment(arguments())
    scripts = tmp_path / "scripts/asr"
    scripts.mkdir(parents=True)
    for name in ("long_runs", "train", "block_adapter", "nemo_hyp", "evaluate", "pool_table", "battery", "ci"):
        (scripts / f"{name}.py").write_text("# synthetic code fingerprint\n")
    src = tmp_path / "src/willspeak"
    src.mkdir(parents=True)
    for name in ("artifacts.py", "evaluate.py", "normalize.py", "us_spelling.json", "g2p.py", "db.py"):
        (src / name).write_text("synthetic dependency fingerprint\n")
    for speaker in runs.SPEAKERS:
        export(exp.export / speaker, speaker)
    exp.prepare()
    exp.a.resume = True
    exp.prepare()
    sf.write(exp.export / "M01/train/0.wav", np.random.default_rng(87).normal(0, 0.1, 8000), 16000)
    with pytest.raises(ValueError, match="manifests changed"):
        exp.prepare()
    original = (exp.report / "experiment.json").read_text()
    exp.models.mkdir(parents=True)
    (exp.models / "partial.pt").write_text("old weights")
    exp.a.restart_changed = True
    exp.prepare()
    assert (exp.report.with_name(exp.report.name + "-previous") / "experiment.json").read_text() == original
    assert (exp.models.with_name(exp.models.name + "-previous") / "partial.pt").read_text() == "old weights"
    assert not (exp.models / "partial.pt").exists()
    assert (exp.report / "experiment.json").read_text() != original
    exp.prepare()


def test_completed_training_can_finish_dev_scoring_without_retraining(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "APP", tmp_path)
    exp = runs.Experiment(arguments())
    recipe = exp.items[0]
    folder = exp.model_dir(recipe, "M01", 17)
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"status": "complete"}))
    (folder / "curve.csv").write_text("epoch,loss\n0,1\n")
    exp.model_file(recipe, "M01", 17).write_text("synthetic weights")
    (folder / "ckpt").mkdir()
    (folder / "ckpt/old").write_text("intermediate weights")
    monkeypatch.setattr(exp, "run", lambda *a: pytest.fail("completed training must not restart"))

    def fake_decode(flags, manifest, audio, out):
        out.mkdir(parents=True)
        (out / "results.json").write_text("{}")

    monkeypatch.setattr(exp, "decode_score", fake_decode)
    exp.train_score(recipe, "M01", 17)
    exp.train_score(recipe, "M01", 17)
    assert not (folder / "ckpt").exists()
    assert (exp.score_file(recipe, "M01", 17, "dev").parent / "training/curve.csv").exists()
    failed = exp.model_dir(recipe, "M02", 17)
    failed.mkdir(parents=True)
    (failed / "run.json").write_text(json.dumps({"status": "failed"}))
    attempts = []
    def retry_run(*args):
        attempts.append(args)
        failed.mkdir(parents=True)
        (failed / "run.json").write_text(json.dumps({"status": "complete"}))
        (failed / "curve.csv").write_text("epoch,loss\n0,1\n")
        exp.model_file(recipe, "M02", 17).write_text("retried weights")
    monkeypatch.setattr(exp, "run", retry_run)
    exp.train_score(recipe, "M02", 17)
    assert attempts and (failed.with_name(failed.name + "-failed") / "run.json").is_file()


def test_one_search_failure_does_not_stop_remaining_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "APP", tmp_path)
    exp = runs.Experiment(arguments())
    exp.report.mkdir(parents=True)
    exp.items = [runs.Recipe("control", 15, "3e-4", adapter=True), runs.Recipe("other", 15, "3e-4", adapter=True)]
    tried = []
    def train(recipe, speaker, seed):
        tried.append((recipe.name, speaker))
        if recipe.name == "control" and speaker == "M01":
            raise RuntimeError("one training crash")
        out = exp.score_file(recipe, speaker, seed, "dev")
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps({"wer": 0.2, "word": {"errors": 2, "n_ref": 10}}))
    monkeypatch.setattr(exp, "train_score", train)
    with pytest.raises(RuntimeError, match="Search trials failed"):
        exp.search()
    assert tried == [(r.name, s) for r in exp.items for s in runs.SPEAKERS]
    assert "one training crash" in (exp.report / "failures.jsonl").read_text()
    assert not (exp.report / "selection.json").exists()
    assert not (exp.report / "summary.json").exists()


def test_low_disk_frees_finished_weights_then_stops_the_study_instead_of_failing_every_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "APP", tmp_path)
    args = arguments()
    args.min_free_gb = 10 ** 6
    exp = runs.Experiment(args)
    recipe = exp.items[0]
    cleaned = []
    monkeypatch.setattr(runs.subprocess, "run", lambda cmd, **kw: cleaned.append(cmd))
    (tmp_path / "scripts/asr").mkdir(parents=True)
    (tmp_path / "scripts/asr/lab_space.py").write_text("")
    monkeypatch.setattr(exp, "run", lambda *a: pytest.fail("no training without room"))
    with pytest.raises(runs.DiskFull) as stop:
        exp.train_score(recipe, "M01", 17)
    assert cleaned and "--long-models" in cleaned[0] and "--resume" in str(stop.value)
    assert not issubclass(runs.DiskFull, RuntimeError)


def test_enough_room_after_cleaning_lets_the_run_go_on(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "APP", tmp_path)
    exp = runs.Experiment(arguments())
    monkeypatch.setattr(runs.shutil, "disk_usage", lambda _p: SimpleNamespace(free=5 * 2 ** 30))
    exp.make_room(tmp_path / "somewhere")
