import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lab_cli", ROOT / "lab.py")
lab = importlib.util.module_from_spec(spec)
sys.modules["lab_cli"] = lab
spec.loader.exec_module(lab)


def mirror(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    for name in ("scripts", "src"):
        (app / name).symlink_to(ROOT / name)
    return app


def dry(app: Path, **overrides) -> "lab.Lab":
    return lab.Lab(app, lab.Settings(app, overrides, env={}), dry=True, stamp="TEST", quiet=True)


def test_settings_precedence_and_types(tmp_path):
    app = tmp_path
    settings = lab.Settings(app, {}, env={})
    assert settings["speakers"] == "M01 M02 M04" and settings["rounds"] == 3 and settings["rerun"] is False
    settings.save("rounds", "5")
    assert lab.Settings(app, {}, env={})["rounds"] == 5
    assert lab.Settings(app, {}, env={"ROUNDS": "7"}).source("rounds") == (7, "env ROUNDS")
    assert lab.Settings(app, {"rounds": "9"}, env={"ROUNDS": "7"}).source("rounds") == (9, "--set")
    assert lab.Settings(app, {"rerun": "1"}, env={})["rerun"] is True
    assert lab.Settings(app, {}, env={"TRIM_MODELS": "1"})["trim_models"] == "on"
    settings.unset("rounds")
    assert lab.Settings(app, {}, env={})["rounds"] == 3
    with pytest.raises(KeyError):
        settings.save("nonsense", "1")
    with pytest.raises(ValueError):
        settings.save("wandb", "sometimes")


def test_merge_flags_replaces_options_and_their_negations():
    base = ["--adapter-after-blocks", "8", "--lr", "3e-4", "--no-keep-stock", "--speed-perturb", "0.85", "1.15"]
    merged = lab.merge_flags(base, ["--lr", "1e-3", "--keep-stock", "--speed-perturb", "0.7", "1.3", "--epochs", "9"])
    assert merged == ["--adapter-after-blocks", "8", "--lr", "1e-3", "--speed-perturb", "0.7", "1.3", "--keep-stock", "--epochs", "9"]
    assert lab.axis_flags("spec_augment", "off") == ["--no-spec-augment"]
    assert lab.axis_flags("time_stretch", "0.75 1.25") == ["--time-stretch", "0.75", "1.25"]
    assert lab.axis_flags("lr", "default") == []


def test_every_experiment_plans_cleanly_with_flags_the_scripts_accept(tmp_path):
    app = mirror(tmp_path)
    first = dry(app)
    ids = list(first.catalog)
    assert {"1", "8b", "15", "W", "B", "A", "N", "F", "L", "T", "E", "S", "C"} <= set(ids)
    commands, notes = lab.plan(lambda: dry(app), ids)
    bad = {st: lines for st, lines in notes.items() if lines}
    assert not bad, bad
    assert len(commands) > 500
    problems, checked = lab.flag_problems(app, commands)
    assert not problems, "\n".join(problems)
    assert checked >= 8, checked


def test_dry_run_leaves_the_repository_alone_and_writes_summaries(tmp_path):
    app = mirror(tmp_path)
    ok = lab.run_stages(dry(app), ["1", "3", "14", "L"])
    assert ok
    assert not (app / "reports").exists() and not (app / "models").exists() and not (app / "data").exists()
    assert (app / "logs" / "lab-TEST-summaries.txt").is_file()


def test_a_stage_refuses_to_overwrite_an_earlier_result(tmp_path):
    app = mirror(tmp_path)
    (app / "reports/lab-03-hypothesis-bounds/bounds").mkdir(parents=True)
    session = dry(app)
    assert not lab.run_stages(session, ["3"])
    assert any("ALREADY EXISTS reports/lab-03-hypothesis-bounds/bounds" in line for line in session.lines)
    assert not session.commands


def test_unknown_ids_are_skipped_and_shortcuts_expand(tmp_path):
    session = dry(mirror(tmp_path))
    assert lab.run_stages(session, ["99"])
    assert any("unknown stage 99" in line for line in session.lines)
    assert lab.resolve(session, ["cheap"]) == lab.DEFAULT_CHEAP.split()
    assert lab.resolve(session, ["l"]) == ["L"] and lab.resolve(session, ["8b"]) == ["8b"]


def test_a_bare_hypotheses_file_satisfies_the_check(tmp_path):
    session = lab.Lab(tmp_path, lab.Settings(tmp_path, {}, env={}), dry=False, quiet=True)
    (tmp_path / "wb.jsonl").write_text("{}\n")
    assert session.need_have("wb.jsonl/hypotheses.jsonl")
    assert not session.need_have("nothing/hypotheses.jsonl")
    assert any("MISSING hypotheses at nothing" in line for line in session.lines)


def test_rerun_moves_an_earlier_attempt_aside_instead_of_refusing(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/out.jsonl").write_text("{}\n")
    (tmp_path / "reports/out").mkdir()
    session = lab.Lab(tmp_path, lab.Settings(tmp_path, {"rerun": "1"}, env={}), dry=False, quiet=True)
    assert session.need_free("reports/out.jsonl")
    assert (tmp_path / "reports/out.jsonl-first-run").is_file() and (tmp_path / "reports/out-first-run").is_dir()
    assert session.need_free("reports/out.jsonl")


def test_rebuild_mode_routes_comparisons_to_the_new_baselines(tmp_path):
    session = dry(mirror(tmp_path), rebuild_missing="1")
    lab.run_stages(session, ["1", "7"])
    text = "\n".join(" ".join(c) for c in session.commands)
    assert "models/lab-baselines/whole-block-M01/parakeet-ft.nemo=0.25" in text
    assert "models/parakeet-ft-M02e-A3" not in text


def test_speakers_and_xflags_reach_the_commands(tmp_path):
    session = dry(mirror(tmp_path), speakers="M04", xflags="--offload-optimizer --seed 5")
    lab.run_stages(session, ["8"])
    trains = [c for c in session.commands if "scripts/asr/train.py" in c]
    assert len(trains) == 1 and "lab-08-ff-M04" in " ".join(trains[0])
    assert "--offload-optimizer" in trains[0] and trains[0][trains[0].index("--seed") + 1] == "5"


def test_wandb_settings_reach_child_processes(tmp_path):
    session = dry(mirror(tmp_path), wandb="offline", wandb_project="proj", wandb_entity="team")
    session.stage = "9"
    env = session.child_env()
    assert env["WILLSPEAK_WANDB"] == "offline" and env["WANDB_PROJECT"] == "proj" and env["WANDB_ENTITY"] == "team"
    assert env["WILLSPEAK_WANDB_GROUP"] == "TEST-9" and "lab" in env["WILLSPEAK_WANDB_TAGS"]
    assert "WANDB_API_KEY" not in lab.Settings(tmp_path, {}, env={}).file


def test_custom_axes_become_arms_with_a_default_to_compare_against():
    spec = {"name": "x", "recipe": "wb", "init": "stock", "axes": {"lr": ["1e-5", "4e-5"], "epochs": ["5", "8"]}, "flags": "", "seeds": [17], "speakers": None, "default": True}
    arms = lab.custom_arms(spec)
    assert [a.label for a in arms] == ["wb-default", "wb-lr1e-5-epochs5", "wb-lr1e-5-epochs8", "wb-lr4e-5-epochs5", "wb-lr4e-5-epochs8"]
    assert arms[2].flags == ["--lr", "1e-5", "--epochs", "8"]


def test_new_validates_flags_against_train_py(tmp_path):
    app = mirror(tmp_path)
    session = dry(app)
    ns = lab.argparse.Namespace(name="bad", title=None, recipe="wb", init="stock", axis=["not-a-flag=1,2"], flags="", seeds=["17"], speakers=None, no_default=False)
    spec = lab.make_spec(ns)
    problems = lab.validate_spec(app, spec)
    text = lab.help_text(app, "scripts/asr/train.py", None, {})
    if text is not None:
        assert problems and "--not-a-flag" in problems[0]
        assert lab.cmd_new(session, ns) == 2 and not (app / lab.EXPERIMENTS_FILE).exists()
    ns.axis, ns.name = ["lr=1e-5,4e-5"], "good"
    assert lab.cmd_new(session, ns) == 0
    saved = lab.custom_experiments(app)
    assert saved[0]["name"] == "good" and saved[0]["axes"] == {"lr": ["1e-5", "4e-5"]}
    again = dry(app)
    assert "c:good" in again.catalog and again.catalog["c:good"].runs == 3 * 1 * 3
    commands, notes = lab.plan(lambda: dry(app), ["c:good"])
    assert not notes["c:good"] and any("big-c-good-wb-lr4e-5-M04-s17" in " ".join(c) for c in commands)


def test_a_sweep_skips_runs_that_already_have_a_decode(tmp_path):
    app = mirror(tmp_path)
    (app / "reports/finetune-torgo").mkdir(parents=True)
    done = "reports/finetune-torgo/big-t-wb-default-M01-s17-test-nemo.jsonl"
    (app / done).write_text("{}\n")
    for s in lab.SEVERE:
        (app / f"data/public/torgo/export/{s}").mkdir(parents=True)
        for name in ("train", "test"):
            (app / f"data/public/torgo/export/{s}/{name}.jsonl").write_text("{}\n")
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo").mkdir(parents=True)
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo/hypotheses.jsonl").write_text("{}\n")
    session = lab.Lab(app, lab.Settings(app, {"seeds": "17"}, env={}), dry=False, quiet=True)
    calls = []
    session.act = lambda: True
    session.run = lambda cmd, env=None: calls.append([str(c) for c in cmd]) or True
    session.stage = "T"
    lab.grid_summary = lambda *a, **k: None
    assert lab.stage_grid("T")(session)
    names = " ".join(" ".join(c) for c in calls if "train.py" in " ".join(c))
    assert "big-t-wb-default-M01-s17" not in names and "big-t-wb-default-M02-s17" in names
    assert any("-- 1 runs" not in line and "already done" in line for line in session.lines)


def test_incomplete_sweep_models_are_moved_aside_and_adapter_decodes_use_their_base(tmp_path):
    app = mirror(tmp_path)
    session = dry(app)
    flags = session.adapter_flags("n", None)
    assert flags == ["--stock", "--adapter", "models/parakeet-ft-n/adapter.pt"]
    assert session.adapter_flags("n", "models/x/parakeet-ft.nemo")[:2] == ["--nemo", "models/x/parakeet-ft.nemo"]
    assert session.adapter_flags("n", "nvidia/parakeet-tdt-1.1b")[:2] == ["--pretrained", "nvidia/parakeet-tdt-1.1b"]


def test_pending_queue_covers_requested_runs_without_B_or_completed_lab_tests(tmp_path):
    session = dry(mirror(tmp_path))
    ids = lab.resolve(session, ["pending"])
    assert ids == ["L", "T", "E", "S", "C", "A", "N", "F", "W",
                   "long-adapter", "acoustic-only", "speaker-replay"]
    assert len(ids) == len(set(ids)) and all(sid in session.catalog for sid in ids)
    assert "B" not in ids and not set(lab.LAB_TESTS) & set(ids)


@pytest.mark.parametrize("sid,study", [("long-adapter", "adapter"), ("acoustic-only", "acoustic"),
                                       ("speaker-replay", "replay")])
def test_long_experiments_keep_stamp_resume_guard_and_resource_flags(tmp_path, sid, study):
    session = dry(mirror(tmp_path), long_flags="--offload-optimizer --batch 4 --grad-accum 4", xflags="--lr 9")
    assert session.catalog[sid].fn(session)
    cmd = session.commands[0]
    assert cmd[1:4] == ["scripts/asr/long_runs.py", study, "TEST"]
    assert "--resume" in cmd and "--dry-run" in cmd and "--offload-optimizer" in cmd
    assert cmd[cmd.index("--cpu-python") + 1] == session.sc
    assert cmd[cmd.index("--train-python") + 1] == str(lab.TRAIN_PY)
    assert "--lr" not in cmd


def test_long_experiment_done_requires_test_and_battery_with_same_stamp(tmp_path):
    session = dry(mirror(tmp_path))
    exp = session.catalog["long-adapter"]
    assert lab.done_state(session, exp) == "new"
    for marker in exp.markers:
        path = session.root / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        assert lab.done_state(session, exp) == ("done" if marker == exp.markers[-1] else "partial")
    other = lab.Lab(session.root, session.cfg, dry=True, stamp="OTHER", quiet=True)
    assert lab.done_state(other, other.catalog["long-adapter"]) == "done"


def test_long_experiment_with_only_a_summary_under_an_old_stamp_is_not_done(tmp_path):
    app = mirror(tmp_path)
    (app / "reports/f2-acoustic-only/OLD").mkdir(parents=True)
    (app / "reports/f2-acoustic-only/OLD/summary.json").write_text("{}")
    session = dry(app)
    assert lab.done_state(session, session.catalog["acoustic-only"]) == "new"


def test_long_flags_cannot_shorten_experiment_to_only_search(tmp_path):
    session = dry(mirror(tmp_path), long_flags="--phase search")
    with pytest.raises(ValueError, match="long_flags accepts only"):
        session.catalog["long-adapter"].fn(session)
    assert not session.commands


def test_pending_queue_continues_after_failure_and_reports_it(tmp_path, monkeypatch):
    session = dry(mirror(tmp_path))
    attempted = []

    def run(_session, stages):
        attempted.extend(stages)
        return stages != ["acoustic-only"]

    monkeypatch.setattr(lab, "run_stages", run)
    assert not lab.run_queue(session, ["pending"])
    assert attempted == lab.PENDING_QUEUE
    assert any("FAILED: acoustic-only" in line for line in session.lines)


def test_the_queue_skips_finished_experiments(tmp_path):
    app = mirror(tmp_path)
    (app / "reports/lab-13-silence-false-inserts").mkdir(parents=True)
    (app / "reports/lab-13-silence-false-inserts/score.json").write_text("{}")
    session = dry(app)
    assert lab.done_state(session, session.catalog["13"]) == "done"
    assert lab.done_state(session, session.catalog["14"]) == "new"
    assert lab.run_queue(session, ["13", "14"])
    assert any("13 already done" in line for line in session.lines)
    assert any("scripts/matcher/route_score.py" in " ".join(c) for c in session.commands)


def test_command_line_vars_set_list_and_run(tmp_path, capsys):
    app = mirror(tmp_path)
    assert lab.main(["--root", str(app), "set", "margin", "3"]) == 0
    assert json.loads((app / lab.SETTINGS_FILE).read_text())["margin"] == 3.0
    assert lab.main(["--root", str(app), "vars"]) == 0
    assert "margin" in capsys.readouterr().out
    assert lab.main(["--root", str(app), "list"]) == 0
    assert lab.main(["--root", str(app), "run", "14", "--dry", "--stamp", "CLI"]) == 0
    assert (app / "logs/lab-CLI-14.txt").is_file()
    assert lab.main(["--root", str(app), "set", "nonsense", "1"]) == 2


def test_the_menu_quits_on_q_and_on_end_of_input(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    assert lab.main(["--root", str(app)]) == 0

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    with pytest.raises(SystemExit):
        lab.main(["--root", str(app)])


def recording(tmp_path):
    session = lab.Lab(tmp_path, lab.Settings(tmp_path, {"rebuild_missing": "1"}, env={}), dry=False, quiet=True)
    calls = []

    def run(cmd, env=None):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        script = next(c for c in cmd if c.startswith("scripts/"))
        out = tmp_path / cmd[cmd.index("--out") + 1]
        if script.endswith("train.py"):
            out.mkdir(parents=True)
            (out / ("adapter.pt" if "--adapter-after-blocks" in cmd else "parakeet-ft.nemo")).write_text("model")
        elif script.endswith("nemo_hyp.py"):
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("hypotheses")
        else:
            out.mkdir(parents=True, exist_ok=True)
            (out / "hypotheses.jsonl").write_text("scored")
            (out / "results.json").write_text("{}")
        return True

    session.run = run
    return session, calls


@pytest.mark.parametrize("speaker,fixed", [("M01", True), ("M02", False), ("M04", True)])
def test_whole_block_baseline_keeps_each_speakers_recipe_and_reuses_finished_outputs(tmp_path, speaker, fixed):
    session, calls = recording(tmp_path)
    assert session.prepare_baseline("whole-block", speaker)
    train = calls[0]
    assert "scripts/asr/train.py" in train and ("--select" in train) == fixed and ("--keep-stock" in train) != fixed
    assert train[train.index("--out") + 1].startswith("models/lab-baselines/")
    assert "scripts/asr/evaluate.py" in calls[-1]
    calls.clear()
    assert session.prepare_baseline("whole-block", speaker) and not calls


def test_adapter_on_f1_baseline_trains_on_the_f1_model(tmp_path):
    session, calls = recording(tmp_path)
    assert session.prepare_baseline("adapter-f1", "M02")
    trains = [c for c in calls if "scripts/asr/train.py" in c]
    assert trains[0][trains[0].index("--export") + 1].endswith("/loso-M02")
    base = "models/lab-baselines/f1-M02/parakeet-ft.nemo"
    assert trains[1][trains[1].index("--base") + 1] == base and "--keep-stock" in trains[1] and "--select" not in trains[1]
    decode = [c for c in calls if "scripts/asr/nemo_hyp.py" in c][-1]
    assert decode[decode.index("--nemo") + 1] == base


def test_an_incomplete_baseline_is_preserved_before_retry(tmp_path):
    session, calls = recording(tmp_path)
    folder = tmp_path / "models/lab-baselines/whole-block-M01"
    folder.mkdir(parents=True)
    assert session.prepare_baseline("whole-block", "M01")
    assert folder.with_name(folder.name + "-first-run").is_dir()
    assert (folder / "parakeet-ft.nemo").is_file() and calls


def corpus(root):
    from willspeak.torgo import DYSARTHRIC

    base = root / "data/public/torgo"
    for speaker in DYSARTHRIC:
        (base / "TORGO" / speaker).mkdir(parents=True)
    return base


def complete_exports(base):
    for speaker in lab.SEVERE:
        for name in (speaker, f"loso-{speaker}", f"loso-severe-{speaker}"):
            folder = base / "export" / name
            folder.mkdir(parents=True)
            for split in ("train", "dev", "test"):
                (folder / f"{split}.jsonl").write_text("original split\n")


def test_fresh_setup_fetches_the_corpus_and_builds_exports_without_research_runs(tmp_path):
    session = dry(tmp_path)
    lab.prepare_exports(session)
    assert [c[1:] for c in session.commands] == [["scripts/corpus/fetch_torgo.py"], ["scripts/corpus/torgo_manifest.py", "--export", "--missing-only",
                                                                                       "--speakers", "M01", "M02", "M04", "--severe-loso"]]


def test_setup_preserves_partial_exports_and_requires_a_transfer(tmp_path):
    base = corpus(tmp_path)
    saved = base / "export/M01/test.jsonl"
    saved.parent.mkdir(parents=True)
    saved.write_text("original split\n")
    session = dry(tmp_path)
    with pytest.raises(RuntimeError, match="will not replace research splits"):
        lab.prepare_exports(session)
    assert saved.read_text() == "original split\n" and not session.commands


def test_setup_adds_missing_pieces_without_replacing_existing_exports(tmp_path):
    base = corpus(tmp_path)
    complete_exports(base)
    session = dry(tmp_path)
    lab.prepare_exports(session)
    assert [c[1:] for c in session.commands] == [["scripts/corpus/torgo_manifest.py"]]
    words = base / "manifests/words.csv"
    words.parent.mkdir()
    words.write_text("path,label,session,speaker\n")
    again = dry(tmp_path)
    lab.prepare_exports(again)
    assert not again.commands
    folder = base / "export/loso-severe-M01"
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    third = dry(tmp_path)
    lab.prepare_exports(third)
    assert len(third.commands) == 1 and "--missing-only" in third.commands[0]
    assert (base / "export/M01/test.jsonl").read_text() == "original split\n"


@pytest.mark.parametrize("stage,speakers", [("A", lab.SEVERE), ("N", lab.SEVERE), ("F", ("M02",))])
def test_missing_pending_baselines_build_all_adapter_seeds(tmp_path, stage, speakers):
    session, calls = recording(tmp_path)
    for speaker in speakers:
        for split in ("train", "dev", "test"):
            path = tmp_path / lab.EX / speaker / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    assert session.prepare_baselines(stage)
    trains = [c for c in calls if "scripts/asr/train.py" in c]
    assert len(trains) == 3 * len(speakers)
    assert {c[c.index("--seed") + 1] for c in trains} == {"17", "18", "19"}
    assert all(session.exists(session.base8(s, seed) + "/results.json") for s in speakers for seed in (17, 18, 19))
    calls.clear()
    assert session.prepare_baselines(stage) and not calls


def test_the_training_idea_stages_exist_and_the_ideas_shortcut_expands(tmp_path):
    session = dry(mirror(tmp_path))
    assert {"P", "U", "R", "K", "G"} <= set(session.catalog)
    assert lab.resolve(session, ["ideas"]) == ["P", "U", "R", "K", "G"]
    for gid in ("P", "G", "R", "K"):
        labels = [a.label for a in lab.GRIDS[gid][1]()]
        assert len(labels) == len(set(labels)), gid


def test_a_pretrained_folder_whose_weights_were_removed_is_moved_aside_and_trained_again(tmp_path, monkeypatch):
    session, calls = recording(tmp_path)
    for speaker in lab.SEVERE:
        for split in ("train", "dev", "test"):
            path = tmp_path / lab.EX / speaker / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        loso = tmp_path / lab.EX / f"loso-severe-{speaker}"
        loso.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"models/parakeet-ft-big-c-pre-{speaker}").mkdir(parents=True)
        (tmp_path / f"models/parakeet-ft-big-c-pre-{speaker}/run.json").write_text("{}")
    session.ensure_severe_exports = lambda: True
    monkeypatch.setattr(lab, "room_for_full_training", lambda lab_, folder: True)
    monkeypatch.setattr(lab, "run_grid", lambda lab_, gid, title, arms, seeds, speakers=None, prepare=None, cleanup=None: bool(prepare()))
    assert lab.run_pretrained_grid(session, "P")
    trains = [c for c in calls if "scripts/asr/train.py" in c]
    assert sorted(c[c.index("--out") + 1] for c in trains) == [f"models/parakeet-ft-big-c-pre-{s}" for s in lab.SEVERE]
    for speaker in lab.SEVERE:
        assert (tmp_path / f"models/parakeet-ft-big-c-pre-{speaker}-first-run/run.json").exists()
        assert (tmp_path / f"models/parakeet-ft-big-c-pre-{speaker}/parakeet-ft.nemo").exists()
    calls.clear()
    assert lab.run_pretrained_grid(session, "P") and not calls


def test_the_soup_stage_averages_five_adapters_per_speaker(tmp_path):
    app = mirror(tmp_path)
    commands, _ = lab.plan(lambda: dry(app), ["U"])
    soups = [c for c in commands if "scripts/asr/adapter_soup.py" in c]
    assert len(soups) == 3
    for c in soups:
        assert c.count("--inputs") == 1 and len([w for w in c if w.endswith("adapter.pt")]) == 6


def test_a_grid_stops_when_the_disk_stays_full_instead_of_failing_every_run(tmp_path):
    app = mirror(tmp_path)
    for s in lab.SEVERE:
        (app / f"data/public/torgo/export/{s}").mkdir(parents=True)
        for name in ("train", "test"):
            (app / f"data/public/torgo/export/{s}/{name}.jsonl").write_text("{}\n")
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo").mkdir(parents=True)
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo/hypotheses.jsonl").write_text("{}\n")
    session = lab.Lab(app, lab.Settings(app, {"seeds": "17"}, env={}), dry=False, quiet=True)
    session.act = lambda: True
    trained = []

    def run(cmd, env=None):
        cmd = [str(c) for c in cmd]
        session.last_output.clear()
        if "scripts/asr/train.py" in cmd:
            trained.append(cmd)
            session.last_output.append("  - 3.9 GB free where /m goes; this run needs about 6 GB (checkpoints + final .nemo).")
            return False
        return True

    session.run = run
    session.stage = "T"
    lab.grid_summary = lambda *a, **k: None
    assert not lab.stage_grid("T")(session)
    assert len(trained) == 2
    assert any("Stopping T" in line for line in session.lines)


def test_nothing_the_lab_imports_needs_python_3_11_datetime_utc():
    import re
    offenders = [str(p.relative_to(ROOT)) for p in list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "src").rglob("*.py"))
                 if re.search(r"from datetime import[^\n]*\bUTC\b", p.read_text(encoding="utf-8"))]
    assert not offenders, offenders


def test_the_followup_queue_expands_and_every_stage_in_it_exists(tmp_path):
    session = dry(mirror(tmp_path))
    ids = lab.resolve(session, ["followup"])
    assert ids == ["24", "P", "G2", "16", "17", "18", "19", "20", "23", "21", "22", "25", "29", "28", "30", "27", "26", "B"]
    assert all(i in session.catalog for i in ids)
    assert all(i in lab.AUTO_REBUILD for i in ids)
    assert "24" in lab.KEEP_BASELINE_WEIGHTS
    assert set(lab.Lab.BASELINE_NEEDS["24"]) == {"stock", "f1", "adapter-f1"}


def test_every_stage_that_loads_baseline_weights_keeps_them():
    import inspect
    loaders = ("lab.wb_m(", "lab.adf1_adapter(")
    for name, fn in vars(lab).items():
        if not name.startswith("stage_") or not callable(fn):
            continue
        st = name.removeprefix("stage_")
        if st in lab.Lab.BASELINE_NEEDS and any(k in inspect.getsource(fn) for k in loaders):
            assert st in lab.KEEP_BASELINE_WEIGHTS, st


def test_the_stacked_grid_adds_whole_block_from_the_pretrained_model(tmp_path):
    arms = {a.label: a for a in lab.GRIDS["P"][1]()}
    assert len(arms) == 8
    for label in ("pre-wb-lr8e-5-ep20", "pre-wb-lr1.6e-4-ep20"):
        assert arms[label].recipe == "wb" and arms[label].init == "pre"
    app = mirror(tmp_path)
    commands, _ = lab.plan(lambda: dry(app), ["P"])
    trains = [c for c in commands if "scripts/asr/train.py" in c]
    assert any("loso-severe-M01" in " ".join(c) for c in trains)
    assert any("--base" in c and "big-c-pre-M01" in " ".join(c) for c in trains)
    wb = [c for c in trains if "--base" in c and any("pre-wb" in w for w in c)]
    assert wb and all("--epochs" in c and "--lr" in c for c in wb)


def test_g2_reuses_the_runs_of_g_and_adds_a_third_seed_and_two_arms_further_out(tmp_path):
    g, g2 = lab.GRIDS["G"], lab.GRIDS["G2"]
    assert [a.label for a in g2[1]()][:4] == [a.label for a in g[1]()]
    assert {a.label for a in g2[1]()} - {a.label for a in g[1]()} == {"wb-lr1.6e-4-ep40", "wb-lr8e-5-ep60"}
    assert g2[2] == [17, 18, 19]
    assert lab.run_tag("G2") == "g" and lab.run_tag("P") == "p"
    app = mirror(tmp_path)
    session = dry(app)
    marks = session.catalog["G2"].markers
    assert marks[-1] == "reports/lab-big-g2/summary.json"
    assert all(m.startswith("reports/finetune-torgo/big-g-") and m.endswith("-test-nemo.jsonl") for m in marks[:-1])
    assert len(marks) - 1 == session.catalog["G2"].runs
    commands, _ = lab.plan(lambda: dry(app), ["G2"])
    outs = [c[c.index("--out") + 1] for c in commands if "scripts/asr/train.py" in c]
    assert outs and all("parakeet-ft-big-g-" in o for o in outs)
    assert any(o.endswith("-s19") for o in outs)


def test_g2_skips_runs_g_already_finished(tmp_path):
    app = mirror(tmp_path)
    for s in lab.SEVERE:
        (app / f"data/public/torgo/export/{s}").mkdir(parents=True)
        for name in ("train", "test"):
            (app / f"data/public/torgo/export/{s}/{name}.jsonl").write_text("{}\n")
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo").mkdir(parents=True)
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo/hypotheses.jsonl").write_text("{}\n")
        for seed in (17, 18):
            (app / f"reports/finetune-torgo/big-g-wb-default-{s}-s{seed}-test-nemo.jsonl").write_text("{}\n")
    session = lab.Lab(app, lab.Settings(app, {"seeds": "17 18"}, env={"WILLSPEAK_WANDB": "off"}), dry=False, quiet=True)
    session.act = lambda: True
    trained = []

    def run(cmd, env=None):
        cmd = [str(c) for c in cmd]
        if "scripts/asr/train.py" in cmd:
            trained.append(cmd[cmd.index("--out") + 1])
        return False

    session.run = run
    lab.run_grid(session, "G2", "t", [lab.Arm("wb-default", "wb")], [17, 18])
    assert trained == []


def test_the_phrase_scoring_stage_scores_three_systems_per_speaker_then_reports(tmp_path):
    app = mirror(tmp_path)
    commands, notes = lab.plan(lambda: dry(app), ["24"])
    scores = [c for c in commands if "scripts/asr/phrase_score.py" in c]
    assert len(scores) == 9
    assert {c[c.index("--out") + 1].split("/")[-1] for c in scores} >= {"M01-stock.json", "M02-F-1.json", "M04-adapter-on-F-1.json"}
    assert all("--pool-glob" in c and "--hyps" in c and c[c.index("--shortlist") + 1] == "15" for c in scores)
    assert any("--adapter" in c for c in scores) and any("--stock" in c for c in scores)
    reports = [c for c in commands if "scripts/asr/phrase_report.py" in c]
    assert len(reports) == 1 and reports[0].count("--system") == 3 and "--export-root" in reports[0]


def test_pass_through_commands_keep_a_leading_option(monkeypatch):
    seen = {}
    monkeypatch.setattr(lab.subprocess, "call", lambda cmd, **kw: seen.setdefault("cmd", cmd) and 0)
    assert lab.main(["doctor", "--fix", "--allow-download"]) == 0
    assert seen["cmd"][-2:] == ["--fix", "--allow-download"]


def test_grid_is_done_only_when_every_run_is_decoded_and_the_summary_exists(tmp_path):
    app = mirror(tmp_path)
    session = dry(app)
    exp = session.catalog["T"]
    assert lab.done_state(session, exp) == "new"
    for marker in exp.markers[:-1]:
        (app / marker).parent.mkdir(parents=True, exist_ok=True)
        (app / marker).write_text("{}\n")
    assert lab.done_state(session, exp) == "partial"
    (app / exp.markers[-1]).parent.mkdir(parents=True, exist_ok=True)
    (app / exp.markers[-1]).write_text("{}")
    assert lab.done_state(session, exp) == "done"


def test_run_grid_skips_its_prerequisites_when_every_run_is_decoded(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    for s in lab.SEVERE:
        (app / f"data/public/torgo/export/{s}").mkdir(parents=True)
        for name in ("train", "test"):
            (app / f"data/public/torgo/export/{s}/{name}.jsonl").write_text("{}\n")
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo").mkdir(parents=True)
        (app / f"reports/finetune-torgo/{s}-test-stock-nemo/hypotheses.jsonl").write_text("{}\n")
    session = lab.Lab(app, lab.Settings(app, {}, env={"WILLSPEAK_WANDB": "off"}), dry=False, quiet=True)
    session.act = lambda: True
    title, arms, seeds, _ = lab.GRIDS["T"]
    for name in lab.grid_run_names(session, "T", arms(), seeds):
        (app / f"reports/finetune-torgo/{name}-test-nemo.jsonl").write_text("{}\n")
    prepared = []
    monkeypatch.setattr(lab, "grid_summary", lambda *a, **k: None)
    monkeypatch.setattr(session, "train_retrying", lambda *a, **k: pytest.fail("nothing should train"))
    assert lab.run_grid(session, "T", title, arms(), seeds, prepare=lambda: prepared.append(1) or True)
    assert not prepared


def test_remaining_lists_unfinished_work_once_and_turns_on_the_hands_off_settings(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    session = dry(app)
    ids = lab.resolve(session, ["remaining"])
    assert ids[:len(lab.REMAINING_QUEUE)] == lab.REMAINING_QUEUE
    assert "B" in ids and "29" in ids and "acoustic-only" in ids
    assert ids.index("28") < ids.index("30") and ids.index("20") < ids.index("23") and ids.index("16") < ids.index("17")
    assert all(st in lab.AUTO_REBUILD for st in lab.REMAINING_QUEUE)
    attempted = []
    monkeypatch.setattr(lab, "run_stage_with_retries", lambda _s, st, _m: attempted.append(st) or True)
    session = dry(app, push_results="0")
    lab.run_queue(session, ["remaining"])
    assert len(attempted) == len(set(attempted))
    assert session.cfg["allow_download"] is True and session.cfg["wait_busy"] is True and session.cfg["rerun"] is True
    assert session.cfg["push_results"] is False


def test_push_results_is_off_by_default_and_never_prompts(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    assert lab.push_results(dry(app)) is False
    calls = []

    class Done:
        returncode, stdout, stderr = 1, "", "fatal: could not read Username"

    monkeypatch.setattr(lab.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or Done())
    session = lab.Lab(app, lab.Settings(app, {"push_results": "1"}, env={}), dry=False, stamp="TEST", quiet=True)
    assert lab.push_results(session) is False
    (cmd, kw), = calls
    assert cmd[:3] == ["git", "push", "origin"] and kw["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_stage_b_is_done_only_after_it_writes_its_own_marker(tmp_path):
    app = mirror(tmp_path)
    session = dry(app)
    for s in lab.SEVERE:
        (app / f"reports/finetune-torgo/{s}-test-stock-110m-nemo").mkdir(parents=True)
    assert lab.done_state(session, session.catalog["B"]) == "new"
    assert session.catalog["B"].markers == [lab.B_DONE]


def test_lab_36_pools_two_systems_per_speaker_plus_t1_then_rescores_three_tables(tmp_path):
    app = mirror(tmp_path)
    commands, notes = lab.plan(lambda: dry(app), ["36"])
    assert not notes["36"], notes["36"]
    pools = [c for c in commands if "scripts/asr/candidate_pool.py" in c]
    assert {c[c.index("--out") + 1].split("/")[-1] for c in pools} == {
        "lab-stock-M01.jsonl", "lab-stock-M02.jsonl", "lab-stock-M04.jsonl",
        "lab-prototype-M01.jsonl", "lab-prototype-M02.jsonl", "lab-prototype-M04.jsonl", "t1-stock.jsonl"}
    assert len(pools) == 7 and all(c[c.index("--beam") + 1] == lab.LM36_BEAM for c in pools)
    assert sum("--adapter" in c for c in pools) == 3 and sum("--stock" in c for c in pools) == 4
    t1 = next(c for c in pools if c[-1].endswith("t1-stock.jsonl"))
    assert t1[t1.index("--manifest") + 1] == lab.T1_MANIFEST and t1[t1.index("--audio-root") + 1] == lab.T1_AUDIO
    rescores = [c for c in commands if "scripts/asr/lm_rescore.py" in c]
    assert {c[c.index("--label") + 1] for c in rescores} == {"stock", "stage P adapter", "stock, T-1 severe sentences"}
    stock = next(c for c in rescores if c[c.index("--label") + 1] == "stock")
    assert stock.count("--pool") == 3 and stock.count("--manifest") == 3 and stock.count("--check-greedy") == 3
    assert stock[stock.index("--cache") + 1] == f"{lab.LM36}/lm-cache" and stock[stock.index("--models") + 1] == "all"
    t1_run = next(c for c in rescores if "T-1" in c[c.index("--label") + 1])
    assert t1_run.count("--pool") == 1 and "=" not in t1_run[t1_run.index("--pool") + 1] and "--check-greedy" not in t1_run
    assert commands.index(pools[-1]) < commands.index(rescores[0])


def test_lab_36_model_list_setting_reaches_the_rescorer(tmp_path):
    app = mirror(tmp_path)
    commands, _ = lab.plan(lambda: dry(app, lm_models="qwen2.5-0.5b qwen3-0.6b"), ["36"])
    rescores = [c for c in commands if "scripts/asr/lm_rescore.py" in c]
    assert rescores and all(c[c.index("--models") + 1] == "qwen2.5-0.5b,qwen3-0.6b" for c in rescores)
