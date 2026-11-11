import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lab_hardening", ROOT / "lab.py")
lab = importlib.util.module_from_spec(spec)
sys.modules["lab_hardening"] = lab
spec.loader.exec_module(lab)


def mirror(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    for name in ("scripts", "src"):
        (app / name).symlink_to(ROOT / name)
    return app


def session(app: Path, dry: bool = True, **overrides):
    return lab.Lab(app, lab.Settings(app, overrides, env={}), dry=dry, stamp="TEST", quiet=True)


def commands_of(app: Path, stage: str, **overrides) -> list[list[str]]:
    s = session(app, **overrides)
    s.stage = stage
    s.catalog[stage].fn(s)
    return s.commands


@pytest.mark.parametrize("stage", ["16", "17", "18", "19", "20", "21", "22", "23", "25", "26", "27", "28", "29", "30"])
def test_new_lab_tests_are_listed_and_plan_without_problems(tmp_path, stage):
    app = mirror(tmp_path)
    s = session(app)
    assert stage in s.catalog and stage in lab.NEW_QUEUE and stage in lab.AUTO_REBUILD
    commands, notes = lab.plan(lambda: session(app), [stage])
    assert commands and not notes[stage]
    problems, checked = lab.flag_problems(app, commands)
    assert not problems and checked >= (1 if stage == "29" else 2)


def test_lab_16_builds_the_validation_set_and_summarises_against_the_real_speakers(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "16")]
    assert any("synth_dysarthria.py validate-set" in c for c in flat)
    assert any("synth_dysarthria.py summarize" in c and "M01=" in c and "M04=" in c for c in flat)
    assert any("nemo_hyp.py --stock" in c for c in flat)


def test_lab_17_has_three_arms_per_speaker_and_a_synthetic_only_pretraining(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "17")]
    assert sum("synth_dysarthria.py build" in c for c in flat) == 9
    assert sum("--only-synthetic" in c for c in flat) == 3
    assert any("lab-17-mix-M01" in c for c in flat) and any("lab-17-then-M02" in c for c in flat) and any("lab-17-sevthen-M04" in c for c in flat)


def test_lab_18_runs_silence_protection_and_the_battery(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "18")]
    assert any("--empty-margin" in c and "lab-18-sil-M01" in c for c in flat)
    assert any("lab-18-synthsil-M04" in c for c in flat)
    assert any("battery.py score" in c and "reports/lab-18-best-system-hardened/battery.json" in c for c in flat)


def test_lab_19_exports_three_int8_recipes_for_two_models(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "19")]
    exports = [c for c in flat if "scripts/asr/export.py" in c]
    assert len(exports) == 18
    assert sum("--int8-ops matmul" in c for c in exports) == 6 and sum("--per-channel" in c for c in exports) == 6
    assert all("--model-dir" in c for c in flat if "evaluate.py" in c and "lab-19-int8-cost" in c)


def test_synthetic_severity_comes_from_the_variable_then_the_calibration_then_a_default(tmp_path):
    app = mirror(tmp_path)
    s = session(app)
    assert lab.synth_severity(s, "M01") == (0.7, "default: lab test 16 has not run")
    folder = app / lab.SYN16
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"calibration": {"M01": {"severity": 0.9}, "pooled": {"severity": 0.5}}}))
    assert lab.synth_severity(s, "M01")[0] == 0.9
    assert lab.synth_severity(s, "M02")[0] == 0.5
    assert lab.synth_severity(session(app, synth_severity="0.4"), "M01")[0] == 0.4
    (folder / "summary.json").write_text("{broken")
    assert lab.synth_severity(s, "M01")[0] == 0.7


def test_a_stage_that_ran_out_of_memory_is_run_again_and_an_unknown_failure_is_not(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    s = session(app, stage_retries="1")
    s.logdir.mkdir(parents=True)
    log = s.logdir / "lab-TEST-9.txt"
    calls = []

    def flaky(_lab, stages):
        calls.append(stages)
        if len(calls) == 1:
            log.write_text("step 3\ntorch.OutOfMemoryError: CUDA out of memory\n")
            _lab.failed_stages.append("9")
            return False
        return True

    monkeypatch.setattr(lab, "run_stages", flaky)
    monkeypatch.setattr(lab.time, "sleep", lambda _x: None)
    assert lab.run_stage_with_retries(s, "9", 14) and len(calls) == 2 and "9" not in s.failed_stages
    assert any("fixing and retrying" in line for line in s.lines)

    calls.clear()
    monkeypatch.setattr(lab, "run_stages", lambda _lab, stages: (calls.append(stages), log.write_text("some other error\n"), False)[2])
    assert not lab.run_stage_with_retries(s, "9", 14) and len(calls) == 1 and "9" in s.failed_stages


def test_retries_stop_at_the_limit(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    s = session(app, stage_retries="2")
    s.logdir.mkdir(parents=True)
    (s.logdir / "lab-TEST-9.txt").write_text("CUDA out of memory\n")
    calls = []
    monkeypatch.setattr(lab, "run_stages", lambda _lab, stages: (calls.append(stages), False)[1])
    monkeypatch.setattr(lab.time, "sleep", lambda _x: None)
    assert not lab.run_stage_with_retries(s, "9", 14) and len(calls) == 3


def test_the_status_file_is_written_atomically_and_not_in_a_dry_run(tmp_path):
    app = mirror(tmp_path)
    real = session(app, dry=False)
    real.logdir.mkdir(parents=True)
    lab.write_status(real, {"done": ["1"], "failed": []})
    data = json.loads((real.logdir / "lab-queue-TEST.status.json").read_text())
    assert data["done"] == ["1"] and "updated" in data and "free_gb" in data
    pretend = session(app)
    (real.logdir / "lab-queue-TEST.status.json").unlink()
    lab.write_status(pretend, {"done": []})
    assert not (real.logdir / "lab-queue-TEST.status.json").exists()


def test_the_doctor_gate_never_blocks_a_dry_run_and_survives_a_broken_doctor(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    assert lab.doctor_gate(session(app), "9", 14) == []
    real = session(app, dry=False)
    monkeypatch.setattr(lab.lab_doctor, "run_checks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert lab.doctor_gate(real, "9", 14) == []
    assert any("DOCTOR: skipped" in line for line in real.lines)


def test_the_doctor_gate_repairs_then_reports_what_is_left(tmp_path, monkeypatch):
    app = mirror(tmp_path)
    real = session(app, dry=False)
    Finding = lab.lab_doctor.Finding
    state = {"n": 0}

    def checks(*_a, **_k):
        state["n"] += 1
        if state["n"] == 1:
            return [Finding("fail", "data", "controls export missing", fix=("exports", "controls")), Finding("fail", "pins", "pin changed")]
        return [Finding("fail", "data", "controls export missing")]

    monkeypatch.setattr(lab.lab_doctor, "run_checks", checks)
    monkeypatch.setattr(lab.lab_doctor, "apply_fixes", lambda *a, **k: ["tried"])
    assert lab.doctor_gate(real, "16", 14) == ["[data] controls export missing"]
    assert state["n"] == 2


def test_models_are_sent_away_only_when_a_place_is_set(tmp_path):
    app = mirror(tmp_path)
    assert not lab.export_models(session(app, dry=False), 40)
    s = session(app, export_to=str(tmp_path / "drive"))
    assert lab.export_models(s, 40)
    cmd = " ".join(s.commands[-1])
    assert "model_export.py export --to" in cmd and "--free-after-verify" in cmd and "--until-free-gb 40" in cmd


def test_export_models_command_asks_for_a_place_instead_of_guessing(tmp_path, capsys):
    app = mirror(tmp_path)
    assert lab.main(["--root", str(app), "export-models", "export"]) == 2
    assert "export_to" in capsys.readouterr().out
    assert lab.main(["--root", str(app), "export-models", "list"]) == 0


def test_the_catalog_markers_make_a_finished_new_test_count_as_done(tmp_path):
    app = mirror(tmp_path)
    s = session(app)
    assert lab.done_state(s, s.catalog["19"]) == "new"
    (app / "reports/lab-19-int8-cost").mkdir(parents=True)
    (app / "reports/lab-19-int8-cost/table.json").write_text("{}")
    assert lab.done_state(s, s.catalog["19"]) == "partial"
    for model in ("whole-block", "adapter-on-F-1"):
        for spk in lab.SEVERE:
            for variant in lab.INT8_VARIANTS:
                (app / f"reports/lab-19-int8-cost/{model}-{spk}-{variant}").mkdir(parents=True)
                (app / f"reports/lab-19-int8-cost/{model}-{spk}-{variant}/results.json").write_text("{}")
    assert lab.done_state(s, s.catalog["19"]) == "done"


def test_lab_20_trains_every_recipe_on_every_fold_then_selects_and_trains_the_choice(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "20")]
    assert sum("cv_folds.py make" in c for c in flat) == 3 and sum("cv_folds.py check" in c for c in flat) == 3
    trains = [c for c in flat if "scripts/asr/train.py" in c and "lab-20-" in c and "-final-" not in c]
    assert len(trains) == 4 * 4 * 3
    assert any("--adapter-after-blocks 4" in c and "lab-20-adf1b4-M01-f0" in c for c in trains)
    assert any("--epochs 30" in c and "lab-20-adf1e30-M04-f3" in c for c in trains)
    assert sum("cv_folds.py merge" in c for c in flat) == 4 * 3
    select = [c for c in flat if "cv_folds.py select" in c]
    assert len(select) == 1 and "--control adf1" in select[0] and select[0].count("--arm ") == 4 and "{speaker}" in select[0]
    assert any("lab-20-final-ad-M01" in c for c in flat)


def test_lab_20_refuses_a_malformed_recipe_list(tmp_path):
    app = mirror(tmp_path)
    with pytest.raises(ValueError, match="cv_arms entry"):
        commands_of(app, "20", cv_arms="adf1:8:15:f1 bad")
    with pytest.raises(ValueError, match="at least one other"):
        commands_of(app, "20", cv_arms="adf1:8:15:f1")
    with pytest.raises(ValueError, match="differ"):
        commands_of(app, "20", cv_arms="a:8:15:f1 a:8:30:f1")


def test_lab_21_speaks_other_speakers_prompts_and_never_his_dev_or_test_prompts(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "21", tts_voices="/voices", tts_extra_text="phrases.txt")]
    makes = [c for c in flat if "tts_synth.py make" in c]
    assert len(makes) == 3 and all("--voice-dir /voices" in c and "--extra-text phrases.txt" in c for c in makes)
    m = next(c for c in makes if "lab-21-tts-M02" in c)
    assert "loso-M02/train.jsonl" in m and "--exclude-from" in m and "M02/dev.jsonl" in m and "M02/test.jsonl" in m
    builds = [c for c in flat if "synth_dysarthria.py build" in c]
    assert len(builds) == 6 and sum("--severity 0.2" in c for c in builds) == 3 and all("--controls data/public/torgo/export/lab-21-tts-" in c for c in builds)
    assert any("battery.py score" in c and "lab-21-tts-synthetic/battery.json" in c for c in flat)


def test_lab_22_adapts_chunk_by_chunk_and_the_decode_of_a_chunk_uses_the_model_trained_before_it(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "22")]
    assert sum("tta.py manifest" in c for c in flat) == 9 and sum("tta.py build" in c for c in flat) == 9
    last = next(c for c in flat if "tta.py build" in c and "--round 3" in c and "lab-22-M04-r3" in c)
    assert "0=" in last and "1=" in last and "2=" in last and "3=" not in last.replace("--round 3", "")
    assert "--agree" in last and "--max-pseudo-share 1" in last
    assert sum("tta.py merge" in c for c in flat) == 3 and sum("tta.py score" in c for c in flat) == 3
    decodes = [c for c in flat if "nemo_hyp.py" in c and "chunk3-manifest" in c]
    assert len(decodes) == 3 and all("lab-22-" in c and "-r3" in c for c in decodes)


def test_lab_23_decodes_the_labelled_clips_with_the_models_that_never_saw_them_then_selects(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "23")]
    decodes = [c for c in flat if "nemo_hyp.py" in c and "cv-M01-all" in c]
    assert len(decodes) == 2 and any("--stock" in c for c in decodes) and any("--nemo" in c for c in decodes)
    selector = [c for c in flat if "hyp_selector.py" in c]
    assert len(selector) == 1 and "--systems-file reports/lab-23-hypothesis-selector/systems.json" in selector[0] and "loso-{speaker}/train.jsonl" in selector[0]
    assert any("pool_table.py" in c and "selector-learned" in c for c in flat)


def test_the_new_lab_tests_have_baselines_markers_and_queue_places(tmp_path):
    s = session(mirror(tmp_path))
    assert lab.NEW_QUEUE.index("20") < lab.NEW_QUEUE.index("23")
    for st in ("20", "21", "22", "23"):
        assert st in lab.Lab.BASELINE_NEEDS and s.catalog[st].markers
    assert "21" in lab.KEEP_BASELINE_WEIGHTS
    assert lab.done_state(s, s.catalog["20"]) == "new"


def test_stage_w_uses_whichever_models_have_answers_and_needs_only_the_stock_model(tmp_path):
    app = mirror(tmp_path)
    s = session(app, dry=False)
    s.dry, s.strict = False, True
    s.act = lambda: True
    s.lines.clear()
    ft = app / lab.FT
    for sp in lab.SEVERE:
        d = ft / "lab-baselines" / f"stock-{sp}"
        d.mkdir(parents=True)
        (d / "hypotheses.jsonl").write_text("")
    for sp in ("M01", "M02"):
        d = ft / "lab-baselines" / f"f1-{sp}"
        d.mkdir(parents=True)
        (d / "hypotheses.jsonl").write_text("")
    calls = []
    s.run = lambda cmd, env=None: (calls.append([str(c) for c in cmd]), True)[1]
    assert lab.stage_W(s)
    text = "\n".join(s.lines)
    assert "models matched: stock" in text and "leaving out F-1: no answers for M04" in text and "leaving out whole-block" in text
    assert calls and all("--model stock=" in " ".join(c) and "F-1=" not in " ".join(c) for c in calls)
    shutil_rm = ft / "lab-baselines" / "stock-M04"
    (shutil_rm / "hypotheses.jsonl").unlink()
    shutil_rm.rmdir()
    s2 = session(app, dry=False)
    s2.dry, s2.strict = False, True
    s2.run = lambda cmd, env=None: True
    assert not lab.stage_W(s2) and any("MISSING the stock model" in line for line in s2.lines)


def test_lab_25_runs_the_self_training_loop_on_the_pretrained_adapter_at_three_seeds(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "25")]
    trains = [c for c in flat if "scripts/asr/train.py" in c]
    assert any("loso-severe-M01" in c and "big-c-pre-M01" in c for c in trains)
    rounds = [c for c in trains if "lab-25-" in c]
    assert len(rounds) == 27
    assert all("--base models/parakeet-ft-big-c-pre-" in c and "--lr 6e-4" in c and "--adapter-after-blocks 8" in c for c in rounds)
    assert {f"--seed {n}" for n in (17, 18, 19)} <= {w for c in rounds for w in [c[c.index("--seed"):c.index("--seed") + 9]]}
    builds = [c for c in flat if "tta.py build" in c]
    assert len(builds) == 27 and all("--agree reports/finetune-torgo/f1-loso-M0" in c and "big-p-pre-ad8-lr6e-4-M0" in c for c in builds)
    assert sum("tta.py merge" in c for c in flat) == 9 and sum("tta.py score" in c for c in flat) == 9
    table = [c for c in flat if "pool_table.py" in c]
    assert len(table) == 1 and "pretrained adapter+TTA s19" in table[0] and "lab-25-M04-s19-tta-test-nemo" in table[0]
    assert any("models/parakeet-ft-big-p-pre-ad8-lr6e-4-M02-s18" in c for c in trains)


def test_lab_26_trains_four_pretraining_variants_then_the_same_adapter_at_three_seeds(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "26")]
    trains = [c for c in flat if "scripts/asr/train.py" in c]
    phase1 = [c for c in trains if "--out models/parakeet-ft-lab-26-pre-" in c]
    assert len(phase1) == 12 and all("loso-severe-M0" in c and "--keep-stock" in c for c in phase1)
    assert sum("--train-parts all" in c for c in phase1) == 3 and sum("--lr 4e-5" in c for c in phase1) == 3
    assert sum("--epochs 20 --patience 6" in c for c in phase1) == 3
    assert sum("--base reports" not in c and "--base models/parakeet-ft-f1-loso-M0" in c for c in phase1) == 3
    adapters = [c for c in trains if "--out models/parakeet-ft-lab-26-" in c and c not in phase1]
    assert len(adapters) == 36 and all("--lr 6e-4" in c and "--base models/parakeet-ft-lab-26-pre-" in c for c in adapters)
    assert sum("models/parakeet-ft-big-p-pre-ad8-lr6e-4-" in c for c in trains) == 9


def test_lab_27_varies_one_tta_setting_at_a_time(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "27")]
    builds = [c for c in flat if "tta.py build" in c]
    assert len(builds) == 3 * (7 + 5 * 3)
    assert sum("--n 8 " in c for c in builds) == 21 and sum("--order id" in c for c in builds) == 9
    assert sum("--agree-min 0 " in c for c in builds) == 9 and sum("--agree-min 0.7" in c for c in builds) == 9
    assert sum("--max-pseudo-share 0.5" in c for c in builds) == 9
    manifests = {c.split("--out ")[1] for c in flat if "tta.py manifest" in c}
    assert len(manifests) == 3 * (7 + 3 + 3)
    rounds = [c for c in flat if "train.py" in c and "lab-27-" in c]
    assert len(rounds) == len(builds) and all("--base models/parakeet-ft-f1-loso-" in c and "--lr 6e-4" not in c for c in rounds)


def test_lab_28_exports_the_fresh_pretrained_adapter_two_ways(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "28")]
    fresh = [c for c in flat if "train.py" in c and "lab-pre-ad8-" in c]
    assert len(fresh) == 3 and all("--base models/parakeet-ft-big-c-pre-" in c and "--lr 6e-4" in c and "--seed 17" in c for c in fresh)
    exports = [c for c in flat if "scripts/asr/export.py" in c]
    assert len(exports) == 6 and all("--nemo models/parakeet-ft-big-c-pre-" in c and "--adapter models/parakeet-ft-lab-pre-ad8-" in c for c in exports)
    assert sum("--int8-ops matmul" in c for c in exports) == 3 and not any("--per-channel" in c for c in exports)
    assert sum("evaluate.py" in c and "--model-dir models/lab-28-export-" in c for c in flat) == 6


def test_lab_29_is_cpu_only_and_passes_every_saved_system(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "29")]
    assert len(flat) == 1 and "scripts/asr/hard_core.py" in flat[0] and "train.py" not in flat[0]
    assert "--system stock=M01:" in flat[0] and "pretrained adapter+TTA s17=M01:reports/finetune-torgo/lab-25-M01-s17-tta-test-nemo" in flat[0]


def test_lab_30_scores_phrases_with_the_pretrained_adapter_and_reports_beside_lab_24(tmp_path):
    flat = [" ".join(c) for c in commands_of(mirror(tmp_path), "30")]
    scores = [c for c in flat if "phrase_score.py" in c]
    assert len(scores) == 3 and all("--nemo models/parakeet-ft-big-c-pre-" in c and "--adapter models/parakeet-ft-lab-pre-ad8-" in c
                                    and "--hyps reports/finetune-torgo/lab-pre-ad8-" in c and "--shortlist 15" in c for c in scores)
    report = [c for c in flat if "phrase_report.py" in c]
    assert len(report) == 1 and "pretrained-adapter=M01:reports/lab-30-phrase-scoring-pretrained/M01-pretrained-adapter.json" in report[0]


def test_lab_19_rebuilds_an_old_table_when_recipe_scores_are_missing(tmp_path):
    app = mirror(tmp_path)
    (app / "reports/lab-19-int8-cost").mkdir(parents=True)
    (app / "reports/lab-19-int8-cost/table.json").write_text("{}")
    s = session(app)
    s.stage = "19"
    s.catalog["19"].fn(s)
    assert any("scripts/asr/export.py" in " ".join(c) and "--per-channel" in c for c in s.commands)
    assert not any("ALREADY EXISTS" in line for line in getattr(s, "lines", []))


def trainings(flat: list[str]) -> list[str]:
    return [c for c in flat if "scripts/asr/train.py" in c]


@pytest.mark.parametrize("stage", ["31", "32", "33"])
def test_acoustic_lab_tests_are_listed_and_plan_without_problems(tmp_path, stage):
    app = mirror(tmp_path)
    s = session(app)
    assert stage in s.catalog and stage in lab.ACOUSTIC_QUEUE and stage in lab.AUTO_REBUILD
    assert s.catalog[stage].markers == [f"reports/{s.catalog[stage].notes.split('/')[1]}/finished.json"]
    commands, notes = lab.plan(lambda: session(app), [stage])
    assert commands and not notes[stage]
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems
    assert lab.resolve(s, ["acoustic"]) == ["31", "32", "33"]


def test_lab_31_trains_each_speakers_acoustic_model_then_three_adapters_on_it(tmp_path):
    flat = trainings([" ".join(c) for c in commands_of(mirror(tmp_path), "31")])
    bases = [c for c in flat if "--train-blocks 24" in c]
    assert len(bases) == 3
    for s, base in zip(lab.SEVERE, bases):
        assert f"--export data/public/torgo/export/{s} " in base and "--freeze-transducer-head" in base
        assert "--lr 1e-5" in base and "--epochs 100" in base and "--target-form stock" in base and "--select last" in base
    adapters = [c for c in flat if "--adapter-after-blocks" in c]
    assert len(adapters) == 9
    assert all("--lr 6e-4" in c and "--adapter-after-blocks 8" in c and "--epochs 15" in c for c in adapters)
    assert sum("--export data/public/torgo/export/M02 " in c and "--base models/parakeet-ft-lab-31-base-M02/parakeet-ft.nemo" in c
               for c in adapters) == 3


def test_lab_32_pretrains_on_the_other_severe_speakers_with_a_settable_length(tmp_path):
    app = mirror(tmp_path)
    flat = trainings([" ".join(c) for c in commands_of(app, "32")])
    bases = [c for c in flat if "--train-blocks 24" in c]
    assert [c.split("--export ")[1].split()[0] for c in bases] == [f"data/public/torgo/export/loso-severe-{s}" for s in lab.SEVERE]
    assert all("--epochs 30" in c for c in bases)
    assert all("--base models/parakeet-ft-lab-32-pre-" in c for c in flat if "--adapter-after-blocks 8" in c)
    longer = trainings([" ".join(c) for c in commands_of(app, "32", acoustic_severe_epochs="60")])
    assert all("--epochs 60" in c for c in longer if "--train-blocks 24" in c)


def test_lab_33_builds_one_pooled_model_and_runs_every_position_seed_by_seed(tmp_path):
    app = mirror(tmp_path)
    flat = [" ".join(c) for c in commands_of(app, "33")]
    assert any("torgo_manifest.py --export --missing-only --speakers M01 M02 M04 --pooled --only pooled-M01-M02-M04" in c for c in flat)
    runs = trainings(flat)
    bases = [c for c in runs if "--train-blocks 24" in c]
    assert len(bases) == 1 and "--export data/public/torgo/export/pooled-M01-M02-M04 " in bases[0] and "--epochs 30" in bases[0]
    adapters = [c for c in runs if "--adapter-after-blocks" in c]
    assert len(adapters) == 45 and all("--base models/parakeet-ft-lab-33-pooled/parakeet-ft.nemo" in c for c in adapters)
    assert all("--seed 17" in c for c in adapters[:15]) and all("--seed 19" in c for c in adapters[30:])
    assert {c.split("--adapter-after-blocks ")[1].split()[0] for c in adapters[:15]} == {"4", "8", "12", "16", "24"}
    assert any("battery.py score" in c and "pooled-acoustic=" in c for c in flat)
    two = trainings([" ".join(c) for c in commands_of(app, "33", pooled_positions="8 16")])
    assert sum("--adapter-after-blocks" in c for c in two) == 18
    with pytest.raises(ValueError):
        lab.pooled_positions(session(app, pooled_positions="8 25"))


def test_the_pooled_export_must_be_built_for_the_same_speakers(tmp_path):
    app = mirror(tmp_path)
    export = app / "data/public/torgo/export/pooled-M01-M02-M04"
    for split in ("train", "dev"):
        (export / split).mkdir(parents=True)
        (export / f"{split}.jsonl").write_text(json.dumps({"audio": f"{split}/a.wav"}) + "\n")
    (export / "pooled.json").write_text(json.dumps({"targets": ["M01", "M02"]}))
    s = session(app, dry=False)
    assert not lab.ensure_pooled_export(s)
    assert any("MISSING a pooled export for M01 M02 M04" in line for line in s.lines)
    (export / "pooled.json").write_text(json.dumps({"targets": ["M01", "M02", "M04"], "train_clips": 1, "train_minutes": 0.1, "dev_clips": 1,
                                                    "bytes": 10, "test_prompt_overlap": 0, "speakers": {"M01": {"train_clips": 1, "train_minutes": 0.1, "dev_clips": 1}}}))
    assert lab.ensure_pooled_export(s)


def test_a_base_model_is_kept_until_every_result_that_needs_it_is_saved(tmp_path):
    app = mirror(tmp_path)
    model = app / "models/parakeet-ft-lab-31-base-M01/parakeet-ft.nemo"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")
    s = session(app, dry=False, trim_models="on")
    lab.release_base(s, "lab-31-base-M01", ["lab-31-ad8-M01-s17"])
    assert model.is_file() and any("keeping models/parakeet-ft-lab-31-base-M01" in line for line in s.lines)
    scores = app / "reports/finetune-torgo/lab-31-ad8-M01-s17-test-nemo"
    scores.mkdir(parents=True)
    (scores / "results.json").write_text("{}")
    lab.release_base(s, "lab-31-base-M01", ["lab-31-ad8-M01-s17"])
    assert not model.exists()


def test_storage_snapshots_are_logged_and_appended_as_json(tmp_path):
    app = mirror(tmp_path)
    folder = app / "models/parakeet-ft-big-x"
    (folder / "ckpt").mkdir(parents=True)
    (folder / "ckpt/a.ckpt").write_bytes(b"0" * 4096)
    (folder / "parakeet-ft.nemo").write_bytes(b"1" * 2048)
    s = session(app, dry=False)
    s.stage = "33"
    record = lab.storage_snapshot(s, "probe")
    assert record["label"] == "probe" and record["stage"] == "33" and record["free_gb"] > 0 and record["models_with_ckpt"] == 1
    assert record["biggest_models"][0]["path"] == "models/parakeet-ft-big-x" and "ckpt" in record["biggest_models"][0]
    lab.storage_snapshot(s, "again")
    lines = (app / "logs/lab-TEST-storage.jsonl").read_text().splitlines()
    assert [json.loads(line)["label"] for line in lines] == ["probe", "again"]
    assert any(line.startswith("STORAGE probe:") for line in s.lines)
    assert lab.storage_snapshot(session(app), "dry") is None


def test_run_records_keep_every_attempt_and_update_runs_in_place(tmp_path):
    app = mirror(tmp_path)
    s = session(app, dry=False)
    lab.note_plan(s, lab.ACOUSTIC31, {"test": "31"})
    lab.note_run(s, lab.ACOUSTIC31, "lab-31-base-M01", kind="base", status="trained")
    lab.note_run(s, lab.ACOUSTIC31, "lab-31-base-M01", wrong=84)
    lab.note_plan(s, lab.ACOUSTIC31, {"test": "31"})
    data = json.loads((app / lab.ACOUSTIC31 / "runs.json").read_text())
    assert len(data["attempts"]) == 2 and set(data["attempts"][0]["sha256"]) == set(lab.ACOUSTIC_CODE)
    assert data["runs"]["lab-31-base-M01"]["status"] == "trained" and data["runs"]["lab-31-base-M01"]["wrong"] == 84


def test_lab_34_is_listed_and_plans_without_problems(tmp_path):
    app = mirror(tmp_path)
    s = session(app)
    assert "34" in s.catalog and "34" in lab.AUTO_REBUILD
    assert s.catalog["34"].markers == ["reports/lab-34-pretrained-adapter-positions/finished.json"]
    commands, notes = lab.plan(lambda: session(app), ["34"])
    assert commands and not notes["34"]
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems
    assert lab.resolve(s, ["positions"]) == ["34", "speaker-replay"]


def test_lab_34_puts_every_position_on_stage_p_pretrained_model_seed_by_seed(tmp_path):
    app = mirror(tmp_path)
    flat = [" ".join(c) for c in commands_of(app, "34")]
    adapters = [c for c in trainings(flat) if "--adapter-after-blocks" in c]
    assert len(adapters) == 36
    for s in lab.SEVERE:
        assert all(f"--base models/parakeet-ft-big-c-pre-{s}/parakeet-ft.nemo" in c for c in adapters if f"export/{s} " in c)
    assert all("--lr 6e-4" in c and "--epochs 15" in c for c in adapters)
    assert all("--seed 17" in c for c in adapters[:12]) and all("--seed 19" in c for c in adapters[24:])
    assert {c.split("--adapter-after-blocks ")[1].split()[0] for c in adapters[:12]} == {"8", "12", "16", "24"}
    assert not any("--train-blocks 24" in c for c in trainings(flat))
    assert any("nemo_hyp.py --nemo models/parakeet-ft-big-c-pre-M01/parakeet-ft.nemo" in c for c in flat)
    assert any("battery.py score" in c and "pretrained+adapter12-M01=" in c and "pretrained-M04=" in c for c in flat)
    two = [c for c in trainings([" ".join(c) for c in commands_of(app, "34", pretrained_positions="8 20")]) if "--adapter-after-blocks" in c]
    assert len(two) == 18
    with pytest.raises(ValueError):
        lab.block_positions(session(app, pretrained_positions="0 8"), "pretrained_positions")


def test_lab_35_is_listed_and_plans_without_problems(tmp_path):
    app = mirror(tmp_path)
    s = session(app)
    assert "35" in s.catalog and "35" in lab.AUTO_REBUILD and "35" in lab.KEEP_BASELINE_WEIGHTS
    assert s.catalog["35"].markers == ["reports/lab-35-silence-preference/table.json"]
    commands, notes = lab.plan(lambda: session(app), ["35"])
    assert commands and not notes["35"]
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems


def test_lab_35_scores_three_systems_and_the_controls_in_both_forms(tmp_path):
    app = mirror(tmp_path)
    flat = [" ".join(c) for c in commands_of(app, "35")]
    scores = [c for c in flat if "reference_score.py" in c]
    assert len(scores) == (3 * len(lab.SEVERE) + 1) * 2
    for form in ("stock", "normalized"):
        assert sum(f"--reference {form} " in c for c in scores) == len(scores) // 2
    for s in lab.SEVERE:
        assert any(f"--adapter models/parakeet-ft-lab-pre-ad8-{s}/adapter.pt" in c and f"--nemo models/parakeet-ft-big-c-pre-{s}/parakeet-ft.nemo" in c
                   for c in scores)
        assert any(f"{s}-whole-block-stock.json" in c and "--hyps" in c for c in scores)
    controls = [c for c in scores if "export/controls/test.jsonl" in c]
    assert len(controls) == 2 and all("--stock" in c and "--hyps" not in c for c in controls)


def test_lab_35_table_reads_each_scored_file(tmp_path):
    app = mirror(tmp_path)
    s = session(app, dry=False)
    out = app / lab.SILENCE35
    out.mkdir(parents=True, exist_ok=True)
    (out / "M02-pretrained-adapter-stock.json").write_text(json.dumps(
        {"summary": {"clips": 37, "median_margin": 1.5, "share_margin_below": {"0.0": 0.25}, "hypotheses": {"empty": 0}}}))
    (out / "controls-stock-normalized.json").write_text(json.dumps(
        {"summary": {"clips": 100, "median_margin": 9.0, "share_margin_below": {"0.0": 0.04}}}))
    rows = {(r["speaker"], r["system"], r["form"]): r for r in lab.silence_rows(s, lab.SILENCE35)}
    assert rows[("M02", "pretrained-adapter", "stock")]["share_preferring_silence"] == 0.25
    assert rows[("M02", "pretrained-adapter", "stock")]["empty_answers"] == 0
    assert rows[("controls", "stock", "normalized")]["empty_answers"] is None
