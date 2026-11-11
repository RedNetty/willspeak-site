from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("willspeak_train_script", ROOT / "scripts/asr/train.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class Loss:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def __float__(self):
        return self.value


def fake_callback(tmp_path, monkeypatch, wer=0.4):
    clock = [0.0]
    monkeypatch.setattr(train.time, "monotonic", lambda: clock[0])
    run = {"configuration": {"select": "dev", "log_every_seconds": 30}, "stock_dev_wer": 0.6}
    trainer = SimpleNamespace(current_epoch=0, max_epochs=2, num_training_batches=4, global_step=0,
                              accumulate_grad_batches=2, callback_metrics={"val_wer": wer},
                              optimizers=[SimpleNamespace(param_groups=[{"lr": 0.001}])])
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    callback = train.curve_recorder(SimpleNamespace(Callback=object), run, tmp_path, torch)
    return callback, trainer, run, clock


def test_training_progress_and_csv_are_consistent(tmp_path, monkeypatch, capsys):
    callback, trainer, run, clock = fake_callback(tmp_path, monkeypatch)
    callback.on_train_epoch_start(trainer, None)
    clock[0] = 60
    trainer.global_step = 2
    callback.on_train_batch_end(trainer, None, {"loss": Loss(0.5)}, None, 3)
    callback.on_validation_start(trainer, None)
    clock[0] = 80
    callback.on_validation_end(trainer, None)
    callback.on_train_epoch_end(trainer, None)
    trainer.current_epoch = 1
    callback.on_train_epoch_start(trainer, None)
    callback.on_validation_start(trainer, None)
    callback.on_validation_end(trainer, None)
    callback.on_train_epoch_end(trainer, None)
    text = capsys.readouterr().out
    assert "epoch 1/2" in text and "Epoch 2/2" in text
    assert "batch 4/4 (100%)" in text and "lr 1.00e-03" in text
    assert "dev WER 40.00%" in text and "base gain +20.00 pp" in text
    assert "new best" in text and "no improvement" in text
    assert "n/a (CPU)" in text
    assert run["curve"][0]["train_loss"] == 1.0
    assert run["curve"][0]["train_seconds"] == 60
    assert run["curve"][0]["validation_seconds"] == 20
    assert run["curve"][0]["epoch_seconds"] == 80
    assert run["curve"][0]["epoch_optimizer_steps"] == 2
    assert run["curve"][0]["learning_rate"] == 0.001
    assert json.loads((tmp_path / "run.json").read_text())["curve"] == run["curve"]
    with (tmp_path / "curve.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[0]["phase"] == "train_all"
    assert rows[1]["train_loss"] == "" and rows[0]["epoch"] == "0"


@pytest.mark.parametrize("wer", [None, -0.1, float("nan"), float("inf")])
def test_training_refuses_invalid_dev_score(tmp_path, monkeypatch, wer):
    callback, trainer, _, _ = fake_callback(tmp_path, monkeypatch, wer)
    callback.on_train_epoch_start(trainer, None)
    with pytest.raises(RuntimeError, match="finite dev WER"):
        callback.on_train_epoch_end(trainer, None)
    assert not (tmp_path / "curve.csv").exists()


def test_training_stops_on_nonfinite_loss(tmp_path, monkeypatch):
    callback, trainer, _, _ = fake_callback(tmp_path, monkeypatch)
    callback.on_train_epoch_start(trainer, None)
    with pytest.raises(RuntimeError, match="Non-finite training loss"):
        callback.on_train_batch_end(trainer, None, {"loss": Loss(float("nan"))}, None, 0)


def test_training_refuses_stale_validation_metrics(tmp_path, monkeypatch):
    callback, trainer, _, _ = fake_callback(tmp_path, monkeypatch)
    callback.on_train_epoch_start(trainer, None)
    with pytest.raises(RuntimeError, match="finite dev WER"):
        callback.on_train_epoch_end(trainer, None)


def test_fixed_schedule_rejects_step_cap(monkeypatch):
    monkeypatch.setattr(train.sys, "argv", ["train.py", "--select", "last", "--no-keep-stock", "--max-steps", "1"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 2


def aug_args(**overrides):
    args = {"no_speed_perturb": False, "speed_perturb": (0.85, 1.15), "time_stretch": None, "white_noise": None,
            "extra_aug_prob": None}
    args.update(overrides)
    return SimpleNamespace(**args)


def install_fake_perturb(monkeypatch, process):
    import sys
    from types import ModuleType

    module = ModuleType("nemo.collections.asr.parts.preprocessing.perturb")
    module.process_augmentations = process
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_default_augmentation_is_the_speed_only_chain_every_earlier_run_used():
    old = {"speed": {"prob": 1.0, "sr": 16000, "resample_type": "kaiser_fast",
                     "min_speed_rate": 0.85, "max_speed_rate": 1.15, "num_rates": 5}}
    assert train.build_augmentor(aug_args()) == old
    assert train.build_augmentor(aug_args(speed_perturb=[0.9, 1.1]))["speed"]["min_speed_rate"] == 0.9
    assert train.build_augmentor(aug_args(no_speed_perturb=True)) is None
    assert train.augmentation_text(aug_args()) == "speed 0.85-1.15 on every clip"
    assert train.augmentation_text(aug_args(no_speed_perturb=True)) == "none"


def test_extra_augmentations_follow_speed_with_their_own_chance():
    chain = train.build_augmentor(aug_args(time_stretch=[0.75, 1.25], white_noise=[-70, -40]))
    assert list(chain) == ["speed", "time_stretch", "white_noise"]
    assert chain["speed"]["prob"] == 1.0
    assert chain["time_stretch"] == {"prob": 0.5, "min_speed_rate": 0.75, "max_speed_rate": 1.25,
                                     "num_rates": 5, "n_fft": 512}
    assert chain["white_noise"] == {"prob": 0.5, "min_level": -70, "max_level": -40}
    assert isinstance(chain["white_noise"]["min_level"], int)
    chain = train.build_augmentor(aug_args(white_noise=[-60, -30], extra_aug_prob=0.25, no_speed_perturb=True))
    assert list(chain) == ["white_noise"] and chain["white_noise"]["prob"] == 0.25
    text = train.augmentation_text(aug_args(time_stretch=[0.75, 1.25], white_noise=[-70, -40]))
    assert text == ("speed 0.85-1.15 on every clip; tempo 0.75-1.25, pitch kept, on 50% of clips; "
                    "white noise -70 to -41 dBFS on 50% of clips")


@pytest.mark.parametrize("flags, message", [
    (["--time-stretch", "0", "1.2"], "--time-stretch"),
    (["--time-stretch", "1.3", "0.8"], "--time-stretch"),
    (["--time-stretch", "nan", "1.2"], "--time-stretch"),
    (["--white-noise", "-70", "-70"], "--white-noise"),
    (["--white-noise", "-40", "-70"], "--white-noise"),
    (["--white-noise", "-70", "5"], "--white-noise"),
    (["--white-noise", "-130", "-40"], "--white-noise"),
    (["--extra-aug-prob", "0.5"], "needs --time-stretch or --white-noise"),
    (["--time-stretch", "0.8", "1.2", "--extra-aug-prob", "0"], "--extra-aug-prob must be"),
    (["--white-noise", "-70", "-40", "--extra-aug-prob", "1.5"], "--extra-aug-prob must be"),
])
def test_augmentation_flags_reject_ranges_that_would_do_nothing_or_fail_late(monkeypatch, capsys, flags, message):
    monkeypatch.setattr(train.sys, "argv", ["train.py", *flags])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_augmentation_check_reports_a_failing_entry_and_leaves_the_random_state_alone(monkeypatch):
    import random

    np = pytest.importorskip("numpy")

    class Chain:
        def __init__(self, fail):
            self.fail = fail

        def perturb(self, clip):
            random.random()
            np.random.rand()
            if self.fail:
                raise TypeError("__init__() got an unexpected keyword argument 'sr'")
            clip._samples = clip._samples[:8000]

    install_fake_perturb(monkeypatch, lambda cfg: Chain(fail="white_noise" in cfg))
    augmentor = train.build_augmentor(aug_args(time_stretch=[0.75, 1.25], white_noise=[-70, -40]))
    random.seed(5)
    np.random.seed(5)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    problems = train.check_augmentor(augmentor)
    assert len(problems) == 1 and problems[0].startswith("white_noise failed on a test clip")
    assert "unexpected keyword argument 'sr'" in problems[0]
    assert random.getstate() == python_state
    assert all(np.array_equal(x, y) for x, y in zip(np.random.get_state(), numpy_state))
    assert train.check_augmentor(None) == []


def test_augmentation_dict_builds_and_runs_in_real_nemo():
    perturb = pytest.importorskip("nemo.collections.asr.parts.preprocessing.perturb")
    pytest.importorskip("librosa")
    augmentor = train.build_augmentor(aug_args(time_stretch=[0.75, 1.25], white_noise=[-70, -40]))
    chain = perturb.process_augmentations(augmentor)
    assert [prob for prob, _ in chain._pipeline] == [1.0, 0.5, 0.5]
    assert [type(p).__name__ for _, p in chain._pipeline] == [
        "SpeedPerturbation", "TimeStretchPerturbation", "WhiteNoisePerturbation"]
    assert train.check_augmentor(augmentor) == []


def test_training_records_interruptions(tmp_path, monkeypatch):
    callback, trainer, run, _ = fake_callback(tmp_path, monkeypatch)
    callback.on_exception(trainer, None, KeyboardInterrupt())
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] == "interrupted" and "KeyboardInterrupt" in saved["error"]
    assert run["finished_at"]


@pytest.mark.parametrize("color", [False, True])
def test_concise_console_preserves_raw_log_and_actionable_warnings(color):
    console, saved = io.StringIO(), io.StringIO()
    tee = train._cli._Tee(console, saved, train._cli.NemoConsole())
    lines = [
        "OneLogger: Setting error_handling_strategy to disabled\n",
        "[NeMo W] If you intend to do training or fine-tuning, please call setup_training_data\n",
        "    Train config:\n", "    manifest_filepath: null\n", "\n",
        "[12:13:22] [train] Epoch 1/15 | dev WER 55.56%\n",
        "[NeMo W] NumbaPerformanceWarning: Grid size 2 will likely result in GPU under-utilization\n",
        "    warn(errors.NumbaPerformanceWarning(msg))\n",
        "[NeMo W] Speed perturbation unavailable: resampy is missing\n",
        "Traceback (most recent call last):\n", "    training_step failed\n", "RuntimeError: CUDA out of memory\n",
    ]
    text = "".join(lines)
    if color:
        text = text.replace("Grid size 2", "\x1b[1mGrid size 2\x1b[0m")
    for chunk in (text[:20], text[20:60], text[60:]):
        tee.write(chunk)
    tee.flush()
    assert saved.getvalue() == text
    visible = console.getvalue()
    assert "Epoch 1/15" in visible and "resampy is missing" in visible
    assert "training_step failed" in visible and "CUDA out of memory" in visible
    assert "Train config" not in visible and "Grid size 2" not in visible and "OneLogger" not in visible


def test_tee_preserves_partial_lines_and_unfiltered_output():
    for predicate in (None, train._cli.NemoConsole()):
        console, saved = io.StringIO(), io.StringIO()
        tee = train._cli._Tee(console, saved, predicate)
        assert tee.write("partial error") == 13
        tee.flush()
        assert console.getvalue() == saved.getvalue() == "partial error"


def test_checkpoint_average_preserves_integer_buffers_and_frozen_weights(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1)
    model.bias.requires_grad_(False)
    model.register_buffer("batches", torch.tensor(7, dtype=torch.int64))
    original_bias = model.bias.detach().clone()
    paths = []
    for number, weight in enumerate((2.0, 6.0)):
        path = tmp_path / f"{number}.ckpt"
        torch.save({"state_dict": {"weight": torch.tensor([[weight]]), "batches": torch.tensor(number)}}, path)
        paths.append(str(path))
    checkpoint = SimpleNamespace(best_k_models=dict(zip(paths, (0.2, 0.3))))
    assert train.average_checkpoints(model, checkpoint, 2, torch) == paths
    assert model.weight.item() == 4 and model.batches.item() == 7
    assert model.batches.dtype == torch.int64 and torch.equal(model.bias, original_bias)
    assert train.average_checkpoints(model, checkpoint, 2, torch, below=0.25) == paths[:1]
    torch.save({"state_dict": {"weight": torch.tensor([[6.0]])}}, paths[1])
    with pytest.raises(RuntimeError, match="different state keys"):
        train.average_checkpoints(model, checkpoint, 2, torch)


def test_curve_callback_with_real_lightning_accumulated_loss(tmp_path):
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("lightning.pytorch")

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("val_wer", torch.tensor(0.25))

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)

    run = {"configuration": {"select": "last", "log_every_seconds": 30}}
    callback = train.curve_recorder(pl, run, tmp_path, torch)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=2, accumulate_grad_batches=2,
                         num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                         enable_model_summary=False, enable_progress_bar=False, callbacks=[callback])
    loader = torch.utils.data.DataLoader(torch.ones(8, 1), batch_size=2)
    trainer.fit(Model(), train_dataloaders=loader, val_dataloaders=loader)
    assert [r["optimizer_steps"] for r in run["curve"]] == [2, 4]
    assert all(r["train_loss"] == 1 and r["dev_wer"] == 0.25 for r in run["curve"])


@pytest.mark.parametrize("max_steps, expected", [(1, [1]), (5, [3, 5])])
def test_step_cap_validates_final_weights_and_saves_checkpoint(tmp_path, max_steps, expected):
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("lightning.pytorch")

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("val_wer", self.trainer.global_step / 100)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0)

    run = {"configuration": {"select": "dev", "log_every_seconds": 30}}
    curve = train.curve_recorder(pl, run, tmp_path, torch)
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=tmp_path / "ckpt", monitor="val_wer", save_top_k=3)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=3, max_steps=max_steps,
                         accumulate_grad_batches=2, num_sanity_val_steps=0, logger=False,
                         enable_progress_bar=False, enable_model_summary=False, callbacks=[curve, checkpoint])
    loader = torch.utils.data.DataLoader(torch.ones(10, 1), batch_size=2)
    trainer.fit(Model(), train_dataloaders=loader, val_dataloaders=loader)
    assert [r["optimizer_steps"] for r in run["curve"]] == expected
    assert [r["dev_wer"] for r in run["curve"]] == [s / 100 for s in expected]
    assert sum(r["epoch_optimizer_steps"] for r in run["curve"]) == max_steps
    assert len(checkpoint.best_k_models) == len(expected)


@pytest.mark.parametrize("extra_augmentation", [False, True])
@pytest.mark.parametrize("selection", ["dev", "last"])
def test_training_workflow_with_real_lightning_and_synthetic_model(tmp_path, monkeypatch, selection,
                                                                  extra_augmentation):
    import sys
    from types import ModuleType

    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("lightning.pytorch")
    config = pytest.importorskip("omegaconf").OmegaConf
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    export, out = tmp_path / "export", tmp_path / "training run"
    rng = np.random.default_rng(17)
    for split, count in (("train", 5), ("dev", 2)):
        (export / split).mkdir(parents=True)
        rows = []
        for i in range(count):
            name = f"{split}/{i}.wav"
            sf.write(export / name, rng.standard_normal(8000) * 0.1, 16000)
            rows.append({"id": f"{split}-{i}", "text": f"{split} phrase {i}", "audio": name, "duration_s": 0.5})
        (export / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (export / "test.jsonl").write_text("must never be opened by training")

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList()
            self.cfg = config.create({"train_ds": {}, "validation_ds": {}, "spec_augment": None})

        def set_trainer(self, trainer):
            self.trainer = trainer

        def setup_training_data(self, cfg):
            rows = Path(cfg.manifest_filepath).read_text().splitlines()
            self.train_loader = torch.utils.data.DataLoader(torch.ones(len(rows), 1), batch_size=cfg.batch_size)

        def setup_validation_data(self, cfg):
            self.val_loader = torch.utils.data.DataLoader(torch.ones(2, 1), batch_size=cfg.batch_size)

        def setup_optimization(self, cfg):
            self.rate = cfg.lr

        def train_dataloader(self):
            return self.train_loader

        def val_dataloader(self):
            return self.val_loader

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("val_wer", self.weight.detach().abs())

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.rate)

        def save_to(self, path):
            Path(path).write_text(str(self.weight.item()))

    modules = {name: ModuleType(name) for name in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.utils")}
    modules["nemo"].collections = modules["nemo.collections"]
    modules["nemo.collections"].asr = modules["nemo.collections.asr"]
    modules["nemo.collections.asr"].models = SimpleNamespace(ASRModel=SimpleNamespace(from_pretrained=lambda _: Model()))
    modules["nemo.utils"].logging = SimpleNamespace(INFO=20, WARNING=30, set_verbosity=lambda _: None)
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = ["train.py", "--export", str(export), "--out", str(out), "--allow-cpu", "--precision", "32-true",
            "--dev-scoring", "nemo", "--batch", "1", "--grad-accum", "2", "--lr", "0.1", "--epochs", "2",
            "--warmup-steps", "0", "--average-top", "2", "--select", selection]
    args += ["--max-steps", "4"] if selection == "dev" else ["--no-keep-stock"]
    checked = []
    if extra_augmentation:
        class Chain:
            def perturb(self, clip):
                pass

        install_fake_perturb(monkeypatch, lambda cfg: checked.append(cfg) or Chain())
        args += ["--time-stretch", "0.75", "1.25", "--white-noise", "-70", "-40", "--extra-aug-prob", "0.25"]
    monkeypatch.setattr(sys, "argv", args)
    assert train.run_training() == 0
    run = json.loads((out / "run.json").read_text())
    if extra_augmentation:
        assert list(run["augmentation"]) == ["speed", "time_stretch", "white_noise"]
        assert run["augmentation"]["white_noise"] == {"prob": 0.25, "min_level": -70, "max_level": -40}
        assert [list(cfg) for cfg in checked] == [["time_stretch"], ["white_noise"]]
        assert all(cfg[name]["prob"] == 1.0 for cfg in checked for name in cfg)
        assert run["configuration"]["extra_aug_prob"] == 0.25
    else:
        assert list(run["augmentation"]) == ["speed"] and not checked
        assert run["configuration"]["time_stretch"] is None and run["configuration"]["white_noise"] is None
    assert run["status"] == "complete" and run["released_dev_wer"] is not None
    assert run["phase_budgets"]["train_all"]["updates_per_epoch"] == 3
    assert sum(row["epoch_optimizer_steps"] for row in run["curve"]) == (4 if selection == "dev" else 6)
    assert run["stop_reason"] == ("step limit (4)" if selection == "dev" else "epoch limit (2)")
    assert len(run["averaged_checkpoints"]) == 2
    assert (out / "parakeet-ft.nemo").is_file()


def test_stock_form_matches_the_models_own_output_and_scores_the_same():
    from willspeak.normalize import normalize, stock_form

    assert stock_form("tomato") == "Tomato."
    assert stock_form("Why?") == "Why?"
    assert stock_form("Don't ask me to carry an oily rag like that.") == "Don't ask me to carry an oily rag like that."
    assert stock_form('the “quick” fox; said "hi"') == "The quick fox said hi."
    assert stock_form("") == ""
    for text in ["tomato", "Don't ask me to carry an oily rag like that.", "Why?"]:
        assert normalize(stock_form(text)) == normalize(text)


def test_write_nemo_manifest_forms_and_skipped_count(tmp_path):
    from willspeak.normalize import normalize, stock_form

    rows = [{"audio": "a.wav", "duration_s": 1.0, "text": "Don't stop."},
            {"audio": "b.wav", "duration_s": 0.5, "text": "tomato"},
            {"audio": "c.wav", "duration_s": 0.5, "text": "..."}]
    out = tmp_path / "m.json"
    assert train.write_nemo_manifest(rows, tmp_path, out, normalize) == 2
    assert train.write_nemo_manifest.skipped == 1
    texts = [json.loads(line)["text"] for line in out.read_text().splitlines()]
    assert texts == ["do not stop", "tomato"]
    assert train.write_nemo_manifest(rows, tmp_path, out, stock_form) == 3
    texts = [json.loads(line)["text"] for line in out.read_text().splitlines()]
    assert texts == ["Don't stop.", "Tomato.", "..."]
    assert train.write_nemo_manifest(rows, tmp_path, out, None) == 3
    assert train.write_nemo_manifest.skipped == 0


def test_target_checker_reports_unspellable_targets():
    class Tok:
        unk_id = 0

        def text_to_ids(self, text):
            return [0 if ch == "~" else 1 for ch in text]

    class Model:
        tokenizer = Tok()

    check = train.target_checker(Model())
    assert check(["tomato", "wa~ter", "fine"]) == ["wa~ter"]
    assert train.target_checker(SimpleNamespace(tokenizer=None)) is None


def test_weight_decay_flag_is_validated(tmp_path):
    import subprocess
    import sys

    script = ROOT / "scripts/asr/train.py"
    r = subprocess.run([sys.executable, str(script), "--weight-decay", "-1", "--check", "--export", str(tmp_path)],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "weight decay" in r.stderr
