import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import wandb_log  # noqa: E402


class FakeRun:
    def __init__(self, fail=False):
        self.fail, self.logged, self.artifacts, self.summary, self.metrics = fail, [], [], {}, []

    def _check(self):
        if self.fail:
            raise RuntimeError("network down")

    def log(self, values):
        self._check()
        self.logged.append(values)

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log_artifact(self, artifact):
        self._check()
        self.artifacts.append(artifact)

    def finish(self, exit_code=0):
        self._check()
        self.exit_code = exit_code


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for key in ("WILLSPEAK_WANDB", "WANDB_API_KEY", "WANDB_MODE", "WILLSPEAK_WANDB_GROUP", "WILLSPEAK_WANDB_TAGS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("HOME", str(tmp_path))
    wandb_log._warned.clear()


def test_off_mode_is_a_pure_noop(monkeypatch):
    monkeypatch.setenv("WILLSPEAK_WANDB", "off")

    def forbidden():
        raise AssertionError("wandb must not be imported when tracking is off")

    monkeypatch.setattr(wandb_log, "_import", forbidden)
    assert wandb_log.mode() == "off"
    assert wandb_log.start("run", {"a": 1}) is None
    for call in (lambda: wandb_log.log(None, {"x": 1}), lambda: wandb_log.summary(None, {"x": 1}),
                 lambda: wandb_log.log_table(None, "t", ["a"], [[1]]), lambda: wandb_log.log_file(None, "x.json"),
                 lambda: wandb_log.finish(None)):
        assert call() is None
    assert wandb_log.log_eval("e", "missing.json") is None


def test_missing_wandb_gives_none_handles(monkeypatch):
    monkeypatch.setenv("WILLSPEAK_WANDB", "online")
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    monkeypatch.setattr(wandb_log, "_import", lambda: None)
    assert wandb_log.available() is False
    assert wandb_log.mode() == "off"
    assert wandb_log.start("run") is None


def test_auto_mode_choices(monkeypatch):
    monkeypatch.setattr(wandb_log, "_import", lambda: object())
    assert wandb_log.mode() == "off"
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert wandb_log.mode() == "offline"
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert wandb_log.mode() == "online"
    monkeypatch.setenv("WILLSPEAK_WANDB", "online")
    monkeypatch.delenv("WANDB_API_KEY")
    assert wandb_log.mode() == "offline"


def test_failing_calls_never_raise(monkeypatch, capsys):
    handle = wandb_log.Handle(FakeRun(fail=True), "run", "offline")
    wandb_log.log(handle, {"x": 1.0}, step_metric="epoch")
    wandb_log.log(handle, {"x": 2.0})
    wandb_log.summary(handle, {"a": object()})
    wandb_log.log_table(handle, "t", ["a"], [[1]])
    wandb_log.finish(handle, "failed")
    assert capsys.readouterr().out.count("[wandb] log failed") == 1

    class Boom:
        @staticmethod
        def init(**kwargs):
            raise RuntimeError("cannot start")

        @staticmethod
        def Settings(**kwargs):
            return None

    monkeypatch.setenv("WILLSPEAK_WANDB", "offline")
    monkeypatch.setattr(wandb_log, "_import", lambda: Boom)
    assert wandb_log.start("run") is None


def test_non_finite_metrics_are_dropped():
    run = FakeRun()
    handle = wandb_log.Handle(run, "run", "offline")
    wandb_log.log(handle, {"a": float("nan"), "b": 2, "c": "text", "d": float("inf"), "e": True})
    assert run.logged == [{"b": 2, "e": 1}]


@pytest.mark.parametrize("name", ["model.nemo", "clip.wav", "weights.pt", "notes.md", "x.ckpt"])
def test_log_file_refuses_private_or_binary_files(tmp_path, name, capsys):
    path = tmp_path / name
    path.write_text("data")
    run = FakeRun()
    wandb_log.log_file(wandb_log.Handle(run, "run", "offline"), path)
    assert run.artifacts == []
    assert "refused" in capsys.readouterr().out


def test_log_file_refuses_large_file(tmp_path, capsys):
    path = tmp_path / "big.json"
    path.write_bytes(b" " * (wandb_log.MAX_FILE_BYTES + 1))
    run = FakeRun()
    wandb_log.log_file(wandb_log.Handle(run, "run", "offline"), path)
    assert run.artifacts == []
    assert "larger than" in capsys.readouterr().out


def test_eval_metrics_read_numbers_only():
    results = {"wer": 0.5, "cer": 0.3, "n_utts": 4,
               "word": {"errors": 5, "n_ref": 10, "sub": 3, "dele": 1, "ins": 1, "rate": 0.5},
               "by_group": {"word": {"errors": 2, "n_ref": 3, "sub": 1, "dele": 0, "ins": 1, "rate": 0.66}},
               "worst": [{"hyp": "private words", "ref": "secret", "id": "x"}],
               "confusions": [{"hyp": "D", "ref": "T", "n": 1}]}
    metrics = wandb_log.eval_metrics(results)
    assert metrics["eval/wer"] == 0.5 and metrics["eval/word/errors"] == 5
    assert metrics["eval/kind/word/ins"] == 1
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_offline_run_end_to_end(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WILLSPEAK_WANDB", "offline")
    monkeypatch.setenv("WANDB_MODE", "offline")
    handle = wandb_log.start("unit-run", {"lr": 1e-4, "nested": {"a": [1, 2]}, "path": Path("x")},
                             group="g", tags=["t"], run_id="a" * 32)
    assert handle is not None and handle.mode == "offline"
    wandb_log.log(handle, {"optimizer_step": 1, "train/loss": 2.0}, step_metric="optimizer_step")
    wandb_log.log(handle, {"epoch": 1, "epoch/dev_wer": 0.5}, step_metric="epoch")
    wandb_log.log_table(handle, "systems", ["system", "wer"], [["a", 0.5]])
    record = tmp_path / "run.json"
    record.write_text(json.dumps({"ok": True}))
    wandb_log.log_file(handle, record, kind="train-record")
    wandb_log.summary(handle, {"released_dev_wer": 0.5, "skip": None})
    wandb_log.finish(handle)
    wandb_log.finish(handle)
    assert list((tmp_path / "wandb").rglob("offline-run-*"))


def test_log_eval_offline(monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WILLSPEAK_WANDB", "offline")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"wer": 0.25, "n_utts": 2, "word": {"errors": 1, "n_ref": 4}}))
    handle = wandb_log.log_eval("eval-run", results, group="M01", run_id="b" * 32)
    assert handle is not None and handle.closed
