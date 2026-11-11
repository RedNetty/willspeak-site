import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import wandb_backfill as backfill  # noqa: E402
import wandb_log  # noqa: E402

LOG = """\
# train_parakeet argv=['--export', 'data/export/M01', '--out', 'models/demo/whole-M01', '--seed', '17'] cwd=/x/app
[15:47:57] [train] --- RUN whole-M01 ---
[15:47:57] [train]   Base model       : nvidia/parakeet-tdt-0.6b-v2
[15:47:57] [train]   Learning rate    : 2.00e-05; warmup 10 updates; minimum 1.00e-06
[15:47:57] [train]   Seed / precision : 17 / bf16-mixed
OneLogger: unrelated line
[15:48:03] [train] stock dev WER 0.7000 (70.00%; base model; a checkpoint must beat this to be released)
[15:48:20] [train] Epoch 1/2 | dev WER 61.11% | base gain +8.89 pp | new best
[15:48:20] [train]   loss 5.8110 | lr 1.09e-05 | updates 5 this epoch / 5 total | train 0m 08s | dev 0m 00s
[15:48:20] [train]   peak GPU allocated 7.19 GB | reserved 7.49 GB
[15:48:44] [train] Epoch 2/2 | dev WER 55.00% | fixed schedule
[15:48:44] [train]   loss 4.6332 | lr 1.00e-06 | updates 5 this epoch / 10 total | train 1h 02m 04s | dev 0m 01s
[15:48:44] [train]   peak GPU allocated 7.56 GB | reserved 8.00 GB
[15:48:58] [train] --- COMPLETE ---
[15:48:58] [train]   Released                : fine-tuned, fixed schedule
[15:48:58] [train]   Training stopped        : epoch limit (2)
[15:48:58] [train]   Completed epochs        : 2
[15:48:58] [train]   Optimizer updates       : 10
[15:48:58] [train]   Base dev WER            : not measured
[15:48:58] [train]   Best checkpoint dev WER : not measured
[15:48:58] [train]   Released model dev WER  : 55.00%
[15:48:58] [train]   Elapsed                 : 0m 54s
"""

FAILED = """\
# train_parakeet argv=['--out', 'models/demo/broken']
[10:00:00] [train] --- RUN broken ---
[10:00:00] [train]   Base model       : nvidia/parakeet-tdt-0.6b-v2
Traceback (most recent call last):
  File "x.py", line 1, in <module>
RuntimeError: out of memory
"""


def test_parse_complete_log():
    parsed = backfill.parse_log(LOG)
    assert parsed["status"] == "complete" and parsed["name"] == "whole-M01"
    assert parsed["out"] == "models/demo/whole-M01"
    assert [e["epoch"] for e in parsed["epochs"]] == [1, 2]
    first, second = parsed["epochs"]
    assert first["dev_wer"] == pytest.approx(0.6111) and first["train_loss"] == 5.811
    assert first["optimizer_steps"] == 5 and first["peak_gpu_gb"] == 7.49
    assert second["train_seconds"] == 3724 and second["optimizer_steps"] == 10
    summary = backfill.log_summary(parsed)
    assert summary["released_dev_wer"] == pytest.approx(0.55) and summary["best_dev_wer"] is None
    assert summary["base_dev_wer"] == pytest.approx(0.7) and summary["elapsed_seconds"] == 54
    assert summary["completed_epochs"] == 2 and summary["optimizer_updates"] == 10


def test_parse_failed_and_garbage_logs():
    parsed = backfill.parse_log(FAILED)
    assert parsed["status"] == "failed" and parsed["epochs"] == []
    assert parsed["error"] == "RuntimeError: out of memory"
    assert backfill.parse_log("")["status"] == "failed"
    assert backfill.parse_log("\x00\x01 not a log\n")["name"] is None
    assert backfill.parse_log(LOG.split("--- COMPLETE")[0])["status"] == "failed"


def test_run_id_is_deterministic_and_content_sensitive():
    first = backfill.run_id_for("runs", "models/a/run.json", b"{}")
    assert first == backfill.run_id_for("runs", "models/a/run.json", b"{}")
    assert len(first) == 32 and int(first, 16) >= 0
    assert first != backfill.run_id_for("runs", "models/a/run.json", b"{ }")
    assert first != backfill.run_id_for("evals", "models/a/run.json", b"{}")
    assert first != backfill.run_id_for("runs", "models/b/run.json", b"{}")


def test_flat_numbers_skips_text_fields():
    flat = backfill.flat_numbers({"a": 1, "provenance": {"x": 2}, "ci": [0.1, 0.2],
                                  "worst": [{"wer": 3}], "s": "text", "ok": True})
    assert flat == {"a": 1, "ci.0": 0.1, "ci.1": 0.2, "ok": 1}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(backfill, "APP", tmp_path)
    run_dir = tmp_path / "models" / "demo" / "whole-M01"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "format": "willspeak-training-v1", "status": "complete", "started_at": "2026-10-06T20:48:03+00:00",
        "configuration": {"out": "models/demo/whole-M01", "base": "nvidia/parakeet-tdt-0.6b-v2", "lr": 2e-5},
        "curve": [{"display_epoch": 1, "optimizer_steps": 5, "dev_wer": 0.6, "train_loss": 5.0,
                   "learning_rate": 1e-5, "train_seconds": 8.0}],
        "released_dev_wer": 0.6, "minutes": 0.9, "parameters": {"trainable": 10, "total": 20}}))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "train_parakeet-20261006-204757-111.log").write_text(LOG)
    (logs / "train_parakeet-20261001-100000-222.log").write_text(FAILED)
    (logs / "train_parakeet-20261002-100000-333.log").write_text(LOG.replace("whole-M01", "other"))
    eval_dir = tmp_path / "reports" / "finetune-torgo" / "lab-01-M01-test-nemo"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results.json").write_text(json.dumps({
        "wer": 0.5, "cer": 0.3, "n_utts": 2, "word": {"errors": 2, "n_ref": 4},
        "worst": [{"hyp": "private", "ref": "secret"}]}))
    lab = tmp_path / "reports" / "lab-01-demo"
    lab.mkdir()
    (lab / "table.json").write_text(json.dumps({
        "baseline": "stock", "provenance": {"inputs": {"a": {"sha256": "x"}}},
        "rows": [{"system": "stock", "pooled_wer": 0.76, "pooled_wrong": 323, "interval": {"ci": [0.6, 0.8]}},
                 {"system": "ft", "pooled_wer": 0.5, "pooled_wrong": 200}]}))
    (lab / "notes.md").write_text("not a table")
    return tmp_path


def test_collection_counts_and_coverage(app):
    items = backfill.build_items(backfill.parse_args([]))
    kinds = [i.kind for i in items]
    assert kinds.count("runs") == 1 and kinds.count("evals") == 1 and kinds.count("tables") == 1
    assert kinds.count("logs") == 2
    table = next(i for i in items if i.kind == "tables")
    assert table.steps == 2 and table.name == "lab-01-demo/table"
    assert [i.run_id for i in items] == [i.run_id for i in backfill.build_items(backfill.parse_args([]))]


def test_what_limit_and_since_filters(app):
    assert {i.kind for i in backfill.build_items(backfill.parse_args(["--what", "evals,tables"]))} == {"evals", "tables"}
    assert len([i for i in backfill.build_items(backfill.parse_args(["--what", "logs", "--limit", "1"]))]) == 1
    newer = backfill.build_items(backfill.parse_args(["--what", "logs", "--since", "20261002"]))
    assert [i.source.endswith("333.log") for i in newer] == [True]
    with pytest.raises(SystemExit):
        backfill.build_items(backfill.parse_args(["--what", "bogus"]))


def test_dry_run_uploads_nothing(app, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not start a tracking run")

    monkeypatch.setattr(wandb_log, "start", forbidden)
    assert backfill.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "whole-M01" in out and "dry run, nothing uploaded: 1 runs, 2 logs, 1 evals, 1 tables" in out


def test_online_without_login_exits_2(app, monkeypatch, tmp_path, capsys):
    pytest.importorskip("wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert backfill.main(["--mode", "online"]) == 2
    assert "wandb login" in capsys.readouterr().out


def test_offline_backfill_end_to_end(app, monkeypatch, tmp_path):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert backfill.main(["--mode", "offline"]) == 0
    assert len(list((tmp_path / "wandb").rglob("offline-run-*"))) == 5


REAL_LOG = ROOT / "logs" / "train_parakeet-20261006-204757-345193.log"


@pytest.mark.skipif(not REAL_LOG.exists(), reason="recorded training log not present")
def test_parse_real_log():
    parsed = backfill.parse_log(REAL_LOG.read_text(errors="replace"))
    assert parsed["status"] == "complete" and parsed["epochs"]
    assert parsed["epochs"][0]["optimizer_steps"] > 0
