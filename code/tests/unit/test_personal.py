import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
sys.path.insert(0, str(ROOT / "src"))
import personal  # noqa: E402

from willspeak.db import LIVE_PROMPT, Store  # noqa: E402

TEXTS = ["I would like some water please.", "Can you open the window?", "Where is my brother today?",
         "The bus was late again this morning.", "Please turn the light off now.", "Thank you for helping me."]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus(root: Path, texts=TEXTS) -> Path:
    store = Store(root)
    sid = store.start_session(label="home", condition="unknown")
    rng = np.random.default_rng(0)
    for i, text in enumerate(texts):
        sf.write(root / "clips" / f"c{i}.wav", rng.standard_normal(16000).astype("float32") * 0.1, 16000,
                 subtype="PCM_16")
        cid = store.add_clip(sid, None, LIVE_PROMPT, f"clips/c{i}.wav", None, 1.0, {"ok": True})
        store.review_clip(cid, text, "accepted")
    store.con.close()
    return root


def torgo_dev(root: Path, texts: list[str]) -> Path:
    (root / "dev").mkdir(parents=True)
    out = []
    for i, text in enumerate(texts):
        sf.write(root / "dev" / f"F01_{i}.wav", np.zeros(8000, dtype="float32"), 16000, subtype="PCM_16")
        out.append({"id": f"F01_{i}", "audio": f"dev/F01_{i}.wav", "text": text, "prompt_text": text,
                    "duration_s": 0.5, "speaker": "F01"})
    (root / "dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    return root


def test_every_accepted_clip_trains_and_dev_is_public_torgo_only(tmp_path):
    data = corpus(tmp_path / "people" / "someone")
    # A split file from an earlier export must not pull any of their clips out of training ...
    (data / "split_assignments.json").write_text(json.dumps({TEXTS[0].lower(): "test", TEXTS[1].lower(): "dev"}))
    before = (data / "split_assignments.json").read_text()
    dev_root = torgo_dev(tmp_path / "torgo", ["Thank you for helping me.", "the quick brown fox", "bring the chair"])
    record = personal.build_personal_export(data, data / "model-export", dev_root, dev_clips=5, trim=False)
    out = data / "model-export"
    train, dev = rows(out / "train.jsonl"), rows(out / "dev.jsonl")
    assert record["train_clips"] == len(train) == len(TEXTS)
    assert all(r["audio"].startswith("train/") and (out / r["audio"]).is_file() for r in train)
    assert not (out / "test").exists() and rows(out / "test.jsonl") == []
    # ... dev is TORGO only, and a TORGO prompt he also recorded is left out (train.py refuses leakage).
    assert [r["text"] for r in dev] == ["the quick brown fox", "bring the chair"]
    assert all(r["id"].startswith("torgo-") and (out / r["audio"]).is_file() for r in dev)
    # ... and the sticky split file a later study would use is untouched.
    assert (data / "split_assignments.json").read_text() == before


def test_export_validates_for_train_py(tmp_path):
    from willspeak.artifacts import validate_export

    data = corpus(tmp_path / "someone")
    dev_root = torgo_dev(tmp_path / "torgo", ["the quick brown fox"])
    personal.build_personal_export(data, data / "model-export", dev_root, trim=False)
    manifests, _ = validate_export(data / "model-export", ("train", "dev"))
    assert len(manifests["train"]) == len(TEXTS) and len(manifests["dev"]) == 1


def test_no_corpus_and_bad_names_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        personal.build_personal_export(tmp_path / "nobody", tmp_path / "out", tmp_path)
    assert not (tmp_path / "nobody").exists()
    for bad in ("Will Smith", "../x", "severe-base", ""):
        with pytest.raises(ValueError):
            personal.Paths(bad)


def test_train_and_deploy_use_stage_p_recipe_and_keep_logs_private(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/asr/personal.py"), "train", "--name", "someone", "--dry-run"]
    env = {"PATH": "/usr/bin:/bin", "WILLSPEAK_LOG_DIR": str(tmp_path / "logs")}
    done = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert "personal base" in done.stdout  # no base yet: it says how to make one and stops
    assert personal.ADAPTER_RECIPE == ["--adapter-after-blocks", "8", "--lr", "6e-4", "--epochs", "15", "--select",
                                       "last", "--no-keep-stock", "--warmup-steps", "10"]
    assert personal.private_log_dir(["train", "--name", "someone"]) == ROOT / "data/people/someone/logs"
    assert personal.private_log_dir(["train", "--data", "/x/y"]) == Path("/x/y/logs")
    assert personal.private_log_dir(["base"]) == ROOT / "data/people/logs"
