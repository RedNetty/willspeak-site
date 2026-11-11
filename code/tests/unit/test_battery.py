import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import battery


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


@pytest.mark.parametrize("kind", battery.KINDS)
def test_every_kind_has_the_requested_length_and_level(kind):
    rng = np.random.default_rng(0)
    x = battery.synth(kind, 2.0, -40.0, rng)
    assert x.dtype == np.float32 and len(x) == 2 * battery.SR and np.isfinite(x).all() and np.abs(x).max() < 1.0
    if kind == "silence":
        assert rms_db(x) < -85
    else:
        assert rms_db(x) == pytest.approx(-40.0, abs=1.0)


def test_same_seed_same_clips_and_unknown_kind_refused():
    a = battery.synth("breath", 1.0, -35.0, np.random.default_rng(7))
    b = battery.synth("breath", 1.0, -35.0, np.random.default_rng(7))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        battery.synth("speech", 1.0, -35.0, np.random.default_rng(0))


def make_real_clips(root: Path):
    rng = np.random.default_rng(1)
    (root / "test").mkdir(parents=True)
    quiet_edges = np.concatenate([rng.normal(size=8000) * 0.001, rng.normal(size=16000) * 0.3, rng.normal(size=8000) * 0.001])
    loud = rng.normal(size=32000) * 0.3
    short = rng.normal(size=4000) * 0.3
    rows = []
    for i, x in enumerate([quiet_edges, loud, short]):
        sf.write(root / "test" / f"{i}.wav", x.astype(np.float32), 16000)
        rows.append({"id": str(i), "audio": f"test/{i}.wav", "text": "yes"})
    return rows


def test_edge_segments_keep_only_the_quiet_ends_of_long_enough_clips(tmp_path):
    rows = make_real_clips(tmp_path)
    found = battery.edge_segments(rows, tmp_path, seconds=0.4)
    assert [name for name, _ in found] == ["head-0", "tail-0"]
    assert all(len(x) == int(0.4 * 16000) for _, x in found)
    assert len(battery.edge_segments(rows, tmp_path, seconds=0.4, limit=1)) == 1


def test_build_writes_wavs_and_a_manifest_that_the_decoders_can_read(tmp_path):
    rows = make_real_clips(tmp_path / "real")
    out = tmp_path / "battery"
    made = battery.build(out, per_kind=2, seed=5, kinds=("white", "taps"), edge_rows=rows, edge_root=tmp_path / "real")
    assert [r["condition"] for r in made] == ["white", "white", "taps", "taps", "edge", "edge"]
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert manifest == made and all(r["text"] == "" and (out / r["audio"]).is_file() for r in manifest)
    from willspeak.evaluate import load_manifest

    assert len(load_manifest(out / "manifest.jsonl")) == 6
    with pytest.raises(FileExistsError):
        battery.build(out, per_kind=1, seed=5)


def test_score_counts_clips_with_words_per_kind():
    rows = [{"id": "a", "condition": "white"}, {"id": "b", "condition": "white"}, {"id": "c", "condition": "hum"}]
    result = battery.score(rows, {"a": "", "b": "Thank you.", "c": "uh huh yeah"})
    assert result["kinds"]["white"] == {"clips": 2, "with_words": 1, "words": 2}
    assert result["kinds"]["hum"] == {"clips": 1, "with_words": 1, "words": 3}
    assert result["total"]["false_insert_rate"] == pytest.approx(2 / 3)
    with pytest.raises(KeyError):
        battery.score(rows, {"a": ""})


def test_command_line_builds_then_scores(tmp_path):
    script = str(ROOT / "scripts" / "asr" / "battery.py")
    out = tmp_path / "bat"
    built = subprocess.run([sys.executable, script, "build", "--out", str(out), "--per-kind", "1", "--kinds", "white", "hum"],
                           capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr
    ids = [json.loads(line)["id"] for line in (out / "manifest.jsonl").read_text().splitlines()]
    hyps = tmp_path / "h.jsonl"
    hyps.write_text("".join(json.dumps({"id": i, "hyp": "oh" if i.startswith("white") else ""}) + "\n" for i in ids))
    result = tmp_path / "score.json"
    scored = subprocess.run([sys.executable, script, "score", "--manifest", str(out / "manifest.jsonl"),
                             "--system", f"tuned={hyps}", "--out", str(result)], capture_output=True, text=True, check=False)
    assert scored.returncode == 0, scored.stderr
    assert json.loads(result.read_text())["systems"]["tuned"]["total"]["with_words"] == 1
