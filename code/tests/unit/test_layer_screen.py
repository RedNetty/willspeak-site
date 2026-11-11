import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "matcher"))
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import layer_screen as ls
from willspeak.matcher import training as S

SPEAKERS = ("M01", "M02", "F03", "F04")
WORDS = [f"word{i}" for i in range(12)]
D = 32


def make_archive(path: Path, noise_by_layer: list[float], seed: int = 0, shuffle: bool = False, flat: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    pattern = rng.normal(size=(len(WORDS), D))
    labels, sessions, speakers, paths, vectors = [], [], [], [], []
    for speaker in SPEAKERS:
        for session in ("A", "B"):
            offset = rng.normal(size=D) * 2
            for w, word in enumerate(WORDS):
                labels.append(word)
                sessions.append(f"{speaker}_{session}")
                speakers.append(speaker)
                paths.append(f"{speaker}/{session}/{word}.wav")
                vectors.append([pattern[w] + offset + rng.normal(size=D) * n for n in noise_by_layer])
    raw = np.array(vectors, dtype=np.float32)
    order = np.random.default_rng(seed + 1).permutation(len(labels)) if shuffle else np.arange(len(labels))
    if flat:
        raw = raw[:, 0, :]
    arrays = {"raw": raw[order], "labels": np.array(labels)[order], "sessions": np.array(sessions)[order],
              "speakers": np.array(speakers)[order], "paths": np.array(paths)[order], "meta": "{}"}
    if not flat:
        arrays["layers"] = np.arange(len(noise_by_layer))
    np.savez(path, **arrays)
    return path


def test_hits_match_the_shipped_cross_session_score():
    rng = np.random.default_rng(1)
    labels = np.array([w for _ in range(3) for w in WORDS])
    sessions = np.array([s for s in "ABC" for _ in WORDS])
    V = rng.normal(size=(len(labels), 8)) + np.tile(np.eye(12, 8) * 3, (3, 1))
    hits, words = ls.cross_session_hits(V, labels, sessions)
    accuracy, n = S.cross_session_top1(V, labels, sessions)
    assert len(hits) == n == len(words) and hits.mean() == pytest.approx(accuracy)


def test_align_keeps_shared_clips_in_the_first_archives_order(tmp_path):
    a = ls.load_archive(str(make_archive(tmp_path / "a.npz", [0.5])))
    b = ls.load_archive(str(make_archive(tmp_path / "b.npz", [0.5], shuffle=True, flat=True)))
    for key in ("paths", "raw", "labels", "sessions", "speakers"):
        b[key] = b[key][:-6]
    both = ls.align({"a": a, "b": b})
    assert len(both["a"]["paths"]) == len(a["paths"]) - 6
    assert list(both["a"]["paths"]) == [p for p in a["paths"] if p in set(b["paths"])]
    assert list(both["a"]["paths"]) == list(both["b"]["paths"]) and list(both["a"]["labels"]) == list(both["b"]["labels"])
    assert np.allclose(both["a"]["raw"][:, 0, :], both["b"]["raw"][:, 0, :])


def test_a_layer_with_word_information_beats_one_without_and_the_gain_has_an_interval(tmp_path):
    first = make_archive(tmp_path / "model.npz", [30.0, 1.0, 0.3])
    flat = make_archive(tmp_path / "flat.npz", [1.0], flat=True)
    archives = ls.align({"m": ls.load_archive(str(first)), "wavlm": ls.load_archive(str(flat))})
    held = list(SPEAKERS)
    systems = {}
    for name, label, column in ls.plan_rows(archives, None):
        d = archives[label]
        systems[name] = ls.score_layer(d["raw"][:, column, :], d, held, dim=8, alpha=0.1)
    rows = {r["system"]: r for r in ls.summarise(systems, {"M01", "M02"}, "wavlm")}
    assert set(rows) == {"m/L0", "m/L1", "m/L2", "wavlm"}
    assert rows["m/L0"]["all"]["accuracy"] < 0.4 < rows["m/L1"]["all"]["accuracy"] < rows["m/L2"]["all"]["accuracy"]
    assert rows["m/L2"]["all"]["accuracy"] > 0.9
    gain = rows["m/L0"]["severe"]["against_baseline"]
    assert gain["gain"] < 0 and gain["ci"][1] < 0 and gain["baseline"] == "wavlm"
    low, high = rows["m/L2"]["severe"]["ci"]
    assert 0 <= low <= rows["m/L2"]["severe"]["accuracy"] <= high <= 1
    assert "against_baseline" not in rows["wavlm"]["severe"]
    assert "wavlm" in ls.render(list(rows.values()), "wavlm")


def test_layer_spec_and_plan():
    assert ls.layer_spec("all") is None and ls.layer_spec("0,4,8-10") == {0, 4, 8, 9, 10}


def test_command_line_screens_layers_and_refuses_to_overwrite(tmp_path):
    model = make_archive(tmp_path / "model.npz", [20.0, 0.5])
    flat = make_archive(tmp_path / "flat.npz", [1.0], flat=True)
    command = [sys.executable, str(ROOT / "scripts" / "matcher" / "layer_screen.py"), "--emb", f"m={model}", "--emb", f"wavlm={flat}",
               "--baseline", "wavlm", "--severe", "M01,M02", "--dim", "8", "--causal", "--out", str(tmp_path / "screen.json")]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    result = json.loads((tmp_path / "screen.json").read_text())
    assert [r["system"] for r in result["rows"]] == ["m/L0", "m/L1", "wavlm"] and result["baseline"] == "wavlm"
    assert all(0 <= r["severe_causal"] <= 1 and r["severe_causal_items"] > 0 for r in result["rows"])
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode != 0
    wrong = [*command[:-2], "--baseline", "nope"]
    assert subprocess.run(wrong, capture_output=True, text=True, check=False).returncode != 0


def test_causal_pool_uses_its_own_query_counts_and_ignores_unscored_speakers():
    per = {"M01": {"hits": np.array([1, 1]), "words": ["a", "b"], "causal": (1.0, 1)},
           "M02": {"hits": np.array([0, 0]), "words": ["a", "b"], "causal": (0.0, 9)},
           "M04": {"hits": np.array([1, 1]), "words": ["a", "b"], "causal": (float("nan"), 0)},
           "F03": {"hits": np.array([1, 1]), "words": ["a", "b"], "causal": (1.0, 100)}}
    row = ls.summarise({"m": per}, {"M01", "M02", "M04"}, None)[0]
    assert row["severe_causal"] == pytest.approx(0.1)
    assert row["severe_causal_items"] == 10


torch = pytest.importorskip("torch")
import embed_layers as el


class FakeEncoder(torch.nn.Module):
    def __init__(self, blocks=3, dim=4):
        super().__init__()
        self.pre_encode = torch.nn.Linear(1, dim)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(dim, dim) for _ in range(blocks))

    def forward(self, x):
        h = self.pre_encode(x)
        for layer in self.layers:
            h = layer(h)
        return h


class FakeSubsampling(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(1, dim)

    def forward(self, x):
        return self.linear(x), torch.tensor([x.shape[1]])


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.encoder.pre_encode = FakeSubsampling(4)

    def forward(self, input_signal, input_signal_length):
        frames = input_signal.shape[1] // 100
        x = input_signal[:, : frames * 100].reshape(1, frames, 100).mean(-1, keepdim=True)
        h = self.encoder.pre_encode(x)[0]
        for layer in self.encoder.layers:
            h = layer(h)
        return h.transpose(1, 2), torch.tensor([frames])


def test_tap_numbers_layers_from_the_subsampling_output_and_pools_the_valid_frames():
    model = FakeModel().eval()
    tap = el.LayerTap(model.encoder)
    assert tap.count == 4
    audio = np.random.default_rng(0).normal(size=1000).astype(np.float32)
    vectors = el.embed_clip(model, tap, audio, [0, 1, 3], "cpu")
    assert vectors.shape == (3, 4)
    x = torch.from_numpy(audio).reshape(1, 10, 100).mean(-1, keepdim=True)
    with torch.no_grad():
        h0 = model.encoder.pre_encode(x)[0]
        h1 = model.encoder.layers[0](h0)
    assert np.allclose(vectors[0], h0[0].mean(0).numpy(), atol=1e-5)
    assert np.allclose(vectors[1], h1[0].mean(0).numpy(), atol=1e-5)
    again = el.embed_clip(model, tap, audio, [0, 1, 3], "cpu")
    assert np.allclose(vectors, again)
    tap.close()
    assert not model.encoder.layers[0]._forward_hooks


def test_parse_layers():
    assert el.parse_layers("all", 25) == list(range(25))
    assert el.parse_layers("0,4,8-10", 25) == [0, 4, 8, 9, 10]
    with pytest.raises(ValueError):
        el.parse_layers("30", 25)


def test_embedding_rejects_empty_frames_and_non_finite_layers():
    model = FakeModel().eval()
    tap = el.LayerTap(model.encoder)
    with pytest.raises(ValueError, match="no encoder frames"):
        el.embed_clip(model, tap, np.zeros(50, dtype=np.float32), [0, 1], "cpu")
    tap.slots = [torch.full((1, 3, 4), float("nan"))] * tap.count
    with pytest.raises(ValueError, match="Layer 0 produced non-finite"):
        tap.pooled(3, [0, 1])
    tap.close()


def test_screen_reports_bad_layers_and_refuses_a_non_finite_baseline(tmp_path, monkeypatch, capsys):
    model = make_archive(tmp_path / "model.npz", [0.5, 1.0])
    archive = dict(np.load(model, allow_pickle=True))
    archive["raw"][0, 0, 0] = np.nan
    np.savez(model, **archive)
    flat = make_archive(tmp_path / "flat.npz", [1.0], flat=True)
    out = tmp_path / "screen.json"
    argv = ["layer_screen", "--emb", f"m={model}", "--emb", f"wavlm={flat}",
            "--baseline", "wavlm", "--dim", "8", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert ls.main() == 0
    assert "SKIPPED m/L0" in capsys.readouterr().out
    result = json.loads(out.read_text())
    assert result["non_finite_skipped"] == {"m/L0": 1}
    assert {r["system"] for r in result["rows"]} == {"m/L1", "wavlm"}
    monkeypatch.setattr(sys, "argv", [*argv[:-2], "--baseline", "m/L0"])
    with pytest.raises(ValueError, match="Baseline.*non-finite"):
        ls.main()


def test_last_session_only_tests_the_final_session(tmp_path):
    d = ls.load_archive(str(make_archive(tmp_path / "a.npz", [0.5])))
    X = d["raw"][:, 0, :]
    speaker = "M01"
    full = ls.score_layer(X, d, [speaker], dim=8, alpha=0.1)[speaker]
    last = ls.score_layer(X, d, [speaker], dim=8, alpha=0.1, last_only=True)[speaker]
    assert ls.last_session(d["sessions"][d["speakers"] == speaker]) == "M01_B"
    assert len(last["hits"]) == len(WORDS) and len(full["hits"]) == 2 * len(WORDS)


def test_archives_with_relative_and_absolute_paths_still_align(tmp_path, monkeypatch):
    monkeypatch.setattr(ls.lab, "ROOT", tmp_path)
    flat = make_archive(tmp_path / "flat.npz", [0.5], flat=True)
    layered = make_archive(tmp_path / "layered.npz", [0.5, 1.0])
    z = dict(np.load(layered, allow_pickle=True))
    z["paths"] = np.array([str(tmp_path / p) for p in z["paths"]])
    np.savez(layered, **z)
    archives = ls.align({"wavlm": ls.load_archive(str(flat)), "model": ls.load_archive(str(layered))})
    assert len(archives["wavlm"]["paths"]) == len(archives["model"]["paths"]) > 0
