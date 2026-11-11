import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
import hyp_selector as hs  # noqa: E402

VOCAB = ["open", "close", "door", "window", "water", "light", "help", "doctor", "music", "phone", "blanket", "tired", "cold", "hungry",
         "morning", "evening", "please", "thanks", "bring", "turn"]


def phrase(i: int) -> str:
    rng = np.random.default_rng(i)
    return " ".join(rng.choice(VOCAB, size=int(rng.integers(2, 6)), replace=False))


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def corrupt(text: str, rng: np.random.Generator, how: str) -> str:
    w = text.split()
    if how == "empty":
        return ""
    if how == "drop":
        return " ".join(w[: max(1, len(w) // 2)])
    if how == "swap":
        i = int(rng.integers(len(w)))
        w[i] = str(rng.choice(VOCAB))
        return " ".join(w)
    return text


def scenario(root: Path, speaker: str = "M01", n_label: int = 120, n_test: int = 50, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    exp = root / hs.EXPORT
    for kind, n, base in (("label", n_label, 0), ("test", n_test, 1000)):
        rows, hyps = [], {"a": [], "b": [], "c": []}
        for i in range(n):
            text = phrase(base + i)
            cid = f"{speaker}-{kind}-{i}"
            dur = 0.5 + 0.4 * len(text.split())
            rows.append({"id": cid, "audio": f"{cid}.wav", "text": text, "duration_s": dur, "speaker": speaker,
                         "kind": "sentence" if len(text.split()) >= 3 else "word"})
            hyps["a"].append(corrupt(text, rng, "empty" if rng.random() < 0.35 else "keep"))
            hyps["b"].append(corrupt(text, rng, "swap" if rng.random() < 0.4 else "keep"))
            hyps["c"].append(corrupt(text, rng, ["swap", "drop", "empty"][int(rng.integers(3))] if rng.random() < 0.7 else "keep"))
        folder = f"cv-{speaker}-all" if kind == "label" else speaker
        write_manifest(exp / folder / "test.jsonl", rows)
        for name, answers in hyps.items():
            (root / "h").mkdir(exist_ok=True)
            (root / "h" / f"{name}-{kind}-{speaker}.jsonl").write_text("".join(json.dumps({"id": r["id"], "hyp": h}) + "\n" for r, h in zip(rows, answers)))


def systems(root: Path):
    return [(n, f"{root}/h/{n}-label-{{speaker}}.jsonl", f"{root}/h/{n}-test-{{speaker}}.jsonl") for n in ("a", "b", "c")]


def test_similarity_and_edit_distance():
    assert hs.similarity(["a", "b"], ["a", "b"]) == 1.0
    assert hs.similarity([], []) == 1.0
    assert hs.similarity(["a", "b"], ["a", "c"]) == pytest.approx(0.5)
    assert hs.similarity(["a"], []) == 0.0


def test_weighted_vote_follows_the_weights_and_the_majority():
    hyps = ["open the door", "open a door", "open the door"]
    assert hs.weighted_vote(hyps, [1, 1, 1]) == "open the door"
    assert hs.weighted_vote(["open a door", "open the door", "open the door"], [3, 1, 1]) == "open a door"
    assert hs.weighted_vote(["open a door", "open the door", "open the door"], [1.0, 0.9, 0.9]) == "open the door"
    assert hs.weighted_vote(["x"], [1.0]) == "x"
    assert hs.weighted_vote(["", "water", "water"], [1, 1, 1]) == "water"
    with pytest.raises(ValueError):
        hs.weighted_vote(["a"], [1, 2])


def test_the_learned_selector_beats_every_single_system_when_failures_are_detectable(tmp_path):
    scenario(tmp_path)
    result = hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "out", folds=4)
    for part in ("labelled", "test"):
        singles = {n: result[part][n]["wrong"] for n in ("a", "b", "c")}
        assert result[part]["learned"]["wrong"] < min(singles.values()), (part, result[part])
        assert result[part]["oracle"]["wrong"] <= result[part]["learned"]["wrong"]
    assert result["order_best_first"][0] in ("a", "b")
    assert (tmp_path / "out/results.json").is_file()
    lines = (tmp_path / "out/hyps/learned-M01.jsonl").read_text().splitlines()
    assert len(lines) == 50 and set(json.loads(lines[0])) == {"id", "hyp"}
    assert (tmp_path / "out/hyps/vote-M01.jsonl").is_file() and (tmp_path / "out/hyps/wvote-M01.jsonl").is_file()


def test_the_labelled_score_is_leave_prompt_out(tmp_path):
    scenario(tmp_path)
    labelled = []
    exp = tmp_path / hs.EXPORT
    rows = hs.lab.read_manifest(exp / "cv-M01-all/test.jsonl")
    hyps = {n: hs.lab.read_hyps(tmp_path / f"h/{n}-label-M01.jsonl") for n in ("a", "b", "c")}
    labelled = hs.load_clips(tmp_path, "M01", exp / "cv-M01-all/test.jsonl", hyps, True)
    folds = hs.grouped_folds(labelled, 4, 17)
    assert sorted(i for f in folds for i in f) == list(range(len(labelled)))
    for f in folds:
        in_fold = {labelled[i].prompt for i in f}
        outside = {labelled[i].prompt for g in folds if g is not f for i in g}
        assert not in_fold & outside
    del rows


def test_a_selector_cannot_learn_from_test_text(tmp_path):
    scenario(tmp_path)
    a = hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "o1", folds=4)
    path = tmp_path / hs.EXPORT / "M01/test.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    for r in rows:
        r["text"] = "completely different words"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    b = hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "o2", folds=4)
    assert (tmp_path / "o1/hyps/learned-M01.jsonl").read_text() == (tmp_path / "o2/hyps/learned-M01.jsonl").read_text()
    assert a["test"]["learned"]["wrong"] != b["test"]["learned"]["wrong"]


def test_refuses_bad_setups(tmp_path):
    scenario(tmp_path)
    with pytest.raises(ValueError, match="two systems"):
        hs.run(tmp_path, ["M01"], systems(tmp_path)[:1], tmp_path / "o")
    with pytest.raises(ValueError, match="unique"):
        hs.run(tmp_path, ["M01"], [("vote", *systems(tmp_path)[0][1:]), systems(tmp_path)[1]], tmp_path / "o")
    hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "o", folds=3)
    with pytest.raises(FileExistsError):
        hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "o", folds=3)
    short = tmp_path / "h/b-test-M01.jsonl"
    short.write_text("\n".join(short.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="no answer"):
        hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "o2", folds=3)


def test_a_phrase_list_that_holds_test_prompts_is_refused_and_a_clean_one_adds_a_feature(tmp_path):
    scenario(tmp_path)
    exp = tmp_path / hs.EXPORT
    other = [{"id": f"o{i}", "audio": "x.wav", "text": phrase(5000 + i), "duration_s": 1.0} for i in range(40)]
    write_manifest(exp / "loso-M01/train.jsonl", other)
    result = hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "o", folds=3, phrase_manifests={"M01": [f"{hs.EXPORT}/loso-M01/train.jsonl"]})
    assert result["phrase_list"] and "phrase_dist" in result["features"]
    leaky = [{"id": "t", "audio": "x.wav", "text": json.loads((exp / "M01/test.jsonl").read_text().splitlines()[0])["text"], "duration_s": 1.0}]
    write_manifest(exp / "loso-M01/train.jsonl", other + leaky)
    with pytest.raises(ValueError, match="test prompts"):
        hs.run(tmp_path, ["M01"], systems(tmp_path), tmp_path / "o3", folds=3, phrase_manifests={"M01": [f"{hs.EXPORT}/loso-M01/train.jsonl"]})


def test_command_line_prints_both_tables(tmp_path, capsys):
    scenario(tmp_path)
    args = ["--speaker", "M01", "--folds", "3", "--out", "out", "--root", str(tmp_path)]
    for n, l, t in systems(tmp_path):
        args += ["--system", f"{n}={l}|{t}"]
    assert hs.main(args) == 0
    text = capsys.readouterr().out
    assert "leave-prompt-out" in text and "learned" in text and "oracle" in text
    with pytest.raises(SystemExit):
        hs.main(["--speaker", "M01", "--system", "broken", "--out", "x"])


def test_a_systems_file_names_each_speakers_files_separately(tmp_path, capsys):
    scenario(tmp_path)
    spec = {n: {"labelled": {"M01": str(tmp_path / f"h/{n}-label-M01.jsonl")}, "test": {"M01": str(tmp_path / f"h/{n}-test-M01.jsonl")}} for n in ("a", "b", "c")}
    (tmp_path / "systems.json").write_text(json.dumps(spec))
    assert hs.main(["--speaker", "M01", "--folds", "3", "--out", "out", "--root", str(tmp_path), "--systems-file", str(tmp_path / "systems.json")]) == 0
    assert "leave-prompt-out" in capsys.readouterr().out
    (tmp_path / "bad.json").write_text(json.dumps({"a": {"labelled": {}}}))
    with pytest.raises(ValueError, match="needs a 'labelled'"):
        hs.read_systems_file(tmp_path / "bad.json")
    with pytest.raises(SystemExit):
        hs.main(["--speaker", "M01", "--out", "x"])
