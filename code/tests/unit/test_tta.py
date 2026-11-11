import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
sys.path.insert(0, str(ROOT / "src"))
import tta as T  # noqa: E402

sf = pytest.importorskip("soundfile")

SECRET = "zebra quantum"
PROMPTS = ["open the door", "water", "please turn on the light", "good morning", "I am cold", "call my brother", "yes", "thank you",
           "bring me a blanket", "what time is it", "my head hurts", "close the window"]


def wav(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.random.default_rng(seed).normal(0, 0.05, 1600), 16000)


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def setup(tmp_path: Path, n_test: int = 24) -> tuple[Path, Path]:
    train = tmp_path / "M01"
    rows = {"train": [], "dev": []}
    for i, text in enumerate(PROMPTS):
        split = "dev" if i in (3, 9) else "train"
        cid = f"M01-{i}"
        wav(train / split / f"{cid}.wav", i)
        rows[split].append({"id": cid, "audio": f"{split}/{cid}.wav", "text": text, "duration_s": 0.1, "speaker": "M01", "kind": "word"})
    write(train / "train.jsonl", rows["train"])
    write(train / "dev.jsonl", rows["dev"])
    test_rows = []
    for i in range(n_test):
        cid = f"T{i:03d}"
        wav(train / "test" / f"{cid}.wav", 100 + i)
        test_rows.append({"id": cid, "audio": f"test/{cid}.wav", "text": SECRET, "duration_s": 0.1, "speaker": "M01", "kind": "word"})
    write(train / "test.jsonl", test_rows)
    return train, train / "test.jsonl"


def hyp_file(path: Path, ids: list[str], answers) -> Path:
    path.write_text("".join(json.dumps({"id": i, "hyp": answers(i)}) + "\n" for i in ids))
    return path


def test_chunks_cover_everything_once_and_are_repeatable():
    ids = [f"c{i}" for i in range(23)]
    a = T.chunk_ids(ids, 4)
    assert sorted(x for c in a for x in c) == sorted(ids) and max(map(len, a)) - min(map(len, a)) <= 1
    assert a == T.chunk_ids(ids, 4) and a != T.chunk_ids(ids, 4, seed=3)
    by_id = T.chunk_ids([f"c{i}" for i in (10, 2, 33, 1, 9, 4, 5, 6)], 2, order="id")
    assert by_id[0] == ["c1", "c2", "c4", "c5"]
    with pytest.raises(ValueError):
        T.chunk_ids(ids, 1)
    with pytest.raises(ValueError):
        T.chunk_ids(ids[:5], 4)
    with pytest.raises(ValueError):
        T.chunk_ids(ids, 4, order="sideways")


def test_round_one_adds_only_the_first_chunks_kept_answers_and_never_reads_test_text(tmp_path):
    train, test = setup(tmp_path)
    ids = [json.loads(l)["id"] for l in test.read_text().splitlines()]
    chunks = T.chunk_ids(ids, 4)
    base = hyp_file(tmp_path / "base.jsonl", ids, lambda i: "" if i == chunks[0][0] else "open the window")
    other = hyp_file(tmp_path / "other.jsonl", ids, lambda i: "open the window")
    out = tmp_path / "r1"
    info = T.build(train, test, 4, 1, {0: str(base)}, [str(other)], out)
    assert info["test_text_read"] is False and info["pseudo_clips"] == len(chunks[0]) - 1
    assert info["gate"]["empty answer"] == 1
    train_rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    pseudo = [r for r in train_rows if r["id"].startswith("tta-")]
    assert len(pseudo) == info["pseudo_clips"] and all((out / r["audio"]).is_file() for r in train_rows)
    assert all(r["audio"].startswith("train/") for r in train_rows)
    assert {r["text"] for r in pseudo} == {"open the window"} and not {r["id"][4:] for r in pseudo} - set(chunks[0])
    everything = "".join(p.read_text() for p in out.glob("*.json*"))
    assert SECRET not in everything and "zebra" not in everything


def test_disagreement_and_dev_prompts_are_dropped_and_the_cap_holds(tmp_path):
    train, test = setup(tmp_path)
    ids = [json.loads(l)["id"] for l in test.read_text().splitlines()]
    chunks = T.chunk_ids(ids, 3)
    first = chunks[0]
    own = hyp_file(tmp_path / "own.jsonl", ids, lambda i: "good morning" if i == first[0] else ("open the door now" if i == first[1] else "close the door"))
    other = hyp_file(tmp_path / "other.jsonl", ids, lambda i: "water" if i == first[1] else "close the door")
    info = T.build(train, test, 3, 1, {0: str(own)}, [str(other)], tmp_path / "r1", agree_min=0.5)
    assert info["gate"]["equals a dev prompt"] == 1 and info["gate"]["the other systems disagree"] >= 1
    capped = T.build(train, test, 3, 1, {0: str(own)}, [], tmp_path / "r1cap", max_pseudo_share=0.2)
    assert capped["pseudo_clips"] == capped["pseudo_cap"] == 2 and capped["pseudo_candidates"] > 2
    with pytest.raises(ValueError, match="above 0"):
        T.build(train, test, 3, 1, {0: str(own)}, [], tmp_path / "x", max_pseudo_share=0)


def test_later_rounds_need_every_earlier_chunk_and_refuse_bad_rounds(tmp_path):
    train, test = setup(tmp_path)
    ids = [json.loads(l)["id"] for l in test.read_text().splitlines()]
    h = hyp_file(tmp_path / "h.jsonl", ids, lambda i: "close the door")
    with pytest.raises(ValueError, match="chunk 1 is missing"):
        T.build(train, test, 4, 2, {0: str(h)}, [], tmp_path / "r2")
    with pytest.raises(ValueError, match="round 0"):
        T.build(train, test, 4, 0, {0: str(h)}, [], tmp_path / "r0")
    with pytest.raises(ValueError, match="1 to 3"):
        T.build(train, test, 4, 4, {0: str(h), 1: str(h), 2: str(h), 3: str(h)}, [], tmp_path / "r4")
    out = tmp_path / "r2"
    info = T.build(train, test, 4, 2, {0: str(h), 1: str(h)}, [], out)
    assert info["from_chunks"] == [0, 1]
    with pytest.raises(FileExistsError):
        T.build(train, test, 4, 2, {0: str(h), 1: str(h)}, [], out)


def test_a_failed_build_leaves_no_partial_folder(tmp_path):
    train, test = setup(tmp_path)
    ids = [json.loads(l)["id"] for l in test.read_text().splitlines()]
    h = hyp_file(tmp_path / "h.jsonl", ids, lambda i: "close the door")
    next((train / "train").iterdir()).unlink()
    with pytest.raises(FileNotFoundError):
        T.build(train, test, 4, 1, {0: str(h)}, [], tmp_path / "r1")
    assert not (tmp_path / "r1").exists() and not list(tmp_path.glob("*.partial"))


def test_a_hypothesis_file_that_misses_a_clip_is_named(tmp_path):
    train, test = setup(tmp_path)
    ids = [json.loads(l)["id"] for l in test.read_text().splitlines()]
    first = T.chunk_ids(ids, 2)[0]
    h = hyp_file(tmp_path / "h.jsonl", [i for i in ids if i != first[0]], lambda i: "x")
    with pytest.raises(ValueError, match="no answer"):
        T.build(train, test, 2, 1, {0: str(h)}, [], tmp_path / "r1")


def test_chunk_manifest_and_merge_round_trip(tmp_path):
    _, test = setup(tmp_path)
    ids = [json.loads(l)["id"] for l in test.read_text().splitlines()]
    chunks = T.chunk_ids(ids, 4)
    got = []
    for k in range(4):
        count = T.write_manifest(test, 4, k, tmp_path / f"c{k}.jsonl")
        assert count == len(chunks[k])
        got += [json.loads(l)["id"] for l in (tmp_path / f"c{k}.jsonl").read_text().splitlines()]
    assert sorted(got) == sorted(ids)
    with pytest.raises(FileExistsError):
        T.write_manifest(test, 4, 0, tmp_path / "c0.jsonl")
    files = {k: str(hyp_file(tmp_path / f"h{k}.jsonl", ids, lambda i, k=k: f"answer {k}")) for k in range(4)}
    merged = T.merge(test, 4, files, tmp_path / "merged.jsonl")
    rows = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in merged.read_text().splitlines()}
    assert len(rows) == len(ids)
    for k in range(4):
        assert all(rows[i] == f"answer {k}" for i in chunks[k])
    with pytest.raises(FileExistsError):
        T.merge(test, 4, files, merged)
    del files[2]
    with pytest.raises(ValueError, match="chunk 2"):
        T.merge(test, 4, files, tmp_path / "m2.jsonl")


def test_command_line(tmp_path, capsys):
    train, test = setup(tmp_path)
    ids = [json.loads(l)["id"] for l in test.read_text().splitlines()]
    h = hyp_file(tmp_path / "h.jsonl", ids, lambda i: "close the door")
    assert T.main(["chunks", "--manifest", str(test), "--n", "4"]) == 0 and "24 clips in 4 chunks" in capsys.readouterr().out
    assert T.main(["manifest", "--manifest", str(test), "--n", "4", "--index", "1", "--out", str(tmp_path / "c1.jsonl")]) == 0
    assert T.main(["build", "--train-export", str(train), "--test-manifest", str(test), "--n", "4", "--round", "1",
                   "--chunk-hyps", f"0={h}", "--agree", str(h), "--out", str(tmp_path / "r1")]) == 0
    assert "pseudo-labelled" in capsys.readouterr().out
    with pytest.raises(ValueError, match="INDEX=PATH"):
        T.main(["merge", "--test-manifest", str(test), "--n", "4", "--chunk-hyps", "oops", "--out", str(tmp_path / "m")])


def test_score_counts_wrong_words_per_chunk(tmp_path, capsys):
    _, test = setup(tmp_path)
    rows = [json.loads(l) for l in test.read_text().splitlines()]
    for i, r in enumerate(rows):
        r["text"] = "open the door"
    write(test, rows)
    ids = [r["id"] for r in rows]
    chunks = T.chunk_ids(ids, 4)
    good = hyp_file(tmp_path / "good.jsonl", ids, lambda i: "open the door")
    bad_first = hyp_file(tmp_path / "bad.jsonl", ids, lambda i: "open a door" if i in chunks[0] else "open the door")
    result = T.score(test, 4, {"good": str(good), "bad": str(bad_first)})
    assert result["systems"]["good"]["wrong"] == 0 and result["systems"]["bad"]["wrong"] == len(chunks[0])
    assert [c["wrong"] for c in result["systems"]["bad"]["by_chunk"]][1:] == [0, 0, 0] and result["systems"]["good"]["words"] == 3 * len(ids)
    assert T.main(["score", "--test-manifest", str(test), "--n", "4", "--system", f"good={good}", "--system", f"bad={bad_first}",
                   "--out", str(tmp_path / "score.json")]) == 0
    assert "chunk 0" in capsys.readouterr().out and (tmp_path / "score.json").is_file()
    with pytest.raises(ValueError, match="NAME=PATH"):
        T.main(["score", "--test-manifest", str(test), "--n", "4", "--system", "nonsense"])
