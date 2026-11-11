import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/asr"))
import cv_folds as cv  # noqa: E402

PROMPTS = ["the cat sat", "a dog barked loudly", "open the door", "tomato", "water", "please turn on the light", "good morning",
           "thank you", "I am cold", "call my brother", "yes", "no", "bring me a blanket", "what time is it", "my head hurts",
           "close the window"]


def make_export(root: Path, takes: int = 2, speaker: str = "M01") -> Path:
    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    train, dev = [], []
    for i, text in enumerate(PROMPTS):
        for t in range(takes):
            cid = f"{speaker}-{i}-{t}"
            split = "dev" if i in (3, 9) else "train"
            (root / split).mkdir(exist_ok=True)
            sf.write(root / split / f"{cid}.wav", rng.normal(0, 0.05, 1600), 16000)
            row = {"id": cid, "audio": f"{split}/{cid}.wav", "text": text, "duration_s": 0.1, "speaker": speaker,
                   "kind": "word" if len(text.split()) == 1 else "sentence"}
            (dev if split == "dev" else train).append(row)
    for split, rows in (("train", train), ("dev", dev), ("test", [{**dev[0], "id": "test-x", "text": "unrelated test words"}])):
        (root / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_folds_keep_whole_prompts_together_and_are_repeatable():
    rows = [{"id": f"{i}-{t}", "text": p} for i, p in enumerate(PROMPTS) for t in range(3)]
    a, b = cv.assign_folds(rows, 4, 17), cv.assign_folds(rows, 4, 17)
    assert a == b and set(a.values()) == {0, 1, 2, 3}
    for p_index, p in enumerate(PROMPTS):
        assert len({a[f"{p_index}-{t}"] for t in range(3)}) == 1
    assert cv.assign_folds(rows, 4, 18) != a
    words = [sum(len(r["text"].split()) for r in rows if a[r["id"]] == k) for k in range(4)]
    assert max(words) - min(words) <= 2 * max(len(p.split()) for p in PROMPTS) * 3
    with pytest.raises(ValueError):
        cv.assign_folds(rows[:3], 5, 17)
    with pytest.raises(ValueError):
        cv.assign_folds(rows, 1, 17)


def test_make_writes_folds_that_pass_the_leak_check_and_costs_no_extra_audio(tmp_path):
    src = make_export(tmp_path / "M01")
    out = tmp_path / "exports"
    cv.make(src, "M01", 4, prefix=str(out / "cv-M01"))
    cv.check("M01", 4, name="cv-M01", base=out)
    folders = sorted(p.name for p in out.iterdir())
    assert folders == ["cv-M01-all", "cv-M01-f0", "cv-M01-f1", "cv-M01-f2", "cv-M01-f3"]
    labelled = len(cv.read_rows(src / "train.jsonl")) + len(cv.read_rows(src / "dev.jsonl"))
    assert len(cv.read_rows(out / "cv-M01-all/test.jsonl")) == labelled
    held = [len(cv.read_rows(out / f"cv-M01-f{k}/test.jsonl")) for k in range(4)]
    assert sum(held) == labelled
    f0 = out / "cv-M01-f0"
    train = cv.read_rows(f0 / "train.jsonl")
    assert not {cv.prompt_key(r) for r in train} & {cv.prompt_key(r) for r in cv.read_rows(f0 / "dev.jsonl")}
    strip = lambda rows: [{k: v for k, v in r.items() if k != "audio"} for r in rows]  # noqa: E731
    assert strip(cv.read_rows(f0 / "dev.jsonl")) == strip(cv.read_rows(f0 / "test.jsonl"))
    assert cv.layout_ok(f0)
    assert all((f0 / r["audio"]).is_file() for r in train)
    assert not any("unrelated test words" in r["text"] for k in range(4) for r in cv.read_rows(out / f"cv-M01-f{k}/train.jsonl"))
    first = sf.read(str(f0 / train[0]["audio"]))[0]
    original = sf.read(str(src / [r for r in cv.read_rows(src / "train.jsonl") if r["id"] == train[0]["id"]][0]["audio"]))[0]
    assert np.allclose(first, original)


def test_make_is_safe_to_repeat_and_refuses_other_settings(tmp_path):
    src = make_export(tmp_path / "M01")
    prefix = str(tmp_path / "exports" / "cv-M01")
    cv.make(src, "M01", 4, prefix=prefix)
    again = cv.make(src, "M01", 4, prefix=prefix)
    assert again["settings"]["folds"] == 4
    with pytest.raises(FileExistsError):
        cv.make(src, "M01", 3, prefix=prefix)
    with pytest.raises(FileExistsError):
        cv.make(src, "M01", 4, seed=99, prefix=prefix)
    assert not list((tmp_path / "exports").glob("*.partial"))


def test_make_names_the_missing_audio(tmp_path):
    src = make_export(tmp_path / "M01")
    next((src / "train").iterdir()).unlink()
    with pytest.raises(FileNotFoundError, match="doctor"):
        cv.make(src, "M01", 4, prefix=str(tmp_path / "exports" / "cv-M01"))


def test_check_catches_a_prompt_on_both_sides(tmp_path):
    src = make_export(tmp_path / "M01")
    out = tmp_path / "exports"
    cv.make(src, "M01", 4, prefix=str(out / "cv-M01"))
    held = cv.read_rows(out / "cv-M01-f1/test.jsonl")
    train = cv.read_rows(out / "cv-M01-f1/train.jsonl")
    cv.write_rows(out / "cv-M01-f1/train.jsonl", train + [{**held[0], "id": "leak"}])
    with pytest.raises(ValueError, match="both sides"):
        cv.check("M01", 4, name="cv-M01", base=out)


def hyp_file(path: Path, labelled: list[dict], wrong_every: int, offset: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, r in enumerate(labelled):
        hyp = "wrong words here" if (i + offset) % wrong_every == 0 else r["text"]
        lines.append(json.dumps({"id": r["id"], "hyp": hyp}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_merge_joins_the_held_out_decodes_and_names_a_missing_clip(tmp_path):
    src = make_export(tmp_path / "M01")
    out = tmp_path / "exports"
    cv.make(src, "M01", 4, prefix=str(out / "cv-M01"))
    ids = cv.fold_ids("M01", 4, name="cv-M01", base=out)
    rows = {r["id"]: r for r in cv.read_rows(out / "cv-M01-all/test.jsonl")}
    for k in range(4):
        hyp_file(tmp_path / f"d/arm-f{k}.jsonl", [rows[i] for i in ids[k]], 5)
    merged = cv.merge("M01", 4, str(tmp_path / "d/arm-f{fold}.jsonl"), tmp_path / "oof/arm-M01.jsonl", base=out)
    assert len(cv.read_rows(merged)) == len(rows)
    with pytest.raises(FileExistsError):
        cv.merge("M01", 4, str(tmp_path / "d/arm-f{fold}.jsonl"), merged, base=out)
    short = tmp_path / "d/arm-f2.jsonl"
    short.write_text("\n".join(short.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="no answer"):
        cv.merge("M01", 4, str(tmp_path / "d/arm-f{fold}.jsonl"), tmp_path / "oof/other.jsonl", base=out)


def arms_for(labelled: dict[str, dict[str, dict]], rates: dict[str, int]) -> dict:
    out = {}
    for arm, every in rates.items():
        out[arm] = {}
        for s, rows in labelled.items():
            out[arm][s] = {cid: ("wrong words here" if (i % every == 0) else r["text"]) for i, (cid, r) in enumerate(rows.items())}
    return out


def labelled_set(n_prompts: int = 40, takes: int = 2) -> dict[str, dict[str, dict]]:
    out = {}
    for s in ("M01", "M02", "M04"):
        out[s] = {f"{s}-{i}-{t}": {"id": f"{s}-{i}-{t}", "text": f"{s.lower()} word{i} extra{i % 5}"} for i in range(n_prompts) for t in range(takes)}
    return out


def test_select_picks_a_clear_winner_and_keeps_the_control_when_it_is_noise():
    labelled = labelled_set()
    clear = cv.select(arms_for(labelled, {"control": 3, "better": 12}), labelled, "control", draws=800)
    assert clear["selected"] == "better" and clear["arms"]["better"]["minus_control"]["ci95_points"][1] < 0
    assert clear["arms"]["control"]["wrong"] > clear["arms"]["better"]["wrong"]
    same = cv.select(arms_for(labelled, {"control": 5, "twin": 5}), labelled, "control", draws=800)
    assert same["selected"] == "control" and "not clearly better" in same["reason"] or "no recipe" in same["reason"]
    worse = cv.select(arms_for(labelled, {"control": 8, "worse": 2}), labelled, "control", draws=800)
    assert worse["selected"] == "control" and worse["best_by_rate"] == "control"


def test_a_gain_on_one_speaker_alone_does_not_win():
    labelled = labelled_set()
    arms = arms_for(labelled, {"control": 3, "lopsided": 3})
    for cid, r in labelled["M01"].items():
        arms["lopsided"]["M01"][cid] = r["text"]
    result = cv.select(arms, labelled, "control", draws=800)
    assert result["arms"]["lopsided"]["minus_control"]["speakers_better"] == 1
    assert result["selected"] == "control"


def test_select_rejects_arms_that_cover_different_prompts():
    labelled = labelled_set(10)
    arms = arms_for(labelled, {"control": 3, "other": 4})
    del arms["other"]["M01"][next(iter(arms["other"]["M01"]))]
    with pytest.raises(Exception):
        cv.select(arms, labelled, "control", draws=100)
    with pytest.raises(ValueError, match="control"):
        cv.select(arms_for(labelled, {"a": 3}), labelled, "missing", draws=100)


def test_command_line_end_to_end(tmp_path, capsys):
    root = tmp_path / "app"
    for s in ("M01", "M02"):
        make_export(root / cv.EXPORT / s, speaker=s)
        assert cv.main(["make", "--export", str(root / cv.EXPORT / s), "--speaker", s, "--folds", "4", "--root", str(root)]) == 0
        assert cv.main(["check", "--speaker", s, "--folds", "4", "--root", str(root)]) == 0
    for arm, every in (("control", 3), ("better", 20)):
        for s in ("M01", "M02"):
            ids = cv.fold_ids(s, 4, root)
            rows = {r["id"]: r for r in cv.read_rows(root / cv.EXPORT / f"cv-{s}-all/test.jsonl")}
            for k in range(4):
                hyp_file(root / f"hyp/{arm}-{s}-f{k}.jsonl", [rows[i] for i in ids[k]], every)
    code = cv.main(["select", "--speaker", "M01", "M02", "--folds", "4", "--control", "control",
                    "--arm", f"control={root}/hyp/control-{{speaker}}-f{{fold}}.jsonl", "--arm", f"better={root}/hyp/better-{{speaker}}-f{{fold}}.jsonl",
                    "--draws", "500", "--min-gain-points", "1", "--out", str(root / "sel.json"), "--root", str(root)])
    assert code == 0
    result = json.loads((root / "sel.json").read_text())
    assert result["selected"] == "better"
    assert "selected: better" in capsys.readouterr().out
    assert cv.main(["merge", "--speaker", "M01", "--folds", "4", "--arm", f"control={root}/hyp/control-M01-f{{fold}}.jsonl",
                    "--out", str(root / "oof"), "--root", str(root)]) == 0
    assert (root / "oof/control-M01.jsonl").is_file()
