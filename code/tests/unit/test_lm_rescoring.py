import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import lm_pool  # noqa: E402
import lm_report  # noqa: E402
import lm_score  # noqa: E402


def cand(norm: str, nll: float | None, views=("greedy",)) -> dict:
    return {"norm": norm, "text": norm, "nll": nll, "views": list(views), "words": len(norm.split()), "lm_text": lm_pool.lm_text(norm)}


def row(clip_id: str, speaker: str, ref: str, cands: list[dict], kind: str = "sentence") -> dict:
    return {"id": clip_id, "speaker": speaker, "kind": kind, "ref": ref, "duration_s": 3.0, "frames": 40, "nll_empty": 9.0, "candidates": cands}


ROWS = [
    row("a1", "M01", "the boy ran to the store", [cand("the boy ran to the stone sore", 10.0), cand("the boy ran to the store", 11.0, ("beam4",))]),
    row("a2", "M01", "i want a glass of water", [cand("i want a glass of water", 4.0), cand("i want a grass of water", 4.5, ("beam4",))]),
    row("b1", "M02", "we went home early", [cand("we went home curly", 6.0), cand("we went home early", 6.5, ("tempo1.15",)),
                                          cand("we went", 5.0, ("beam8",))]),
    row("b2", "M02", "she is at school", [cand("she is at school", 3.0), cand("sheep is at school", 3.2, ("pitch0.95",))]),
    row("c1", "M04", "open the door", [cand("open a door", 5.0), cand("open the door", 5.5, ("beam4",))]),
    row("c2", "M04", "tomato", [cand("potato", 2.0), cand("tomato", 2.2, ("beam4",))], kind="word"),
]
LOGP = {
    "The boy ran to the stone sore.": -45.0, "The boy ran to the store.": -25.0, "I want a glass of water.": -20.0,
    "I want a grass of water.": -33.0, "We went home curly.": -30.0, "We went home early.": -18.0, "We went.": -9.0,
    "She is at school.": -12.0, "Sheep is at school.": -22.0, "Open a door.": -14.0, "Open the door.": -9.0, "Potato.": -9.0, "Tomato.": -9.5,
}


def write_pool(path: Path, rows=ROWS) -> str:
    path.write_text(json.dumps({"format": "willspeak-lm-pool-v1", "views": ["greedy", "beam4"], "rows": rows}), encoding="utf-8")
    return str(path)


def write_scores(path: Path, tag: str = "qwen25-05b-q8", logp=LOGP) -> str:
    lm = lm_score.get(tag)
    data = {"format": "willspeak-lm-scores-v1", "model": lm.__dict__, "runtime": "test", "file_mb": lm.mb, "peak_rss_mb": 600.0, "threads": 4,
            "tokens_per_second": 100.0, "load_seconds": 0.5, "machine": {"node": "test"},
            "texts": {text: [value, len(text.split()) + 1, 0.01] for text, value in logp.items()}}
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_views_parse_put_greedy_first_and_refuse_nonsense():
    views = lm_pool.parse_views("beam4, tempo1.15 pitch0.95")
    assert [v[0] for v in views] == ["greedy", "beam4", "tempo1.15", "pitch0.95"]
    assert views[1][1:] == ("beam", 4.0) and views[2][1:] == ("tempo", 1.15)
    assert [v[0] for v in lm_pool.parse_views(lm_pool.DEFAULT_VIEWS)][0] == "greedy"
    for bad in ("beam1", "beam4.5", "tempo1", "pitch3", "loud2", "greedy2", "beam4 beam4"):
        with pytest.raises(ValueError):
            lm_pool.parse_views(bad)


def test_the_language_model_sees_one_form_of_every_answer():
    assert lm_pool.lm_text("the boy ran") == "The boy ran."
    assert lm_pool.lm_text("so i went") == "So I went."
    assert lm_pool.lm_text("i am here") == "I am here."
    assert lm_pool.lm_text("") == ""


def test_answers_that_differ_only_in_form_are_one_candidate_with_the_better_audio_score():
    found = {"The boy ran.": ["greedy"], "the boy ran": ["beam4"], "The boy can.": ["beam4", "tempo1.15"], "": ["pitch0.95"]}
    out = lm_pool.group_candidates(found, [7.0, 5.5, 6.0, 99.0], empty=8.0)
    assert [c["norm"] for c in out] == ["the boy ran", "the boy can", ""]
    assert out[0]["nll"] == 5.5 and out[0]["text"] == "the boy ran" and out[0]["views"] == ["greedy", "beam4"]
    assert out[2]["nll"] == 8.0 and out[2]["words"] == 0 and out[2]["lm_text"] == ""
    assert lm_pool.group_candidates({"too long": ["beam4"]}, [float("inf")], empty=3.0)[0]["nll"] is None


def test_far_worse_answers_are_dropped_but_never_the_first_decode():
    cands = [cand("first", 30.0), cand("best", 5.0, ("beam4",)), cand("close", 14.9, ("beam8",)), cand("far", 15.1, ("beam8",)),
             cand("no alignment", None, ("beam8",))]
    kept, dropped = lm_pool.prune(cands, 10.0)
    assert [c["norm"] for c in kept] == ["first", "best", "close"] and dropped == 2
    assert lm_pool.prune(cands, 0.0) == (cands, 0)


def test_pitch_view_changes_length_by_the_inverse_factor():
    audio = np.sin(np.linspace(0, 200, 16000)).astype("float32")
    longer = lm_pool.stretch(audio, 1 / 0.95)
    assert longer.dtype == np.float32 and abs(len(longer) - 16000 / 0.95) <= 2


def test_catalog_tags_are_unique_and_every_group_names_real_models():
    assert len({m.tag for m in lm_score.CATALOG}) == len(lm_score.CATALOG)
    grouped = [t for tags in lm_score.GROUPS.values() for t in tags.split()]
    assert len(set(grouped)) == len(grouped) and all(lm_score.get(t) for t in grouped)
    assert lm_score.DEFAULT_TAGS == lm_score.GROUPS["pi"] and "qwen25-05b-q8" in lm_score.GROUPS["pi"].split()
    assert all(lm_score.get(t).mb <= 1700 for t in lm_score.GROUPS["pi"].split())
    assert all(1700 < lm_score.get(t).mb <= 3200 for t in lm_score.GROUPS["large"].split())
    assert all(m.file.endswith(".gguf") for m in lm_score.CATALOG if m.backend == "gguf")
    assert [m.tag for m in lm_score.CATALOG if m.backend == "hf"] == ["qwen25-05b-fp32"]
    with pytest.raises(KeyError):
        lm_score.get("gpt-9")


def test_group_names_expand_to_their_models_once_each():
    pi = lm_score.GROUPS["pi"].split()
    assert lm_score.expand("") == pi and lm_score.expand("pi qwen25-05b-q8") == pi
    assert lm_score.expand(["qwen35-4b-q4", "large"])[0] == "qwen35-4b-q4"
    assert sorted(lm_score.expand("large qwen35-4b-q4")) == sorted(lm_score.GROUPS["large"].split())
    assert lm_score.group_of("qwen38-27b-q4") == "ceiling" and lm_score.group_of("smollm2-135m-q8") == "extra"
    with pytest.raises(KeyError):
        lm_score.expand("pi gpt-9")


class ScriptedScorer(lm_score.Scorer):
    vocab = 1000

    def __init__(self, means: dict[tuple, float], bos=1, eos=2):
        self.means, self.bos, self.eos = means, bos, eos

    def specials(self):
        return self.bos, self.eos

    def tokens(self, text: str) -> list[int]:
        return [9] if text == "\n" else [100 + i for i, _word in enumerate(text.split())]

    def piece(self, ids: list[int]) -> str:
        return "".join({1: "<s>", 2: "</s>", 9: "\n"}[i] for i in ids)

    def logps(self, seq: list[int], first: int) -> list[float]:
        return [self.means[tuple(seq[:first])]] * (len(seq) - first)


def test_what_goes_before_the_first_word_is_the_simplest_choice_that_reads_well():
    plain = ScriptedScorer({(1,): -2.3, (2,): -7.0, (9,): -2.4, (1, 9): -2.2, (2, 9): -6.0})
    found = lm_score.settle(plain)
    assert found["before_the_first_word"] == "start token" and plain.start == [1] and found["text"] == "<s>"
    assert found["tried"]["end token"] == -7.0 and found["check_mean_logp"] == -2.3 and found["chance"] == pytest.approx(-6.91, abs=0.01)
    spoiled = ScriptedScorer({(1,): -6.7, (9,): -3.1, (1, 9): -7.4}, eos=1)
    assert lm_score.settle(spoiled)["before_the_first_word"] == "new line" and spoiled.start == [9]
    assert list(lm_score.settle(spoiled)["tried"]) == ["start token", "new line", "start token, new line"]
    logp, count = spoiled.score("three short words")
    assert count == 3 and logp == pytest.approx(-9.3)


def test_a_model_the_runtime_cannot_read_is_refused():
    for value in (-9.5, float("nan")):
        broken = ScriptedScorer({(1,): value, (2,): value, (9,): value, (1, 9): value, (2, 9): value})
        with pytest.raises(ValueError, match="does not read the model correctly"):
            lm_score.settle(broken)


def test_log_softmax_pick_matches_the_definition():
    logits = [2.0, 0.5, -1.0]
    total = math.log(sum(math.exp(x) for x in logits))
    assert lm_score.log_softmax_pick(logits, 1) == pytest.approx(0.5 - total)
    assert lm_score.log_softmax_pick(np.array([1000.0, 0.0]), 0) == pytest.approx(0.0, abs=1e-9)


def test_scoring_collects_each_distinct_sentence_once(tmp_path):
    rows = lm_score.read_pools([write_pool(tmp_path / "p.json"), write_pool(tmp_path / "q.json", ROWS[:1] + [row("z", "M01", "hello", [cand("", 1.0)])])])
    texts = lm_score.pool_texts(rows)
    assert len(texts) == len(set(texts)) == len(LOGP) and "" not in texts
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
        lm_score.read_pools([str(tmp_path / "bad.json")])


def test_memorisation_prompts_are_first_halves_of_distinct_sentences():
    prompts = lm_score.halves(ROWS + ROWS)
    assert ("The boy ran", ["to", "the", "store"]) in prompts and ("I want a", ["glass", "of", "water"]) in prompts
    assert len(prompts) == 4 and all(len(start.split()) + len(rest) >= 4 for start, rest in prompts)
    assert lm_score.plain_words(" over the Lazy dog. And more")[:4] == ["over", "the", "lazy", "dog"]


class FakeScorer:
    version, architecture = "fake 1.0", "fake"

    def __init__(self, start=(1,)):
        self.asked: list[str] = []
        self.settled = {"before_the_first_word": "start token", "text": "<s>", "ids": list(start), "check_mean_logp": -2.0}

    def score(self, text: str) -> tuple[float, int]:
        self.asked.append(text)
        return -float(len(text)), len(text.split())

    def complete(self, text: str, max_tokens: int) -> str:
        self.asked.append(text)
        return " to the store and back" if text == "The boy ran" else " something else"


def run_lm_score(monkeypatch, tmp_path, scorer, *argv) -> int:
    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"not a model")
    monkeypatch.setattr(lm_score, "load", lambda lm, root, threads, ctx: scorer)
    monkeypatch.setattr(lm_score, "model_path", lambda lm, root=None: weights)
    monkeypatch.setattr(sys, "argv", ["lm_score.py", *argv])
    return lm_score.main()


def test_scores_are_extended_when_pools_are_added_and_never_silently_replaced(tmp_path, monkeypatch):
    first, both = write_pool(tmp_path / "a.json", ROWS[:2]), write_pool(tmp_path / "b.json", ROWS[2:])
    out = tmp_path / "scores" / "qwen25-05b-q8.json"
    scorer = FakeScorer()
    assert run_lm_score(monkeypatch, tmp_path, scorer, "score", "--model", "qwen25-05b-q8", "--pool", first, "--out", str(out)) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["format"] == "willspeak-lm-scores-v1" and len(data["texts"]) == 4 and data["threads"] == 4 and data["start"]["text"] == "<s>"
    assert data["texts"]["The boy ran to the store."][:2] == [-25.0, 6]
    with pytest.raises(FileExistsError):
        run_lm_score(monkeypatch, tmp_path, scorer, "score", "--model", "qwen25-05b-q8", "--pool", first, "--out", str(out))
    scorer.asked.clear()
    assert run_lm_score(monkeypatch, tmp_path, scorer, "score", "--model", "qwen25-05b-q8", "--pool", first, both, "--out", str(out), "--extend") == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["texts"]) == len(LOGP) and data["scored_now"] == len(LOGP) - 4
    assert "The boy ran to the store." not in scorer.asked and "Open the door." in scorer.asked
    scorer.asked.clear()
    assert run_lm_score(monkeypatch, tmp_path, scorer, "score", "--model", "qwen25-05b-q8", "--pool", first, both, "--out", str(out), "--extend") == 0
    assert scorer.asked == []
    with pytest.raises(ValueError):
        run_lm_score(monkeypatch, tmp_path, scorer, "score", "--model", "qwen25-05b-q4", "--pool", first, "--out", str(out), "--extend")
    more = write_pool(tmp_path / "c.json", [row("z", "M01", "hello there", [cand("hello there", 1.0)])])
    with pytest.raises(ValueError, match="before the first word"):
        run_lm_score(monkeypatch, tmp_path, FakeScorer(start=(9,)), "score", "--model", "qwen25-05b-q8", "--pool", more, "--out", str(out), "--extend")


def test_pending_says_whether_a_model_still_has_something_to_score(tmp_path, monkeypatch):
    pool, later = write_pool(tmp_path / "a.json", ROWS[:2]), write_pool(tmp_path / "b.json", ROWS[2:])
    scores, done = tmp_path / "s.json", tmp_path / "d.json"
    assert lm_score.pending([pool], scores, done)
    assert run_lm_score(monkeypatch, tmp_path, FakeScorer(), "score", "--model", "qwen25-05b-q8", "--pool", pool, "--out", str(scores)) == 0
    assert lm_score.pending([pool], scores, done)
    assert run_lm_score(monkeypatch, tmp_path, FakeScorer(), "complete", "--model", "qwen25-05b-q8", "--pool", pool, "--out", str(done)) == 0
    assert not lm_score.pending([pool], scores, done) and lm_score.pending([pool, later], scores, done)


def test_fetch_refuses_a_file_the_disk_cannot_hold_and_release_removes_only_that_model(tmp_path, monkeypatch, capsys):
    called: list[str] = []
    monkeypatch.setattr(lm_score, "fetch", lambda lm, root=None: called.append(lm.tag))
    monkeypatch.setattr(lm_score, "free_mb", lambda path: 2300.0)
    monkeypatch.setattr(sys, "argv", ["lm_score.py", "fetch", "--dir", str(tmp_path), "--models", "lfm25-230m-q4", "qwen25-05b-q8"])
    assert lm_score.main() == 1 and called == ["lfm25-230m-q4"]
    assert "qwen25-05b-q8: FAILED no space left" in capsys.readouterr().out
    for tag in ("lfm25-230m-q4", "qwen25-05b-q8"):
        (tmp_path / tag).mkdir()
        (tmp_path / tag / "weights.gguf").write_bytes(b"x" * 10)
    monkeypatch.setattr(sys, "argv", ["lm_score.py", "release", "--dir", str(tmp_path), "--models", "lfm25-230m-q4", "gemma3-1b-q4"])
    assert lm_score.main() == 0
    assert not (tmp_path / "lfm25-230m-q4").exists() and (tmp_path / "qwen25-05b-q8" / "weights.gguf").is_file()


def test_the_memorisation_check_counts_sentences_finished_word_for_word(tmp_path, monkeypatch):
    pool = write_pool(tmp_path / "a.json")
    out = tmp_path / "done.json"
    assert run_lm_score(monkeypatch, tmp_path, FakeScorer(), "complete", "--model", "qwen25-05b-q8", "--pool", pool, "--out", str(out)) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["prompts"] == 4 and data["finished"] == 1 and [r["start"] for r in data["rows"] if r["finished"]] == ["The boy ran"]
    scorer = FakeScorer()
    assert run_lm_score(monkeypatch, tmp_path, scorer, "complete", "--model", "qwen25-05b-q8", "--pool", pool, "--out", str(out), "--extend") == 0
    assert scorer.asked == []


def test_the_language_model_changes_the_choice_only_when_its_weight_is_on():
    clips, _info = lm_report.read_clips([write_pool(Path(TMP) / "one.json")])
    lms = lm_report.lm_values(clips, {t: [v, 1, 0.0] for t, v in LOGP.items()})
    first = clips[0]
    assert lm_report.choose(first, lms[0], 0.0, 0.0) == 0
    assert lm_report.choose(first, lms[0], 0.7, 3.0) == 1
    short = clips[2]
    assert lm_report.choose(short, lms[2], 0.0, 0.0) == 2
    assert lm_report.choose(short, lms[2], 0.0, 2.0) == 0
    assert lm_report.choose(short, lms[2], 0.7, 3.0) == 2
    assert lm_report.choose(short, lms[2], 0.7, 6.0) == 1


def test_weights_are_chosen_without_the_held_out_speaker(tmp_path):
    clips, _info = lm_report.read_clips([write_pool(tmp_path / "p.json")])
    lms = lm_report.lm_values(clips, {t: [v, 1, 0.0] for t, v in LOGP.items()})
    folds, unit = lm_report.fold_of(clips)
    assert unit == "speaker" and sorted(set(folds)) == ["M01", "M02", "M04"]
    picks, chosen = lm_report.cross_fit(clips, lms, folds, lm_report.GRID_WEIGHT, lm_report.GRID_BONUS)
    assert set(chosen) == {"M01", "M02", "M04"}
    assert chosen["M02"] == [0.2, 0.0]
    sentences = [(c, p) for c, p in zip(clips, picks) if c.kind == "sentence"]
    assert sum(int(c.right[p]) for c, p in sentences) == 4
    spoiled = [lm.copy() for lm in lms]
    spoiled[0][:] = [0.0, -500.0]
    spoiled[1][:] = [-500.0, 0.0]
    _again, chosen_again = lm_report.cross_fit(clips, spoiled, folds, lm_report.GRID_WEIGHT, lm_report.GRID_BONUS)
    assert chosen_again["M01"] == chosen["M01"]
    assert chosen_again["M02"] != chosen["M02"]


def test_one_speaker_is_split_by_prompt_halves(tmp_path):
    rows = [dict(r, speaker="M02") for r in ROWS]
    clips, _info = lm_report.read_clips([write_pool(tmp_path / "p.json", rows)])
    folds, unit = lm_report.fold_of(clips)
    assert unit == "half of the prompts" and set(folds) <= {"half 1", "half 2"}
    same = [f for c, f in zip(clips, folds) if c.ref == "open the door"]
    assert len(set(same)) == 1


def test_report_counts_whole_sentences_and_the_pool_ceiling(tmp_path, capsys, monkeypatch):
    pool = write_pool(tmp_path / "pool.json")
    scores = write_scores(tmp_path / "qwen25-05b-q8.json")
    out = tmp_path / "table.json"
    monkeypatch.setattr(sys, "argv", ["lm_report.py", "--system", "stock, test=" + pool, "--scores", scores, "--out", str(out)])
    assert lm_report.main() == 0
    table = json.loads(out.read_text(encoding="utf-8"))
    rows = {r["label"]: r for r in table["systems"]["stock, test"]["rows"]}
    assert rows[lm_report.FIRST]["sentence"] == {"clips": 5, "right": 2, "wrong_words": 4, "words": 23}
    assert rows[lm_report.BEST]["sentence"]["right"] == 5 and rows[lm_report.BEST]["word"]["right"] == 1
    lm = rows["qwen25-05b-q8"]
    assert lm["sentence"]["right"] == 4 and lm["against_first_decode"]["gained"] == 2 and lm["against_first_decode"]["lost"] == 0
    assert lm["against_first_decode"]["right_gain"] == pytest.approx(0.4)
    assert lm["against_first_decode"]["right_gain_ci"][0] <= 0.4 <= lm["against_first_decode"]["right_gain_ci"][1]
    assert lm["preset"]["weights"] == [0.7, 3.0] and lm["preset"]["sentence"]["right"] == 4
    assert rows[lm_report.AUDIO]["sentence"]["right"] == 2 and set(rows[lm_report.AUDIO]["weights"]) == {"M01", "M02", "M04"}
    assert table["models"][0]["tag"] == "qwen25-05b-q8" and table["models"][0]["seconds_per_clip_median"] > 0
    shown = capsys.readouterr().out
    assert "sentences right" in shown and "best answer in the pool" in shown
    with pytest.raises(FileExistsError):
        lm_report.main()


def test_report_refuses_scores_that_miss_a_candidate(tmp_path, monkeypatch):
    pool = write_pool(tmp_path / "pool.json")
    scores = write_scores(tmp_path / "s.json", logp={k: v for k, v in LOGP.items() if "store" not in k})
    monkeypatch.setattr(sys, "argv", ["lm_report.py", "--system", "stock=" + pool, "--scores", scores, "--out", str(tmp_path / "t.json")])
    with pytest.raises(ValueError, match="no language-model score"):
        lm_report.main()


def test_scripts_print_help_without_torch_or_llama():
    for script, sub in (("lm_pool.py", []), ("lm_report.py", []), ("lm_score.py", ["score"]), ("lm_score.py", ["list"]),
                        ("lm_score.py", ["fetch"]), ("lm_score.py", ["check"]), ("lm_score.py", ["release"])):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "asr" / script), *sub, "--help"], capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-400:]


def load_lab():
    spec = importlib.util.spec_from_file_location("lab_cli_lm", ROOT / "lab.py")
    lab = importlib.util.module_from_spec(spec)
    sys.modules["lab_cli_lm"] = lab
    spec.loader.exec_module(lab)
    return lab


def plan(tmp_path, **overrides):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    for name in ("scripts", "src"):
        (app / name).symlink_to(ROOT / name)
    commands, notes = lab.plan(lambda: lab.Lab(app, lab.Settings(app, overrides, env={}), dry=True, stamp="TEST", quiet=True), ["36"])
    assert notes == {"36": []}
    return lab, app, commands


def test_stage_36_pools_three_systems_scores_every_model_and_reports_once(tmp_path):
    lab, app, commands = plan(tmp_path, trim_models="off")
    pools = [c for c in commands if "scripts/asr/lm_pool.py" in c]
    outs = [c[c.index("--out") + 1].split("/")[-1] for c in pools]
    assert len(pools) == 1 + 3 * (3 + 1 + 4) and len(set(outs)) == len(outs)
    assert {"stock-severe.json", "stock-M01-train.json", "stock-M04-test.json", "prototype-M02-test.json", "prototype-M04-f3.json"} <= set(outs)
    assert sum("--adapter" in c for c in pools) == 3 * 5 and all("--views" in c for c in pools)
    trained = [c[c.index("--out") + 1] for c in commands if "scripts/asr/train.py" in c]
    assert sum("big-c-pre-" in t for t in trained) == 3 and sum("lab-36-pre-ad8-" in t for t in trained) == 12 and sum("lab-pre-ad8-" in t for t in trained) == 3
    tags = lm_score.GROUPS["pi"].split()
    lm = [c for c in commands if "scripts/asr/lm_score.py" in c]
    scores = [c for c in lm if "score" in c]
    assert [c[c.index("--model") + 1] for c in scores] == tags
    assert all(c[c.index("--threads") + 1] == "8" and "--extend" in c and len(c[c.index("--pool") + 1: c.index("--threads")]) == len(pools)
               for c in scores)
    assert [c[2:] for c in lm][:3] == [["fetch", "--models", tags[0]], scores[0][2:], ["complete", *scores[0][3:-1], scores[0][-1].replace("scores", "completions")]]
    assert len([c for c in lm if "complete" in c]) == len(tags) and not [c for c in lm if "release" in c]
    assert commands.index(lm[0]) > max(commands.index(c) for c in pools)
    reports = [c for c in commands if "scripts/asr/lm_report.py" in c]
    assert len(reports) == 1 and reports[0].count("--system") == 5
    folded = next(x for x in reports[0] if x.startswith("stage P's recipe, every labelled clip held out="))
    assert folded.count(",") + 1 - 2 == 3 * 5 - 1
    problems, checked = lab.flag_problems(app, commands)
    assert not problems, problems


def test_stage_36_stock_alone_needs_no_training(tmp_path):
    _lab, _app, commands = plan(tmp_path, lm_systems="stock", lm_models="qwen25-05b-q8 qwen25-05b-fp32", trim_models="off")
    assert not [c for c in commands if "scripts/asr/train.py" in c]
    assert len([c for c in commands if "scripts/asr/lm_pool.py" in c]) == 1
    assert [c[c.index("--model") + 1] for c in commands if "score" in c] == ["qwen25-05b-q8", "qwen25-05b-fp32"]


def test_stage_36_frees_each_model_after_its_scores_when_disk_is_short(tmp_path):
    lab, app, commands = plan(tmp_path, lm_systems="stock", lm_models="large qwen25-05b-q8", trim_models="on")
    lm = [c[2:] for c in commands if "scripts/asr/lm_score.py" in c]
    tags = [*lm_score.GROUPS["large"].split(), "qwen25-05b-q8"]
    assert [c[0] for c in lm] == ["fetch", "score", "complete", "release"] * len(tags)
    assert [c[2] for c in lm if c[0] in ("fetch", "release")] == [t for t in tags for _ in range(2)]
    problems, _checked = lab.flag_problems(app, commands)
    assert not problems, problems


def test_stage_36_refuses_unknown_systems_and_models(tmp_path):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    for name in ("scripts", "src"):
        (app / name).symlink_to(ROOT / name)
    for bad in ({"lm_systems": "everything"}, {"lm_models": "gpt-9"}, {"lm_models": "pi huge"}):
        session = lab.Lab(app, lab.Settings(app, bad, env={}), dry=True, stamp="TEST", quiet=True)
        with pytest.raises((ValueError, KeyError)):
            lab.stage_36(session)


TMP = None


@pytest.fixture(autouse=True)
def _scratch(tmp_path):
    global TMP
    TMP = str(tmp_path)


def test_each_pretrained_base_frees_its_checkpoints_before_the_next_one_trains(tmp_path, monkeypatch):
    lab = load_lab()
    app = tmp_path / "app"
    app.mkdir()
    session = lab.Lab(app, lab.Settings(app, {}, env={}), dry=False, stamp="TEST", quiet=True)
    monkeypatch.setattr(session, "ensure_severe_exports", lambda: True)
    monkeypatch.setattr(lab, "room_for_full_training", lambda _lab, folder: len(list((app / "models").glob("*/ckpt"))) == 0)
    import lab_space

    monkeypatch.setattr(lab_space, "nemo_intact", lambda path: True)
    trained: list[str] = []

    def train(export, name, recipe, extra=None, **kw):
        folder = app / "models" / f"parakeet-ft-{name}"
        (folder / "ckpt").mkdir(parents=True)
        (folder / "ckpt" / "last.ckpt").write_bytes(b"x" * 100)
        (folder / "parakeet-ft.nemo").write_bytes(b"model")
        (folder / "run.json").write_text('{"status": "complete"}', encoding="utf-8")
        trained.append(name)
        return True

    monkeypatch.setattr(session, "train", train)
    assert lab.ensure_pretrained_models(session)
    assert trained == [f"big-c-pre-{s}" for s in session.spk]
    assert not list((app / "models").glob("*/ckpt")) and len(list((app / "models").glob("*/parakeet-ft.nemo"))) == len(session.spk)
