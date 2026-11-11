import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import lm_rescore as lr  # noqa: E402

A, B, C = "The boy ran to the store.", "The boy ran to the stone sore.", ""


def cand(text, acoustic, *sources):
    return {"text": text, "sources": list(sources), "n_words": len(text.split()), "acoustic": acoustic}


def row(clip_id, *cands):
    return {"id": clip_id, "kind": "sentence", "candidates": list(cands)}


LM = {A: -18.0, B: -27.0}


def test_the_language_model_breaks_a_close_acoustic_tie_the_worked_example():
    r = row("c1", cand(B, -40.2, "greedy"), cand(A, -41.0, "beam:2"))
    assert lr.choose(r, 0.0, 0.0, lr.NO_LM)["text"] == B  # on sound alone the wrong one wins by 0.8
    assert lr.choose(r, 0.7, 3.0, LM)["text"] == A  # -41 - 12.6 + 18 = -35.6 beats -40.2 - 18.9 + 21 = -38.1
    assert lr.candidate_total(cand(A, -41.0, "g"), 0.7, 3.0, LM) == pytest.approx(-35.6)


def test_ties_keep_the_earlier_candidate_and_unscored_candidates_are_skipped():
    r = row("c1", cand("one", -5.0, "greedy"), cand("two", -5.0, "beam:1"), cand("three", None, "slow"))
    assert lr.choose(r, 0.0, 0.0, lr.NO_LM)["text"] == "one"
    only_none = row("c2", cand("x", None, "greedy"), cand("y", None, "hop"))
    assert lr.choose(only_none, 0.7, 3.0, {"x": -1.0, "y": -1.0})["text"] == "x"


def test_empty_text_scores_zero_with_the_lm_and_the_bonus_is_what_keeps_it_down():
    r = row("c1", cand(C, -3.0, "greedy"), cand(A, -9.0, "beam:1"))
    assert lr.lm_score(LM, C) == 0.0
    assert lr.choose(r, 0.7, 0.0, LM)["text"] == C  # -3 beats -9 - 12.6
    assert lr.choose(r, 0.7, 3.0, LM)["text"] == C  # -9 - 12.6 + 18 = -3.6 still loses to -3.0
    assert lr.choose(r, 0.7, 4.0, LM)["text"] == A  # -9 - 12.6 + 24 = 2.4 beats -3.0


def test_fit_weights_picks_the_fewest_wrong_words_and_breaks_ties_toward_the_audio():
    ref_a = lr.normalize(A)
    train = [(row("c1", cand(B, -40.2, "greedy"), cand(A, -41.0, "beam:2")), ref_a)]
    lam, bonus = lr.fit_weights(train, LM, lambdas=(0.0, 0.3, 0.7), bonuses=(0.0, 3.0))
    # lambda 0.3: -41 - 5.4 + 0 = -46.4 vs -40.2 - 8.1 = -48.3 -> A wins already, so the smallest lambda that works is 0.3, bonus 0
    assert (lam, bonus) == (0.3, 0.0)
    unhelpful = [(row("c1", cand(A, -1.0, "greedy")), ref_a)]
    assert lr.fit_weights(unhelpful, LM, lambdas=(0.3, 0.7), bonuses=(0.0, 3.0)) == (0.3, 0.0)


def test_leave_one_out_fits_each_speaker_on_the_others_only():
    ref_a = lr.normalize(A)
    pools = {"M01": {"a": row("a", cand(B, -40.2, "greedy"), cand(A, -41.0, "beam:2"))},
             "M02": {"b": row("b", cand(A, -40.2, "greedy"), cand(B, -41.0, "beam:2"))},
             "M04": {"c": row("c", cand(B, -40.2, "greedy"), cand(A, -41.0, "beam:2"))}}
    refs = {"M01": {"a": ref_a}, "M02": {"b": ref_a}, "M04": {"c": ref_a}}
    weights = lr.leave_one_out(pools, refs, LM, lambdas=(0.0, 0.7), bonuses=(0.0,))
    assert weights["M02"] == (0.7, 0.0)  # fitted on M01 and M04, where the LM is needed
    assert weights["M01"] == (0.7, 0.0)  # M02 is right either way, M04 needs the LM
    with pytest.raises(ValueError):
        lr.leave_one_out({"M01": pools["M01"]}, refs, LM)


def test_count_and_compare_give_exact_sentences_and_word_changes_with_intervals():
    refs = {"M01": {"a": "the boy ran to the store", "b": "yes"}}
    kinds = {"M01": {"a": "sentence", "b": "word"}}
    greedy = lr.count({"M01": {"a": B, "b": ""}}, refs, kinds)
    better = lr.count({"M01": {"a": A, "b": "Yes."}}, refs, kinds)
    g, b = greedy.pooled(), better.pooled()
    assert (g["wrong"], g["words"], g["exact_sentences"], g["sentence_clips"], g["empty"]) == (3, 7, 0, 1, 1)
    assert (b["wrong"], b["exact_sentences"], b["exact_words"], b["empty"]) == (0, 1, 1, 0)
    change = lr.compare(greedy, better)
    assert change["wrong_change"] == -3 and change["exact_sentences_change"] == 1
    assert change["gain_points"] == pytest.approx(300 / 7)
    assert change["exact_sentences_ci"] == [1.0, 1.0]  # one sentence clip, so every resample moves by one
    assert greedy.by_speaker()["M01"]["wrong"] == 3


def test_paired_interval_is_deterministic_and_brackets_the_mean_change():
    base = np.array([0, 0, 0, 1, 0, 0, 1, 0])
    new = np.array([1, 0, 1, 1, 0, 1, 1, 0])
    low, high = lr.paired_count_interval(base, new)
    assert low <= 3 <= high and (low, high) == tuple(lr.paired_count_interval(base, new))
    with pytest.raises(ValueError):
        lr.paired_count_interval(base, new[:3])


def test_oracle_and_greedy_candidates():
    r = row("c1", cand(B, -1.0, "greedy"), cand(A, -9.0, "beam:3"))
    assert lr.oracle_candidate(r, lr.normalize(A))["text"] == A
    assert lr.greedy_candidate(r)["text"] == B


def test_parse_pairs_and_single_pool_split_by_speaker(tmp_path):
    assert lr.parse_pairs(["M01=a.jsonl", "M02=b.jsonl"], "pool") == {"M01": "a.jsonl", "M02": "b.jsonl"}
    assert lr.parse_pairs(["only.jsonl"], "pool") == {"": "only.jsonl"}
    with pytest.raises(ValueError):
        lr.parse_pairs(["a.jsonl", "b.jsonl"], "pool")
    manifest = {"x": {"speaker": "F01", "text": "a"}, "y": {"speaker": "M01", "text": "b"}, "z": {"speaker": "M01", "text": "c"}}
    pool = {"x": row("x", cand("a", -1, "greedy")), "z": row("z", cand("c", -1, "greedy"))}
    pools, manifests = lr.split_by_speaker(manifest, pool)
    assert set(pools) == {"F01", "M01"} and list(pools["M01"]) == ["z"] and set(manifests["M01"]) == {"y", "z"}
    with pytest.raises(ValueError):
        lr.split_by_speaker({"x": {"text": "a"}}, pool)


def test_roster_tags_are_unique_and_selectable():
    assert len({m.tag for m in lr.ROSTER}) == len(lr.ROSTER)
    assert [m.tag for m in lr.select_models("qwen2.5-0.5b,gpt2")] == ["qwen2.5-0.5b", "gpt2"]
    assert lr.select_models("all") == list(lr.ROSTER)
    with pytest.raises(KeyError):
        lr.select_models("nope")


def test_read_pool_rejects_duplicates_and_empty_candidate_lists(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text(json.dumps(row("a", cand("x", -1, "greedy"))) + "\n" + json.dumps(row("a", cand("y", -1, "greedy"))) + "\n")
    with pytest.raises(ValueError):
        lr.read_pool(path)
    path.write_text(json.dumps({"id": "a", "kind": "sentence", "candidates": []}) + "\n")
    with pytest.raises(ValueError):
        lr.read_pool(path)


def test_cached_scores_reuse_the_cache_without_loading_a_model(tmp_path, monkeypatch):
    entry = lr.BY_TAG["gpt2"]
    cache = tmp_path / "lm-cache"
    cache.mkdir()
    (cache / "gpt2.json").write_text(json.dumps({"model": entry.name, "scores": {A: -18.0}}))
    monkeypatch.setattr(lr, "load_lm", lambda *a, **k: pytest.fail("the model must not load when every sentence is cached"))
    scores, info = lr.cached_scores(entry, [A, C], cache, "cpu", lambda *a: None)
    assert scores[A] == -18.0 and info["scored"] == 0
    (cache / "gpt2.json").write_text(json.dumps({"model": "someone/else", "scores": {}}))
    with pytest.raises(ValueError):
        lr.cached_scores(entry, [A], cache, "cpu", lambda *a: None)


def test_sentence_logprobs_with_a_tiny_fake_model():
    torch = pytest.importorskip("torch")

    class Tok:
        bos_token_id, eos_token_id, pad_token_id = None, 7, None

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 7 for c in text]}

    class Model:
        def __call__(self, input_ids, attention_mask):
            # Uniform over 8 symbols: every token costs log(1/8), so a text of n tokens scores n * log(1/8).
            logits = torch.zeros(*input_ids.shape, 8)
            return type("Out", (), {"logits": logits})()

    scores = lr.sentence_logprobs(Tok(), Model(), ["ab", "abcd"], "cpu", batch=2)
    assert scores[0] == pytest.approx(2 * np.log(1 / 8)) and scores[1] == pytest.approx(4 * np.log(1 / 8))
