import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import candidate_pool as cp  # noqa: E402


def test_merge_keeps_the_first_spelling_and_every_source_of_a_repeated_answer():
    answers = [("greedy", "The boy ran."), ("beam:1", "the boy ran"), ("beam:2", "The boy ran to the store."), ("slow", ""), ("hop", "The boy ran.")]
    cands = cp.merge_candidates(answers)
    assert [c["text"] for c in cands] == ["The boy ran.", "The boy ran to the store.", ""]
    assert cands[0]["sources"] == ["greedy", "beam:1", "hop"] and cands[0]["n_words"] == 3
    assert cands[2]["n_words"] == 0 and cands[2]["sources"] == ["slow"]


def test_slow_ratio_turns_a_speed_into_resampling_factors():
    assert cp.slow_ratio(0.95) == (20, 19)
    assert cp.slow_ratio(0.9) == (10, 9)
    for bad in (1.0, 0.2, 2.0):
        with pytest.raises(ValueError):
            cp.slow_ratio(bad)


def test_slow_audio_lasts_longer_by_the_right_factor():
    pytest.importorskip("scipy")
    audio = np.sin(np.linspace(0, 200 * np.pi, 16000)).astype("float32")
    slow = cp.slow_audio(audio, 0.95)
    assert slow.dtype == np.float32 and abs(len(slow) - 16000 * 20 / 19) <= 1


def test_nbest_texts_handles_every_shape_nemo_returns():
    hyp = SimpleNamespace(text="one")
    nbest = SimpleNamespace(n_best_hypotheses=[SimpleNamespace(text="one"), SimpleNamespace(text="two")])
    assert cp.nbest_texts(hyp) == ["one"] and cp.nbest_texts(nbest) == ["one", "two"]
    assert cp.nbest_texts([hyp, SimpleNamespace(text=None)]) == ["one", ""]
    assert cp.nbest_texts("plain") == ["plain"]


def test_unpack_transcribe_picks_the_list_with_one_entry_per_clip():
    best = [SimpleNamespace(text="a"), SimpleNamespace(text="b")]
    full = [SimpleNamespace(n_best_hypotheses=[best[0]]), SimpleNamespace(n_best_hypotheses=[best[1]])]
    assert cp.unpack_transcribe((best, full), 2) is full
    assert cp.unpack_transcribe(best, 2) is best
    with pytest.raises(RuntimeError):
        cp.unpack_transcribe(best, 3)
    with pytest.raises(RuntimeError):
        cp.unpack_transcribe(([], []), 2)


def test_beam_decode_falls_through_the_strategies_and_restores_the_original_decoding():
    pytest.importorskip("torch")
    pytest.importorskip("omegaconf")
    calls = []

    class Model:
        cfg = SimpleNamespace(decoding={"strategy": "greedy_batch", "beam": {"beam_size": 1}})

        def change_decoding_strategy(self, cfg, verbose=True):
            calls.append(cfg["strategy"] if isinstance(cfg, dict) else cfg.strategy)
            if calls[-1] == "malsd_batch":
                raise ValueError("unknown strategy")

        def transcribe(self, paths, batch_size, verbose=False, return_hypotheses=False):
            return [SimpleNamespace(n_best_hypotheses=[SimpleNamespace(text=f"{p} best"), SimpleNamespace(text=f"{p} second")]) for p in paths]

    import omegaconf

    model = Model()
    model.cfg.decoding = omegaconf.OmegaConf.create({"strategy": "greedy_batch", "beam": {"beam_size": 1}})
    texts, strategy = cp.beam_decode(model, ["x", "y"], 4, 2, lambda *a: None)
    assert strategy == "maes" and texts == [["x best", "x second"], ["y best", "y second"]]
    assert calls == ["malsd_batch", "maes", "greedy_batch"]  # the failed one, the one that worked, then the original put back
