from __future__ import annotations

import numpy as np

from willspeak.matcher import scoring as SC


def test_exercise_1_lengths_dots_cosines():
    a, b = np.array([3.0, 4.0]), np.array([4.0, 3.0])
    assert np.linalg.norm(a) == 5 and np.linalg.norm(b) == 5
    assert float(a @ b) == 24
    ua, ub = SC.l2_normalize(a), SC.l2_normalize(b)
    assert np.isclose(float(ua @ ub), 0.96, atol=1e-6)
    assert np.isclose(SC.cosine_scores(SC.l2_normalize(np.array([[1.0, 0.0]])), SC.l2_normalize(np.array([0.0, 1.0])))[0], 0.0)
    assert np.isclose(float(SC.l2_normalize(np.array([6.0, 8.0])) @ ua), 1.0)
    assert not np.isnan(SC.l2_normalize(np.zeros(3))).any()


def test_exercise_2_bank_at_query():
    bank = SC.l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))
    q = SC.l2_normalize(np.array([0.8, 0.6]))
    assert np.allclose(SC.cosine_scores(bank, q), [0.8, 0.6, 0.96], atol=1e-6)


def test_exercise_3_center_by_session():
    R = np.array([[5, 1], [7, 1], [6, 2], [105, 1], [107, 1]], np.float32)
    sess = np.array(["A", "A", "A", "B", "B"])
    Rn, means = SC.center_by_session(R, sess)
    assert np.allclose(R[:3].mean(0), [6, 4 / 3], atol=1e-5)
    assert np.allclose(Rn[:3], R[:3] - np.array([6, 4 / 3]), atol=1e-5)
    assert np.allclose(Rn[3:], [[-1, 0], [1, 0]], atol=1e-5)
    assert set(means) == {"A", "B"} and np.isclose(np.linalg.norm(means["B"]), 1.0, atol=1e-5)


def test_exercise_4_shrunk_session_mean():
    prior = np.array([10.0, 10.0])
    assert np.allclose(SC.shrunk_session_mean(None, 0, prior, 20), prior)
    total, n = SC.update_running_sum(None, 0, np.array([2.0, 2.0]))
    assert np.allclose(total, [2, 2]) and n == 1
    total, n = SC.update_running_sum(total, n, np.array([4.0, 0.0]))
    assert np.allclose(total, [6.0, 2.0]) and n == 2
    assert np.allclose(SC.shrunk_session_mean(total, n, prior, 2), [(20 + 6) / 4, (20 + 2) / 4])
    assert np.allclose(SC.shrunk_session_mean(total, n, prior, 0), [3.0, 1.0])
    big = np.array([1.0, 1.0]) * 1000
    assert np.allclose(SC.shrunk_session_mean(big, 1000, prior, 20), [(200 + 1000) / 1020] * 2)


def test_best_per_label_keeps_the_take_view():
    s = np.array([0.2, 0.9, 0.5, 0.4])
    labels = ["water", "water", "help", "help"]
    best = SC.best_per_label(s, labels, ["s1", "s2", "s1", "s2"])
    assert best["water"] == {"score": 0.9, "own_takes": 2, "session": "s2"}
    assert best["help"]["score"] == 0.5 and best["help"]["session"] == "s1"
    assert SC.best_per_label(np.array([]), [], []) == {}


def test_prototype_is_the_mean_of_a_labels_takes():
    vecs = SC.l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.1]]))
    labs, protos = SC.label_prototypes(vecs, ["water", "water", "help", "help"])
    assert labs == ["water", "help"] and protos.shape == (2, 2)
    assert np.allclose(protos[0], SC.l2_normalize(np.array([0.5, 0.5])), atol=1e-6)
    assert np.isclose(np.linalg.norm(protos[1]), 1.0, atol=1e-6)
    assert SC.label_prototypes(np.zeros((0, 2)), []) == ([], ) or SC.label_prototypes(np.zeros((0, 2)), [])[0] == []
    q = SC.l2_normalize(np.array([1.0, 1.0]))
    assert float(protos[0] @ q) > max(float(vecs[0] @ q), float(vecs[1] @ q))


def test_score_labels_uses_the_prototype_and_keeps_take_detail():
    take_scores = np.array([0.2, 0.9, 0.5, 0.4])
    labels = ["water", "water", "help", "help"]
    proto_labels = ["water", "help"]
    proto_scores = np.array([0.7, 0.45])
    out = SC.score_labels(proto_scores, proto_labels, take_scores, labels, ["s1", "s2", "s1", "s2"])
    assert out["water"] == {"score": 0.7, "own_takes": 2, "session": "s2", "best_take": 0.9}
    assert out["help"]["score"] == 0.45 and out["help"]["own_takes"] == 2
    cands, top, margin = SC.rank_candidates(out, topk=2)
    assert [c["text"] for c in cands] == ["water", "help"] and np.isclose(top, 0.7) and np.isclose(margin, 0.25)


def test_asr_vote_exact_gate_and_ranking():
    best = {"water": {"score": 0.50, "own_takes": 2, "session": "s1"},
            "help": {"score": 0.48, "own_takes": 1, "session": "s1"},
            "no": {"score": 0.10, "own_takes": 3, "session": "s2"}}
    assert SC.apply_asr_vote(dict(best), "wotter") is None
    assert SC.apply_asr_vote(best, "help") == "help"
    std = np.std([0.50, 0.48, 0.10])
    assert np.isclose(best["help"]["score"], 0.48 + 0.5 * (std + 1e-6), atol=1e-6)
    cands, top, margin = SC.rank_candidates(best, topk=2)
    assert [c["text"] for c in cands] == ["help", "water"] and len(cands) == 2
    assert np.isclose(top, round(best["help"]["score"], 3)) and np.isclose(margin, round(best["help"]["score"] - 0.5, 3))
    assert SC.rank_candidates({}, 3) == ([], 0.0, 0.0)
    one = {"only": {"score": 0.7, "own_takes": 1, "session": None}}
    assert SC.apply_asr_vote(one, "only") == "only"
    assert SC.rank_candidates(one)[2] == 0.0
