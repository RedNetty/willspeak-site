from __future__ import annotations

import json

import numpy as np
import pytest

from willspeak.matcher import training as S
from willspeak.matcher.engine import Projection

D, WORDS, SPK, SESS, TAKES = 768, 30, 5, 3, 3


def synth(seed=0, word_scale=1.0, speaker_scale=1.5, session_scale=2.0, noise=0.6):
    rng = np.random.default_rng(seed)
    word_v = rng.normal(size=(WORDS, D)) * word_scale
    spk_v = rng.normal(size=(SPK, D)) * speaker_scale
    basis = np.linalg.qr(rng.normal(size=(D, 12)))[0]
    X, lab, spk, sess = [], [], [], []
    for p in range(SPK):
        for s in range(SESS):
            day = basis @ (rng.normal(size=12) * session_scale * np.sqrt(D / 12))
            for w in range(WORDS):
                off = day + basis @ (rng.normal(size=12) * session_scale * np.sqrt(D / 12))
                for _ in range(TAKES):
                    X.append(word_v[w] + spk_v[p] + off + rng.normal(size=D) * noise)
                    lab.append(f"w{w}"); spk.append(f"p{p}"); sess.append(f"p{p}s{s}")
    return np.array(X, np.float32), np.array(lab), np.array(spk), np.array(sess)


@pytest.fixture(scope="module")
def data():
    return synth()


def _fit_on_others(X, lab, spk, sess, held, kind):
    m = spk != held
    Xc = S.session_center(X[m], sess[m])
    W, mu = S.lda_fit(Xc, lab[m], dim=32, alpha=0.1)
    P = (Xc - mu) @ W
    cls = np.array([f"{a}|{b}" for a, b in zip(spk[m], lab[m])])
    if kind == "wccn":
        T = S.wccn_fit(P, cls, alpha=0.1)
    elif kind == "nap":
        T, _ = S.nap_fit(P, cls, sess[m], k=8)
    else:
        T = np.eye(W.shape[1], dtype=np.float32)
    return W, mu, T


def test_lda_beats_raw_cross_session(data):
    X, lab, spk, sess = data
    held = "p0"
    m = spk == held
    raw, _ = S.cross_session_top1(X[m], lab[m], sess[m])
    W, mu, _T = _fit_on_others(X, lab, spk, sess, held, "none")
    Xc = S.session_center(X[m], sess[m])
    lda, n = S.cross_session_top1((Xc - mu) @ W, lab[m], sess[m])
    assert n > 0
    assert raw < 0.5, f"synthetic sessions are meant to break raw cosine, got {raw:.2f}"
    assert lda > raw + 0.2, f"LDA {lda:.2f} vs raw {raw:.2f}"


@pytest.mark.parametrize("kind", ["wccn", "nap"])
def test_session_transform_helps_held_out_speaker(data, kind):
    X, lab, spk, sess = data
    held = "p1"
    m = spk == held
    W, mu, T = _fit_on_others(X, lab, spk, sess, held, kind)
    P = (X[m] - mu) @ W
    before, _ = S.cross_session_top1(P, lab[m], sess[m])
    after, _ = S.cross_session_top1(P @ T, lab[m], sess[m])
    probe_b, chance = S.session_probe_accuracy(P, sess[m], labels=lab[m])
    probe_a, _ = S.session_probe_accuracy(P @ T, sess[m], labels=lab[m])
    assert after >= before - 1e-9, f"{kind}: {before:.3f} -> {after:.3f}"
    assert after > 0.9, f"{kind}: cross-session top-1 only {after:.3f}"
    if kind == "nap":
        assert probe_a < probe_b, f"{kind}: session probe {probe_b:.2f} -> {probe_a:.2f} (chance {chance:.2f})"
    else:
        assert np.isfinite(probe_a)


def test_wccn_whitens_within_class():
    rng = np.random.default_rng(1)
    d = 16
    A = rng.normal(size=(d, d)); cov = A @ A.T / d + 0.1 * np.eye(d)
    L = np.linalg.cholesky(cov)
    cls = np.repeat(np.arange(40), 25)
    means = rng.normal(size=(40, d)) * 5
    P = means[cls] + rng.normal(size=(len(cls), d)) @ L.T
    T = S.wccn_fit(P, cls, alpha=0.0)
    Pw = P @ T
    Sw, _, _ = S._scatter(Pw, cls)
    Sw /= (len(cls) - 40)
    assert np.allclose(Sw, np.eye(d), atol=0.15)


def test_scatter_includes_singleton_class():
    X = np.array([[0., 0.], [2., 0.], [0., 3.]])
    labels = np.array(["water", "water", "help"])
    within, between, mean = S._scatter(X, labels)
    centered = X - X.mean(0)
    assert np.allclose(mean, [2 / 3, 1])
    assert np.allclose(within, [[2, 0], [0, 0]])
    assert np.allclose(within + between, centered.T @ centered)


@pytest.mark.parametrize("vectors,labels,kwargs", [
    (np.empty((0, 2)), np.array([]), {}),
    (np.ones((3, 2)), np.array(["a", "b"]), {}),
    (np.array([[np.nan, 0], [0, 1]]), np.array(["a", "b"]), {}),
    (np.ones((2, 2)), np.array(["a", "a"]), {}),
    (np.ones((2, 2)), np.array(["a", "b"]), {"alpha": -0.1}),
    (np.ones((2, 2)), np.array(["a", "b"]), {"dim": 0}),
])
def test_lda_refuses_invalid_fit(vectors, labels, kwargs):
    with pytest.raises(ValueError):
        S.lda_fit(vectors, labels, **kwargs)


def test_causal_center_cannot_use_future_queries():
    queries = np.array([[1, 2], [3, 4], [5, 6]], np.float32)
    prior = np.array([10, 20], np.float32)
    centered = S.causal_session_center(queries, prior, warmup=2)
    assert np.allclose(centered[0], queries[0] - (2 * prior + queries[0]) / 3)
    changed = queries.copy()
    changed[-1] += 10000
    assert np.array_equal(centered[:2], S.causal_session_center(changed, prior, warmup=2)[:2])


def test_causal_cross_session_excludes_unknown_labels_from_accuracy():
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0]], np.float32)
    labels = np.array(["water", "help", "water", "help", "unknown"])
    sessions = np.array(["one", "one", "two", "two", "two"])
    score, n = S.causal_cross_session_top1(vectors, labels, sessions)
    assert n == 4 and score == 1
    score, n = S.causal_cross_session_top1(vectors, labels, np.array(["one"] * 5))
    assert n == 0 and np.isnan(score)


def test_projection_file_roundtrip_applies_T(tmp_path):
    rng = np.random.default_rng(2)
    W = rng.normal(size=(D, 8)).astype(np.float32)
    T = np.diag(np.arange(1, 9, dtype=np.float32))
    mean = rng.normal(size=D).astype(np.float32)
    p = tmp_path / "proj.npz"
    np.savez(p, W=W, mean=mean, T=T, meta=json.dumps({"train": "synthetic", "session_transform": "wccn"}))
    proj = Projection(p)
    x = rng.normal(size=D).astype(np.float32)
    assert np.allclose(proj(x), ((x - mean) @ W) @ T, atol=1e-4)
    assert proj.dim == 8 and "wccn" in proj.describe()
    p2 = tmp_path / "old.npz"
    np.savez(p2, W=W, mean=mean, meta=json.dumps({"train": "old"}))
    old = Projection(p2)
    assert old.T is None and np.allclose(old(x), (x - mean) @ W, atol=1e-4)
