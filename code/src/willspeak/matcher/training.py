from __future__ import annotations

import numpy as np


def session_center(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    Xc = X.astype(np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        Xc[m] -= Xc[m].mean(0)
    return Xc


def _scatter(X: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = X.shape[1]
    mu = X.mean(0)
    Sw = np.zeros((d, d)); Sb = np.zeros((d, d))
    for c in np.unique(classes):
        m = classes == c
        Xc = X[m]
        mc = Xc.mean(0)
        D = Xc - mc
        Sw += D.T @ D
        Sb += len(Xc) * np.outer(mc - mu, mc - mu)
    return Sw, Sb, mu


def shrink(S: np.ndarray, alpha: float) -> np.ndarray:
    d = S.shape[0]
    return (1 - alpha) * S + alpha * (np.trace(S) / d) * np.eye(d)


def inv_sqrt(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-10, None)
    return (V / np.sqrt(w)) @ V.T


def lda_fit(X: np.ndarray, classes: np.ndarray, dim: int = 128, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, np.float64)
    classes = np.asarray(classes)
    if (X.ndim != 2 or not X.size or classes.ndim != 1 or len(classes) != len(X)
            or not np.isfinite(X).all() or not np.isfinite(alpha) or not 0 <= alpha <= 1 or dim < 1):
        raise ValueError("LDA needs finite vectors, matching labels, a positive dimension, and shrinkage in [0, 1]")
    if len(np.unique(classes)) < 2:
        raise ValueError("LDA needs at least two distinct labels")
    Sw, Sb, mu = _scatter(X, classes)
    n = len(X)
    Sw = shrink(Sw / max(n, 1), alpha)
    Sb = Sb / max(n, 1)
    Wh = inv_sqrt(Sw)
    M = Wh @ Sb @ Wh
    w, U = np.linalg.eigh((M + M.T) / 2)
    order = np.argsort(w)[::-1][:dim]
    W = Wh @ U[:, order]
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    return W.astype(np.float32), mu.astype(np.float32)


def wccn_fit(P: np.ndarray, classes: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    P = np.asarray(P, np.float64)
    Sw, _, _ = _scatter(P, classes)
    n_pairs = sum(max(0, int((classes == c).sum()) - 1) for c in np.unique(classes))
    Sw = shrink(Sw / max(n_pairs, 1), alpha)
    L = np.linalg.cholesky(np.linalg.inv(Sw))
    return L.astype(np.float32)


def nap_fit(P: np.ndarray, classes: np.ndarray, sessions: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, np.float64)
    d = P.shape[1]
    Sn = np.zeros((d, d))
    for c in np.unique(classes):
        m = classes == c
        sess_c = sessions[m]
        if len(np.unique(sess_c)) < 2:
            continue
        mc = P[m].mean(0)
        for s in np.unique(sess_c):
            ms = P[m][sess_c == s].mean(0) - mc
            Sn += np.outer(ms, ms)
    w, V = np.linalg.eigh((Sn + Sn.T) / 2)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    Vk = V[:, :k]
    T = np.eye(d) - Vk @ Vk.T
    return T.astype(np.float32), w


def load_embeddings(paths) -> dict:
    from pathlib import Path
    raw, lab, sess, spk = [], [], [], []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        n = len(z["raw"])
        raw.append(z["raw"].astype(np.float32))
        lab.append(z["labels"].astype(str))
        sess.append(z["sessions"].astype(str) if "sessions" in z.files else np.array(["s0"] * n))
        spk.append(z["speakers"].astype(str) if "speakers" in z.files else np.array([Path(p).stem] * n))
    return {"raw": np.concatenate(raw), "labels": np.concatenate(lab),
            "sessions": np.concatenate(sess), "speakers": np.concatenate(spk)}


def l2n(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def cross_session_top1(V: np.ndarray, labels: np.ndarray, sessions: np.ndarray) -> tuple[float, int]:
    V = l2n(np.asarray(V, np.float64))
    hits = n = 0
    for s in np.unique(sessions):
        en_idx = []
        seen = set()
        for i in np.where(sessions == s)[0]:
            if labels[i] not in seen:
                seen.add(labels[i]); en_idx.append(i)
        en_idx = np.array(en_idx)
        te_idx = [i for i in np.where(sessions != s)[0] if labels[i] in seen]
        if len(te_idx) == 0 or len(en_idx) < 2:
            continue
        S = V[te_idx] @ V[en_idx].T
        pred = labels[en_idx][S.argmax(1)]
        hits += int((pred == labels[te_idx]).sum()); n += len(te_idx)
    return (hits / n if n else float("nan")), n


def causal_session_center(X: np.ndarray, prior: np.ndarray, warmup: int = 20) -> np.ndarray:
    from .scoring import shrunk_session_mean, update_running_sum

    if warmup < 0:
        raise ValueError("Session warmup must be nonnegative")
    centered = np.asarray(X, np.float32).copy()
    total, n = None, 0
    for i, vector in enumerate(X):
        total, n = update_running_sum(total, n, vector)
        centered[i] -= shrunk_session_mean(total, n, prior, warmup)
    return centered


def causal_cross_session_top1(X: np.ndarray, labels: np.ndarray, sessions: np.ndarray,
                             project=None, warmup: int = 20) -> tuple[float, int]:
    from .scoring import label_prototypes

    X, labels, sessions = np.asarray(X, np.float32), np.asarray(labels), np.asarray(sessions)
    project = project if project is not None else lambda values: values
    hits = n = 0
    for enrolled in np.unique(sessions):
        bank = sessions == enrolled
        if len(np.unique(labels[bank])) < 2:
            continue
        prior = X[bank].mean(0)
        bank_labels, prototypes = label_prototypes(l2n(project(X[bank] - prior)), labels[bank])
        bank_labels = np.asarray(bank_labels)
        for testing in np.unique(sessions[~bank]):
            queries = sessions == testing
            vectors = l2n(project(causal_session_center(X[queries], prior, warmup)))
            known = np.isin(labels[queries], bank_labels)
            predictions = bank_labels[(vectors[known] @ prototypes.T).argmax(1)]
            hits += int((predictions == labels[queries][known]).sum())
            n += int(known.sum())
    return (hits / n if n else float("nan")), n


def session_probe_accuracy(V: np.ndarray, sessions: np.ndarray,
                           labels: np.ndarray | None = None,
                           seed: int = 0) -> tuple[float, float]:
    V = np.asarray(V, np.float64).copy()
    sessions = np.asarray(sessions)
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(V):
            raise ValueError("labels and vectors must have the same length")
        for label in np.unique(labels):
            mask = labels == label
            V[mask] -= V[mask].mean(0)
    V = l2n(V)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(V))
    tr, te = idx[: len(V) // 2], idx[len(V) // 2:]
    sess = np.unique(sessions)
    if len(sess) < 2 or len(te) == 0:
        return float("nan"), float("nan")
    C = np.stack([V[tr][sessions[tr] == s].mean(0) if (sessions[tr] == s).any() else np.zeros(V.shape[1]) for s in sess])
    pred = sess[(V[te] @ C.T).argmax(1)]
    return float((pred == sessions[te]).mean()), 1.0 / len(sess)
