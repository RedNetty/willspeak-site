from __future__ import annotations

import numpy as np


def l2_normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def cosine_scores(bank: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.asarray(bank, np.float32) @ np.asarray(q, np.float32)


def center_by_session(R: np.ndarray, sessions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    R = np.asarray(R, np.float32)
    sessions = np.asarray(sessions)
    Rn = R.copy()
    means: dict[str, np.ndarray] = {}
    for s in np.unique(sessions):
        m = sessions == s
        mu = R[m].mean(0)
        Rn[m] -= mu
        means[str(s)] = l2_normalize(mu)
    return Rn, means


def update_running_sum(total: np.ndarray | None, n: int, v: np.ndarray) -> tuple[np.ndarray, int]:
    v = np.asarray(v, np.float32)
    if total is None:
        return v.copy(), 1
    return (total + v).astype(np.float32), n + 1


def shrunk_session_mean(total: np.ndarray | None, n: int, prior: np.ndarray, n0: float) -> np.ndarray:
    prior = np.asarray(prior, np.float32)
    if total is None or n <= 0:
        return prior.copy()
    return ((n0 * prior + np.asarray(total, np.float32)) / (n0 + n)).astype(np.float32)




def best_per_label(scores: np.ndarray, labels: list[str] | np.ndarray,
                   sessions: list[str] | np.ndarray | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i, lab in enumerate(labels):
        sess = None if sessions is None else str(sessions[i])
        d = out.get(lab)
        if d is None:
            out[lab] = {"score": float(scores[i]), "own_takes": 1, "session": sess}
        else:
            d["own_takes"] += 1
            if scores[i] > d["score"]:
                d["score"] = float(scores[i]); d["session"] = sess
    return out


def label_prototypes(vecs: np.ndarray, labels: list[str] | np.ndarray) -> tuple[list[str], np.ndarray]:
    vecs = np.asarray(vecs, np.float32)
    order: list[str] = []
    for lab in labels:
        if lab not in order:
            order.append(lab)
    if not order:
        return [], np.zeros((0, vecs.shape[1] if vecs.ndim == 2 else 0), np.float32)
    labels = np.asarray(labels)
    protos = np.stack([vecs[labels == lab].mean(0) for lab in order])
    return order, l2_normalize(protos)


def score_labels(proto_scores: np.ndarray, proto_labels: list[str], take_scores: np.ndarray,
                 labels: list[str] | np.ndarray, sessions: list[str] | np.ndarray | None = None) -> dict[str, dict]:
    best = best_per_label(take_scores, labels, sessions)
    out: dict[str, dict] = {}
    for i, lab in enumerate(proto_labels):
        d = best.get(lab, {"score": float(proto_scores[i]), "own_takes": 0, "session": None})
        out[lab] = {"score": float(proto_scores[i]), "own_takes": d["own_takes"],
                    "session": d["session"], "best_take": float(d["score"])}
    return out


def apply_asr_vote(best: dict[str, dict], asr_text: str | None) -> str | None:
    if not asr_text or asr_text not in best or not best:
        return None
    vals = np.array([d["score"] for d in best.values()], np.float32)
    best[asr_text]["score"] = float(best[asr_text]["score"] + 0.5 * (vals.std() + 1e-6))
    return asr_text


def rank_candidates(best: dict[str, dict], topk: int = 3) -> tuple[list[dict], float, float]:
    ranked = sorted(best.items(), key=lambda kv: -kv[1]["score"])
    top = ranked[0][1]["score"] if ranked else 0.0
    second = ranked[1][1]["score"] if len(ranked) > 1 else top
    cands = [{"text": t, "score": round(d["score"], 3), "own_takes": d["own_takes"], "session": d.get("session"),
              "best_take": round(d.get("best_take", d["score"]), 3)}
             for t, d in ranked[:topk]]
    return cands, round(top, 3), round(top - second, 3)
