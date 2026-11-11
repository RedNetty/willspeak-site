from __future__ import annotations

import math

NEG_INF = float("-inf")


def _add(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    return a + math.log1p(math.exp(b - a)) if a >= b else b + math.log1p(math.exp(a - b))


def lattice_nll_lists(blank_lp, label_lp, durations_lp, durations: list[int]) -> float:
    n_frames = len(blank_lp)
    if n_frames == 0:
        return float("inf")
    n_labels = len(label_lp[0])
    alpha = [[NEG_INF] * (n_labels + 1) for _ in range(n_frames)]
    alpha[0][0] = 0.0
    finish = NEG_INF
    steps = list(enumerate(durations))
    for t in range(n_frames):
        row = alpha[t]
        for u in range(n_labels + 1):
            here = row[u]
            if here == NEG_INF:
                continue
            blank_here = here + blank_lp[t][u]
            dur = durations_lp[t][u]
            for n, d in steps:
                if d >= 1:
                    arc = blank_here + dur[n]
                    land = t + d
                    if land < n_frames:
                        alpha[land][u] = _add(alpha[land][u], arc)
                    elif land == n_frames and u == n_labels:
                        finish = _add(finish, arc)
            if u < n_labels:
                label_here = here + label_lp[t][u]
                for n, d in steps:
                    land = t + d
                    if land < n_frames:
                        alpha[land][u + 1] = _add(alpha[land][u + 1], label_here + dur[n])
    return float("inf") if finish == NEG_INF else -finish
