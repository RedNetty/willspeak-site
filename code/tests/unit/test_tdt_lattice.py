import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import tdt_lattice as tl

DURATIONS = [0, 1, 2]


def random_joint(frames, labels, vocab=4, durations=DURATIONS, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(frames, labels + 1, vocab + 1 + len(durations), generator=g)


def brute_force(tokens, dur, targets, durations, blank):
    frames, labels = tokens.shape[0], len(targets)
    total = [0.0]

    def walk(t, u, logp):
        for n, d in enumerate(durations):
            if d >= 1:
                if t + d == frames and u == labels:
                    total[0] += math.exp(logp + float(tokens[t, u, blank] + dur[t, u, n]))
                elif t + d < frames:
                    walk(t + d, u, logp + float(tokens[t, u, blank] + dur[t, u, n]))
            if u < labels and t + d < frames:
                walk(t + d, u + 1, logp + float(tokens[t, u, targets[u]] + dur[t, u, n]))

    walk(0, 0, 0.0)
    return -math.log(total[0]) if total[0] > 0 else float("inf")


def nemo_reference(acts, duration_acts, labels, act_len, label_len, durations, blank):
    T, U = act_len, label_len + 1
    log_alpha = torch.zeros(T, U)
    for t in range(T):
        for u in range(U):
            if u == 0:
                if t == 0:
                    log_alpha[t, u] = 0.0
                else:
                    log_alpha[t, u] = -1000.0
                    for n, d in enumerate(durations):
                        if t - d >= 0 and d > 0:
                            tmp = log_alpha[t - d, u] + acts[t - d, u, blank] + duration_acts[t - d, u, n]
                            log_alpha[t, u] = torch.logsumexp(torch.stack([tmp, log_alpha[t, u]]), 0)
            else:
                log_alpha[t, u] = -1000.0
                for n, d in enumerate(durations):
                    if t - d >= 0:
                        if d > 0:
                            tmp = log_alpha[t - d, u] + acts[t - d, u, blank] + duration_acts[t - d, u, n]
                            log_alpha[t, u] = torch.logsumexp(torch.stack([tmp, log_alpha[t, u]]), 0)
                        tmp = log_alpha[t - d, u - 1] + acts[t - d, u - 1, labels[u - 1]] + duration_acts[t - d, u - 1, n]
                        log_alpha[t, u] = torch.logsumexp(torch.stack([tmp, log_alpha[t, u]]), 0)
    tt = torch.tensor(-1000.0)
    for n, d in enumerate(durations):
        if act_len - d >= 0 and d > 0:
            bb = log_alpha[act_len - d, label_len] + acts[act_len - d, label_len, blank] + duration_acts[act_len - d, label_len, n]
            tt = torch.logsumexp(torch.stack([bb, tt]), 0)
    return -tt


@pytest.mark.parametrize("frames,targets", [(3, []), (4, [1]), (4, [2, 0]), (5, [3, 3]), (6, [0, 1, 2])])
def test_dp_matches_brute_force_and_nemo_reference(frames, targets):
    vocab = 4
    logits = random_joint(frames, len(targets), vocab)
    tokens, dur = tl.split_logprobs(logits, len(DURATIONS), sigma=0.05)
    got = float(tl.sequence_nll(tokens, dur, targets, DURATIONS, blank=vocab))
    assert got == pytest.approx(brute_force(tokens, dur, targets, DURATIONS, vocab), abs=1e-4)
    ref = float(nemo_reference(tokens, dur, targets, frames, len(targets), DURATIONS, vocab))
    assert got == pytest.approx(ref, abs=1e-4)


def test_empty_path_counts_the_ways_to_tile_the_clip_with_blank_durations():
    vocab, frames, durations = 3, 5, [0, 1, 2, 3, 4]
    tokens = torch.zeros(frames, 1, vocab + 1)
    dur = torch.zeros(frames, 1, len(durations))
    nll = float(tl.sequence_nll(tokens, dur, [], durations, blank=vocab))
    assert nll == pytest.approx(-math.log(15), abs=1e-5)


def test_empty_nll_uses_only_the_start_state_logits():
    vocab = 4
    logits = random_joint(6, 3, vocab)
    full = tl.sequence_nll(*tl.split_logprobs(logits, 3), [], DURATIONS, vocab)
    assert float(tl.empty_nll(logits[:, 0, :], DURATIONS, vocab)) == pytest.approx(float(full), abs=1e-5)


def test_empty_path_gets_cheaper_when_blank_is_favoured():
    vocab, frames = 4, 8
    logits = random_joint(frames, 0, vocab)[:, 0, :]
    boosted = logits.clone()
    boosted[:, vocab] += 5.0
    assert float(tl.empty_nll(boosted, DURATIONS, vocab)) < float(tl.empty_nll(logits, DURATIONS, vocab))


def test_a_clip_too_short_for_its_words_has_no_alignment():
    vocab = 4
    logits = random_joint(2, 3, vocab, durations=[1, 2])
    tokens, dur = tl.split_logprobs(logits, 2)
    assert math.isinf(float(tl.sequence_nll(tokens, dur, [0, 1, 2], [1, 2], vocab)))


def test_gradient_reaches_the_logits_and_the_hinge_is_zero_beyond_the_margin():
    vocab = 4
    logits = random_joint(6, 0, vocab)[:, 0, :].clone().requires_grad_(True)
    nll = tl.empty_nll(logits, DURATIONS, vocab)
    loss = tl.silence_hinge(nll, margin=float(nll.detach()) + 1.0)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
    assert float(tl.silence_hinge(torch.tensor(30.0), 20.0)) == 0.0
    assert float(tl.silence_hinge(torch.tensor(5.0), 20.0)) == 15.0


def test_bad_shapes_are_refused():
    with pytest.raises(ValueError):
        tl.split_logprobs(torch.zeros(3, 4), 5)
    tokens, dur = tl.split_logprobs(random_joint(4, 1, 4), 3)
    with pytest.raises(ValueError, match="fewer prediction steps"):
        tl.sequence_nll(tokens, dur, [1, 2, 3], DURATIONS, 4)
