from __future__ import annotations

import torch


def split_logprobs(logits: torch.Tensor, n_durations: int, sigma: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if n_durations < 1 or logits.shape[-1] <= n_durations + 1:
        raise ValueError(f"Joint output of width {logits.shape[-1]} cannot hold {n_durations} durations and a vocabulary")
    logits = logits.float()
    tokens = torch.log_softmax(logits[..., :-n_durations], dim=-1) - sigma
    durations = torch.log_softmax(logits[..., -n_durations:], dim=-1)
    return tokens, durations


def reduce_logprobs(tokens: torch.Tensor, durations_lp: torch.Tensor, targets: list[int], blank: int
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_labels = len(targets)
    if tokens.shape[1] < n_labels + 1 or durations_lp.shape[:2] != tokens.shape[:2]:
        raise ValueError("The joint output has fewer prediction steps than the transcript needs")
    blank_lp = tokens[:, : n_labels + 1, blank]
    if n_labels:
        index = torch.tensor(targets, device=tokens.device, dtype=torch.long)
        label_lp = tokens[:, :n_labels, :].gather(2, index.view(1, n_labels, 1).expand(tokens.shape[0], n_labels, 1)).squeeze(2)
    else:
        label_lp = tokens.new_zeros((tokens.shape[0], 0))
    return blank_lp, label_lp, durations_lp[:, : n_labels + 1, :]


def lattice_nll(blank_lp: torch.Tensor, label_lp: torch.Tensor, durations_lp: torch.Tensor, durations: list[int]
                ) -> torch.Tensor:
    n_frames, n_labels = blank_lp.shape[0], label_lp.shape[1]
    if len(durations) != durations_lp.shape[-1]:
        raise ValueError("durations does not match the duration logits")
    incoming: list[list[list[torch.Tensor]]] = [[[] for _ in range(n_labels + 1)] for _ in range(n_frames)]
    incoming[0][0].append(blank_lp.new_zeros(()))
    finish: list[torch.Tensor] = []
    for t in range(n_frames):
        for u in range(n_labels + 1):
            terms = incoming[t][u]
            if not terms:
                continue
            here = terms[0] if len(terms) == 1 else torch.logsumexp(torch.stack(terms), dim=0)
            blank_here = here + blank_lp[t, u]
            for n, d in enumerate(durations):
                if d >= 1:
                    arc = blank_here + durations_lp[t, u, n]
                    if t + d < n_frames:
                        incoming[t + d][u].append(arc)
                    elif t + d == n_frames and u == n_labels:
                        finish.append(arc)
            if u < n_labels:
                label_here = here + label_lp[t, u]
                for n, d in enumerate(durations):
                    if t + d < n_frames:
                        incoming[t + d][u + 1].append(label_here + durations_lp[t, u, n])
    if not finish:
        return blank_lp.new_full((), float("inf"))
    return -torch.logsumexp(torch.stack(finish), dim=0)


def sequence_nll(tokens: torch.Tensor, durations_lp: torch.Tensor, targets: list[int], durations: list[int],
                 blank: int) -> torch.Tensor:
    blank_lp, label_lp, dur_lp = reduce_logprobs(tokens, durations_lp, targets, blank)
    return lattice_nll(blank_lp, label_lp, dur_lp, durations)


def empty_nll(logits_u0: torch.Tensor, durations: list[int], blank: int, sigma: float = 0.0) -> torch.Tensor:
    tokens, dur = split_logprobs(logits_u0.unsqueeze(1), len(durations), sigma)
    return sequence_nll(tokens, dur, [], durations, blank)


def silence_hinge(nll_empty: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.relu(margin - nll_empty)
