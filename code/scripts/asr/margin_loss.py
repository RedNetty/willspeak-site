from __future__ import annotations

import torch

import tdt_lattice


class SilenceAwareLoss(torch.nn.Module):
    def __init__(self, inner, margin: float = 0.0, weight: float = 1.0, report=None, report_every: int = 200):
        super().__init__()
        raw = getattr(inner, "_loss", None)
        durations = list(getattr(raw, "durations", None) or [])
        if not durations:
            raise ValueError("The silence-aware loss needs a TDT model: its loss has no durations")
        if margin < 0 or weight < 0:
            raise ValueError("margin and weight cannot be negative")
        self.inner = inner
        self.durations = [int(d) for d in durations]
        self.sigma = float(getattr(raw, "sigma", 0.0))
        self.blank = int(raw.blank)
        self.margin, self.weight = float(margin), float(weight)
        self.report, self.report_every = report, report_every
        self._calls = 0
        self._checked = False
        self._reset()

    @property
    def reduction(self):
        return self.inner.reduction

    @reduction.setter
    def reduction(self, value):
        self.inner.reduction = value

    def reduce(self, losses, target_lengths):
        lengths = ([length.clamp_min(1) for length in target_lengths]
                   if isinstance(target_lengths, list) else target_lengths.clamp_min(1))
        return self.inner.reduce(losses, lengths)

    def _reset(self) -> None:
        self._stats = {"rows": 0, "empty_rows": 0, "hinge_active": 0, "nll_empty_sum": 0.0, "nll_empty_min": float("inf")}

    def _nll_empty(self, log_probs: torch.Tensor, b: int, frames: int) -> torch.Tensor:
        return tdt_lattice.empty_nll(log_probs[b, :frames, 0, :], self.durations, self.blank, self.sigma)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        if not self.training:
            return self.inner(log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths)
        words = target_lengths > 0
        reduction = self.reduction
        batch = log_probs.shape[0]
        saved = self.inner.reduction
        self.inner.reduction = None
        try:
            kept = words.nonzero().squeeze(1)
            inner_values = None
            if len(kept):
                tl_kept = target_lengths[kept]
                lp = log_probs[kept][:, :, : int(tl_kept.max()) + 1].contiguous()
                inner_values = self.inner(log_probs=lp, targets=targets[kept], input_lengths=input_lengths[kept], target_lengths=tl_kept)
        finally:
            self.inner.reduction = saved
        position = {int(b): i for i, b in enumerate(kept.tolist())}
        per_clip = []
        for b in range(batch):
            frames = int(input_lengths[b])
            if b in position:
                value = inner_values[position[b]]
                if self.margin > 0:
                    empty = self._nll_empty(log_probs, b, frames)
                    hinge = tdt_lattice.silence_hinge(empty, self.margin)
                    value = value + self.weight * hinge
                    self._note(empty, hinge > 0, False)
            else:
                value = self._nll_empty(log_probs, b, frames)
                self._note(value, False, True)
            per_clip.append(value)
        per_clip = torch.stack([v.float() for v in per_clip])
        if not bool(torch.isfinite(per_clip).all()):
            rows = []
            for b in torch.nonzero(~torch.isfinite(per_clip)).flatten().tolist():
                frames = int(input_lengths[b])
                finite_in = bool(torch.isfinite(log_probs[b, :frames]).all())
                rows.append(f"row {b}: {frames} frames, {int(target_lengths[b])} target tokens, loss {float(per_clip[b])}, "
                            f"joint output finite: {finite_in}, path: {'with words' if b in position else 'empty transcript'}")
            raise RuntimeError("Non-finite per-clip loss in " + "; ".join(rows))
        self._maybe_check(log_probs, targets, input_lengths, target_lengths, inner_values, position)
        self._calls += 1
        if self.report and self._calls % self.report_every == 0:
            self._flush()
        return per_clip if reduction is None else self.reduce(per_clip, target_lengths)

    def _note(self, empty: torch.Tensor, active, is_empty_row: bool) -> None:
        s = self._stats
        s["rows"] += 1
        s["empty_rows"] += int(is_empty_row)
        s["hinge_active"] += int(bool(active))
        value = float(empty.detach())
        s["nll_empty_sum"] += value
        s["nll_empty_min"] = min(s["nll_empty_min"], value)

    def _flush(self) -> None:
        s = dict(self._stats)
        s["nll_empty_mean"] = s["nll_empty_sum"] / s["rows"] if s["rows"] else None
        self.report(s)
        self._reset()

    def _maybe_check(self, log_probs, targets, input_lengths, target_lengths, inner_values, position) -> None:
        if self._checked or not position or self.report is None:
            return
        self._checked = True
        try:
            diffs = []
            with torch.no_grad():
                for b, i in list(position.items())[:2]:
                    n = int(target_lengths[b])
                    tok, dur = tdt_lattice.split_logprobs(log_probs[b, : int(input_lengths[b]), : n + 1], len(self.durations), self.sigma)
                    mine = tdt_lattice.sequence_nll(tok.cpu(), dur.cpu(), targets[b, :n].tolist(), self.durations, self.blank)
                    diffs.append(abs(float(mine) - float(inner_values[i])))
            self.report({"first_batch_check_max_abs_diff": max(diffs), "clips": len(diffs)})
        except Exception as error:  # noqa: BLE001
            self.report({"first_batch_check_skipped": f"{type(error).__name__}: {error}"})


def install(model, margin: float, weight: float, report=None, report_every: int = 200) -> SilenceAwareLoss:
    wrapper = SilenceAwareLoss(model.loss, margin=margin, weight=weight, report=report, report_every=report_every)
    model.loss = wrapper
    if getattr(model.joint, "fuse_loss_wer", False):
        model.joint.set_loss(wrapper)
    return wrapper
