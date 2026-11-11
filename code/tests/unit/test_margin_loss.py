import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import margin_loss as ml
import tdt_lattice as tl

VOCAB, DURATIONS, SIGMA = 5, [0, 1, 2], 0.05


class FakeRNNTLoss(torch.nn.Module):
    def __init__(self, reduction="mean_batch"):
        super().__init__()
        self.reduction = reduction
        self._loss = SimpleNamespace(durations=DURATIONS, sigma=SIGMA, blank=VOCAB)
        self.calls = 0

    def reduce(self, losses, target_lengths):
        if isinstance(losses, list):
            losses = torch.cat(losses, 0)
            target_lengths = torch.cat(target_lengths, 0)
        if self.reduction == "mean_batch":
            return losses.mean()
        if self.reduction == "mean":
            return (losses / target_lengths).mean()
        if self.reduction == "mean_volume":
            return losses.sum() / target_lengths.sum()
        return losses.sum() if self.reduction == "sum" else losses

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        self.calls += 1
        if int(target_lengths.min()) == 0:
            raise AssertionError("NeMo's kernel is never given an empty target")
        if log_probs.shape[2] != int(target_lengths.max()) + 1:
            raise ValueError("Output length mismatch")
        per = []
        for b in range(log_probs.shape[0]):
            n = int(target_lengths[b])
            tok, dur = tl.split_logprobs(log_probs[b, : int(input_lengths[b]), : n + 1], len(DURATIONS), SIGMA)
            per.append(tl.sequence_nll(tok, dur, targets[b, :n].tolist(), DURATIONS, VOCAB))
        per = torch.stack(per)
        return per if self.reduction is None else self.reduce(per, target_lengths)


def batch(lengths=(2, 3, 0), frames=(8, 10, 9), seed=0):
    g = torch.Generator().manual_seed(seed)
    b, t, u = len(lengths), max(frames), max(lengths) + 1
    logits = torch.randn(b, t, u, VOCAB + 1 + len(DURATIONS), generator=g).requires_grad_(True)
    targets = torch.randint(0, VOCAB, (b, max(max(lengths), 1)), generator=g)
    return logits, targets, torch.tensor(frames), torch.tensor(lengths)


def empty_of(logits, b, frames):
    return tl.empty_nll(logits[b, :frames, 0, :], DURATIONS, VOCAB, SIGMA)


def test_empty_clips_get_the_lattice_loss_and_never_reach_the_inner_loss():
    logits, targets, frames, lengths = batch(lengths=(2, 3, 0))
    inner = FakeRNNTLoss()
    wrapper = ml.SilenceAwareLoss(inner)
    got = wrapper(logits, targets, frames, lengths)
    first_two = inner(logits[:2], targets[:2], frames[:2], lengths[:2])
    expected = (first_two * 2 + empty_of(logits, 2, int(frames[2]))) / 3
    assert got.item() == pytest.approx(expected.item(), abs=1e-4)
    assert wrapper.reduction == "mean_batch" and inner.reduction == "mean_batch"


def test_a_batch_of_only_empty_clips_works():
    logits, targets, frames, lengths = batch(lengths=(0, 0), frames=(7, 9))
    wrapper = ml.SilenceAwareLoss(FakeRNNTLoss())
    got = wrapper(logits, targets, frames, lengths)
    assert got.item() == pytest.approx(((empty_of(logits, 0, 7) + empty_of(logits, 1, 9)) / 2).item(), abs=1e-4)


def test_margin_adds_a_hinge_only_to_clips_with_words_and_only_below_the_margin():
    logits, targets, frames, lengths = batch(lengths=(2, 3, 1))
    inner = FakeRNNTLoss()
    base = inner(logits, targets, frames, lengths)
    empties = torch.stack([empty_of(logits, b, int(frames[b])) for b in range(3)])
    margin, weight = float(empties.detach().max()) + 3.0, 0.5
    got = ml.SilenceAwareLoss(inner, margin=margin, weight=weight)(logits, targets, frames, lengths)
    assert got.item() == pytest.approx((base + weight * torch.relu(margin - empties).mean()).item(), abs=1e-4)
    tiny = ml.SilenceAwareLoss(inner, margin=1e-6, weight=weight)(logits, targets, frames, lengths)
    assert tiny.item() == pytest.approx(base.item(), abs=1e-4)


def test_gradient_pushes_the_empty_path_down_where_the_hinge_is_active():
    logits, targets, frames, lengths = batch(lengths=(2, 2), frames=(8, 8))
    wrapper = ml.SilenceAwareLoss(FakeRNNTLoss(), margin=1000.0, weight=1.0)
    before = [float(empty_of(logits.detach(), b, 8)) for b in range(2)]
    wrapper(logits, targets, frames, lengths).backward()
    stepped = (logits - 0.01 * logits.grad).detach()
    after = [float(empty_of(stepped, b, 8)) for b in range(2)]
    assert all(a > b for a, b in zip(after, before))


def test_the_fused_joints_calling_pattern_works():
    logits, targets, frames, lengths = batch(lengths=(2, 3, 0))
    wrapper = ml.SilenceAwareLoss(FakeRNNTLoss())
    wrapper.reduction = None
    per_clip = wrapper(logits, targets, frames, lengths)
    wrapper.reduction = "mean_batch"
    assert per_clip.shape == (3,)
    assert wrapper.reduce([per_clip], [lengths]).item() == pytest.approx(per_clip.mean().item())


def test_evaluation_mode_passes_through_untouched():
    logits, targets, frames, lengths = batch(lengths=(2, 3, 1))
    inner = FakeRNNTLoss()
    wrapper = ml.SilenceAwareLoss(inner, margin=1000.0, weight=5.0).eval()
    assert torch.allclose(wrapper(logits, targets, frames, lengths), inner(logits, targets, frames, lengths))


@pytest.mark.parametrize("reduction", ["mean", "mean_volume", "mean_batch", "sum"])
@pytest.mark.parametrize("target_sizes", [(2, 0), (0, 0)])
def test_fused_reduction_and_backward_are_finite_with_empty_targets(reduction, target_sizes):
    logits, targets, frames, lengths = batch(lengths=target_sizes)
    inner = FakeRNNTLoss(reduction=reduction)
    wrapper = ml.SilenceAwareLoss(inner)
    wrapper.reduction = None
    per_clip = wrapper(logits, targets, frames, lengths)
    wrapper.reduction = reduction
    got = wrapper.reduce([per_clip[:1], per_clip[1:]], [lengths[:1], lengths[1:]])
    assert torch.isfinite(got)
    assert torch.allclose(got, inner.reduce(per_clip, lengths.clamp_min(1)))
    got.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.allclose(wrapper(logits.detach(), targets, frames, lengths), got.detach())
    assert torch.equal(lengths, torch.tensor(target_sizes))


@pytest.mark.parametrize("reduction", ["mean", "mean_volume", "mean_batch", "sum"])
def test_word_only_reductions_match_nemo(reduction):
    logits, targets, frames, lengths = batch(lengths=(2, 3))
    inner = FakeRNNTLoss(reduction=reduction)
    assert torch.allclose(ml.SilenceAwareLoss(inner)(logits, targets, frames, lengths),
                          inner(logits, targets, frames, lengths))


def test_non_tdt_models_and_negative_margins_are_refused():
    plain = FakeRNNTLoss()
    plain._loss = SimpleNamespace(blank=VOCAB)
    with pytest.raises(ValueError, match="TDT"):
        ml.SilenceAwareLoss(plain)
    with pytest.raises(ValueError):
        ml.SilenceAwareLoss(FakeRNNTLoss(), margin=-1.0)


def test_reports_the_first_batch_check_and_running_statistics():
    reports = []
    logits, targets, frames, lengths = batch(lengths=(2, 3, 0))
    wrapper = ml.SilenceAwareLoss(FakeRNNTLoss(), margin=1000.0, report=reports.append, report_every=2)
    wrapper(logits, targets, frames, lengths)
    wrapper(logits, targets, frames, lengths)
    check = next(r for r in reports if "first_batch_check_max_abs_diff" in r)
    assert check["first_batch_check_max_abs_diff"] < 1e-4 and check["clips"] == 2
    stats = next(r for r in reports if "rows" in r)
    assert stats["rows"] == 6 and stats["empty_rows"] == 2 and stats["hinge_active"] == 4


def test_install_replaces_both_places_nemo_keeps_the_loss():
    class Joint:
        fuse_loss_wer = True
        loss = None

        def set_loss(self, value):
            self.loss = value

    model = torch.nn.Module()
    model.loss = FakeRNNTLoss()
    model.joint = Joint()
    wrapper = ml.install(model, margin=2.0, weight=1.0)
    assert model.loss is wrapper and model.joint.loss is wrapper and wrapper.margin == 2.0
