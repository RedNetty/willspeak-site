import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import reference_score as rs
import tdt_lattice as tl

D, VOCAB, DURATIONS = 8, 6, [0, 1, 2]
FRAME = 1600


class FakeDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB + 1, D)

    def forward(self, targets, target_length, states=None):
        y = self.embed(targets)
        y = torch.cat([torch.zeros(y.shape[0], 1, D), y], dim=1)
        return y.transpose(1, 2), target_length, None

    def predict(self, y=None, state=None, add_sos=True, batch_size=None):
        return torch.zeros(batch_size, 1, D), None


class FakeJoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f, self.g = torch.nn.Linear(D, 12), torch.nn.Linear(D, 12)
        self.out = torch.nn.Linear(12, VOCAB + 1 + len(DURATIONS))

    def forward(self, encoder_outputs, decoder_outputs):
        enc, dec = encoder_outputs.transpose(1, 2), decoder_outputs.transpose(1, 2)
        return self.out(torch.relu(self.f(enc).unsqueeze(2) + self.g(dec).unsqueeze(1)))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.decoder, self.joint = FakeDecoder(), FakeJoint()
        self.proj = torch.nn.Linear(1, D)
        self.loss = SimpleNamespace(_loss=SimpleNamespace(durations=DURATIONS, sigma=0.05, blank=VOCAB))

    def forward(self, input_signal, input_signal_length):
        frames = input_signal.shape[1] // FRAME
        pooled = input_signal[:, : frames * FRAME].reshape(input_signal.shape[0], frames, FRAME).abs().mean(-1, keepdim=True)
        return self.proj(pooled).transpose(1, 2), input_signal_length // FRAME


def clip(seconds_in_frames, seed):
    return np.random.default_rng(seed).normal(size=seconds_in_frames * FRAME).astype(np.float32)


def run(model, audio, refs, hyps=None):
    return rs.score_batch(model, audio, refs, hyps, "cpu", 0.05, DURATIONS, VOCAB)


@torch.no_grad()
def test_empty_and_reference_scores_equal_the_lattice_on_the_models_own_logits():
    model = FakeModel().eval()
    audio = [clip(6, 1), clip(9, 2)]
    refs = [[1, 2], [3, 3, 4]]
    rows = run(model, audio, refs)
    for b, (a, ref) in enumerate(zip(audio, refs)):
        signal = torch.from_numpy(a).unsqueeze(0)
        encoded, enc_len = model.forward(signal, torch.tensor([len(a)]))
        g, _, _ = model.decoder(torch.tensor([ref]), torch.tensor([len(ref)]))
        logits = model.joint(encoded, g)[0]
        tok, dur = tl.split_logprobs(logits, len(DURATIONS), 0.05)
        assert rows[b]["nll_ref"] == pytest.approx(float(tl.sequence_nll(tok, dur, ref, DURATIONS, VOCAB)), abs=1e-4)
        assert rows[b]["nll_empty"] == pytest.approx(float(tl.empty_nll(logits[:, 0, :], DURATIONS, VOCAB, 0.05)), abs=1e-4)
        assert rows[b]["margin"] == pytest.approx(rows[b]["nll_empty"] - rows[b]["nll_ref"])
        assert rows[b]["frames"] == int(enc_len[0]) and rows[b]["tokens"] == len(ref)


def test_a_clip_scores_the_same_alone_and_padded_inside_a_longer_batch():
    model = FakeModel().eval()
    short, long = clip(5, 3), clip(11, 4)
    alone = run(model, [short], [[2, 4]])[0]
    batched = run(model, [short, long], [[2, 4], [1, 1, 5, 2]])[0]
    assert batched["nll_ref"] == pytest.approx(alone["nll_ref"], abs=1e-4)
    assert batched["nll_empty"] == pytest.approx(alone["nll_empty"], abs=1e-4)


def test_empty_hypotheses_take_the_empty_score_and_others_are_scored_as_words():
    model = FakeModel().eval()
    audio = [clip(7, 5), clip(7, 6)]
    rows = run(model, audio, [[1, 2], [3]], hyps=[[], [4, 4]])
    assert rows[0]["nll_hyp"] == pytest.approx(rows[0]["nll_empty"])
    assert rows[1]["nll_hyp"] != pytest.approx(rows[1]["nll_empty"])


def test_pad_makes_a_rectangle_and_lengths():
    targets, lengths = rs.pad([[1, 2, 3], [4], []])
    assert targets.tolist() == [[1, 2, 3], [4, 0, 0], [0, 0, 0]] and lengths.tolist() == [3, 1, 0]
    assert rs.pad([[], []])[0].shape == (2, 0)


def test_summary_separates_search_misses_from_model_errors():
    rows = [
        {"nll_ref": 5.0, "nll_empty": 40.0, "margin": 35.0, "nll_hyp": 9.0, "hyp_empty": False, "errors": 2},
        {"nll_ref": 12.0, "nll_empty": 14.0, "margin": 2.0, "nll_hyp": 7.0, "hyp_empty": False, "errors": 1},
        {"nll_ref": 15.0, "nll_empty": 11.0, "margin": -4.0, "nll_hyp": 11.0, "hyp_empty": True, "errors": 3},
        {"nll_ref": 3.0, "nll_empty": 30.0, "margin": 27.0, "nll_hyp": 3.0, "hyp_empty": False, "errors": 0},
    ]
    summary = rs.summarise(rows)
    assert summary["clips"] == 4 and summary["median_margin"] == pytest.approx(14.5)
    assert summary["share_margin_below"]["0.0"] == 0.25 and summary["share_margin_below"]["10.0"] == 0.5
    h = summary["hypotheses"]
    assert h["empty"] == 1 and h["mean_margin_on_empty"] == -4.0
    assert h["wrong_nonempty"] == 2 and h["truth_more_likely_than_answer"] == 1 and h["answer_more_likely_than_truth"] == 1


def test_reference_text_forms():
    assert rs.reference_text("The little schoolhouse stood empty.", "normalized") == "the little schoolhouse stood empty"
    assert rs.reference_text("tomato", "stock") == "Tomato."
    assert rs.reference_text("The little schoolhouse stood empty.", "stock") == "The little schoolhouse stood empty."
    assert rs.reference_text("tomato", "raw") == "tomato"
