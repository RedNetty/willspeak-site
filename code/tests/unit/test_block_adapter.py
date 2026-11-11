import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import block_adapter as ba

WIDTH = 16


class StandInBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(WIDTH, WIDTH)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        out = torch.tanh(self.lin(x))
        if cache_last_channel is not None:
            return out, cache_last_channel, cache_last_time
        return out


class StandInEncoder(torch.nn.Module):
    def __init__(self, blocks=4):
        super().__init__()
        self.d_model = WIDTH
        self.layers = torch.nn.ModuleList(StandInBlock() for _ in range(blocks))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x=x, att_mask=None, pos_emb=None, pad_mask=None)
        return x


class StandInModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StandInEncoder()
        self.joint = torch.nn.Linear(WIDTH, 4)

    def forward(self, x):
        return self.joint(self.encoder(x))


def make_model(seed=0):
    torch.manual_seed(seed)
    return StandInModel()


def test_real_size_is_1_049_600_numbers():
    assert sum(p.numel() for p in ba.AdapterLayer(1024).parameters()) == 1_049_600


def test_zero_start_changes_nothing():
    model = make_model()
    x = torch.randn(2, 5, WIDTH)
    before = model(x)
    ba.add_adapter(model, 2)
    assert torch.equal(model(x), before)


def test_adapter_goes_after_the_first_n_blocks():
    model = make_model()
    ba.add_adapter(model, 2)
    kinds = [type(layer).__name__ for layer in model.encoder.layers]
    assert kinds == ["StandInBlock", "BlockThenAdapter", "StandInBlock", "StandInBlock"]


def test_position_out_of_range_is_refused():
    for bad in (0, 5):
        with pytest.raises(ValueError):
            ba.add_adapter(make_model(), bad)


def test_second_adapter_is_refused():
    model = make_model()
    ba.add_adapter(model, 2)
    with pytest.raises(ValueError):
        ba.add_adapter(model, 3)


def test_only_the_adapter_trains_and_gradients_reach_it_through_the_blocks_above():
    model = make_model()
    adapter = ba.add_adapter(model, 2)
    ba.freeze_all_but_adapter(model, adapter)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"encoder.layers.1.adapter.linear.weight", "encoder.layers.1.adapter.linear.bias"}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == WIDTH * WIDTH + WIDTH
    model(torch.randn(2, 5, WIDTH)).sum().backward()
    assert adapter.linear.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if n not in trainable)


def test_one_step_moves_the_output_and_only_the_adapter():
    model = make_model()
    adapter = ba.add_adapter(model, 2)
    ba.freeze_all_but_adapter(model, adapter)
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    x = torch.randn(2, 5, WIDTH)
    before = model(x).detach()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    model(x).pow(2).sum().backward()
    optimizer.step()
    assert not torch.equal(model(x), before)
    for n, p in model.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n])


def test_streaming_tuple_is_handled():
    model = make_model()
    ba.add_adapter(model, 2)
    block = model.encoder.layers[1]
    x = torch.randn(1, 3, WIDTH)
    cache_a, cache_b = torch.zeros(1), torch.ones(1)
    out = block(x=x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=cache_a, cache_last_time=cache_b)
    assert isinstance(out, tuple) and len(out) == 3
    assert out[1] is cache_a and out[2] is cache_b


def test_saved_adapter_reproduces_the_trained_model(tmp_path):
    trained = make_model()
    adapter = ba.add_adapter(trained, 2)
    with torch.no_grad():
        adapter.linear.weight.normal_()
        adapter.linear.bias.normal_()
    base = {"name": "some-base"}
    ba.save_adapter(tmp_path / "adapter.pt", adapter, 2, base)
    payload = ba.read_adapter(tmp_path / "adapter.pt")
    assert payload["after_blocks"] == 2 and payload["width"] == WIDTH

    fresh = make_model()
    ba.check_base(payload, base)
    ba.attach_saved_adapter(fresh, payload)
    x = torch.randn(2, 5, WIDTH)
    assert torch.equal(fresh(x), trained(x))


def test_adapter_for_another_base_is_refused(tmp_path):
    adapter = ba.add_adapter(make_model(), 2)
    ba.save_adapter(tmp_path / "adapter.pt", adapter, 2, {"sha256": "aaa", "size": 1})
    payload = ba.read_adapter(tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="different base model"):
        ba.check_base(payload, {"name": "nvidia/parakeet-tdt-0.6b-v2"})
    with pytest.raises(ValueError, match="different base model"):
        ba.check_base(payload, {"sha256": "bbb", "size": 1})
    ba.check_base(payload, {"sha256": "aaa", "size": 1})


def test_a_file_that_is_not_an_adapter_is_refused(tmp_path):
    torch.save({"format": "something-else"}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a willspeak adapter file"):
        ba.read_adapter(tmp_path / "other.pt")


def test_train_refuses_to_mix_the_adapter_with_other_training_choices():
    script = ROOT / "scripts" / "asr" / "train.py"
    for other in (["--train-parts", "all"], ["--freeze-encoder-layers", "16"]):
        result = subprocess.run(
            [sys.executable, str(script), "--check", "--adapter-after-blocks", "8", *other],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode != 0
        assert "trains only the adapter" in result.stderr
