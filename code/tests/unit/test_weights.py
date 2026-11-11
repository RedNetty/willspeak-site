import io
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import nemo_hyp
import weights


def make_state(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "preprocessor.featurizer.window": torch.hann_window(8),
        "encoder.layers.0.conv.weight": torch.randn(4, 4, generator=g),
        "encoder.layers.0.norm.num_batches_tracked": torch.tensor(7),
        "encoder.layers.8.conv.weight": torch.randn(4, 4, generator=g),
        "decoder.embed.weight": torch.randn(3, 4, generator=g),
    }


def write_nemo(path: Path, state: dict) -> Path:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    data = buffer.getvalue()
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("./model_weights.ckpt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        config = b"sample_rate: 16000\n"
        info = tarfile.TarInfo("./model_config.yaml")
        info.size = len(config)
        tar.addfile(info, io.BytesIO(config))
    return path


def tuned_from(base: dict, moves: dict[str, torch.Tensor], steps: int = 12) -> dict:
    out = {k: v.clone() for k, v in base.items()}
    for key, delta in moves.items():
        out[key] = out[key] + delta
    out["encoder.layers.0.norm.num_batches_tracked"] = torch.tensor(steps)
    return out


def test_read_state_round_trip(tmp_path):
    state = make_state()
    loaded = weights.read_state(write_nemo(tmp_path / "m.nemo", state))
    assert loaded.keys() == state.keys()
    assert all(torch.equal(loaded[k], state[k]) for k in state)


def test_read_state_needs_a_nemo_checkpoint(tmp_path):
    path = tmp_path / "other.tar"
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("readme.txt")
        info.size = 0
        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(ValueError, match="model_weights.ckpt"):
        weights.read_state(path)


def test_halfway_is_the_midpoint_and_untouched_tensors_stay_exact(tmp_path):
    base = make_state()
    tuned = tuned_from(base, {"encoder.layers.0.conv.weight": torch.ones(4, 4)})
    blended, report = weights.blend(base, [(write_nemo(tmp_path / "t.nemo", tuned), 0.5)])
    assert torch.allclose(blended["encoder.layers.0.conv.weight"], base["encoder.layers.0.conv.weight"] + 0.5)
    for key in ("preprocessor.featurizer.window", "encoder.layers.8.conv.weight", "decoder.embed.weight"):
        assert torch.equal(blended[key], base[key])
    assert report["tensors_changed"] == 1 and report["tensors"] == len(base)


def test_integer_buffers_come_from_the_base(tmp_path):
    base = make_state()
    tuned = tuned_from(base, {"decoder.embed.weight": torch.ones(3, 4)}, steps=99)
    blended, _ = weights.blend(base, [(write_nemo(tmp_path / "t.nemo", tuned), 1.0)])
    assert int(blended["encoder.layers.0.norm.num_batches_tracked"]) == 7


def test_weight_one_is_the_tuned_model_and_zero_is_the_base(tmp_path):
    base = make_state(0)
    tuned = tuned_from(base, {"encoder.layers.0.conv.weight": torch.randn(4, 4), "decoder.embed.weight": torch.randn(3, 4)})
    path = write_nemo(tmp_path / "t.nemo", tuned)
    one, _ = weights.blend(base, [(path, 1.0)])
    zero, _ = weights.blend(base, [(path, 0.0)])
    for key in base:
        if base[key].is_floating_point():
            assert torch.allclose(one[key], tuned[key], atol=1e-6)
        assert torch.equal(zero[key], base[key])


def test_two_models_merge_as_the_sum_of_their_changes(tmp_path):
    base = make_state()
    a = tuned_from(base, {"encoder.layers.0.conv.weight": torch.ones(4, 4)})
    b = tuned_from(base, {"encoder.layers.0.conv.weight": 3 * torch.ones(4, 4), "decoder.embed.weight": torch.ones(3, 4)})
    merged, report = weights.blend(base, [(write_nemo(tmp_path / "a.nemo", a), 0.5), (write_nemo(tmp_path / "b.nemo", b), 0.5)])
    assert torch.allclose(merged["encoder.layers.0.conv.weight"], base["encoder.layers.0.conv.weight"] + 2.0)
    assert torch.allclose(merged["decoder.embed.weight"], base["decoder.embed.weight"] + 0.5)
    assert report["weights_sum"] == 1.0 and len(report["files"]) == 2


def test_weights_need_not_sum_to_one(tmp_path):
    base = make_state()
    tuned = tuned_from(base, {"decoder.embed.weight": torch.ones(3, 4)})
    path = write_nemo(tmp_path / "t.nemo", tuned)
    over, _ = weights.blend(base, [(path, 1.5)])
    assert torch.allclose(over["decoder.embed.weight"], base["decoder.embed.weight"] + 1.5)


def test_a_checkpoint_with_other_tensors_is_refused(tmp_path):
    base = make_state()
    wrong = dict(base)
    wrong["extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="does not have the base model's tensors"):
        weights.blend(base, [(write_nemo(tmp_path / "w.nemo", wrong), 0.5)])
    reshaped = dict(base)
    reshaped["decoder.embed.weight"] = torch.zeros(5, 4)
    with pytest.raises(ValueError, match="shape"):
        weights.blend(base, [(write_nemo(tmp_path / "r.nemo", reshaped), 0.5)])


def test_non_finite_values_and_weights_are_refused(tmp_path):
    base = make_state()
    broken = tuned_from(base, {"decoder.embed.weight": torch.full((3, 4), float("nan"))})
    with pytest.raises(ValueError, match="non-finite"):
        weights.blend(base, [(write_nemo(tmp_path / "n.nemo", broken), 0.5)])
    fine = write_nemo(tmp_path / "f.nemo", tuned_from(base, {"decoder.embed.weight": torch.ones(3, 4)}))
    with pytest.raises(ValueError, match="finite number"):
        weights.blend(base, [(fine, float("inf"))])
    with pytest.raises(ValueError, match="at least one"):
        weights.blend(base, [])


def test_blend_does_not_change_the_base(tmp_path):
    base = make_state()
    before = {k: v.clone() for k, v in base.items()}
    weights.blend(base, [(write_nemo(tmp_path / "t.nemo", tuned_from(base, {"decoder.embed.weight": torch.ones(3, 4)})), 0.5)])
    assert all(torch.equal(base[k], before[k]) for k in base)


def test_block_grouping():
    assert weights.block_of("encoder.layers.3.self_attn.weight") == "block03"
    assert weights.block_of("encoder.pre_encode.conv.0.weight") == "encoder.pre_encode"
    assert weights.block_of("decoder.prediction.embed.weight") == "decoder"
    assert weights.block_of("joint.joint_net.2.weight") == "joint"


def test_geometry_cosines_tell_apart_agreeing_unrelated_and_opposing_changes():
    base = make_state()
    base["decoder.embed.weight"] = torch.zeros(3, 4)
    up = tuned_from(base, {"decoder.embed.weight": torch.ones(3, 4)})
    same = tuned_from(base, {"decoder.embed.weight": 2 * torch.ones(3, 4)})
    down = tuned_from(base, {"decoder.embed.weight": -torch.ones(3, 4)})
    other = tuned_from(base, {"encoder.layers.8.conv.weight": torch.ones(4, 4)})
    result = weights.geometry(base, {"up": up, "same": same, "down": down, "other": other})
    cos = result["cosine_between_changes"]
    assert cos["up~same"] == pytest.approx(1.0)
    assert cos["up~down"] == pytest.approx(-1.0)
    assert cos["up~other"] == pytest.approx(0.0)
    assert result["change_size"]["same"]["total_l2"] == pytest.approx(2 * result["change_size"]["up"]["total_l2"])
    assert "block08" in result["change_size"]["other"]["per_block_l2"]


def test_mix_flags_parse_and_reject_junk():
    assert nemo_hyp.parse_mix(["a/b.nemo=0.25", "c.nemo=-1"]) == [("a/b.nemo", 0.25), ("c.nemo", -1.0)]
    assert nemo_hyp.parse_mix([r"C:\models\m.nemo=0.5"]) == [(r"C:\models\m.nemo", 0.5)]
    for bad in ("nofile", "=0.5", "m.nemo=half"):
        with pytest.raises(ValueError):
            nemo_hyp.parse_mix([bad])


@pytest.mark.parametrize("stock", [True, False])
def test_nemo_loader_moves_model_to_cpu_before_blending(tmp_path, monkeypatch, stock):
    events = []
    base = make_state()
    tuned = tuned_from(base, {"decoder.embed.weight": torch.ones(3, 4)})
    checkpoint = write_nemo(tmp_path / "tuned.nemo", tuned)

    class LoadedModel:
        def cpu(self):
            events.append("cpu")
            return self

        def state_dict(self):
            assert events == ["cpu"]
            events.append("blend")
            return base

        def load_state_dict(self, state):
            assert torch.allclose(state["decoder.embed.weight"], base["decoder.embed.weight"] + 0.5)
            events.append("loaded")

    model = LoadedModel()
    nemo = ModuleType("nemo")
    collections = ModuleType("nemo.collections")
    asr = ModuleType("nemo.collections.asr")
    asr.models = SimpleNamespace(ASRModel=SimpleNamespace(
        from_pretrained=lambda name: model, restore_from=lambda path: model))
    nemo.collections = collections
    collections.asr = asr
    for name, module in (("nemo", nemo), ("nemo.collections", collections), ("nemo.collections.asr", asr)):
        monkeypatch.setitem(sys.modules, name, module)
    args = SimpleNamespace(stock=stock, pretrained=None, nemo="base.nemo", mix=[f"{checkpoint}=0.5"], adapter=None)
    loaded, record = nemo_hyp.load_model(args, {})
    assert loaded is model and record["mix"]["tensors_changed"] == 1
    assert events == ["cpu", "blend", "loaded"]


def test_stock_nemo_ignores_a_snapshot_link_whose_file_was_deleted(tmp_path, monkeypatch):
    snap = tmp_path / "models--nvidia--demo" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "demo.nemo").symlink_to(tmp_path / "gone-blob")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        weights.stock_nemo("nvidia/demo")
    (tmp_path / "blob").write_bytes(b"x")
    (snap / "demo.nemo").unlink()
    (snap / "demo.nemo").symlink_to(tmp_path / "blob")
    assert weights.stock_nemo("nvidia/demo").name == "demo.nemo"
