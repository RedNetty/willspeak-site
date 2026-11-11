import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests.unit.test_margin_loss import FakeRNNTLoss
from tests.unit.test_training import train

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "asr" / "train.py"


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in train.BLOCK_PARTS:
            setattr(self, name, torch.nn.Linear(2, 2))


def stand_in(blocks=12):
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.layers = torch.nn.ModuleList(Block() for _ in range(blocks))
    return model


def open_blocks(model):
    return [i for i, b in enumerate(model.encoder.layers) if any(p.requires_grad for p in b.parameters())]


def test_default_window_is_still_the_bottom_eight():
    model = stand_in()
    assert train.train_lower_parts(model, train.BLOCK_PARTS) == 8
    assert open_blocks(model) == list(range(8))


def test_the_window_can_move_and_resize():
    model = stand_in()
    assert train.train_lower_parts(model, train.BLOCK_PARTS, 4, 6) == 4
    assert open_blocks(model) == [6, 7, 8, 9]
    assert train.train_lower_parts(model, train.BLOCK_PARTS, 2) == 2
    assert open_blocks(model) == [0, 1]


def test_only_the_named_parts_open_inside_the_window():
    model = stand_in()
    parts = train.parse_train_parts("norm_feed_forward1,feed_forward1,norm_feed_forward2,feed_forward2")
    train.train_lower_parts(model, parts, 3)
    block = model.encoder.layers[1]
    assert all(p.requires_grad for p in block.feed_forward1.parameters())
    assert not any(p.requires_grad for p in block.self_attn.parameters())
    assert not any(p.requires_grad for p in model.encoder.layers[5].feed_forward1.parameters())


def test_a_window_past_the_end_is_refused():
    with pytest.raises(ValueError, match="do not exist"):
        train.train_lower_parts(stand_in(12), train.BLOCK_PARTS, 8, 8)
    with pytest.raises(ValueError):
        train.train_lower_parts(stand_in(12), train.BLOCK_PARTS, 0)


@pytest.mark.parametrize("flags,message", [
    (["--train-first-block", "2"], "only apply together with --train-parts"),
    (["--train-parts", "all", "--train-blocks", "25"], "at most 24"),
    (["--train-parts", "all", "--train-first-block", "20", "--train-blocks", "8"], "at most 24"),
    (["--empty-margin", "-1"], "cannot be negative"),
    (["--empty-clips-per-kind", "-3"], "cannot be negative"),
    (["--empty-margin", "5", "--curriculum"], "not combined with --curriculum"),
])
def test_bad_combinations_stop_before_anything_is_read(flags, message):
    result = subprocess.run([sys.executable, str(SCRIPT), "--check", *flags], capture_output=True, text=True, check=False)
    assert result.returncode != 0 and message in result.stderr


def test_workflow_adds_empty_clips_and_installs_the_silence_loss(tmp_path, monkeypatch):
    pl = pytest.importorskip("lightning.pytorch")
    config = pytest.importorskip("omegaconf").OmegaConf
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    export, out = tmp_path / "export", tmp_path / "run"
    rng = np.random.default_rng(3)
    for split, count in (("train", 5), ("dev", 2)):
        (export / split).mkdir(parents=True)
        rows = []
        for i in range(count):
            name = f"{split}/{i}.wav"
            sf.write(export / name, rng.standard_normal(8000) * 0.1, 16000)
            rows.append({"id": f"{split}-{i}", "text": f"{split} phrase {i}", "audio": name, "duration_s": 0.5})
        (export / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (export / "test.jsonl").write_text("must never be opened by training")
    made = {}

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            made["model"] = self
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList()
            self.loss = FakeRNNTLoss()
            self.joint = SimpleNamespace(fuse_loss_wer=False)
            self.cfg = config.create({"train_ds": {}, "validation_ds": {}, "spec_augment": None})

        def set_trainer(self, trainer):
            self.trainer = trainer

        def setup_training_data(self, cfg):
            made["train_manifest"] = Path(cfg.manifest_filepath).read_text().splitlines()
            self.train_loader = torch.utils.data.DataLoader(torch.ones(len(made["train_manifest"]), 1), batch_size=cfg.batch_size)

        def setup_validation_data(self, cfg):
            self.val_loader = torch.utils.data.DataLoader(torch.ones(2, 1), batch_size=cfg.batch_size)

        def setup_optimization(self, cfg):
            self.rate = cfg.lr

        def train_dataloader(self):
            return self.train_loader

        def val_dataloader(self):
            return self.val_loader

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def validation_step(self, batch, batch_idx):
            self.log("val_wer", self.weight.detach().abs())

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=self.rate)

        def save_to(self, path):
            Path(path).write_text(str(self.weight.item()))

    modules = {name: ModuleType(name) for name in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.utils")}
    modules["nemo"].collections = modules["nemo.collections"]
    modules["nemo.collections"].asr = modules["nemo.collections.asr"]
    modules["nemo.collections.asr"].models = SimpleNamespace(ASRModel=SimpleNamespace(from_pretrained=lambda _: Model()))
    modules["nemo.utils"].logging = SimpleNamespace(INFO=20, WARNING=30, set_verbosity=lambda _: None)
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--export", str(export), "--out", str(out), "--allow-cpu", "--precision", "32-true", "--dev-scoring", "nemo",
        "--batch", "1", "--grad-accum", "2", "--lr", "0.1", "--epochs", "2", "--warmup-steps", "0", "--average-top", "2",
        "--select", "last", "--no-keep-stock", "--empty-clips-per-kind", "1", "--empty-margin", "5", "--empty-weight", "0.5"])
    assert train.run_training() == 0
    lines = [json.loads(line) for line in (out / "train_all.json").read_text().splitlines()]
    empties = [r for r in lines if r["text"] == ""]
    assert len(lines) == 5 + 7 and len(empties) == 7 and {r["condition"] for r in empties} >= {"white", "hum", "silence"}
    assert all(Path(r["audio_filepath"]).is_file() and r["audio_filepath"].startswith(str(out)) for r in empties)
    assert len((out / "empty-clips" / "manifest.jsonl").read_text().splitlines()) == 7
    assert all(json.loads(line)["text"] for line in (out / "dev.json").read_text().splitlines())
    run = json.loads((out / "run.json").read_text())
    assert run["silence_loss"] == {"margin_nats": 5.0, "weight": 0.5, "empty_clips": 7, "empty_clips_seed": 1000}
    assert run["phase_budgets"]["train_all"]["updates_per_epoch"] == 6
    assert type(made["model"].loss).__name__ == "SilenceAwareLoss" and made["model"].loss.margin == 5.0
