import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))

import adapter_soup  # noqa: E402
import block_adapter  # noqa: E402

BASE = {"name": "nvidia/parakeet-tdt-0.6b-v2"}


def payload(value: float, after_blocks: int = 8, width: int = 4, base: dict | None = None) -> dict:
    layer = block_adapter.AdapterLayer(width)
    torch.nn.init.constant_(layer.linear.weight, value)
    torch.nn.init.constant_(layer.linear.bias, value / 2)
    return {"format": block_adapter.FORMAT, "after_blocks": after_blocks, "width": width, "base": dict(base or BASE),
            "state_dict": {k: v.detach().clone() for k, v in layer.state_dict().items()}}


def test_the_average_is_the_mean_of_each_tensor():
    soup = adapter_soup.average_payloads([payload(1.0), payload(2.0), payload(6.0)])
    assert torch.allclose(soup["state_dict"]["linear.weight"], torch.full((4, 4), 3.0))
    assert torch.allclose(soup["state_dict"]["linear.bias"], torch.full((4,), 1.5))
    assert soup["after_blocks"] == 8 and soup["width"] == 4 and soup["base"] == BASE


def test_weights_shift_the_average():
    soup = adapter_soup.average_payloads([payload(0.0), payload(4.0)], weights=[1, 3])
    assert torch.allclose(soup["state_dict"]["linear.weight"], torch.full((4, 4), 3.0))


@pytest.mark.parametrize("other", [payload(1.0, after_blocks=6), payload(1.0, width=8), payload(1.0, base={"name": "another"})])
def test_adapters_from_different_positions_widths_or_bases_are_refused(other):
    with pytest.raises(ValueError, match="differs"):
        adapter_soup.average_payloads([payload(1.0), other])


def test_bad_weights_are_refused():
    with pytest.raises(ValueError):
        adapter_soup.average_payloads([payload(1.0), payload(2.0)], weights=[1.0])
    with pytest.raises(ValueError):
        adapter_soup.average_payloads([payload(1.0)], weights=[0.0])
    with pytest.raises(ValueError):
        adapter_soup.average_payloads([])


def test_the_command_writes_a_file_the_decoder_can_read(tmp_path):
    paths = []
    for i, v in enumerate((1.0, 3.0)):
        p = tmp_path / f"a{i}" / "adapter.pt"
        p.parent.mkdir()
        torch.save(payload(v), p)
        paths.append(str(p))
    out = tmp_path / "soup" / "adapter.pt"
    assert adapter_soup.main(["--inputs", *paths, "--out", str(out)]) == 0
    back = block_adapter.read_adapter(out)
    assert torch.allclose(back["state_dict"]["linear.weight"], torch.full((4, 4), 2.0))
    assert not out.with_suffix(".pt.tmp").exists()
