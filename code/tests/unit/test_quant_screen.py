import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.unit.test_weights import write_nemo

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import quant_screen as qs

CHANNELS, KERNEL = 8, 9
SCALES = np.array([1, 1, 1, 1, 0.001, 1, 0.003, 1], np.float32)


def depthwise_weight() -> np.ndarray:
    w = np.random.default_rng(0).normal(size=(CHANNELS, 1, KERNEL)).astype(np.float32) * SCALES[:, None, None]
    w[3] *= 20
    return w


def test_dead_channels_are_the_slices_that_hold_only_the_zero_point():
    q = np.full((3, 1, 4), 7)
    q[1, 0, 2] = 8
    assert qs.dead_channels(q, 7) == [0, 2]
    assert qs.dead_channels(np.zeros((2, 5)), 0) == [0, 1]
    assert qs.dead_channels(np.array([[1, 7], [7, 7]]), 7, axis=1) == [1]


def test_quantize_rejects_an_unknown_scheme_and_keeps_zero_exact():
    with pytest.raises(ValueError):
        qs.quantize(np.ones(3), "float8")
    for scheme in ("uint8", "int8"):
        w = np.array([-1.0, 0.0, 0.5, 2.0], np.float32)
        q, scale, zero = qs.quantize(w, scheme)
        assert (q[1] - zero) * scale == 0.0


def conv_model(path: Path, name: str = "/layers.3/conv/depthwise_conv/Conv"):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, CHANNELS, 32])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, CHANNELS, 32])
    node = helper.make_node("Conv", ["x", "w"], ["y"], name=name, group=CHANNELS, pads=[4, 4], kernel_shape=[KERNEL])
    graph = helper.make_graph([node], "g", [x], [y], [numpy_helper.from_array(depthwise_weight(), "w")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


@pytest.mark.parametrize("scheme,kind", [("uint8", "QUInt8"), ("int8", "QInt8")])
def test_formula_matches_onnx_runtimes_dynamic_quantizer(tmp_path, scheme, kind):
    pytest.importorskip("onnx")
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic

    import onnx

    source = conv_model(tmp_path / "dw.onnx")
    quantized = tmp_path / "dw.q.onnx"
    quantize_dynamic(str(source), str(quantized), weight_type=getattr(QuantType, kind))
    init = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(quantized)).graph.initializer}
    q, scale, zero = qs.quantize(depthwise_weight(), scheme)
    assert np.array_equal(init["w_quantized"].astype(np.int32), q)
    assert scale == pytest.approx(float(init["w_scale"])) and zero == int(init["w_zero_point"])
    found = qs.screen_onnx(quantized)
    assert found["blocks"]["3"]["dead_installed"] == len(qs.dead_channels(q, zero)) == 2
    assert found["tensors"][0]["dead"] == [4, 6]


def state_with(raw_gamma: float | None = None) -> dict:
    w = torch.from_numpy(depthwise_weight())
    linear = torch.randn(16, 16)
    linear[2] = linear[2] * 1e-4
    linear[5] = linear[5] * 40
    gamma = torch.ones(CHANNELS)
    if raw_gamma is not None:
        gamma[[4, 6]] = raw_gamma
    return {"encoder.layers.1.conv.depthwise_conv.weight": w, "encoder.layers.1.conv.batch_norm.weight": gamma,
            "encoder.layers.1.conv.batch_norm.running_var": torch.ones(CHANNELS),
            "encoder.layers.1.feed_forward1.linear1.weight": linear, "decoder.prediction.embed.weight": torch.randn(5, 5),
            "encoder.layers.1.conv.batch_norm.num_batches_tracked": torch.tensor(3)}


def test_screen_state_counts_dead_channels_raw_and_after_folding_batchnorm():
    plain = qs.screen_state(state_with())["blocks"]["1"]
    assert plain["depthwise_channels"] == CHANNELS and plain["dead_raw"] == 2 and plain["dead_folded"] == 2
    assert plain["other_dead_rows"] >= 1
    rescued = qs.screen_state(state_with(raw_gamma=500.0))["blocks"]["1"]
    assert rescued["dead_raw"] == 2 and rescued["dead_folded"] == 0
    assert set(qs.screen_state(state_with())["blocks"]) == {"1"}


def test_render_names_each_model_and_hides_clean_blocks():
    clean = {"scheme": "uint8", "blocks": {"0": {"depthwise_channels": 8, "dead_raw": 0, "dead_folded": 0, "other_dead_rows": 0,
                                                   "relative_error_max": 0.01}}, "tensors": []}
    bad = qs.screen_state(state_with())
    text = qs.render({"stock": clean, "tuned": bad})
    assert "stock" in text and "tuned" in text and "\n1 " in text and "\n0 " not in text


def test_command_line_screens_checkpoints_and_exported_files(tmp_path):
    pytest.importorskip("onnx")
    nemo = write_nemo(tmp_path / "tuned.nemo", state_with())
    exported = tmp_path / "q.onnx"
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(conv_model(tmp_path / "dw.onnx")), str(exported), weight_type=QuantType.QUInt8)
    out = tmp_path / "screen.json"
    command = [sys.executable, str(ROOT / "scripts" / "asr" / "quant_screen.py"), "--nemo", f"tuned={nemo}", "--onnx", f"file={exported}",
               "--out", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr + done.stdout
    result = json.loads(out.read_text())["results"]
    assert result["tuned"]["blocks"]["1"]["dead_raw"] == 2 and result["file"]["blocks"]["3"]["dead_installed"] == 2
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode != 0
    assert subprocess.run([sys.executable, str(ROOT / "scripts" / "asr" / "quant_screen.py")], capture_output=True, text=True, check=False).returncode != 0


def test_cached_models_are_found_in_the_huggingface_cache(tmp_path, monkeypatch):
    import weights

    snapshot = tmp_path / "hub" / "models--nvidia--parakeet-tdt-0.6b-v3" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    write_nemo(snapshot / "parakeet-tdt-0.6b-v3.nemo", state_with())
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert weights.stock_nemo("nvidia/parakeet-tdt-0.6b-v3").name == "parakeet-tdt-0.6b-v3.nemo"
    with pytest.raises(FileNotFoundError, match="parakeet-tdt-1.1b"):
        weights.stock_nemo("nvidia/parakeet-tdt-1.1b")
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "asr" / "quant_screen.py"), "--cached", "v3=nvidia/parakeet-tdt-0.6b-v3"],
                          capture_output=True, text=True, check=False, env={**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub")})
    assert done.returncode == 0 and "v3: " in done.stdout, done.stderr + done.stdout


def test_dangling_optional_cache_link_does_not_lose_other_screen_results(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    snapshot = cache / "models--nvidia--missing" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "missing.nemo").symlink_to(tmp_path / "deleted-blob")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    model = write_nemo(tmp_path / "good.nemo", state_with())
    out = tmp_path / "screen.json"
    monkeypatch.setattr(sys, "argv", ["quant_screen", "--cached", "missing=nvidia/missing",
                                      "--nemo", f"good={model}", "--out", str(out)])
    assert qs.main() == 0
    result = json.loads(out.read_text())
    assert set(result["results"]) == {"good"} and set(result["missing_cached"]) == {"missing"}
