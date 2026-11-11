import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import export  # noqa: E402

IN, OUT = 6, 4
CH, K = 4, 3


def graph(tmp_path: Path, matmul_w: np.ndarray, matmul_zp: np.ndarray, conv_w: np.ndarray, conv_zp: np.ndarray) -> Path:
    inits = [numpy_helper.from_array(matmul_w, "mm_w"), numpy_helper.from_array(np.float32(0.1), "mm_s"), numpy_helper.from_array(matmul_zp, "mm_zp"),
             numpy_helper.from_array(conv_w, "cv_w"), numpy_helper.from_array(np.float32(0.1), "cv_s"), numpy_helper.from_array(conv_zp, "cv_zp")]
    nodes = [helper.make_node("MatMulInteger", ["x", "mm_w", "x_zp", "mm_zp"], ["y"], name="linear"),
             helper.make_node("ConvInteger", ["x2", "cv_w", "x_zp", "cv_zp"], ["y2"], name="depthwise")]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.UINT8, [1, IN]),
                                       helper.make_tensor_value_info("x2", TensorProto.UINT8, [1, CH, 8]),
                                       helper.make_tensor_value_info("x_zp", TensorProto.UINT8, [])],
                          [helper.make_tensor_value_info("y", TensorProto.INT32, None), helper.make_tensor_value_info("y2", TensorProto.INT32, None)],
                          initializer=inits)
    path = tmp_path / "encoder.int8.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]), str(path))
    return path


def test_per_tensor_zero_points_count_dead_tensors(tmp_path):
    mm = np.full((IN, OUT), 7, np.uint8)
    mm[:, 0] = 9
    cv = np.full((CH, 1, K), 3, np.uint8) + np.arange(K, dtype=np.uint8)
    result = export.int8_weight_check(graph(tmp_path, mm, np.uint8(7), cv, np.uint8(3)))
    assert result["tensors"] == 2 and result["mostly_zero"] == 1
    assert result["worst"][0]["node"] == "linear" and result["worst"][0]["at_zero_point"] == 0.75


def test_per_channel_zero_points_follow_each_layouts_channel_axis(tmp_path):
    mm_zp = np.array([1, 2, 3, 4], np.uint8)
    mm = np.tile(mm_zp, (IN, 1))
    mm[0, :] = 200
    cv_zp = np.array([10, 20, 30, 40], np.uint8)
    cv = np.tile(cv_zp[:, None, None], (1, 1, K))
    cv[1:] += 1
    result = export.int8_weight_check(graph(tmp_path, mm, mm_zp, cv, cv_zp))
    by = {t["node"]: t["at_zero_point"] for t in result["worst"]} | {"depthwise": None}
    assert result["tensors"] == 2
    assert by["linear"] == pytest.approx(5 / 6, abs=1e-4)
    assert result["mostly_zero"] == 1


def test_a_zero_point_count_that_matches_no_axis_is_skipped_not_crashed(tmp_path):
    mm = np.zeros((IN, OUT), np.uint8)
    cv = np.zeros((CH, 1, K), np.uint8)
    result = export.int8_weight_check(graph(tmp_path, mm, np.zeros(5, np.uint8), cv, np.zeros(CH, np.uint8)))
    assert result["tensors"] == 1
