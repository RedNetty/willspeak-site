#!/usr/bin/env python3
"""Will this checkpoint survive int8? Which channels does the quantizer flatten? Lab test 15."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

import labkit as lab

BN_EPS = 1e-5
BLOCK = re.compile(r"layers[._/](\d+)")


def quantize(w: np.ndarray, scheme: str = "uint8") -> tuple[np.ndarray, float, int]:
    w = np.asarray(w, np.float32)
    if scheme == "uint8":
        low, high = min(float(w.min()), 0.0), max(float(w.max()), 0.0)
        scale = (high - low) / 255 or 1.0
        zero = int(np.clip(np.round(-low / scale), 0, 255))
        return np.clip(np.round(w / scale) + zero, 0, 255).astype(np.int32), float(scale), zero
    if scheme == "int8":
        scale = float(np.abs(w).max()) / 127 or 1.0
        return np.clip(np.round(w / scale), -128, 127).astype(np.int32), scale, 0
    raise ValueError(f"scheme must be uint8 or int8, not {scheme!r}")


def dead_channels(q: np.ndarray, zero: int, axis: int = 0) -> list[int]:
    flat = np.moveaxis(q, axis, 0).reshape(q.shape[axis], -1)
    return [int(c) for c in np.where((flat == zero).all(1))[0]]


def block_of(name: str) -> str:
    match = BLOCK.search(name)
    return match.group(1) if match else ("subsampling" if "pre_encode" in name else "other")


def fold_batchnorm(weight: np.ndarray, state: dict, prefix: str) -> np.ndarray | None:
    keys = [f"{prefix}.{k}" for k in ("weight", "running_var")]
    if not all(k in state for k in keys):
        return None
    gamma, var = (state[k].float().numpy() for k in keys)
    return weight * (gamma / np.sqrt(var + BN_EPS)).reshape(-1, *([1] * (weight.ndim - 1)))


def screen_state(state: dict, scheme: str = "uint8") -> dict:
    blocks: dict[str, dict] = {}
    tensors = []
    for name, value in state.items():
        if not name.startswith("encoder.") or not name.endswith(".weight") or not hasattr(value, "ndim") or value.ndim < 2:
            continue
        if not value.dtype.is_floating_point:
            continue
        w = value.float().numpy()
        depthwise = w.ndim >= 3 and w.shape[1] == 1 and w.shape[0] > 1
        b = blocks.setdefault(block_of(name), {"depthwise_channels": 0, "dead_raw": 0, "dead_folded": 0, "other_dead_rows": 0,
                                               "relative_error_max": 0.0})
        q, scale, zero = quantize(w, scheme)
        error = float(np.linalg.norm((q - zero) * scale - w) / (np.linalg.norm(w) or 1.0))
        b["relative_error_max"] = max(b["relative_error_max"], error)
        raw = dead_channels(q, zero)
        if depthwise:
            folded = fold_batchnorm(w, state, name.rsplit(".depthwise_conv.weight", 1)[0] + ".batch_norm") if name.endswith(".depthwise_conv.weight") else None
            folded_dead = raw
            if folded is not None:
                folded_q, _, folded_zero = quantize(folded, scheme)
                folded_dead = dead_channels(folded_q, folded_zero)
            b["depthwise_channels"] += w.shape[0]
            b["dead_raw"] += len(raw)
            b["dead_folded"] += len(folded_dead)
            if raw or folded_dead:
                tensors.append({"tensor": name, "channels": int(w.shape[0]), "dead_raw": raw, "dead_folded": folded_dead})
        else:
            b["other_dead_rows"] += len(raw)
            if raw:
                tensors.append({"tensor": name, "rows": int(w.shape[0]), "dead_rows": raw})
    return {"scheme": scheme, "blocks": blocks, "tensors": tensors}


def screen_onnx(path: str | Path) -> dict:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(lab.here(path)))
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    blocks: dict[str, dict] = {}
    tensors = []
    for node in model.graph.node:
        if node.op_type not in ("ConvInteger", "MatMulInteger") or len(node.input) < 2 or node.input[1] not in init:
            continue
        q = init[node.input[1]].astype(np.int32)
        zero = int(init[node.input[3]]) if len(node.input) > 3 and node.input[3] in init else 0
        group = next((int(a.i) for a in node.attribute if a.name == "group"), 1)
        axis = 1 if node.op_type == "MatMulInteger" else 0
        dead = dead_channels(q, zero, axis)
        b = blocks.setdefault(block_of(node.name or node.input[1]), {"depthwise_channels": 0, "dead_installed": 0, "other_dead_rows": 0})
        if node.op_type == "ConvInteger" and group > 1:
            b["depthwise_channels"] += q.shape[0]
            b["dead_installed"] += len(dead)
        else:
            b["other_dead_rows"] += len(dead)
        if dead:
            tensors.append({"node": node.name, "channels": int(q.shape[axis]), "dead": dead, "depthwise": group > 1})
    if not blocks:
        raise ValueError(f"{path} has no ConvInteger or MatMulInteger nodes; is it an int8 export (encoder.int8.onnx)?")
    return {"scheme": "stored in the file", "blocks": blocks, "tensors": tensors}


def dead_counts(info: dict) -> tuple[int, ...]:
    return (info["dead_installed"],) if "dead_installed" in info else (info["dead_folded"], info["dead_raw"])


def render(results: dict[str, dict]) -> str:
    labels = list(results)
    order = sorted({b for r in results.values() for b in r["blocks"]}, key=lambda b: (not b.isdigit(), int(b) if b.isdigit() else 0, b))
    column = max(max(len(label) for label in labels), 14) + 2
    out = ["dead depthwise channels per block: one number for an exported file, 'folded/raw' for a simulated checkpoint "
           "(BatchNorm folded in or not); blocks with none anywhere are left out",
           f"{'block':<12}" + "".join(f"{label:>{column}}" for label in labels)]
    for b in order:
        cells, quiet = [], True
        for label in labels:
            info = results[label]["blocks"].get(b)
            if info is None:
                cells.append("-")
                continue
            quiet = quiet and not any(dead_counts(info)) and not info["other_dead_rows"]
            cells.append("/".join(str(n) for n in dead_counts(info)) + f" of {info['depthwise_channels']}")
        if not quiet:
            out.append(f"{b:<12}" + "".join(f"{c:>{column}}" for c in cells))
    for label in labels:
        blocks = results[label]["blocks"].values()
        dead = sum(dead_counts(b)[0] for b in blocks)
        out.append(f"{label}: {dead} dead depthwise channels, {sum(b['other_dead_rows'] for b in blocks)} dead rows in other weights")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stock", action="store_true", help="include the stock Parakeet-TDT-0.6B-v2 from the Hugging Face cache")
    ap.add_argument("--cached", action="append", default=[], metavar="LABEL=HF_NAME",
                    help="another downloaded model, e.g. v3=nvidia/parakeet-tdt-0.6b-v3 (it must have been decoded once so NeMo cached it)")
    ap.add_argument("--nemo", action="append", default=[], metavar="LABEL=FILE.nemo", help="a checkpoint to screen; repeat")
    ap.add_argument("--onnx", action="append", default=[], metavar="LABEL=encoder.int8.onnx", help="an exported int8 encoder to read; repeat")
    ap.add_argument("--scheme", default="uint8", choices=["uint8", "int8"], help="weight type to simulate (export.py quantizes the encoder as uint8)")
    ap.add_argument("--out", help="also write the result as .json (refuses to overwrite)")
    a = ap.parse_args()
    if not (a.stock or a.cached or a.nemo or a.onnx):
        ap.error("give --stock, --cached, --nemo or --onnx")
    if a.out:
        lab.refuse_existing(a.out)
    results: dict[str, dict] = {}
    sources: dict[str, str] = {}
    missing_cached: dict[str, str] = {}

    def add(label: str, value: dict, path) -> None:
        if label in results:
            raise ValueError(f"Label {label!r} used twice")
        results[label] = value
        sources[label] = str(path)

    if a.stock:
        import weights

        path = weights.stock_nemo()
        add("stock", screen_state(weights.read_state(path), a.scheme), path)
    for text in a.cached:
        import weights

        label, _, name = text.partition("=")
        if not name:
            ap.error(f"--cached wants LABEL=HF_NAME, got {text!r}")
        try:
            path = weights.stock_nemo(name)
        except FileNotFoundError as error:
            missing_cached[label] = str(error)
            print(f"SKIPPED cached {label}: {error}")
            continue
        add(label, screen_state(weights.read_state(path), a.scheme), path)
    for text in a.nemo:
        import weights

        label, _, path = text.partition("=")
        if not path:
            ap.error(f"--nemo wants LABEL=FILE.nemo, got {text!r}")
        add(label, screen_state(weights.read_state(path), a.scheme), path)
    for text in a.onnx:
        label, _, path = text.partition("=")
        if not path:
            ap.error(f"--onnx wants LABEL=FILE, got {text!r}")
        add(label, screen_onnx(path), path)
    if not results:
        raise ValueError("No models were screened; restore a checkpoint or an int8 export first")
    print(render(results))
    if a.out:
        lab.save_json(a.out, {"results": results, "missing_cached": missing_cached,
                              "provenance": lab.provenance(__file__, sources)})
        print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "quant_screen"))
