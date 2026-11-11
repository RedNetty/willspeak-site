#!/usr/bin/env python3
"""Export a Parakeet-TDT checkpoint to the ONNX trio sherpa-onnx runs on the CPU box, plus tokens.txt.

The default quantisation follows sherpa-onnx's parakeet-tdt-0.6b-v2 export recipe (Apache-2.0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import _cli
from willspeak.artifacts import file_record, new_directory, timestamp, versions, write_json

STOCK = "nvidia/parakeet-tdt-0.6b-v2"
PARTS = ("encoder", "decoder", "joiner")


def add_meta(path: Path, meta: dict) -> None:
    import onnx

    m = onnx.load(str(path))
    while len(m.metadata_props):
        m.metadata_props.pop()
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key = k
        p.value = str(v)
    if path.name == "encoder.onnx":
        onnx.save(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="encoder.weights")
    else:
        onnx.save(m, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--nemo", help="fine-tuned .nemo file")
    source.add_argument("--stock", action="store_true", help=f"export {STOCK} instead")
    ap.add_argument("--adapter", help="adapter.pt from train.py --adapter-after-blocks, added to the base "
                                      "model (--stock or --nemo, whichever it was trained on) before export")
    ap.add_argument("--out", help="output directory")
    ap.add_argument("--no-int8", action="store_true", help="skip quantisation (fp32 only)")
    ap.add_argument("--int8-parts", nargs="+", choices=PARTS, default=list(PARTS),
                    help="which of the three graphs to quantise (default all); the others stay fp32. "
                         "The decoder and joiner are small, so leaving them fp32 costs little")
    ap.add_argument("--int8-ops", choices=("all", "matmul"), default="all",
                    help="'all' (default, the recipe of every recorded export) quantises matrix products and "
                         "convolutions; 'matmul' leaves every convolution in fp32, including the depthwise "
                         "convolutions whose folded BatchNorm scales the weight check reports on")
    ap.add_argument("--per-channel", action="store_true",
                    help="one int8 scale per output channel instead of one per tensor")
    ap.add_argument("--check-audio", help="a 16 kHz recording to decode after export (default: the stock "
                                          "model's test_wavs/ when present)")
    a = ap.parse_args()
    from willspeak.asr.parakeet import ParakeetASR
    from willspeak.paths import MODELS_DIR

    if a.stock and a.adapter:
        default_out = (ROOT / a.adapter).parent / "onnx-int8"
    elif a.stock:
        default_out = MODELS_DIR / "parakeet-tdt-0.6b-v2-int8"
    elif a.adapter:
        default_out = (ROOT / a.adapter).parent / "onnx-int8"
    else:
        default_out = (ROOT / a.nemo).parent / "onnx-int8"
    out = ROOT / a.out if a.out else default_out
    if out.exists():
        raise FileExistsError(f"Export destination already exists; choose a new name: {out}")
    if not ParakeetASR.installed():
        raise ImportError("sherpa-onnx is required to validate an export before releasing it")
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        print(f'missing dependency: {e}\n  python -m pip install -e ".[train]"')
        return 1

    if a.stock:
        model = nemo_asr.models.ASRModel.from_pretrained(STOCK)
        base_id = {"name": STOCK}
    else:
        nemo = ROOT / a.nemo
        if not nemo.exists():
            print(f"No trained model at {nemo}\n"
                  "  train.py writes parakeet-ft.nemo only when a run finishes; read that run's log in logs/")
            return 1
        source_record = file_record(nemo)
        base_id = source_record
        model = nemo_asr.models.ASRModel.restore_from(str(nemo))
    adapter_record = None
    if a.adapter:
        import block_adapter

        adapter_path = ROOT / a.adapter
        if not adapter_path.is_file():
            print(f"No adapter at {adapter_path}")
            return 1
        payload = block_adapter.read_adapter(adapter_path)
        block_adapter.check_base(payload, base_id)
        block_adapter.attach_saved_adapter(model, payload)
        adapter_record = file_record(adapter_path)
        print(f"Added the adapter from {a.adapter}: after encoder blocks 0-{payload['after_blocks'] - 1}")
    model.cpu().eval()
    check_audio = None
    if a.check_audio:
        check_audio = ROOT / a.check_audio
        if not check_audio.is_file():
            print(f"No recording at {check_audio}")
            return 1
    else:
        wavs = sorted((MODELS_DIR / "parakeet-tdt-0.6b-v2-int8" / "test_wavs").glob("*.wav"))
        check_audio = wavs[0] if wavs else None
    int8_parts = [] if a.no_int8 else a.int8_parts
    with new_directory(out) as stage:
        decodes, weight_check = export_model(model, stage, int8_parts, int8_ops=a.int8_ops,
                                             per_channel=a.per_channel, check_audio=check_audio)
        provenance = {
            "format": "willspeak-export-v1",
            "created_at": timestamp(),
            "source": STOCK if a.stock else str(nemo),
            "int8": not a.no_int8,
            "int8_parts": list(int8_parts),
            "int8_ops": a.int8_ops if int8_parts else None,
            "per_channel": bool(a.per_channel) if int8_parts else None,
            "packages": versions(),
            "script": file_record(Path(__file__)),
            "smoke_decode": decodes,
            "int8_weight_check": weight_check,
        }
        if adapter_record:
            provenance["adapter_file"] = adapter_record
            provenance["adapter_after_blocks"] = int(payload["after_blocks"])
        if not a.stock:
            if file_record(nemo) != source_record:
                raise ValueError("Source checkpoint changed during export")
            provenance["source_file"] = source_record
            if (nemo.parent / "run.json").is_file():
                import shutil

                shutil.copyfile(nemo.parent / "run.json", stage / "training.json")
        if a.adapter and (adapter_path.parent / "run.json").is_file():
            import shutil

            shutil.copyfile(adapter_path.parent / "run.json", stage / "training.json")
        write_json(stage / "export.json", provenance)
    print(f"Export validated: {out}\nRun: python scripts/asr/check.py --model-dir {out} --audio <sample.wav>")
    return 0


def int8_weight_check(path: Path, dead_fraction: float = 0.5) -> dict:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path), load_external_data=True)
    init = {t.name: t for t in m.graph.initializer}
    tensors = []
    for node in m.graph.node:
        if node.op_type not in ("ConvInteger", "MatMulInteger"):
            continue
        if len(node.input) < 4 or node.input[1] not in init or node.input[3] not in init:
            continue
        w = numpy_helper.to_array(init[node.input[1]])
        zp = numpy_helper.to_array(init[node.input[3]])
        if w.dtype not in (np.uint8, np.int8):
            continue
        if zp.size == 1:
            at_zero = float(np.mean(w == zp.reshape(()).item()))
        else:
            axes = [i for i, n in enumerate(w.shape) if n == zp.size]
            if not axes:
                continue
            shape = [1] * w.ndim
            shape[axes[-1] if node.op_type == "MatMulInteger" else axes[0]] = -1
            at_zero = float(np.mean(w == zp.reshape(shape)))
        tensors.append({"node": node.name or node.input[1], "op": node.op_type,
                        "values": int(w.size), "at_zero_point": round(at_zero, 4)})
    dead = sorted((t for t in tensors if t["at_zero_point"] > dead_fraction),
                  key=lambda t: -t["at_zero_point"])
    return {"tensors": len(tensors), "mostly_zero": len(dead), "threshold": dead_fraction, "worst": dead[:8]}


def export_model(model, out: Path, int8_parts: list[str], *, int8_ops: str = "all",
                 per_channel: bool = False, check_audio: Path | None = None) -> tuple[dict, dict | None]:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with open(out / "tokens.txt", "w", encoding="utf-8") as f:
        i = -1
        for i, s in enumerate(model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i + 1}\n")
    print(f"wrote tokens.txt ({i + 2} entries)")

    with torch.no_grad():
        model.encoder.export(str(out / "encoder.onnx"))
        model.decoder.export(str(out / "decoder.onnx"))
        model.joint.export(str(out / "joiner.onnx"))
    normalize = model.cfg.preprocessor.normalize
    meta = {
        "vocab_size": model.decoder.vocab_size,
        "normalize_type": "" if normalize == "NA" else normalize,
        "pred_rnn_layers": model.decoder.pred_rnn_layers,
        "pred_hidden": model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "2",
        "model_author": "NeMo",
        "url": STOCK,
        "comment": "willspeak export; transducer branch only",
        "feat_dim": int(model.cfg.preprocessor.features),
    }
    add_meta(out / "encoder.onnx", meta)
    if int8_parts:
        import logging

        root = logging.getLogger()
        previous = root.level
        root.setLevel(logging.ERROR)
        try:
            for part in int8_parts:
                print(f"quantizing {part} to int8 ({int8_ops} ops, {'per channel' if per_channel else 'per tensor'}) ...",
                      flush=True)
                options = {}
                if int8_ops == "matmul":
                    options["op_types_to_quantize"] = ["MatMul"]
                if per_channel:
                    options["per_channel"] = True
                quantize_dynamic(
                    model_input=str(out / f"{part}.onnx"),
                    model_output=str(out / f"{part}.int8.onnx"),
                    weight_type=QuantType.QUInt8 if part == "encoder" else QuantType.QInt8,
                    **options,
                )
        finally:
            root.setLevel(previous)
        if "encoder" in int8_parts:
            add_meta(out / "encoder.int8.onnx", meta)
            weight_check = int8_weight_check(out / "encoder.int8.onnx")
            print(f"int8 weight check: {weight_check['mostly_zero']} of {weight_check['tensors']} quantised weight "
                  f"tensors have more than {weight_check['threshold']:.0%} of their values at the zero point")
            for t in weight_check["worst"]:
                print(f"  {t['op']} {t['node']}: {t['at_zero_point']:.1%} of {t['values']} values are zero")
            if weight_check["mostly_zero"]:
                print("  Those weights no longer act on the device. Compare --int8-ops matmul or --per-channel "
                      "on the F-2 clips before trusting this export (docs/training.md, Export).")
        else:
            weight_check = None
        keep = {f"{part}.int8.onnx" if part in int8_parts else f"{part}.onnx" for part in PARTS} | {"tokens.txt"}
        if "encoder" not in int8_parts:
            keep.add("encoder.weights")
        for generated in out.iterdir():
            if generated.name not in keep and generated.is_file():
                generated.unlink()
    else:
        weight_check = None
    import numpy as np
    import soundfile as sf

    from willspeak.asr.parakeet import ParakeetASR

    asr = ParakeetASR(out)
    decodes = {"note": "not an accuracy evaluation"}
    sample = out / "_silence.wav"
    sf.write(sample, np.zeros(16000, np.float32), 16000)
    try:
        result = asr.transcribe(str(sample))
        decodes["silence"] = {"text": result["text"], "latency_ms": result["latency_ms"]}
        print(f"Validated sherpa-onnx decode: {result['latency_ms']} ms including model load; "
              f"silence -> {result['text']!r}")
        if result["text"]:
            print("  WARNING: the model answers one second of silence; the stock export says nothing")
    finally:
        sample.unlink(missing_ok=True)
    if check_audio is not None:
        result = asr.transcribe(str(check_audio))
        decodes["recording"] = {"file": str(check_audio), "text": result["text"],
                                "latency_ms": result["latency_ms"]}
        print(f"Recording {check_audio.name} -> {result['text']!r} ({result['latency_ms']} ms)")
        if not result["text"]:
            print("  WARNING: empty transcript on a real recording; the encoder may be damaged")
    return decodes, weight_check


if __name__ == "__main__":
    sys.exit(_cli.run(main, "export_parakeet"))
