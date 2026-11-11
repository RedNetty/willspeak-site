#!/usr/bin/env python3
"""Transcribe a manifest with a NeMo checkpoint, in NeMo itself, and write the hypotheses for ``evaluate.py --hyp``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _cli
from willspeak.artifacts import contained_file, file_record, timestamp, versions, write_json
from willspeak.evaluate import load_manifest

STOCK = "nvidia/parakeet-tdt-0.6b-v2"


def text_of(item) -> str:
    text = getattr(item, "text", item)
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError(f"NeMo returned a non-text hypothesis: {type(text).__name__}")
    return text


def here(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def add_model_args(ap: argparse.ArgumentParser) -> None:
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--nemo", help="checkpoint to load (.nemo)")
    source.add_argument("--stock", action="store_true", help=f"load {STOCK} instead")
    source.add_argument("--pretrained", help="load a NeMo model name supplied to train.py --base")
    ap.add_argument("--adapter", help="adapter.pt from train.py --adapter-after-blocks, added to the model")
    ap.add_argument("--mix", action="append", default=[], metavar="FILE=WEIGHT",
                    help="move the weights toward another .nemo by WEIGHT of the way (repeatable); see the top of this file")


def parse_mix(items: list[str]) -> list[tuple[str, float]]:
    out = []
    for item in items:
        path, sep, weight = item.rpartition("=")
        if not sep or not path:
            raise ValueError(f"--mix wants FILE=WEIGHT, got {item!r}")
        try:
            out.append((path, float(weight)))
        except ValueError:
            raise ValueError(f"--mix weight must be a number, got {weight!r}") from None
    return out


def model_inputs(a) -> tuple[str | None, dict[str, Path]]:
    pretrained = STOCK if a.stock else a.pretrained
    paths: dict[str, Path] = {}
    if pretrained is None:
        paths["nemo"] = here(a.nemo)
    if a.adapter:
        paths["adapter"] = here(a.adapter)
    for i, (path, _) in enumerate(parse_mix(a.mix)):
        paths[f"mix{i}"] = here(path)
    return pretrained, paths


def load_model(a, files: dict[str, dict]) -> tuple[object, dict]:
    import nemo.collections.asr as nemo_asr

    pretrained = STOCK if a.stock else a.pretrained
    if pretrained is not None:
        model = nemo_asr.models.ASRModel.from_pretrained(pretrained)
    else:
        model = nemo_asr.models.ASRModel.restore_from(str(here(a.nemo)))
    model = model.cpu()
    record: dict = {"base": pretrained or str(here(a.nemo))}
    mix = parse_mix(a.mix)
    if mix:
        import weights

        state, report = weights.blend(model.state_dict(), [(here(path), w) for path, w in mix])
        model.load_state_dict(state)
        record["mix"] = report
        print(f"Mixed in {len(mix)} checkpoint(s): {report['tensors_changed']} of {report['tensors']} tensors moved "
              f"(weights {[w for _, w in mix]}, sum {report['weights_sum']:g})")
    if a.adapter:
        import block_adapter

        payload = block_adapter.read_adapter(here(a.adapter))
        base = {"name": pretrained} if pretrained is not None else files["nemo"]
        if mix:
            raise ValueError("--adapter with --mix: the adapter records one base model; train it on the mixed model first")
        block_adapter.check_base(payload, base)
        block_adapter.attach_saved_adapter(model, payload)
        print(f"Added the adapter from {a.adapter}: after encoder blocks 0-{payload['after_blocks'] - 1}")
    return model, record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True, help="hypotheses file to write (.jsonl)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="show NeMo setup details in the console")
    a = ap.parse_args()
    if a.batch < 1:
        ap.error("--batch must be positive")

    manifest, audio_root, out = here(a.manifest), here(a.audio_root), here(a.out)
    provenance_path = out.with_suffix(".provenance.json")
    if out.exists() or provenance_path.exists():
        raise FileExistsError(f"Hypotheses file already exists; choose a new name: {out}")
    rows = list(load_manifest(manifest).values())
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    input_paths = {"manifest": manifest}
    input_paths.update({f"audio/{r['audio']}": contained_file(audio_root, r["audio"]) for r in rows})
    inputs = {key: file_record(path) for key, path in input_paths.items()}
    paths = [str(input_paths[f"audio/{r['audio']}"]) for r in rows]
    pretrained, model_paths = model_inputs(a)
    model_files = {key: file_record(path) for key, path in model_paths.items()}

    import torch

    model, record = load_model(a, model_files)
    gpu = torch.cuda.is_available()
    model = model.cuda() if gpu else model.cpu()
    model.eval()
    print(f"Transcribing {len(paths)} clips with NeMo on the {'GPU' if gpu else 'CPU'}…")
    with torch.no_grad():
        try:
            result = model.transcribe(paths, batch_size=a.batch, verbose=False)
        except TypeError:
            result = model.transcribe(paths, batch_size=a.batch)
    if isinstance(result, tuple):
        result = result[0]
    hyps = [text_of(h) for h in result]
    if len(hyps) != len(rows):
        raise RuntimeError(f"NeMo returned {len(hyps)} transcripts for {len(rows)} clips")
    if inputs != {key: file_record(path) for key, path in input_paths.items()}:
        raise ValueError("Manifest or audio changed during decoding; no hypotheses will be saved")
    if model_files != {key: file_record(path) for key, path in model_paths.items()}:
        raise ValueError("Model or adapter changed during decoding; no hypotheses will be saved")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write("".join(json.dumps({"id": r["id"], "hyp": h}) + "\n" for r, h in zip(rows, hyps)))
    try:
        provenance = {
            "format": "willspeak-nemo-hypotheses-v1", "created_at": timestamp(),
            "inputs": inputs, "model_files": model_files, "base": record["base"],
            "device": "cuda" if gpu else "cpu", "batch": a.batch, "packages": versions(),
            "script": file_record(Path(__file__)), "hypotheses": file_record(out),
        }
        if "mix" in record:
            provenance["mix"] = record["mix"]
        write_json(provenance_path, provenance)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    print(f"wrote {out}\nDecode provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "nemo_hyp",
                      console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))
