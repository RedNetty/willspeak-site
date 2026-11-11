#!/usr/bin/env python3
"""Embed single-word clips with every layer of a Parakeet encoder."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cli
import labkit as lab
import nemo_hyp
from embed_manifest import rows_from_csv


class LayerTap:
    def __init__(self, encoder):
        modules = [encoder.pre_encode, *encoder.layers]
        self.count = len(modules)
        self.slots: list = [None] * self.count
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(modules)]

    def _hook(self, index: int):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            self.slots[index] = value.detach()
        return hook

    def pooled(self, frames: int, chosen: list[int]) -> np.ndarray:
        if frames <= 0:
            raise ValueError("The clip has no encoder frames; cannot pool an empty layer")
        out = []
        for i in chosen:
            value = self.slots[i]
            if value is None:
                raise RuntimeError(f"Layer {i} produced no output; run a forward pass first")
            vector = value[0, :frames].float().mean(0).cpu().numpy()
            if not np.isfinite(vector).all():
                raise ValueError(f"Layer {i} produced non-finite values")
            out.append(vector)
        self.slots = [None] * self.count
        return np.stack(out)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def parse_layers(spec: str, count: int) -> list[int]:
    if spec == "all":
        return list(range(count))
    chosen: list[int] = []
    for part in spec.split(","):
        low, _, high = part.strip().partition("-")
        chosen += list(range(int(low), int(high or low) + 1))
    if not chosen or min(chosen) < 0 or max(chosen) >= count:
        raise ValueError(f"--layers {spec!r}: this encoder has layers 0 to {count - 1}")
    return sorted(set(chosen))


def embed_clip(model, tap: LayerTap, audio: np.ndarray, chosen: list[int], device) -> np.ndarray:
    import torch

    signal = torch.from_numpy(audio).unsqueeze(0).to(device)
    length = torch.tensor([len(audio)], device=device)
    with torch.inference_mode():
        _, enc_len = model.forward(input_signal=signal, input_signal_length=length)
    return tap.pooled(int(enc_len[0]), chosen)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    nemo_hyp.add_model_args(ap)
    ap.add_argument("--manifest", required=True, help="CSV: path,label,session,speaker (words.csv)")
    ap.add_argument("--layers", default="all", help="'all' or a list such as 0,4,8-12")
    ap.add_argument("--limit", type=int, default=0, help="debug: embed only the first N clips")
    ap.add_argument("--out", required=True, help="output .npz (refuses to overwrite)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    lab.refuse_existing(a.out)
    rows = rows_from_csv(lab.here(a.manifest))
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        print("nothing to embed")
        return 1
    _, model_paths = nemo_hyp.model_inputs(a)
    from willspeak.artifacts import file_record

    model_files = {k: file_record(p) for k, p in model_paths.items()}

    import torch

    model, record = nemo_hyp.load_model(a, model_files)
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    model = (model.cuda() if gpu else model.cpu()).eval()
    tap = LayerTap(model.encoder)
    chosen = parse_layers(a.layers, tap.count)
    print(f"{len(rows)} clips · layers {chosen[0]} to {chosen[-1]} ({len(chosen)} of {tap.count}) · {device}")

    import soundfile as sf

    from willspeak import audio as A

    vectors, kept = [], []
    started = time.time()
    for i, row in enumerate(rows):
        try:
            x, rate = sf.read(row["path"], dtype="float32")
            vectors.append(embed_clip(model, tap, A.to_mono_16k(x, rate), chosen, device))
            kept.append(row)
        except Exception as error:  # noqa: BLE001
            print(f"  skip {row['path']}: {type(error).__name__}: {error}")
            tap.slots = [None] * tap.count
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)} ({(i + 1) / (time.time() - started):.1f} clips/s)")
    tap.close()
    if not kept:
        print("every clip failed")
        return 1
    out = lab.here(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": record, "model_files": model_files, "layers": chosen, "pool": "mean over encoder frames",
            "numbering": "0 = subsampling output, k = output of encoder block k - 1", "manifest": str(a.manifest),
            "n": len(kept), "skipped": len(rows) - len(kept), "created": time.strftime("%Y-%m-%d")}
    np.savez(out, raw=np.stack(vectors).astype(np.float32), layers=np.array(chosen),
             labels=np.array([r["label"] for r in kept]), sessions=np.array([r["session"] for r in kept]),
             speakers=np.array([r["speaker"] for r in kept]), paths=np.array([r["path"] for r in kept]), meta=json.dumps(meta))
    print(f"wrote {out}  [{len(kept)} clips x {len(chosen)} layers x {np.stack(vectors).shape[-1]}]")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "embed_layers", console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))
