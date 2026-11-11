#!/usr/bin/env python3
"""Decode each clip several ways, keep every distinct answer, and score each answer against the audio. Lab test 36."""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
from fractions import Fraction

import labkit as lab
import nemo_hyp
from willspeak.artifacts import contained_file, file_record, timestamp, versions
from willspeak.normalize import normalize, stock_form

DEFAULT_VIEWS = "greedy,beam4,beam8,tempo1.1,tempo1.2,tempo1.3,pitch0.9,pitch0.95,pitch1.05"
VIEW = re.compile(r"(greedy|beam|tempo|pitch)(\d+(?:\.\d+)?)?$")
GPU_NEEDS_GB = 3.5


def parse_views(text: str) -> list[tuple[str, str, float]]:
    views: list[tuple[str, str, float]] = []
    for name in [v for v in re.split(r"[,\s]+", text) if v]:
        m = VIEW.fullmatch(name)
        if not m:
            raise ValueError(f"View {name!r} is not greedy, beamN, tempoF or pitchF (for example beam4, tempo1.15, pitch0.95)")
        kind, number = m.group(1), m.group(2)
        if kind == "greedy":
            if number:
                raise ValueError("greedy takes no number")
            value = 0.0
        elif kind == "beam":
            if not number or "." in number or int(number) < 2:
                raise ValueError(f"{name!r}: the beam width is a whole number of at least 2")
            value = float(int(number))
        else:
            if not number or not 0.5 <= float(number) <= 2.0 or float(number) == 1.0:
                raise ValueError(f"{name!r}: the factor is between 0.5 and 2 and not 1")
            value = float(number)
        if name in [v[0] for v in views]:
            raise ValueError(f"View {name!r} is listed twice")
        views.append((name, kind, value))
    if "greedy" in [v[0] for v in views]:
        views.sort(key=lambda v: v[0] != "greedy")
    else:
        views.insert(0, ("greedy", "greedy", 0.0))
    return views


def lm_text(norm: str) -> str:
    return stock_form(" ".join("I" if w == "i" else w for w in norm.split()))


def stretch(audio, ratio: float):
    from scipy.signal import resample_poly

    frac = Fraction(ratio).limit_denominator(200)
    return resample_poly(audio.astype("float64"), frac.numerator, frac.denominator).astype("float32")


def read_clip(path):
    import soundfile as sf

    from willspeak.audio import to_mono_16k

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(data, rate)


def group_candidates(found: dict[str, list[str]], nlls: list[float], empty: float) -> list[dict]:
    groups: dict[str, dict] = {}
    for (text, views), nll in zip(found.items(), nlls):
        norm = normalize(text)
        value = empty if not norm else nll
        g = groups.setdefault(norm, {"norm": norm, "text": text, "nll": value, "views": []})
        if value < g["nll"]:
            g["text"], g["nll"] = text, value
        g["views"] += [v for v in views if v not in g["views"]]
    out = []
    for g in groups.values():
        out.append({"norm": g["norm"], "text": g["text"], "nll": None if g["nll"] == float("inf") else g["nll"], "views": g["views"],
                    "words": len(g["norm"].split()), "lm_text": lm_text(g["norm"])})
    return out


def prune(cands: list[dict], keep_within: float) -> tuple[list[dict], int]:
    if keep_within <= 0:
        return cands, 0
    best = min(c["nll"] for c in cands if c["nll"] is not None)
    kept = [c for i, c in enumerate(cands) if i == 0 or (c["nll"] is not None and c["nll"] <= best + keep_within)]
    return kept, len(cands) - len(kept)


class Decoder:
    def __init__(self, model, views: list[tuple[str, str, float]], device: str):
        import torch
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding
        from omegaconf import open_dict

        self.torch, self.model, self.views, self.device = torch, model, views, device
        self.featurizer = model.preprocessor.featurizer
        self.hop = int(self.featurizer.hop_length)
        self.beams: dict[str, object] = {}
        for name, kind, value in views:
            if kind != "beam":
                continue
            cfg = copy.deepcopy(model.cfg.decoding)
            with open_dict(cfg):
                cfg.strategy = "malsd_batch"
                cfg.beam.beam_size = int(value)
                cfg.beam.return_best_hypothesis = False
                cfg.beam.allow_cuda_graphs = False
            self.beams[name] = RNNTBPEDecoding(decoding_cfg=cfg, decoder=model.decoder, joint=model.joint, tokenizer=model.tokenizer)

    def encode(self, audio, hop: int | None = None):
        torch = self.torch
        self.featurizer.hop_length = hop or self.hop
        try:
            with torch.no_grad():
                signal = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                encoded, length = self.model.forward(input_signal=signal, input_signal_length=torch.tensor([len(audio)]).to(self.device))
        finally:
            self.featurizer.hop_length = self.hop
        return encoded, length

    def answers(self, decoding, encoded, length) -> list[str]:
        result = decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=length, return_hypotheses=True)
        first = result[0]
        return [nemo_hyp.text_of(h) for h in (first if isinstance(first, list) else [first])]

    def pool(self, audio) -> tuple[dict[str, list[str]], object, int]:
        encoded, length = self.encode(audio)
        found: dict[str, list[str]] = {}
        for name, kind, value in self.views:
            if kind == "greedy":
                texts = self.answers(self.model.decoding, encoded, length)
            elif kind == "beam":
                texts = self.answers(self.beams[name], encoded, length)
            elif kind == "tempo":
                texts = self.answers(self.model.decoding, *self.encode(audio, round(self.hop * value)))
            else:
                texts = self.answers(self.model.decoding, *self.encode(stretch(audio, 1.0 / value), round(self.hop / value)))
            for text in texts:
                found.setdefault(text.strip(), [])
                if name not in found[text.strip()]:
                    found[text.strip()].append(name)
        return found, encoded, int(length[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    nemo_hyp.add_model_args(ap)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--views", default=DEFAULT_VIEWS,
                    help="the decodes that propose answers: greedy (always first), beamN (every answer of an N-wide beam search), "
                         "tempoF (the model hears the clip F times faster, pitch unchanged), pitchF (pitch times F, speed unchanged)")
    ap.add_argument("--keep-within", type=float, default=15.0,
                    help="drop answers whose audio score is more than this many nats worse than the clip's best (the first decode always "
                         "stays; 0 keeps everything)")
    ap.add_argument("--kinds", default="sentence,word", help="which clips are pooled: sentence, word or sentence,word")
    ap.add_argument("--limit", type=int, default=0, help="pool only the first N clips (0 = all)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = its default)")
    ap.add_argument("--cpu", action="store_true", help="stay on the CPU even when a GPU is free")
    ap.add_argument("--out", required=True, help="pool .json to write (refuses to overwrite)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    kinds = [k for k in a.kinds.split(",") if k]
    if not kinds or any(k not in lab.KINDS for k in kinds) or a.limit < 0 or a.threads < 0 or a.keep_within < 0:
        ap.error("--kinds must be sentence, word or sentence,word; --limit, --threads and --keep-within cannot be negative")
    views = parse_views(a.views)
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lab.refuse_existing(a.out)
    manifest = lab.read_manifest(a.manifest)
    ids = [i for i, row in manifest.items() if lab.clip_kind(row) in kinds and normalize(row["text"])]
    ids = ids[: a.limit] if a.limit else ids
    if not ids:
        raise ValueError(f"No {' or '.join(kinds)} clips in {a.manifest}")
    audio_root = lab.here(a.audio_root)
    audio_paths = {i: contained_file(audio_root, manifest[i]["audio"]) for i in ids}
    _, model_paths = nemo_hyp.model_inputs(a)
    model_files = {k: file_record(p) for k, p in model_paths.items()}

    import torch

    import phrase_score

    if a.threads:
        torch.set_num_threads(a.threads)
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 2**30
        if free < GPU_NEEDS_GB:
            raise ValueError(f"The GPU has {free:.1f} GB free and the model needs about {GPU_NEEDS_GB:g}: close what is using it, or pass --cpu")
    model, record = nemo_hyp.load_model(a, model_files)
    inner = getattr(getattr(model, "loss", None), "_loss", None)
    if not hasattr(inner, "durations") or not list(inner.durations):
        raise ValueError("This script needs a TDT model (its loss has no durations)")
    sigma, durations, blank = float(getattr(inner, "sigma", 0.0)), [int(d) for d in inner.durations], int(inner.blank)
    gpu = torch.cuda.is_available() and not a.cpu
    device = "cuda" if gpu else "cpu"
    model = model.cuda() if gpu else model.cpu()
    model.eval()
    if hasattr(model.joint, "set_fuse_loss_wer"):
        model.joint.set_fuse_loss_wer(False)
    decoder = Decoder(model, views, device)
    print(f"on the {device}; views {' '.join(v[0] for v in views)}; {len(ids)} clips")

    rows = []
    started = time.monotonic()
    for n, clip_id in enumerate(ids):
        row = manifest[clip_id]
        audio = read_clip(audio_paths[clip_id])
        t0 = time.monotonic()
        found, encoded, frames = decoder.pool(audio)
        t1 = time.monotonic()
        nlls, empty = phrase_score.score_encoded(model, encoded, frames, list(found), sigma, durations, blank)
        t2 = time.monotonic()
        cands, dropped = prune(group_candidates(found, nlls, empty), a.keep_within)
        rows.append({"id": clip_id, "speaker": str(row.get("speaker", "")), "kind": lab.clip_kind(row), "ref": normalize(row["text"]),
                     "duration_s": round(len(audio) / 16000, 3), "frames": frames, "nll_empty": empty, "candidates": cands, "dropped": dropped,
                     "seconds": {"decode": round(t1 - t0, 3), "score": round(t2 - t1, 3)}})
        if n % 25 == 24 or n + 1 == len(ids):
            print(f"  {n + 1}/{len(ids)} clips, {time.monotonic() - started:.0f} s", flush=True)
    sizes = [len(r["candidates"]) for r in rows]
    held = sum(1 for r in rows if any(c["norm"] == r["ref"] for c in r["candidates"]))
    first = sum(1 for r in rows if r["candidates"][0]["norm"] == r["ref"])
    print(f"{sum(sizes) / len(sizes):.1f} distinct answers kept per clip (most {max(sizes)}; {sum(r['dropped'] for r in rows)} dropped as more than "
          f"{a.keep_within:g} nats behind); the first decode is right on {first} of {len(rows)} clips, and some answer in the pool is right on {held}")
    lab.save_json(a.out, {
        "format": "willspeak-lm-pool-v1", "created_at": timestamp(), "model": record, "views": [v[0] for v in views], "kinds": kinds,
        "keep_within": a.keep_within, "device": device, "sigma": sigma, "durations": durations, "rows": rows, "model_files": model_files, "packages": versions(),
        "provenance": lab.provenance(__file__, {"manifest": a.manifest})})
    print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "lm_pool", console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))
