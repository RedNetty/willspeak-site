#!/usr/bin/env python3
"""Decode each clip several ways with one model, keep every distinct answer, and score each one against the audio. Lab test 36.

One run is one model on one manifest. For every clip the model is asked four times:

- greedy, the normal decode and the system's current answer;
- beam search that keeps its top N answers instead of one (``--beam``);
- the audio played a little slower and lower (``--slow 0.95``), then greedy;
- features taken every 11.5 ms instead of 10 (``--hop-ms 11.5``), then greedy.

Answers that are the same after ``willspeak.normalize`` are one candidate; the first spelling seen is kept, greedy's first.
Every candidate is then scored with the same model's full-sum TDT log-likelihood of that text given the original audio
(``phrase_score.score_texts``), so the scores are comparable across the candidates of a clip.

The pool holds no reference transcripts. ``lm_rescore.py`` reads them from the manifest when it counts.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import labkit as lab
import nemo_hyp
from willspeak.artifacts import contained_file, file_record, timestamp, versions, write_json
from willspeak.normalize import normalize

BEAM_STRATEGIES = ("malsd_batch", "maes", "beam")
VIEWS = ("slow", "hop")


def merge_candidates(answers: list[tuple[str, str]]) -> list[dict]:
    """Collapse (source, text) pairs into distinct candidates, in first-seen order, keeping the first spelling of each."""
    out: list[dict] = []
    seen: dict[str, dict] = {}
    for source, text in answers:
        key = normalize(text)
        if key in seen:
            seen[key]["sources"].append(source)
            continue
        cand = {"text": text, "sources": [source], "n_words": len(key.split())}
        seen[key] = cand
        out.append(cand)
    return out


def slow_ratio(slow: float) -> tuple[int, int]:
    """Resampling factors (up, down) that make the audio last 1/slow as long, e.g. 0.95 -> (20, 19)."""
    if not 0.5 <= slow <= 1.5 or slow == 1.0:
        raise ValueError(f"--slow must be between 0.5 and 1.5 and not 1.0, got {slow}")
    frac = Fraction(str(slow)).limit_denominator(100)
    return frac.denominator, frac.numerator


def slow_audio(audio, slow: float):
    from scipy.signal import resample_poly

    up, down = slow_ratio(slow)
    return resample_poly(audio, up, down).astype("float32")


def nbest_texts(item) -> list[str]:
    """The texts of one clip's beam result, however NeMo packaged it."""
    inner = getattr(item, "n_best_hypotheses", None)
    if inner is not None:
        return [nemo_hyp.text_of(h) for h in inner]
    if isinstance(item, (list, tuple)):
        return [nemo_hyp.text_of(h) for h in item]
    return [nemo_hyp.text_of(item)]


def unpack_transcribe(result, n_clips: int):
    """NeMo's transcribe returns a list, or a tuple of lists in some versions; take the list that has one entry per clip."""
    if isinstance(result, tuple):
        lists = [r for r in result if isinstance(r, list) and len(r) == n_clips]
        if not lists:
            raise RuntimeError(f"NeMo returned {len(result)} lists, none with {n_clips} entries")
        # Prefer the one whose items carry n-best lists.
        for r in lists:
            if r and getattr(r[0], "n_best_hypotheses", None) is not None:
                return r
        return lists[0]
    if len(result) != n_clips:
        raise RuntimeError(f"NeMo returned {len(result)} transcripts for {n_clips} clips")
    return result


def transcribe(model, paths: list[str], batch: int, hypotheses: bool = False) -> list:
    import torch

    kwargs = {"return_hypotheses": True} if hypotheses else {}
    with torch.no_grad():
        try:
            result = model.transcribe(paths, batch_size=batch, verbose=False, **kwargs)
        except TypeError:
            result = model.transcribe(paths, batch_size=batch, **kwargs)
    return unpack_transcribe(result, len(paths))


def beam_config(model, strategy: str, beam: int):
    from omegaconf import open_dict

    cfg = copy.deepcopy(model.cfg.decoding)
    with open_dict(cfg):
        cfg.strategy = strategy
        if "beam" not in cfg or cfg.beam is None:
            cfg.beam = {}
        cfg.beam.beam_size = beam
        cfg.beam.return_best_hypothesis = False
        if strategy == "malsd_batch":
            cfg.beam.allow_cuda_graphs = False
    return cfg


def beam_decode(model, paths: list[str], beam: int, batch: int, say) -> tuple[list[list[str]], str]:
    """Try the TDT beam strategies in order; return each clip's n-best texts and the strategy that worked."""
    original = copy.deepcopy(model.cfg.decoding)
    try:
        last_error: Exception | None = None
        for strategy in BEAM_STRATEGIES:
            try:
                model.change_decoding_strategy(beam_config(model, strategy, beam), verbose=False)
            except TypeError:
                model.change_decoding_strategy(beam_config(model, strategy, beam))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                say(f"beam strategy {strategy} not available here ({type(exc).__name__}: {str(exc)[:120]}); trying the next")
                continue
            try:
                result = transcribe(model, paths, batch, hypotheses=True)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                say(f"beam strategy {strategy} failed while decoding ({type(exc).__name__}: {str(exc)[:120]}); trying the next")
                continue
            texts = [nbest_texts(item) for item in result]
            if all(len(t) <= 1 for t in texts):
                say(f"beam strategy {strategy} returned one answer per clip, not {beam}; the pool will carry only its best")
            return texts, strategy
        raise RuntimeError(f"No TDT beam search strategy worked in this NeMo: {last_error}")
    finally:
        try:
            model.change_decoding_strategy(original, verbose=False)
        except TypeError:
            model.change_decoding_strategy(original)


def with_hop(model, hop_ms: float, device: str):
    """Swap in a feature front end with another window stride; returns the original to put back."""
    from omegaconf import open_dict

    cfg = copy.deepcopy(model.cfg.preprocessor)
    with open_dict(cfg):
        cfg.window_stride = hop_ms / 1000.0
    original = model.preprocessor
    replacement = model.from_config_dict(cfg).to(device)
    replacement.eval()
    model.preprocessor = replacement
    return original


def write_wavs(folder: Path, ids: list[str], audio_paths: dict[str, Path], slow: float) -> list[str]:
    import soundfile as sf

    out = []
    for clip_id in ids:
        path = folder / f"{clip_id}.wav"
        sf.write(str(path), slow_audio(lab.read_audio(audio_paths[clip_id]), slow), 16000, subtype="PCM_16")
        out.append(str(path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    nemo_hyp.add_model_args(ap)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True, help="pool .jsonl to write (refuses to overwrite); a .provenance.json goes beside it")
    ap.add_argument("--kinds", default="sentence,word", help="which clips to pool: sentence, word or sentence,word")
    ap.add_argument("--beam", type=int, default=4, help="beam size and how many beam answers to keep (0 = no beam search)")
    ap.add_argument("--views", default=",".join(VIEWS), help="audio views to decode greedy as well: slow, hop, both or none")
    ap.add_argument("--slow", type=float, default=0.95, help="speed of the slow view (0.95 = 5%% slower and lower)")
    ap.add_argument("--hop-ms", type=float, default=11.5, help="window stride of the wide-hop view in ms (the model's own is 10)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="pool only the first N clips (0 = all)")
    ap.add_argument("--verbose", action="store_true", help="show NeMo setup details in the console")
    a = ap.parse_args()
    kinds = [k for k in a.kinds.split(",") if k]
    views = [v for v in a.views.split(",") if v and v != "none"]
    if not kinds or any(k not in lab.KINDS for k in kinds):
        ap.error("--kinds must be sentence, word or sentence,word")
    if any(v not in VIEWS for v in views):
        ap.error(f"--views must name only {', '.join(VIEWS)}, or none")
    if a.beam < 0 or a.batch < 1 or a.limit < 0 or a.hop_ms <= 0:
        ap.error("--beam and --limit must be 0 or more; --batch and --hop-ms must be positive")
    if "slow" in views:
        slow_ratio(a.slow)
    lab.refuse_existing(a.out, lab.here(a.out).with_suffix(".provenance.json"))

    manifest = lab.read_manifest(a.manifest)
    ids = [i for i, row in manifest.items() if lab.clip_kind(row) in kinds]
    ids = ids[: a.limit] if a.limit else ids
    if not ids:
        raise ValueError(f"No {' or '.join(kinds)} clips in {a.manifest}")
    audio_root = lab.here(a.audio_root)
    audio_paths = {i: contained_file(audio_root, manifest[i]["audio"]) for i in ids}
    inputs = {"manifest": file_record(lab.here(a.manifest)), **{f"audio/{manifest[i]['audio']}": file_record(audio_paths[i]) for i in ids}}
    pretrained, model_paths = nemo_hyp.model_inputs(a)
    model_files = {k: file_record(p) for k, p in model_paths.items()}

    import torch

    model, record = nemo_hyp.load_model(a, model_files)
    inner = getattr(getattr(model, "loss", None), "_loss", None)
    if not hasattr(inner, "durations") or not list(inner.durations):
        raise ValueError("This script needs a TDT model (its loss has no durations)")
    sigma, durations, blank = float(getattr(inner, "sigma", 0.0)), [int(d) for d in inner.durations], int(inner.blank)
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    model = model.cuda() if gpu else model.cpu()
    model.eval()
    if hasattr(model.joint, "set_fuse_loss_wer"):
        model.joint.set_fuse_loss_wer(False)
    say = print
    say(f"{len(ids)} {' and '.join(kinds)} clips on the {device}; beam {a.beam}, views {', '.join(views) or 'none'}")

    paths = [str(audio_paths[i]) for i in ids]
    answers: dict[str, list[tuple[str, str]]] = {i: [] for i in ids}
    started = time.time()

    say("greedy decode")
    for clip_id, item in zip(ids, transcribe(model, paths, a.batch)):
        answers[clip_id].append(("greedy", nemo_hyp.text_of(item)))

    strategy = None
    if a.beam > 0:
        say(f"beam search, size {a.beam}, keeping every answer")
        nbest, strategy = beam_decode(model, paths, a.beam, a.batch, say)
        for clip_id, texts in zip(ids, nbest):
            for rank, text in enumerate(texts[: a.beam], start=1):
                answers[clip_id].append((f"beam:{rank}", text))

    if "slow" in views:
        up, down = slow_ratio(a.slow)
        say(f"slow view: audio resampled {up}/{down} (played at {a.slow:g} speed), then greedy")
        with tempfile.TemporaryDirectory(prefix="willspeak-slow-") as tmp:
            slow_paths = write_wavs(Path(tmp), ids, audio_paths, a.slow)
            for clip_id, item in zip(ids, transcribe(model, slow_paths, a.batch)):
                answers[clip_id].append(("slow", nemo_hyp.text_of(item)))

    if "hop" in views:
        say(f"wide-hop view: features every {a.hop_ms:g} ms, then greedy")
        original = with_hop(model, a.hop_ms, device)
        try:
            for clip_id, item in zip(ids, transcribe(model, paths, a.batch)):
                answers[clip_id].append(("hop", nemo_hyp.text_of(item)))
        finally:
            model.preprocessor = original

    say(f"decodes done in {time.time() - started:.0f} s; scoring every candidate against the audio")
    import phrase_score

    rows = []
    n_cands = 0
    for n, clip_id in enumerate(ids):
        cands = merge_candidates(answers[clip_id])
        nlls, empty, frames = phrase_score.score_texts(model, lab.read_audio(audio_paths[clip_id]), [c["text"] for c in cands],
                                                       device, sigma, durations, blank)
        for cand, nll in zip(cands, nlls):
            cand["acoustic"] = None if nll == float("inf") or nll != nll else -float(nll)
        n_cands += len(cands)
        rows.append({"id": clip_id, "kind": lab.clip_kind(manifest[clip_id]), "frames": int(frames),
                     "acoustic_empty": None if empty == float("inf") else -float(empty), "candidates": cands})
        if n % 25 == 24:
            say(f"  {n + 1}/{len(ids)}", flush=True)

    if inputs != {"manifest": file_record(lab.here(a.manifest)), **{f"audio/{manifest[i]['audio']}": file_record(audio_paths[i]) for i in ids}}:
        raise ValueError("Manifest or audio changed during the run; nothing saved")
    if model_files != {k: file_record(p) for k, p in model_paths.items()}:
        raise ValueError("Model or adapter changed during the run; nothing saved")
    out = lab.here(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
    try:
        write_json(out.with_suffix(".provenance.json"), {
            "format": "willspeak-candidate-pool-v1", "created_at": timestamp(), "inputs": inputs, "model_files": model_files,
            "base": record["base"], **({"mix": record["mix"]} if "mix" in record else {}),
            "kinds": kinds, "beam": {"size": a.beam, "strategy": strategy}, "views": {v: (a.slow if v == "slow" else a.hop_ms) for v in views},
            "acoustic": {"sigma": sigma, "durations": durations, "blank": blank}, "device": device, "batch": a.batch,
            "clips": len(rows), "candidates": n_cands, "packages": versions(),
            "script": file_record(Path(__file__)), "pool": file_record(out)})
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    say(f"wrote {out}: {len(rows)} clips, {n_cands} candidates ({n_cands / len(rows):.1f} per clip), {time.time() - started:.0f} s in all")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "candidate_pool", console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))
