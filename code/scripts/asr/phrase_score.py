#!/usr/bin/env python3
"""Score candidate phrases against the audio, with the model's own likelihood."""
from __future__ import annotations

import argparse
import sys

import labkit as lab
import lattice_fast
import nemo_hyp
import phrasebook
import reference_score
import tdt_lattice
from phrase_pick import build_pool, pick_candidates, rank_pool
from willspeak.artifacts import contained_file, file_record, timestamp, versions
from willspeak.normalize import normalize


def score_encoded(model, encoded, frames: int, texts: list[str], sigma: float, durations: list[int], blank: int, chunk: int = 8
                  ) -> tuple[list[float], float]:
    import torch

    n_dur = len(durations)
    device = encoded.device
    token_lists = [list(model.tokenizer.text_to_ids(t)) for t in texts]
    out: list[float] = []
    with torch.no_grad():
        def nll(tok_dur, tokens: list[int]) -> float:
            blank_lp, label_lp, dur_lp = tdt_lattice.reduce_logprobs(tok_dur[0], tok_dur[1], tokens, blank)
            return lattice_fast.lattice_nll_lists(blank_lp.cpu().tolist(), label_lp.cpu().tolist(), dur_lp.cpu().tolist(), durations)

        g0 = model.decoder.predict(None, batch_size=1, add_sos=False)[0].transpose(1, 2).to(device)
        empty_logits = model.joint(encoder_outputs=encoded, decoder_outputs=g0)
        empty = nll(tdt_lattice.split_logprobs(empty_logits[0, :frames], n_dur, sigma), [])
        del empty_logits
        for start in range(0, len(token_lists), chunk):
            part = token_lists[start : start + chunk]
            targets, target_len = reference_score.pad(part)
            g, _, _ = model.decoder(targets=targets.to(device), target_length=target_len.to(device))
            logits = model.joint(encoder_outputs=encoded.expand(len(part), -1, -1), decoder_outputs=g)
            for b, tokens in enumerate(part):
                out.append(nll(tdt_lattice.split_logprobs(logits[b, :frames], n_dur, sigma), tokens) if tokens else empty)
            del logits, g
    return out, empty


def score_texts(model, audio, texts: list[str], device, sigma: float, durations: list[int], blank: int, chunk: int = 8
                ) -> tuple[list[float], float, int]:
    import torch

    with torch.no_grad():
        signal = torch.from_numpy(audio).unsqueeze(0).to(device)
        encoded, enc_len = model.forward(input_signal=signal, input_signal_length=torch.tensor([len(audio)]).to(device))
    frames = int(enc_len[0])
    out, empty = score_encoded(model, encoded, frames, texts, sigma, durations, blank, chunk)
    return out, empty, frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    nemo_hyp.add_model_args(ap)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--pool-glob", nargs="+", required=True, help="manifests whose prompts form the phrase list")
    ap.add_argument("--kinds", default="sentence,word", help="which prompts form the list, and which clips are scored")
    ap.add_argument("--hyps", required=True, help="the same model's decoded hypotheses (a folder or .jsonl); the shortlist is built from them")
    ap.add_argument("--shortlist", type=int, default=15, help="phrases nearest to the answer that are scored for each clip")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips (0 = all)")
    ap.add_argument("--out", required=True, help="results .json to write (refuses to overwrite)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    kinds = [k for k in a.kinds.split(",") if k]
    if not kinds or any(k not in ("sentence", "word") for k in kinds) or a.shortlist < 1 or a.limit < 0:
        ap.error("--kinds must be sentence, word or sentence,word; --shortlist must be positive")
    lab.refuse_existing(a.out)
    manifest = lab.read_manifest(a.manifest)
    ids = [i for i, row in manifest.items() if phrasebook.kind_of(row) in kinds and "(live)" not in str(row.get("text", ""))]
    ids = ids[: a.limit] if a.limit else ids
    hyps = lab.read_hyps(a.hyps)
    if not set(ids) <= set(hyps):
        raise ValueError("--hyps does not cover every scored clip")
    pool = build_pool(a.pool_glob, kinds)
    pool_norms = {p.norm for p in pool}
    missing = [i for i in ids if normalize(manifest[i]["text"]) not in pool_norms]
    if missing:
        raise ValueError(f"{len(missing)} test prompts are missing from the list (the pool glob must include the test manifest): {missing[:5]}")
    print(f"phrase list: {len(pool)} distinct {' and '.join(kinds)} prompts; scoring {len(ids)} clips, shortlist {a.shortlist}")
    audio_root = lab.here(a.audio_root)
    audio_paths = {i: contained_file(audio_root, manifest[i]["audio"]) for i in ids}
    _, model_paths = nemo_hyp.model_inputs(a)
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
    print(f"on the {device}; sigma {sigma}, durations {durations}, blank id {blank}")

    rows = []
    for n, clip_id in enumerate(ids):
        true_norm = normalize(manifest[clip_id]["text"])
        hyp_norm = normalize(hyps[clip_id])
        ranking = rank_pool(phrasebook.Item(hyps[clip_id]), pool)
        cands = pick_candidates(ranking, pool, true_norm, hyp_norm, a.shortlist)
        scores, empty, frames = score_texts(model, lab.read_audio(audio_paths[clip_id]), [c["text"] for c in cands], device, sigma, durations, blank)
        if n == 0:
            tokens = list(model.tokenizer.text_to_ids(cands[0]["text"]))
            with torch.no_grad():
                signal = torch.from_numpy(lab.read_audio(audio_paths[clip_id])).unsqueeze(0).to(device)
                enc, enc_len = model.forward(input_signal=signal, input_signal_length=torch.tensor([signal.shape[1]]).to(device))
                g, _, _ = model.decoder(targets=torch.tensor([tokens]).to(device), target_length=torch.tensor([len(tokens)]).to(device))
                lg = model.joint(encoder_outputs=enc, decoder_outputs=g)
                tok, dur = tdt_lattice.split_logprobs(lg[0, : int(enc_len[0])], len(durations), sigma)
                slow = float(tdt_lattice.sequence_nll(tok, dur, tokens, durations, blank))
            print(f"check: list-based lattice {scores[0]:.4f} nats, torch lattice {slow:.4f} nats, difference {abs(scores[0] - slow):.6f} (should be about 0.0001 or less)")
        for c, s in zip(cands, scores):
            c["nll"] = None if s == float("inf") else s
        rows.append({"id": clip_id, "kind": phrasebook.kind_of(manifest[clip_id]), "ref": true_norm, "hyp": hyp_norm, "frames": frames,
                     "nll_empty": empty, "candidates": cands})
        if n % 25 == 24:
            print(f"  {n + 1}/{len(ids)}", flush=True)
    lab.save_json(a.out, {
        "format": "willspeak-phrase-score-v1", "created_at": timestamp(), "model": record, "kinds": kinds, "shortlist": a.shortlist,
        "pool": len(pool), "sigma": sigma, "durations": durations, "rows": rows,
        "model_files": model_files, "packages": versions(),
        "provenance": lab.provenance(__file__, {"manifest": a.manifest, "hypotheses": a.hyps})})
    print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "phrase_score", console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))
