#!/usr/bin/env python3
"""How likely does a Parakeet-TDT model think the true words are, and how likely is it to say nothing? Lab test 6."""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import labkit as lab
import nemo_hyp
import tdt_lattice
from willspeak.artifacts import contained_file, file_record, timestamp, versions
from willspeak.normalize import normalize, stock_form

MARGIN_FLAGS = (0.0, 5.0, 10.0)


def reference_text(text: str, form: str) -> str:
    if form == "normalized":
        return normalize(text)
    if form == "stock":
        return stock_form(text)
    return text


def summarise(rows: list[dict]) -> dict:
    margins = [r["margin"] for r in rows if r["margin"] is not None and r["margin"] == r["margin"]]
    out = {
        "clips": len(rows),
        "mean_nll_ref": statistics.fmean(r["nll_ref"] for r in rows),
        "mean_nll_empty": statistics.fmean(r["nll_empty"] for r in rows),
        "median_margin": statistics.median(margins) if margins else None,
        "share_margin_below": {str(f): sum(1 for m in margins if m < f) / len(margins) for f in MARGIN_FLAGS} if margins else {},
    }
    scored = [r for r in rows if "nll_hyp" in r]
    if scored:
        empty = [r for r in scored if r["hyp_empty"]]
        wrong = [r for r in scored if r["errors"] > 0 and not r["hyp_empty"]]
        out["hypotheses"] = {
            "empty": len(empty),
            "mean_margin_on_empty": statistics.fmean(r["margin"] for r in empty) if empty else None,
            "wrong_nonempty": len(wrong),
            "truth_more_likely_than_answer": sum(1 for r in wrong if r["nll_ref"] < r["nll_hyp"]),
            "answer_more_likely_than_truth": sum(1 for r in wrong if r["nll_ref"] >= r["nll_hyp"]),
        }
    return out


def pad(token_lists: list[list[int]]):
    import torch

    width = max((len(t) for t in token_lists), default=0)
    targets = torch.zeros(len(token_lists), width, dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        if tokens:
            targets[i, : len(tokens)] = torch.tensor(tokens)
    return targets, torch.tensor([len(t) for t in token_lists], dtype=torch.long)


def score_batch(model, audio: list, reference_tokens: list[list[int]], hyp_tokens: list[list[int]] | None,
                device, sigma: float, durations: list[int], blank: int, check_loss: bool = False) -> list[dict]:
    import torch

    batch = len(audio)
    lengths = torch.tensor([len(a) for a in audio])
    signal = torch.zeros(batch, int(lengths.max()))
    for i, a in enumerate(audio):
        signal[i, : len(a)] = torch.from_numpy(a)
    with torch.no_grad():
        encoded, enc_len = model.forward(input_signal=signal.to(device), input_signal_length=lengths.to(device))
        n_dur = len(durations)

        def lattice(logits, b, tokens) -> float:
            tok, dur = tdt_lattice.split_logprobs(logits[b, : int(enc_len[b])], n_dur, sigma)
            parts = tdt_lattice.reduce_logprobs(tok, dur, tokens, blank)
            return float(tdt_lattice.lattice_nll(*(p.cpu() for p in parts), durations))

        g0 = model.decoder.predict(None, batch_size=batch, add_sos=False)[0].transpose(1, 2).to(encoded.device)
        empty_logits = model.joint(encoder_outputs=encoded, decoder_outputs=g0)
        empty = [lattice(empty_logits, b, []) for b in range(batch)]
        del empty_logits

        def teacher_forced(token_lists):
            targets, target_len = pad(token_lists)
            g, _, _ = model.decoder(targets=targets.to(device), target_length=target_len.to(device))
            logits = model.joint(encoder_outputs=encoded, decoder_outputs=g)
            values = [empty[b] if not token_lists[b] else lattice(logits, b, token_lists[b]) for b in range(batch)]
            note = None
            if check_loss:
                note = nemo_loss_difference(model, logits, targets, enc_len, target_len, values, token_lists)
            return values, note

        reference, note = teacher_forced(reference_tokens)
        answer = teacher_forced(hyp_tokens)[0] if hyp_tokens is not None else None
    if note:
        print(note)
    rows = []
    for b in range(batch):
        row = {"nll_ref": reference[b], "nll_empty": empty[b], "margin": empty[b] - reference[b], "frames": int(enc_len[b]),
               "tokens": len(reference_tokens[b])}
        if answer is not None:
            row["nll_hyp"] = answer[b]
        rows.append(row)
    return rows


def nemo_loss_difference(model, logits, targets, enc_len, target_len, own: list[float], token_lists) -> str:
    try:
        loss = model.loss
        saved = loss.reduction
        loss.reduction = None
        try:
            theirs = loss(log_probs=logits, targets=targets.to(logits.device), input_lengths=enc_len,
                          target_lengths=target_len.to(logits.device))
        finally:
            loss.reduction = saved
        diffs = [abs(float(t) - o) for t, o, tokens in zip(theirs, own, token_lists) if tokens]
        return (f"check against NeMo's loss on the first batch: largest difference {max(diffs):.4f} nats over {len(diffs)} clips "
                "(about 0.01 or less means the same lattice; much larger means sigma or the arcs differ)") if diffs else ""
    except Exception as error:  # noqa: BLE001
        return f"check against NeMo's loss skipped: {type(error).__name__}: {error}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    nemo_hyp.add_model_args(ap)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--hyps", help="the same model's decoded hypotheses (a folder or .jsonl) to score as well")
    ap.add_argument("--reference", default="normalized", choices=["normalized", "stock", "raw"],
                    help="form of the true words: normalized (lower case, no punctuation, the form the fine-tunes train on), "
                         "stock (capital first letter and a closing mark, the stock model's own form) or raw (the manifest text)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips (0 = all)")
    ap.add_argument("--out", required=True, help="results .json to write (refuses to overwrite)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.batch < 1 or a.limit < 0:
        ap.error("--batch must be positive and --limit cannot be negative")
    lab.refuse_existing(a.out)
    manifest = lab.read_manifest(a.manifest)
    ids = list(manifest)[: a.limit] if a.limit else list(manifest)
    audio_root = lab.here(a.audio_root)
    audio_paths = {i: contained_file(audio_root, manifest[i]["audio"]) for i in ids}
    hyps = lab.read_hyps(a.hyps) if a.hyps else None
    if hyps is not None and not set(ids) <= set(hyps):
        raise ValueError("--hyps does not cover every scored clip")
    _, model_paths = nemo_hyp.model_inputs(a)
    model_files = {k: file_record(p) for k, p in model_paths.items()}

    import torch

    model, record = nemo_hyp.load_model(a, model_files)
    inner = getattr(getattr(model, "loss", None), "_loss", None)
    if not hasattr(inner, "durations") or not list(inner.durations):
        raise ValueError("This script needs a TDT model (its loss has no durations); see candidates.py, loss column")
    sigma, durations, blank = float(getattr(inner, "sigma", 0.0)), [int(d) for d in inner.durations], int(inner.blank)
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    model = model.cuda() if gpu else model.cpu()
    model.eval()
    if hasattr(model.joint, "set_fuse_loss_wer"):
        model.joint.set_fuse_loss_wer(False)
    print(f"Scoring {len(ids)} clips on the {device}; sigma {sigma}, durations {durations}, blank id {blank}")

    def tokens_of(text: str) -> list[int]:
        return list(model.tokenizer.text_to_ids(text))

    rows = []
    for start in range(0, len(ids), a.batch):
        chunk = ids[start : start + a.batch]
        audio = [lab.read_audio(audio_paths[i]) for i in chunk]
        texts = [reference_text(manifest[i]["text"], a.reference) for i in chunk]
        ref_tokens = [tokens_of(t) for t in texts]
        if not all(ref_tokens):
            raise ValueError(f"A reference has no tokens: {[i for i, t in zip(chunk, ref_tokens) if not t]}")
        hyp_tokens = [tokens_of(hyps[i]) if normalize(hyps[i]).split() else [] for i in chunk] if hyps is not None else None
        scored = score_batch(model, audio, ref_tokens, hyp_tokens, device, sigma, durations, blank,
                             check_loss=gpu and start == 0)
        for i, row in zip(chunk, scored):
            row["id"] = i
            if hyps is not None:
                row["hyp_empty"] = not normalize(hyps[i]).split()
                row["errors"] = lab.clip_counts({i: manifest[i]}, {i: hyps[i]})[i][0]
            rows.append(row)
        if (start // a.batch) % 20 == 19:
            print(f"  {len(rows)}/{len(ids)}")
    unreachable = [r["id"] for r in rows if r["nll_ref"] == float("inf")]
    if unreachable:
        print(f"note: {len(unreachable)} references have no alignment (clip shorter than its words); left out of the summary: {unreachable}")
    summary = summarise([r for r in rows if r["nll_ref"] != float("inf")])
    summary["no_alignment"] = unreachable
    rows = [{k: (None if isinstance(v, float) and v in (float("inf"), float("-inf")) or v != v else v) for k, v in r.items()} for r in rows]
    print(json.dumps(summary, indent=2))
    inputs = {"manifest": a.manifest, **({"hypotheses": a.hyps} if a.hyps else {})}
    lab.save_json(a.out, {
        "format": "willspeak-reference-score-v1", "created_at": timestamp(), "model": record, "reference": a.reference,
        "sigma": sigma, "durations": durations, "summary": summary, "rows": rows,
        "model_files": model_files, "packages": versions(), "provenance": lab.provenance(__file__, inputs)})
    print(f"wrote {lab.here(a.out)}")
    return 0


if __name__ == "__main__":
    import _cli

    sys.exit(_cli.run(main, "reference_score",
                      console_filter_factory=None if "--verbose" in sys.argv else _cli.NemoConsole))

