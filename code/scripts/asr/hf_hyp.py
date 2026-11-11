#!/usr/bin/env python3
"""Transcribe a manifest with a Hugging Face speech model and write hypotheses for ``evaluate.py --hyp``."""
from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _cli
import labkit as lab
from willspeak.artifacts import contained_file, file_record, timestamp, versions, write_json
from willspeak.evaluate import load_manifest

KINDS = ("qwen3-asr", "cohere", "granite")
GRANITE_PROMPTS = {
    "granite-4.0": "<|audio|>can you transcribe the speech into a written format?",
    "granite-speech-4.1": "<|audio|>transcribe the speech with proper punctuation and capitalization.",
}


def granite_prompt(name: str) -> str:
    for key, prompt in GRANITE_PROMPTS.items():
        if key in name:
            return prompt
    raise ValueError(f"No known prompt for {name}; add it to GRANITE_PROMPTS from the model card")


def here(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load_qwen(name: str, device: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(name, dtype=torch.bfloat16 if device != "cpu" else torch.float32,
                                          device_map=device, max_new_tokens=256)

    def run(path: Path) -> str:
        return (model.transcribe(audio=str(path))[0].text or "")

    return run


def load_cohere(name: str, device: str):
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(name)
    model = CohereAsrForConditionalGeneration.from_pretrained(name, device_map=device).eval()

    def run(path: Path) -> str:
        audio = lab.read_audio(path)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
        inputs.to(model.device, dtype=model.dtype)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        text = processor.decode(outputs, skip_special_tokens=True)
        return (text[0] if isinstance(text, list) else text) or ""

    return run


def load_granite(name: str, device: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(name)
    tokenizer = processor.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        name, device_map=device, torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32).eval()
    chat = [{"role": "user", "content": granite_prompt(name)}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def run(path: Path) -> str:
        wav = torch.from_numpy(lab.read_audio(path)).unsqueeze(0)
        inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        new_tokens = outputs[0, inputs["input_ids"].shape[-1]:].unsqueeze(0)
        return (tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0] or "")

    return run


LOADERS = {"qwen3-asr": load_qwen, "cohere": load_cohere, "granite": load_granite}
PACKAGES = ("transformers", "qwen-asr", "soundfile", "torch")


def package_versions() -> dict:
    found = dict(versions())
    for package in PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--model", required=True, help="Hugging Face model name, as candidates.py lists it")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True, help="hypotheses file to write (.jsonl)")
    a = ap.parse_args()

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

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run = LOADERS[a.kind](a.model, device)
    hyps = []
    print(f"Transcribing {len(rows)} clips with {a.model} on {device}…")
    for i, r in enumerate(rows, 1):
        try:
            hyps.append(run(input_paths[f"audio/{r['audio']}"]))
        except Exception as error:
            raise RuntimeError(f"{a.model} failed on clip {r['id']}: {error}") from error
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")
    if inputs != {key: file_record(path) for key, path in input_paths.items()}:
        raise ValueError("Manifest or audio changed during decoding; no hypotheses will be saved")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write("".join(json.dumps({"id": r["id"], "hyp": h}) + "\n" for r, h in zip(rows, hyps)))
    try:
        write_json(provenance_path, {
            "format": "willspeak-hf-hypotheses-v1", "created_at": timestamp(), "inputs": inputs,
            "model": a.model, "kind": a.kind, "device": device, "packages": package_versions(),
            "script": file_record(Path(__file__)), "hypotheses": file_record(out),
        })
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    print(f"wrote {out}\nDecode provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli.run(main, "hf_hyp"))
