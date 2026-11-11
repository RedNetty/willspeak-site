#!/usr/bin/env python3
"""Load Parakeet and decode one audio file or a generated silence clip."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from willspeak.asr.parakeet import ParakeetASR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="optional WAV or audio file to transcribe")
    parser.add_argument("--model-dir", type=Path, help="Parakeet ONNX model directory")
    args = parser.parse_args()

    model_dir = args.model_dir
    if model_dir is not None and not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    recognizer = ParakeetASR(model_dir)
    if not recognizer.installed():
        print(f"sherpa-onnx is not installed in {sys.executable}", file=sys.stderr)
        return 1
    if not recognizer.model_present():
        print(f"Parakeet model files are missing from {recognizer.model_dir}", file=sys.stderr)
        if args.model_dir is None:
            print("Run: python scripts/fetch_models.py --parakeet-only", file=sys.stderr)
        else:
            print("A fine-tuned model folder is written by scripts/asr/export.py; run that first", file=sys.stderr)
        return 1

    temporary_path: Path | None = None
    audio_path = args.audio
    try:
        if audio_path is None:
            with tempfile.NamedTemporaryFile(prefix="willspeak-check-", suffix=".wav", delete=False) as handle:
                temporary_path = Path(handle.name)
            sf.write(temporary_path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
            audio_path = temporary_path
        if not audio_path.is_file():
            print(f"Audio file not found: {audio_path}", file=sys.stderr)
            return 1
        result = recognizer.transcribe(str(audio_path))
        print(json.dumps(result, indent=2))
        print("Parakeet loaded and decoded successfully.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Parakeet check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
