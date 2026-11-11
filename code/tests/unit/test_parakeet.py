from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from willspeak.asr.parakeet import ParakeetASR, model_files


def make_model_dir(path):
    path.mkdir()
    for name in ("encoder.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (path / name).write_bytes(b"model")
    return path


def test_model_files_prefers_int8(tmp_path):
    files = model_files(make_model_dir(tmp_path / "parakeet"))
    assert files is not None
    assert files["encoder"].name == "encoder.int8.onnx"
    assert set(files) == {"encoder", "decoder", "joiner", "tokens"}


def test_model_files_rejects_incomplete_directory(tmp_path):
    model_dir = tmp_path / "incomplete"
    model_dir.mkdir()
    (model_dir / "encoder.int8.onnx").write_bytes(b"model")
    assert model_files(model_dir) is None


def test_transcribe_builds_sherpa_recognizer_and_decodes(monkeypatch, tmp_path):
    model_dir = make_model_dir(tmp_path / "parakeet")
    wav_path = tmp_path / "sample.wav"
    sf.write(wav_path, np.zeros(3200, dtype=np.float32), 16000)
    captured = {}

    class Stream:
        result = SimpleNamespace(text="test phrase")

        def accept_waveform(self, sample_rate, samples):
            captured["sample_rate"] = sample_rate
            captured["samples"] = samples

    class Recognizer:
        @staticmethod
        def from_transducer(**kwargs):
            captured["config"] = kwargs
            return Recognizer()

        def create_stream(self):
            return Stream()

        def decode_stream(self, stream):
            captured["decoded"] = stream

    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=Recognizer))
    result = ParakeetASR(model_dir).transcribe(str(wav_path))

    assert captured["config"]["decoding_method"] == "greedy_search"
    assert captured["config"]["model_type"] == "nemo_transducer"
    assert captured["config"]["feature_dim"] == 128
    assert captured["sample_rate"] == 16000
    assert result["text"] == "test phrase"
    assert result["model"].startswith("parakeet:")
