import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import candidates
import hf_hyp
import labkit as lab


def test_tags_are_unique_and_every_candidate_has_a_loader():
    assert len({c.tag for c in candidates.CANDIDATES}) == len(candidates.CANDIDATES)
    for c in candidates.CANDIDATES:
        assert c.family in ("nemo", "hf") and c.role in ("screen", "probe")
        if c.family == "hf":
            assert c.kind in hf_hyp.KINDS
        if c.finetune:
            assert c.family == "nemo" and c.loss in ("tdt", "rnnt")


def test_select_filters():
    assert {c.tag for c in candidates.select("hf", "probe")} == {"cohere", "granite41", "granite4"}
    assert [c.tag for c in candidates.select(tags=["v3", "qwen3"])] == ["v3", "qwen3"]
    with pytest.raises(KeyError):
        candidates.select(tags=["v9"])


def test_command_line_lines_split_on_a_pipe():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/asr/candidates.py"), "--family", "nemo", "--finetune", "--lines"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert "v3|tdt|nvidia/parakeet-tdt-0.6b-v3" in out
    assert not any(line.startswith("ctc06|") for line in out)
    assert all(line.count("|") == 2 for line in out)


def test_granite_prompts_follow_each_card():
    assert "written format" in hf_hyp.granite_prompt("ibm-granite/granite-4.0-1b-speech")
    assert "punctuation" in hf_hyp.granite_prompt("ibm-granite/granite-speech-4.1-2b")
    with pytest.raises(ValueError):
        hf_hyp.granite_prompt("someone/else")


def test_audio_must_already_be_16k_mono(tmp_path):
    sf = pytest.importorskip("soundfile")
    good = tmp_path / "good.wav"
    sf.write(good, np.zeros(1600, np.float32), 16000)
    assert lab.read_audio(good).shape == (1600,)
    wrong_rate = tmp_path / "rate.wav"
    sf.write(wrong_rate, np.zeros(1600, np.float32), 44100)
    with pytest.raises(ValueError, match="44100"):
        lab.read_audio(wrong_rate)
    stereo = tmp_path / "stereo.wav"
    sf.write(stereo, np.zeros((1600, 2), np.float32), 16000)
    with pytest.raises(ValueError, match="channels"):
        lab.read_audio(stereo)
