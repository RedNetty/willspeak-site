import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import labkit as lab
import phrasebook
import pseudo_label as pl

sf = pytest.importorskip("soundfile")

PHRASES = ["i want water", "turn on the light", "call my mother please", "good morning", "open the door",
           "where is the bathroom", "help me", "thank you very much", "i am cold", "please close the window",
           "what time is it", "bring me a blanket"]

CLIPS = {
    "c0": ("i want water", "i want water"),
    "c1": ("turn on the light", "turn on the lite"),
    "c2": ("call my mother please", ""),
    "c3": ("good morning", "xyzzy plugh quux"),
    "c4": ("open the door", "open the doors"),
    "c5": ("where is the bathroom", "where is the bath room"),
    "c6": ("help me", "help me"),
    "c7": ("thank you very much", "thank you very much"),
    "c8": ("what time is it", "what time is it"),
    "c9": ("bring me a blanket", "bring me a blanket"),
    "c10": ("some sentence nobody listed", "i want water"),
    "c11": ("i am cold", "i am cold"),
}


def items(texts):
    return [phrasebook.Item(t) for t in texts]


def fixture(tmp_path, with_text=True):
    folder = tmp_path / "M09"
    rows = []
    for i, (clip, (truth, _)) in enumerate(CLIPS.items()):
        (folder / "train").mkdir(parents=True, exist_ok=True)
        sf.write(folder / "train" / f"{clip}.wav", np.random.default_rng(i).normal(size=1600).astype("float32") * 0.1, 16000)
        row = {"id": clip, "audio": f"train/{clip}.wav", "duration_s": 0.1, "speaker": "M09", "kind": "sentence"}
        if with_text:
            row["text"] = truth
        rows.append(row)
    manifest = folder / "train.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    hyps = tmp_path / "hyps"
    hyps.mkdir()
    (hyps / "hypotheses.jsonl").write_text("".join(json.dumps({"id": c, "hyp": h}) + "\n" for c, (_, h) in CLIPS.items()), encoding="utf-8")
    listing = tmp_path / "phrases.txt"
    listing.write_text("\n".join(PHRASES) + "\n", encoding="utf-8")
    test = tmp_path / "test.jsonl"
    test.write_text(json.dumps({"id": "t0", "text": "do not use this prompt", "audio": "x.wav"}) + "\n", encoding="utf-8")
    return manifest, hyps, listing, test


def run(tmp_path, manifest, hyps, listing, out, *extra):
    command = [sys.executable, str(ROOT / "scripts" / "asr" / "pseudo_label.py"), "--manifest", str(manifest), "--hyps", str(hyps),
               "--phrases", str(listing), "--out", str(out), "--dev-share", "0.25", *extra]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_gate_keeps_a_near_match_and_gives_reasons_for_the_rest():
    phrases = items(PHRASES)
    ok = pl.gate("turn on the lite", phrases, 0.5, 0.1)
    assert ok["accepted"] and ok["phrase"] == "turn on the light"
    far = pl.gate("xyzzy plugh quux", phrases, 0.5, 0.1)
    assert not far["accepted"] and far["reason"] == "too far from every phrase"
    assert not pl.gate("", phrases, 0.5, 0.1)["accepted"] and pl.gate("  ", phrases, 0.5, 0.1)["reason"] == "empty answer"
    close = items(["turn on the light", "turn on the lights", "good morning"])
    assert pl.gate("turn on the lite", close, 0.5, 0.1)["reason"] == "two phrases about equally near"
    with pytest.raises(ValueError):
        pl.gate("hello", items(["hello"]), 0.5, 0.1)


def test_dev_phrases_is_stable_never_empty_and_leaves_train_something():
    phrases = [f"phrase number {i}" for i in range(8)]
    held = pl.dev_phrases(phrases, 0.12)
    assert held == pl.dev_phrases(list(reversed(phrases)), 0.12) and len(held) == 1
    assert len(pl.dev_phrases(phrases, 0.5)) == 4 and len(pl.dev_phrases(phrases[:5], 0.01)) == 1


def test_export_labels_with_the_phrase_and_never_splits_a_phrase(tmp_path):
    manifest, hyps, listing, _ = fixture(tmp_path)
    done = run(tmp_path, manifest, hyps, listing, tmp_path / "out")
    assert done.returncode == 0, done.stderr
    train = lab.read_manifest(tmp_path / "out" / "train.jsonl")
    dev = lab.read_manifest(tmp_path / "out" / "dev.jsonl")
    assert train and dev and not set(train) & set(dev)
    assert {r["text"] for r in train.values()}.isdisjoint({r["text"] for r in dev.values()})
    everything = {**train, **dev}
    assert everything["c1"]["text"] == "turn on the light"
    assert everything["c10"]["text"] == "i want water"
    assert "c2" not in everything and "c3" not in everything
    assert all("prompt_text" not in r for r in everything.values())
    assert all((tmp_path / "out" / r["audio"]).is_file() for r in everything.values())
    assert (tmp_path / "out" / "phrases.txt").read_text().split("\n")[0] == "i want water"


def test_the_loop_runs_when_the_manifest_has_no_text_at_all(tmp_path):
    manifest, hyps, listing, _ = fixture(tmp_path, with_text=False)
    assert run(tmp_path, manifest, hyps, listing, tmp_path / "blind").returncode == 0
    audited = run(tmp_path, manifest, hyps, listing, tmp_path / "blind2", "--audit")
    assert audited.returncode != 0


def test_audit_counts_right_and_wrong_labels_without_changing_the_export(tmp_path):
    manifest, hyps, listing, _ = fixture(tmp_path)
    plain = run(tmp_path, manifest, hyps, listing, tmp_path / "plain")
    audited = run(tmp_path, manifest, hyps, listing, tmp_path / "audited", "--audit")
    assert plain.returncode == 0 and audited.returncode == 0, audited.stderr
    for name in ("train.jsonl", "dev.jsonl"):
        assert (tmp_path / "plain" / name).read_text() == (tmp_path / "audited" / name).read_text()
    summary = json.loads((tmp_path / "audited" / "selection.json").read_text())
    audit = summary["audit"]
    assert audit["wrong"] == 1 and audit["wrong_because_true_phrase_not_on_list"] == 1
    assert audit["right"] == audit["kept"] - 1 and audit["true_phrase_on_list"] == 11
    assert audit["precision"] == pytest.approx(audit["right"] / audit["kept"])
    assert json.loads((tmp_path / "plain" / "selection.json").read_text())["audit"] is None


def test_refuses_a_phrase_list_that_holds_a_test_prompt_and_an_existing_folder(tmp_path):
    manifest, hyps, listing, test = fixture(tmp_path)
    listing.write_text("\n".join([*PHRASES, "do not use this prompt"]) + "\n", encoding="utf-8")
    leaked = run(tmp_path, manifest, hyps, listing, tmp_path / "leak", "--test-manifest", str(test))
    assert leaked.returncode != 0 and "test prompts" in (leaked.stderr + leaked.stdout)
    listing.write_text("\n".join(PHRASES) + "\n", encoding="utf-8")
    assert run(tmp_path, manifest, hyps, listing, tmp_path / "fine", "--test-manifest", str(test)).returncode == 0
    again = run(tmp_path, manifest, hyps, listing, tmp_path / "fine")
    assert again.returncode != 0 and "Already exists" in (again.stderr + again.stdout)
