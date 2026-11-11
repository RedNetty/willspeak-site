import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import labkit as lab
import pool_table

SPEAKERS = ("M01", "M02", "M04")
TEXTS = {
    "M01": ["i want water", "turn on the light", "yes", "call my mother please"],
    "M02": ["good morning", "i am cold", "open the door"],
    "M04": ["tomato", "where is the bathroom", "help me", "thank you very much", "no"],
}


def write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def make_export(root: Path) -> Path:
    for speaker, texts in TEXTS.items():
        write_jsonl(root / speaker / "test.jsonl", [{"id": f"{speaker}-{i}", "text": t, "audio": f"test/{i}.wav"} for i, t in enumerate(texts)])
    return root


def make_system(root: Path, name: str, edits=None) -> dict[str, str]:
    edits = edits or {}
    folders = {}
    for speaker, texts in TEXTS.items():
        rows = [{"id": f"{speaker}-{i}", "hyp": edits.get(f"{speaker}-{i}", t)} for i, t in enumerate(texts)]
        write_jsonl(root / name / speaker / "hypotheses.jsonl", rows)
        folders[speaker] = str(root / name / speaker)
    return folders


def test_bar_words_is_floor_of_thirty_five_percent():
    assert lab.bar_words(425) == 148
    assert lab.bar_words(100) == 35
    assert lab.bar_words(20) == 7


def test_read_hyps_takes_a_folder_or_a_file_and_refuses_duplicates(tmp_path):
    good = write_jsonl(tmp_path / "a" / "hypotheses.jsonl", [{"id": "x", "hyp": "hello"}, {"id": "y", "hyp": ""}])
    assert lab.read_hyps(good.parent) == {"x": "hello", "y": ""}
    assert lab.read_hyps(good) == {"x": "hello", "y": ""}
    dup = write_jsonl(tmp_path / "b.jsonl", [{"id": "x", "hyp": "a"}, {"id": "x", "hyp": "b"}])
    with pytest.raises(ValueError, match="Duplicate"):
        lab.read_hyps(dup)
    null = write_jsonl(tmp_path / "c.jsonl", [{"id": "x", "hyp": None}])
    with pytest.raises(ValueError, match="non-text"):
        lab.read_hyps(null)
    with pytest.raises(FileNotFoundError):
        lab.read_hyps(tmp_path / "missing")


def test_parse_system():
    assert lab.parse_system("w=M01:a/b,M02:c") == ("w", {"M01": "a/b", "M02": "c"})
    for bad in ("nolabel", "x=M01", "x=M01:a,M01:b"):
        with pytest.raises(ValueError):
            lab.parse_system(bad)


def test_clip_counts_and_empty_share():
    manifest = {"a": {"text": "hello world"}, "b": {"text": "tomato"}, "c": {"text": "one two three"}}
    hyps = {"a": "hello word", "b": "", "c": "One, two, three!"}
    counts = lab.clip_counts(manifest, hyps)
    assert counts == {"a": (1, 2), "b": (1, 1), "c": (0, 3)}
    assert lab.totals(counts) == (2, 6)
    assert lab.empty_share(manifest, hyps) == (1, 3)
    with pytest.raises(KeyError):
        lab.clip_counts(manifest, {"a": "x"})


def test_british_spellings_are_not_errors():
    assert lab.clip_counts({"a": {"text": "the colour red"}}, {"a": "the color red"}) == {"a": (0, 3)}


def test_pooled_counts_and_interval(tmp_path):
    export = make_export(tmp_path / "export")
    manifests = {s: lab.read_manifest(export / s / "test.jsonl") for s in SPEAKERS}
    perfect = lab.load_pooled("perfect", make_system(tmp_path, "perfect"), manifests)
    broken = lab.load_pooled("broken", make_system(tmp_path, "broken", {"M01-0": "i want", "M02-1": "", "M04-0": "potato", "M04-1": ""}), manifests)
    assert perfect.pooled_wrong() == 0 and perfect.pooled_words() == sum(len(t.split()) for ts in TEXTS.values() for t in ts)
    assert broken.wrong("M01") == 1 and broken.wrong("M02") == 3 and broken.wrong("M04") == 1 + 4
    assert broken.pooled_empty() == (2, 12)
    single = lab.pooled_interval(broken)
    assert single["ci"][0] <= single["wer"] <= single["ci"][1]
    pair = lab.pooled_interval(perfect, broken)
    assert pair["gain_over_other"] > 0 and pair["gain_ci"][0] <= pair["gain_over_other"] <= pair["gain_ci"][1]
    by_prompt = lab.pooled_interval(perfect, broken, by="prompt")
    assert by_prompt["unit"] == "prompt" and by_prompt["gain_over_other"] == pair["gain_over_other"]
    with pytest.raises(ValueError):
        lab.pooled_interval(perfect, by="speaker")


def test_a_system_missing_a_speaker_is_refused(tmp_path):
    export = make_export(tmp_path / "export")
    manifests = {s: lab.read_manifest(export / s / "test.jsonl") for s in SPEAKERS}
    folders = make_system(tmp_path, "partial")
    del folders["M02"]
    with pytest.raises(ValueError, match="no hypotheses"):
        lab.load_pooled("partial", folders, manifests)


def test_ranks_and_spearman():
    assert lab.ranks([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]
    assert lab.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert lab.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(lab.spearman([1, 1, 1], [1, 2, 3]))
    with pytest.raises(ValueError):
        lab.spearman([1], [1])


def test_verdict_needs_a_big_enough_gap_and_two_speakers_agreeing():
    assert pool_table.verdict(8.0, {"M01": 9.0, "M02": 7.0, "M04": 8.0}) == "better"
    assert pool_table.verdict(8.0, {"M01": 20.0, "M02": -3.0, "M04": 1.0}) == "better"
    assert pool_table.verdict(8.0, {"M01": 20.0, "M02": -3.0, "M04": -1.0}) == "unclear"
    assert pool_table.verdict(3.0, {"M01": 3.0, "M02": 3.0, "M04": 3.0}) == "unclear"
    assert pool_table.verdict(-9.0, {"M01": -9.0, "M02": -8.0, "M04": 1.0}) == "worse"
    assert pool_table.verdict(5.0, {"M01": 5.0}, noise_points=4.5) == "better"


def test_pool_table_builds_and_renders(tmp_path):
    export = make_export(tmp_path / "export")
    manifests = {s: lab.read_manifest(export / s / "test.jsonl") for s in SPEAKERS}
    systems = {
        "stock": lab.load_pooled("stock", make_system(tmp_path, "stock", {"M01-0": "", "M01-1": "", "M02-0": "", "M04-1": "", "M04-3": "x"}), manifests),
        "tuned": lab.load_pooled("tuned", make_system(tmp_path, "tuned"), manifests),
    }
    rows = pool_table.build(systems, "stock")
    stock, tuned = rows
    assert "gain_points" not in stock and tuned["gain_points"] > 0 and tuned["against"] == "stock"
    assert tuned["meets_bar"] and not stock["meets_bar"]
    text = pool_table.render(rows, list(SPEAKERS), "clip")
    assert "tuned against stock" in text and "bar: 35%" in text
    with pytest.raises(ValueError, match="Baseline"):
        pool_table.build(systems, "nope")


def test_pool_table_command_line_writes_a_refusing_json(tmp_path):
    export = make_export(tmp_path / "export")
    stock = make_system(tmp_path, "stock", {"M01-0": ""})
    tuned = make_system(tmp_path, "tuned")
    out = tmp_path / "out" / "table.json"
    command = [sys.executable, str(ROOT / "scripts" / "asr" / "pool_table.py"), "--export", str(export),
               "--system", "stock=" + ",".join(f"{s}:{p}" for s, p in stock.items()),
               "--system", "tuned=" + ",".join(f"{s}:{p}" for s, p in tuned.items()), "--out", str(out)]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    table = json.loads(out.read_text())
    assert [r["system"] for r in table["rows"]] == ["stock", "tuned"] and table["provenance"]["format"] == "willspeak-lab-v1"
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0 and "Already exists" in (again.stdout + again.stderr)


def test_provenance_accepts_a_symlinked_input_like_the_huggingface_cache(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"weights")
    link = tmp_path / "model.nemo"
    link.symlink_to(blob)
    script = tmp_path / "script.py"
    script.write_text("pass")
    record = lab.provenance(script, {"model": link})
    assert record["inputs"]["model"]["size"] == 7


def make_kind_export(root: Path) -> Path:
    for speaker, texts in TEXTS.items():
        write_jsonl(root / speaker / "test.jsonl",
                    [{"id": f"{speaker}-{i}", "text": t, "audio": f"test/{i}.wav", "kind": "word" if len(t.split()) == 1 else "sentence"}
                     for i, t in enumerate(texts)])
    return root


def test_only_kind_splits_sentences_from_single_words():
    manifest = {"a": {"text": "yes", "kind": "word"}, "b": {"text": "i am cold", "prompt_set": "torgo-sentence"}, "c": {"text": "no", "kind": "word"}}
    assert list(lab.only_kind(manifest, "sentence")) == ["b"]
    assert list(lab.only_kind(manifest, "word")) == ["a", "c"]
    assert lab.only_kind(manifest, "all") == manifest
    with pytest.raises(ValueError):
        lab.only_kind(manifest, "phrase")


def test_pool_table_adds_a_sentences_only_table(tmp_path):
    export = make_kind_export(tmp_path / "export")
    stock = make_system(tmp_path, "stock", {"M01-0": "", "M02-1": "", "M04-0": "potato"})
    tuned = make_system(tmp_path, "tuned", {"M01-2": "no"})
    out = tmp_path / "out" / "table.json"
    command = [sys.executable, str(ROOT / "scripts" / "asr" / "pool_table.py"), "--export", str(export),
               "--system", "stock=" + ",".join(f"{s}:{p}" for s, p in stock.items()),
               "--system", "tuned=" + ",".join(f"{s}:{p}" for s, p in tuned.items()), "--out", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "secondary score: sentences only" in done.stdout
    table = json.loads(out.read_text())
    sentence_words = sum(len(t.split()) for texts in TEXTS.values() for t in texts if len(t.split()) > 1)
    tuned_all = next(r for r in table["rows"] if r["system"] == "tuned")
    tuned_sentences = next(r for r in table["sentences"] if r["system"] == "tuned")
    assert tuned_all["pooled_wrong"] == 1 and tuned_sentences["pooled_wrong"] == 0
    assert tuned_sentences["pooled_words"] == sentence_words
    only = subprocess.run([*command[:-2], "--kind", "sentence"], capture_output=True, text=True, check=False)
    assert only.returncode == 0 and "sentence clips only" in only.stdout and "secondary score" not in only.stdout


def test_sentence_rescore_groups_speakers_and_seeds_and_ranks_by_sentences(tmp_path):
    import sentence_rescore as rescore

    export = make_kind_export(tmp_path / "export")
    manifests = {s: lab.read_manifest(export / s / "test.jsonl") for s in SPEAKERS}
    reports = tmp_path / "reports"
    for seed in (17, 18):
        for s, texts in TEXTS.items():
            good = [{"id": f"{s}-{i}", "hyp": t} for i, t in enumerate(texts)]
            words_wrong = [{"id": r["id"], "hyp": "zzz" if len(r["hyp"].split()) == 1 else r["hyp"]} for r in good]
            write_jsonl(reports / "ft" / f"big-a-{s}-s{seed}-test-nemo.jsonl", words_wrong)
            write_jsonl(reports / "ft" / f"big-b-{s}-s{seed}-test-nemo" / "hypotheses.jsonl",
                        [{"id": r["id"], "hyp": "" if len(r["hyp"].split()) > 1 else r["hyp"]} for r in good])
            write_jsonl(reports / "ft" / f"big-c-{s}-s{seed}-test-nemo.jsonl", words_wrong)
    write_jsonl(reports / "ft" / "big-d-M01-s17-test-nemo.jsonl", [{"id": "M01-0", "hyp": "x"}])
    scored = rescore.from_decodes(reports, manifests)
    rows = rescore.families(scored, list(SPEAKERS))
    assert [r["family"] for r in rows] == ["ft/big-a-{S}{seed}", "ft/big-b-{S}{seed}"]
    best, worst = rows
    assert best["seeds"] == ["17", "18"] and best["sentence_mean"] == 0 and best["word_wer"] == 1.0
    assert best["same_runs_as"] == ["ft/big-c-{S}{seed}"]
    assert worst["sentence_wer"] == 1.0 and worst["word_mean"] == 0
    assert "meets 35%" in rescore.render(rows, list(SPEAKERS), "test")


def test_sentence_rescore_reads_saved_results_without_manifests(tmp_path):
    import sentence_rescore as rescore

    reports = tmp_path / "reports"
    for s, n in (("M01", (169, 122, 47)), ("M02", (95, 68, 27)), ("M04", (161, 113, 48))):
        path = reports / "ft" / f"x-{s}-s17-test-nemo" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"word": {"errors": 50, "n_ref": n[0]},
                                    "by_group": {"torgo-sentence": {"errors": 20, "n_ref": n[1]}, "torgo-word": {"errors": 30, "n_ref": n[2]}}}))
    rows = rescore.families(rescore.from_results(reports, rescore.FALLBACK_WORDS), list(SPEAKERS))
    assert len(rows) == 1 and rows[0]["sentence_mean"] == 60 and rows[0]["words"]["sentence"] == 303
