import json
import subprocess
import sys
from pathlib import Path

from tests.unit.test_lab import SPEAKERS, TEXTS, make_export, make_system

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import hyp_bounds
import labkit as lab


def test_vote_follows_the_majority_and_ties_go_to_the_pivot():
    assert hyp_bounds.vote(["a b c", "a b c", "a b c"]) == "a b c"
    assert hyp_bounds.vote(["a b c", "a x c", "a b c"]) == "a b c"
    assert hyp_bounds.vote(["a x c", "a b c", "a b c"]) == "a b c"
    assert hyp_bounds.vote(["a b", "a x"]) == "a b"


def test_vote_handles_deletions_and_insertions():
    assert hyp_bounds.vote(["a b c", "a c", "a c"]) == "a c"
    assert hyp_bounds.vote(["a b c", "a c", "a b c"]) == "a b c"
    assert hyp_bounds.vote(["a c", "a b c", "a b c"]) == "a b c"
    assert hyp_bounds.vote(["a c", "a b c"]) == "a c"
    assert hyp_bounds.vote(["", "", ""]) == ""
    assert hyp_bounds.vote(["", "yes", "yes"]) == "yes"


def test_vote_normalises_the_way_scoring_does():
    assert hyp_bounds.vote(["Hello, World!", "hello world", "hello world"]) == "hello world"


def test_oracle_is_never_worse_than_the_best_system(tmp_path):
    export = make_export(tmp_path / "export")
    manifests = {s: lab.read_manifest(export / s / "test.jsonl") for s in SPEAKERS}
    first = lab.load_pooled("first", make_system(tmp_path, "first", {"M01-0": "", "M02-0": "", "M04-0": ""}), manifests)
    second = lab.load_pooled("second", make_system(tmp_path, "second", {"M01-1": "", "M02-1": "", "M04-1": ""}), manifests)
    hyps = {name: {s: lab.read_hyps(tmp_path / name / s) for s in SPEAKERS} for name in ("first", "second")}
    report = hyp_bounds.group_report(("first", "second"), {"first": first, "second": second}, manifests, hyps, "first")
    assert report["oracle_pooled"] == 0
    assert report["oracle_pooled"] <= min(first.pooled_wrong(), second.pooled_wrong())
    assert report["pivot"] == "first" and report["pooled_words"] == first.pooled_words()


def test_command_line_reports_and_refuses_to_overwrite(tmp_path):
    export = make_export(tmp_path / "export")
    systems = {"a": make_system(tmp_path, "a", {"M01-0": ""}), "b": make_system(tmp_path, "b", {"M02-1": ""}),
               "c": make_system(tmp_path, "c", {"M04-0": "potato"})}
    out = tmp_path / "bounds"
    command = [sys.executable, str(ROOT / "scripts" / "asr" / "hyp_bounds.py"), "--export", str(export), "--out", str(out)]
    for label, folders in systems.items():
        command += ["--system", f"{label}=" + ",".join(f"{s}:{p}" for s, p in folders.items())]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    result = json.loads((out / "results.json").read_text())
    assert result["pooled_words"] == sum(len(t.split()) for ts in TEXTS.values() for t in ts)
    assert result["groups"][0]["oracle_pooled"] == 0
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode != 0
    one = subprocess.run(command[:6] + command[6:8], capture_output=True, text=True, check=False)
    assert one.returncode != 0 and "at least two" in one.stderr
