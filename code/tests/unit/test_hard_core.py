import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import hard_core  # noqa: E402

CLIPS = {
    "M01": [("a1", "sentence", 2.0, "the cat sat"), ("a2", "word", 0.8, "tomato"), ("a3", "word", 0.9, "paper")],
    "M02": [("b1", "sentence", 4.0, "we went home early"), ("b2", "word", 0.7, "juice")],
}
ANSWERS = {
    "stock": {"a1": "the cat sat", "a2": "potato", "a3": "", "b1": "we went home", "b2": "chews"},
    "better": {"a1": "the cat sat", "a2": "tomato", "a3": "", "b1": "we went home early", "b2": "juice"},
}


def build(tmp_path: Path) -> tuple[str, dict]:
    export = tmp_path / "export"
    systems: dict[str, dict[str, str]] = {label: {} for label in ANSWERS}
    for s, rows in CLIPS.items():
        (export / s).mkdir(parents=True)
        (export / s / "test.jsonl").write_text("".join(json.dumps({"id": i, "audio": f"test/{i}.wav", "text": t, "kind": k, "duration_s": d}) + "\n"
                                                        for i, k, d, t in rows), encoding="utf-8")
        for label, answers in ANSWERS.items():
            folder = tmp_path / label / s
            folder.mkdir(parents=True)
            (folder / "hypotheses.jsonl").write_text("".join(json.dumps({"id": i, "hyp": answers[i]}) + "\n" for i, *_ in rows), encoding="utf-8")
            systems[label][s] = str(folder)
    return str(export), systems


def test_oracle_shared_errors_and_breakdowns(tmp_path):
    export, systems = build(tmp_path)
    r = hard_core.analyse(systems, export, top=2)
    assert r["words"] == 10 and r["pooled"] == {"stock": 4, "better": 1} and r["best_system"] == "better"
    assert r["oracle"] == 1 and r["best_errors_no_system_fixes"] == 1 and r["best_errors_fixable_by_another_system"] == 0
    assert r["clips_every_system_fully_wrong"] == 1 and r["words_in_them"] == 1
    assert r["clips_every_system_right"] == 1 and r["words_in_those"] == 3
    assert r["by_kind"]["word"] == {"clips": 3, "words": 3, "oracle": 1, "best": 1, "every_system_wrong": 1}
    assert r["by_duration"]["over 3 s"]["best"] == 0 and r["by_duration"]["under 1.5 s"]["clips"] == 3
    assert [c["id"] for c in r["hardest"]] == ["a3", "a2"]
    assert r["hardest"][0]["answers"] == {"stock": "", "better": ""}


def test_a_system_missing_a_speaker_is_refused(tmp_path):
    export, systems = build(tmp_path)
    del systems["better"]["M02"]
    with pytest.raises(ValueError, match="does not cover"):
        hard_core.analyse(systems, export)


def test_the_command_writes_the_json_with_provenance(tmp_path, capsys):
    export, systems = build(tmp_path)
    out = tmp_path / "hard_core.json"
    args = [f"--system={label}=" + ",".join(f"{s}:{p}" for s, p in per.items()) for label, per in systems.items()]
    assert hard_core.main([*args, "--export-root", export, "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["oracle"] == 1 and set(saved["provenance"]["inputs"]) == {"stock:M01", "stock:M02", "better:M01", "better:M02"}
    assert "hardest clips" in capsys.readouterr().out
