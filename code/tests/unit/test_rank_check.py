import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_lab import SPEAKERS, make_export, make_system

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import labkit as lab
import rank_check

CLIPS = ["M01-0", "M01-1", "M01-2", "M02-0", "M02-1", "M04-0", "M04-1"]


def wrong(n: int) -> dict[str, str]:
    return {clip: "" for clip in CLIPS[:n]}


def pooled(tmp_path, name: str, errors: int) -> lab.Pooled:
    export = make_export(tmp_path / "export")
    manifests = {s: lab.read_manifest(export / s / "test.jsonl") for s in SPEAKERS}
    return lab.load_pooled(name, make_system(tmp_path, name, wrong(errors)), manifests)


def test_concordance_counts_pairs():
    assert rank_check.concordance([1, 2, 3], [10, 20, 30]) == {"concordant": 3, "discordant": 0, "ties": 0}
    assert rank_check.concordance([1, 2, 3], [30, 20, 10]) == {"concordant": 0, "discordant": 3, "ties": 0}
    assert rank_check.concordance([1, 1, 3], [5, 6, 7]) == {"concordant": 2, "discordant": 0, "ties": 1}
    with pytest.raises(ValueError):
        rank_check.concordance([1, 2], [1])


def test_same_order_gives_correlation_one_and_reversed_gives_minus_one(tmp_path):
    stock = {c: pooled(tmp_path, f"s-{c}", n) for c, n in (("a", 1), ("b", 3), ("c", 6))}
    same = {c: pooled(tmp_path, f"t-{c}", n) for c, n in (("a", 0), ("b", 1), ("c", 2))}
    result = rank_check.build(stock, same)
    assert result["scopes"]["pooled"]["spearman"] == pytest.approx(1.0)
    assert result["scopes"]["pooled"]["pairs"]["concordant"] == 3
    assert result["scopes"]["pooled"]["chance_of_this_exact_order"] == pytest.approx(1 / 6)
    reversed_ = {c: pooled(tmp_path, f"r-{c}", n) for c, n in (("a", 2), ("b", 1), ("c", 0))}
    flipped = rank_check.build(stock, reversed_)
    assert flipped["scopes"]["pooled"]["spearman"] == pytest.approx(-1.0)
    assert flipped["gain_points_pooled"]["c"] > flipped["gain_points_pooled"]["a"]
    assert set(flipped["scopes"]) == {"pooled", *SPEAKERS}
    assert "Spearman" in rank_check.render(flipped)


def test_build_refuses_two_candidates_or_mismatched_lists(tmp_path):
    two = {c: pooled(tmp_path, f"x-{c}", n) for c, n in (("a", 1), ("b", 2))}
    with pytest.raises(ValueError, match="at least 3"):
        rank_check.build(two, two)
    three = {c: pooled(tmp_path, f"y-{c}", n) for c, n in (("a", 1), ("b", 2), ("c", 3))}
    other = {c: pooled(tmp_path, f"z-{c}", n) for c, n in (("a", 1), ("c", 2), ("b", 3))}
    with pytest.raises(ValueError, match="same candidates"):
        rank_check.build(three, other)


def test_command_line_writes_the_result_once(tmp_path):
    export = make_export(tmp_path / "export")
    command = [sys.executable, str(ROOT / "scripts" / "asr" / "rank_check.py"), "--export", str(export),
               "--speakers", ",".join(SPEAKERS), "--out", str(tmp_path / "rank.json")]
    for c, (s, t) in {"a": (1, 0), "b": (3, 1), "c": (6, 2)}.items():
        stock = make_system(tmp_path, f"stock-{c}", wrong(s))
        tuned = make_system(tmp_path, f"tuned-{c}", wrong(t))
        command += ["--stock", f"{c}=" + ",".join(f"{k}:{v}" for k, v in stock.items()),
                    "--tuned", f"{c}=" + ",".join(f"{k}:{v}" for k, v in tuned.items())]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads((tmp_path / "rank.json").read_text())["scopes"]["pooled"]["spearman"] == pytest.approx(1.0)
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0 and "Already exists" in (again.stderr + again.stdout)
