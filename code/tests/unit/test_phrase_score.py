import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "asr"))
import lattice_fast as lf
import phrase_report as pr

DURATIONS = [0, 1, 2]


def random_tables(frames, labels, seed):
    rng = random.Random(seed)

    def logp(n):
        raw = [rng.random() + 0.05 for _ in range(n)]
        s = sum(raw)
        return [math.log(x / s) for x in raw]

    blank = [[logp(3)[0] for _u in range(labels + 1)] for _t in range(frames)]
    label = [[logp(3)[0] for _u in range(labels)] for _t in range(frames)]
    dur = [[logp(len(DURATIONS)) for _u in range(labels + 1)] for _t in range(frames)]
    return blank, label, dur


def brute_force(blank, label, dur, durations):
    frames, labels = len(blank), len(label[0])
    total = [0.0]

    def walk(t, u, logp):
        for n, d in enumerate(durations):
            if d >= 1:
                arc = logp + blank[t][u] + dur[t][u][n]
                if t + d == frames and u == labels:
                    total[0] += math.exp(arc)
                elif t + d < frames:
                    walk(t + d, u, arc)
            if u < labels and t + d < frames:
                walk(t + d, u + 1, logp + label[t][u] + dur[t][u][n])

    walk(0, 0, 0.0)
    return -math.log(total[0]) if total[0] > 0 else float("inf")


@pytest.mark.parametrize("frames,labels,seed", [(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 4), (3, 2, 5)])
def test_list_lattice_matches_brute_force(frames, labels, seed):
    blank, label, dur = random_tables(frames, labels, seed)
    assert lf.lattice_nll_lists(blank, label, dur, DURATIONS) == pytest.approx(brute_force(blank, label, dur, DURATIONS), abs=1e-9)


def test_list_lattice_no_alignment():
    blank, label, dur = random_tables(1, 3, 9)
    assert brute_force(blank, label, dur, [1]) == float("inf")
    assert lf.lattice_nll_lists(blank, label, dur[:], [1]) == float("inf")


def test_list_lattice_matches_torch_lattice():
    torch = pytest.importorskip("torch")
    import tdt_lattice as tl

    g = torch.Generator().manual_seed(3)
    logits = torch.randn(6, 4, 5 + 1 + len(DURATIONS), generator=g)
    tok, dur = tl.split_logprobs(logits, len(DURATIONS), 0.05)
    targets = [1, 3, 2]
    slow = float(tl.sequence_nll(tok, dur, targets, DURATIONS, 5))
    b, l, d = tl.reduce_logprobs(tok, dur, targets, 5)
    assert lf.lattice_nll_lists(b.tolist(), l.tolist(), d.tolist(), DURATIONS) == pytest.approx(slow, abs=1e-4)


def test_pick_candidates_keeps_true_phrase_and_answer(monkeypatch):
    import types

    if "willspeak.g2p" not in sys.modules:
        try:
            import willspeak.g2p  # noqa: F401
        except Exception:  # noqa: BLE001
            fake = types.ModuleType("willspeak.g2p")
            fake.SIL = "_"
            fake.text_phones = lambda text: (list(text.replace(" ", "")), [])
            monkeypatch.setitem(sys.modules, "willspeak.g2p", fake)
    ps = pytest.importorskip("phrase_pick")
    pb = pytest.importorskip("phrasebook")
    pool = [pb.Item(t) for t in ["open the door", "close the door", "tomato", "potato", "i want water", "turn the page"]]
    ranking = ps.rank_pool(pb.Item("pot ato"), pool)
    cands = ps.pick_candidates(ranking, pool, "i want water", "pot ato", 2)
    texts = [c["text"] for c in cands]
    assert len(texts) == len(set(texts))
    assert "i want water" in texts and sum(c["true"] for c in cands) == 1
    assert cands[-1]["hyp"] and not cands[-1]["in_pool"] and cands[-1]["rank"] is None
    ranks = [c["rank"] for c in cands if c["in_pool"] and not c["true"]]
    assert ranks == sorted(ranks) and len(cands) in (4, 5)
    true = next(c for c in cands if c["true"])
    assert true["rank"] == [i for i, _d in ranking].index(pool.index(next(p for p in pool if p.norm == "i want water")))


def clip(cid, ref, hyp, cands, empty=50.0, kind="sentence"):
    return {"id": cid, "kind": kind, "ref": ref, "hyp": hyp, "frames": 10, "nll_empty": empty, "candidates": cands}


def cand(text, rank, nll, true=False, hyp=False, in_pool=True, dist=0.1):
    return {"text": text, "dist": dist, "rank": rank, "true": true, "hyp": hyp, "in_pool": in_pool, "nll": nll}


def write_table(tmp_path, name, rows, shortlist=2):
    path = tmp_path / name
    path.write_text(json.dumps({"shortlist": shortlist, "rows": rows}), encoding="utf-8")
    return str(path)


def test_word_errors():
    assert pr.word_errors("a b c".split(), "a x c".split()) == 1
    assert pr.word_errors("a b".split(), []) == 2
    assert pr.word_errors([], "x".split()) == 1


def test_candidate_sets_closed_open():
    row = clip("1", "tomato", "tomatoe", [cand("tomatoe", 0, 9, hyp=True, in_pool=False), cand("potato", 0, 20), cand("tomato", 1, 10, true=True),
                                          cand("banana", 2, 30), cand("apple", 3, 40)])
    sets = pr.candidate_sets(row, 2)
    assert [c["text"] for c in sets["closed"]] == ["potato", "tomato"]
    assert [c["text"] for c in sets["open"]] == ["potato", "banana"]
    row2 = clip("2", "apple", "x", [cand("x", None, 5, hyp=True, in_pool=False), cand("potato", 0, 20), cand("tomato", 1, 10), cand("apple", 3, 12, true=True)])
    assert [c["text"] for c in pr.candidate_sets(row2, 2)["closed"]] == ["potato", "tomato"]
    assert "apple" in [c["text"] for c in pr.candidate_sets(row2, 2)["closed+"]]


def test_decide_acoustic_prefers_likelier_phrase_and_rule_rejects_costly_swap():
    row = clip("1", "tomato", "tomatoe", [cand("tomatoe", None, 9, hyp=True, in_pool=False), cand("potato", 0, 20, dist=0.1),
                                          cand("tomato", 1, 12, true=True, dist=0.2)])
    cs = pr.candidate_sets(row, 2)["closed"]
    assert pr.decide(row, cs, ("text", None)) == "potato"
    assert pr.decide(row, cs, ("ac", 0.0, None)) == "tomato"
    assert pr.decide(row, cs, ("ac", 0.0, 5.0)) == "tomato"
    assert pr.decide(row, cs, ("ac", 0.0, 1.0)) == "tomatoe"
    assert pr.decide(row, cs, ("ac", 100.0, None)) == "potato"


def test_decide_without_candidates_returns_raw():
    row = clip("1", "a", "b", [])
    assert pr.decide(row, [], ("ac", 0.0, None)) == "b"


def test_report_runs_and_selects_on_other_speakers(tmp_path, capsys):
    def rows(spk, n):
        out = []
        for i in range(n):
            ref = f"phrase {spk} {i}"
            out.append(clip(str(i), ref, "garbled words", [cand("garbled words", None, 30, hyp=True, in_pool=False),
                                                              cand("other one", 0, 40), cand(ref, 1, 20, true=True)]))
        return out

    files = {s: write_table(tmp_path, f"{s}.json", rows(s, 6)) for s in ("M01", "M02", "M04")}
    spec = "toy=" + ",".join(f"{s}:{p}" for s, p in files.items())
    assert pr.main(["--system", spec, "--boot", "50", "--out", str(tmp_path / "out")]) == 0
    text = capsys.readouterr().out
    assert "acoustic lam=0, no rule" in text and "chosen on others" in text
    result = json.loads((tmp_path / "out" / "results.json").read_text())
    sel = result["systems"]["toy"]["selection"]["ac"]
    assert sel["by_coverage"]["1"]["chosen"][0] < sel["by_coverage"]["1"]["raw"][0]
    assert set(sel["picks"]) == {"M01", "M02", "M04"}
    with pytest.raises(FileExistsError):
        pr.main(["--system", spec, "--boot", "5", "--out", str(tmp_path / "out")])


def test_coverage(tmp_path):
    def put(spk, split, texts):
        d = tmp_path / spk
        d.mkdir(exist_ok=True)
        (d / f"{split}.jsonl").write_text("\n".join(json.dumps({"id": str(i), "text": t, "kind": "sentence"}) for i, t in enumerate(texts)) + "\n")

    put("M01", "test", ["a b c", "d e f", "g h i"])
    put("M01", "train", ["a b c"])
    put("M02", "test", ["d e f"])
    put("M02", "train", [])
    cov = pr.coverage(str(tmp_path), ["M01"], ["sentence"])
    assert cov["M01"]["test_clips"] == 3
    assert cov["M01"]["own_training_prompts"] == pytest.approx(100 / 3)
    assert cov["M01"]["other_speakers"] == pytest.approx(100 / 3)
    assert cov["M01"]["neither"] == pytest.approx(100 / 3)
