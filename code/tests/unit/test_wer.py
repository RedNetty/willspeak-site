from collections import Counter

import pytest

from willspeak.evaluate import DEL, INS, OK, SUB, align, evaluate
from willspeak.normalize import normalize


def counts(ref: str, hyp: str) -> Counter:
    return Counter(op for op, _, _ in align(ref.split(), hyp.split()))


def utt_wer(ref: str, hyp: str) -> float:
    c = counts(ref, hyp)
    n_ref = c[OK] + c[SUB] + c[DEL]
    err = c[SUB] + c[DEL] + c[INS]
    return err / n_ref if n_ref else 0.0


def test_identical_is_all_ok():
    c = counts("the cat sat on the mat", "the cat sat on the mat")
    assert c[OK] == 6
    assert c[SUB] == c[DEL] == c[INS] == 0


@pytest.mark.parametrize("hyp, expected", [
    ("the dog sat", (1, 0, 0)),
    ("the sat", (0, 1, 0)),
    ("the big cat sat", (0, 0, 1)),
])
def test_one_edit(hyp, expected):
    c = counts("the cat sat", hyp)
    assert (c[SUB], c[DEL], c[INS]) == expected
    assert utt_wer("the cat sat", hyp) == pytest.approx(1 / 3)


def test_empty_hypothesis_is_all_deletions():
    c = counts("the cat sat", "")
    assert c[DEL] == 3
    assert utt_wer("the cat sat", "") == pytest.approx(1.0)


def test_wer_can_exceed_one():
    assert utt_wer("yes", "yes yes yes yes") == pytest.approx(3.0)


def test_alignment_conserves_tokens():
    for ref, hyp in [
        ("the cat sat on the mat", "a cat sat on mat now"),
        ("i need help", ""),
        ("", "i need help"),
        ("one two three four", "one three two four five"),
    ]:
        r, h = ref.split(), hyp.split()
        c = counts(ref, hyp)
        assert c[OK] + c[SUB] + c[DEL] == len(r)
        assert c[OK] + c[SUB] + c[INS] == len(h)


def test_corpus_wer_is_sum_then_divide():
    pairs = [
        {"ref": "the cat sat", "hyp": "the dog sat"},
        {"ref": "yes", "hyp": "yes yes yes yes"},
    ]
    res = evaluate(pairs)
    assert res.wer == pytest.approx(4 / 4)

    mean_of_rates = (1 / 3 + 3 / 1) / 2
    assert res.wer != pytest.approx(mean_of_rates)


@pytest.mark.parametrize("hyp, expected", [("", 0), ("hello", 1), ("hello there", 2)])
def test_silent_reference_counts_hallucinated_words(hyp, expected):
    result = evaluate([{"ref": "", "hyp": hyp}])
    assert result.wer == expected
    assert result.word["ins"] == expected
    assert result.cer == len(hyp.replace(" ", ""))


def test_group_breakout():
    pairs = [
        {"ref": "the cat sat", "hyp": "the cat sat", "group": "good-day"},
        {"ref": "the cat sat", "hyp": "the dog mat", "group": "bad-day"},
    ]
    res = evaluate(pairs)
    assert res.by_group["good-day"]["rate"] == pytest.approx(0.0, abs=1e-4)
    assert res.by_group["bad-day"]["rate"] == pytest.approx(2 / 3, abs=1e-4)


@pytest.mark.parametrize("raw,expected", [
    ("I'm hungry.", "i am hungry"),
    ("Im hungry", "i am hungry"),
    ("Don't!", "do not"),
    ("  Turn   me  over  ", "turn me over"),
    ("It costs 8 dollars", "it costs eight dollars"),
    ("Room 21", "room twenty one"),
    ("that’s it", "that is it"),
    ("Those musicians harmonize marvellously.", "those musicians harmonize marvelously"),
    ("The colour grey", "the color gray"),
])
def test_normalizer_cases(raw, expected):
    assert normalize(raw) == expected


def test_split_keys_ignore_the_spelling_table():
    from willspeak.torgo import split_key
    assert split_key("The colour grey") == "the colour grey"
    assert normalize("The colour grey", us_spelling=False) == "the colour grey"


def test_normalizer_makes_punctuation_free():
    assert normalize("Yes, please -- now!") == normalize("yes please now")


def test_evaluate_normalizes_both_sides():
    assert evaluate([{"ref": "I'm hungry", "hyp": "im hungry"}]).wer == pytest.approx(0.0)


BATTERY = [
    ("the cat sat on the mat", "the cat sat on the mat"),
    ("the cat sat on the mat", "the cat sat on a mat"),
    ("the cat sat on the mat", "cat sat mat"),
    ("the cat sat on the mat", "the the cat sat on the the mat"),
    ("yes", "yes yes yes yes"),
    ("i need help please", ""),
    ("turn me over", "turn me over now please"),
    ("one two three four five", "one three two five four"),
]


def test_agrees_with_jiwer():
    jiwer = pytest.importorskip("jiwer", reason="pip install -r requirements-dev.txt")

    refs = [normalize(r) for r, _ in BATTERY]
    hyps = [normalize(h) for _, h in BATTERY]

    ours = evaluate([{"ref": r, "hyp": h} for r, h in zip(refs, hyps)]).wer
    theirs = jiwer.wer(refs, hyps)
    assert ours == pytest.approx(theirs, abs=1e-4)


@pytest.mark.parametrize("ref,hyp", BATTERY)
def test_agrees_with_jiwer_per_utterance(ref, hyp):
    jiwer = pytest.importorskip("jiwer", reason="pip install -r requirements-dev.txt")
    r, h = normalize(ref), normalize(hyp)
    if not r:
        pytest.skip("jiwer rejects an empty reference")
    assert utt_wer(r, h) == pytest.approx(jiwer.wer(r, h), abs=1e-4)
