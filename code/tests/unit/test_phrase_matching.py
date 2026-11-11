import pytest

from willspeak.handsfree import contains_phrase, similarity

STOP = "purple elephant"
SIM = 0.70


def test_similarity_bounds():
    assert similarity("purple elephant", "purple elephant") == pytest.approx(1.0)
    assert similarity("", "anything") == 0.0
    assert 0.0 <= similarity("purple elephant", "people elegant") < 1.0


def test_phrase_found_exactly():
    assert contains_phrase("purple elephant", STOP, SIM)


def test_phrase_found_when_embedded():
    assert contains_phrase("um purple elephant", STOP, SIM)
    assert contains_phrase("no wait purple elephant please", STOP, SIM)


def test_phrase_found_when_garbled():
    assert contains_phrase("purpul elefant", STOP, SIM)


def test_ordinary_speech_is_not_the_stop_phrase():
    for text in ["can i have some water", "i need help please", "turn me over", ""]:
        assert not contains_phrase(text, STOP, SIM)


def test_phrase_mid_word_substring():
    assert not contains_phrase("concatenate", "cat", 0.99)
    assert not contains_phrase("unstoppable", "stop", 0.85)
    assert contains_phrase("please, stop!", "stop", 0.99)


def test_text_shorter_than_phrase_true_case():
    assert contains_phrase("hither", "hi there", 0.7)


def test_text_shorter_than_phrase_false_case():
    assert not contains_phrase("yo", "hello there friend", 0.7)


def test_empty_inputs():
    assert not contains_phrase("", STOP, SIM)
    assert not contains_phrase(STOP, "", SIM)
