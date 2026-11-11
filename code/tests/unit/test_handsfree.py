import time

import pytest

from willspeak.handsfree import (
    DEFAULTS,
    Settings,
    contains_phrase,
    decide,
    detect_phrase,
    norm_text,
    similarity,
)

STOP = "purple elephant"


@pytest.fixture
def s():
    return dict(DEFAULTS)


def match(label, score, margin=0.2, own_takes=3, ok=True):
    return {"ok": ok, "margin": margin,
            "candidates": [{"text": label, "score": score, "own_takes": own_takes}]}


@pytest.mark.parametrize("raw,expected", [
    ("Purple Elephant!", "purple elephant"),
    ("  purple   elephant  ", "purple elephant"),
    ("it’s fine", "it's fine"),
    ("", ""),
])
def test_norm_text(raw, expected):
    assert norm_text(raw) == expected


def test_similarity_bounds():
    assert similarity("purple elephant", "purple elephant") == pytest.approx(1.0)
    assert similarity("", "anything") == 0.0
    assert 0.0 <= similarity("purple elephant", "people elegant") < 1.0


def test_phrase_found_exactly_or_embedded(s):
    assert contains_phrase("purple elephant", STOP, s["stop_text_sim"])
    assert contains_phrase("um purple elephant", STOP, s["stop_text_sim"])
    assert contains_phrase("no wait purple elephant please", STOP, s["stop_text_sim"])


def test_garbled_phrase_respects_configured_threshold(s):
    assert not contains_phrase("purpul elefant", STOP, s["stop_text_sim"])
    assert contains_phrase("purpul elefant", STOP, 0.70)


def test_zero_margin_disables_margin_detector(s):
    s["stop_match_margin"] = 0
    assert detect_phrase("mmph", match(STOP, 0.2, margin=0.1), STOP, s, 0.85) is None
    assert detect_phrase("mmph", match(STOP, 0.8, margin=0), STOP, s, 0.85) == "matcher"


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "6"])
def test_invalid_settings_leave_memory_and_disk_unchanged(tmp_path, value):
    st = Settings(tmp_path)
    st.update({"speak_delay_s": 1})
    before = st.path.read_bytes()
    with pytest.raises(ValueError):
        st.update({"stop_phrase": "green tractor", "speak_delay_s": value})
    assert st.path.read_bytes() == before
    assert st.get()["stop_phrase"] == DEFAULTS["stop_phrase"]


def test_zero_retraction_window_is_disabled(s, monkeypatch):
    monkeypatch.setattr(time, "time", lambda: 123)
    s["retract_window_s"] = 0
    assert not decide(STOP, None, s, 123)["retract"]


def test_ordinary_speech_is_not_the_stop_phrase(s):
    for text in ["can i have some water", "i need help please", "turn me over", ""]:
        assert not contains_phrase(text, STOP, s["stop_text_sim"])


def test_detector_fires_on_transcript(s):
    assert detect_phrase("purple elephant", None, STOP, s, s["stop_text_sim"]) == "text"


def test_detector_fires_on_matcher_score(s):
    m = match(STOP, score=0.62, margin=0.01)
    assert detect_phrase("mmph", m, STOP, s, s["stop_text_sim"]) == "matcher"


def test_detector_fires_on_matcher_margin(s):
    m = match(STOP, score=0.30, margin=0.15)
    assert detect_phrase("mmph", m, STOP, s, s["stop_text_sim"]) == "matcher"


def test_detector_ignores_unrecorded_labels(s):
    m = match(STOP, score=0.95, margin=0.5, own_takes=0)
    assert detect_phrase("mmph", m, STOP, s, s["stop_text_sim"]) is None


def test_detector_ignores_weak_matches(s):
    m = match(STOP, score=0.20, margin=0.01)
    assert detect_phrase("mmph", m, STOP, s, s["stop_text_sim"]) is None


def test_detector_ignores_a_different_top_label(s):
    m = match("i need help", score=0.99, margin=0.5)
    assert detect_phrase("mmph", m, STOP, s, s["stop_text_sim"]) is None


def test_empty_phrase_never_fires(s):
    assert detect_phrase("anything at all", None, "", s, s["stop_text_sim"]) is None


def test_speaks_the_transcript(s):
    d = decide("i need help", None, s, None)
    assert d["action"] == "speak"
    assert d["text"] == "i need help"


def test_stop_phrase_with_nothing_recently_spoken(s):
    d = decide("purple elephant", None, s, None)
    assert d["action"] == "stop"
    assert d["retract"] is False


def test_stop_phrase_inside_the_retract_window(s):
    d = decide("purple elephant", None, s, time.time() - 1.0)
    assert d["action"] == "stop"
    assert d["retract"] is True
    assert d["retract_text"] == s["retract_text"]


@pytest.mark.parametrize("now, expected", [(100, True), (101, True), (106, True), (106.01, False), (99, False)])
def test_retraction_window_boundaries(s, now, expected):
    s["retract_window_s"] = 6
    assert decide(STOP, None, s, 100, now=now)["retract"] is expected


def test_single_label_cannot_trigger_stop_by_margin(s):
    from willspeak.matcher.scoring import rank_candidates

    cands, _, margin = rank_candidates({norm_text(STOP): {"score": 0.2, "own_takes": 1}})
    assert decide("ordinary sentence", {"ok": True, "candidates": cands, "margin": margin}, s, None)["action"] == "speak"


def test_sleep_and_wake(s):
    assert decide(s["sleep_phrase"], None, s, None)["action"] == "sleep"
    assert decide(s["wake_phrase"], None, s, None, awake=False)["action"] == "wake"


@pytest.mark.parametrize("text", [
    "i want to go to sleep", "can i go to sleep now", "go to sleep",
    "mom i am going to sleep", "boys go to sleep",
])
def test_everyday_sleep_talk_is_spoken_not_obeyed(s, text):
    assert decide(text, None, s, None)["action"] == "speak"


@pytest.mark.parametrize("text", ["did you wake up", "time to wake up", "wake up", "boys wake up"])
def test_other_people_cannot_wake_the_device_with_everyday_words(s, text):
    assert decide(text, None, s, None, awake=False)["action"] == "ignore"


@pytest.mark.parametrize("text", ["voys go to sleep", "noise go to sleep", "voice goto sleep", "voice go to sleep now"])
def test_garbled_or_merged_sleep_phrase_still_counts(s, text):
    assert decide(text, None, s, None)["action"] == "sleep"


def test_split_and_merged_tokens_still_match_the_stop_phrase(s):
    assert contains_phrase("purple ele phant", STOP, s["stop_text_sim"])
    assert contains_phrase("um purpleelephant", STOP, s["stop_text_sim"])


def test_a_dropped_phrase_word_needs_the_rest_near_exact():
    assert not contains_phrase("go to sleep", "voice go to sleep", 0.70)
    assert contains_phrase("hither", "hi there", 0.70)


def test_asleep_ignores_everything_else(s):
    for text in ["i need help", STOP, "hello there"]:
        d = decide(text, None, s, None, awake=False)
        assert d["action"] == "ignore"
        assert d["asleep"] is True


def test_nothing_heard(s):
    d = decide("   ", None, s, None)
    assert d["action"] == "ignore"
    assert d["reason"] == "nothing heard"


def test_vocab_override_is_off_by_default(s):
    m = match("i need help", score=0.99, margin=0.9)
    d = decide("i need kelp", m, s, None)
    assert d["action"] == "speak"
    assert d["text"] == "i need kelp"
    assert d["override"] is None


def test_vocab_override_when_enabled_and_very_sure(s):
    s["vocab_override"] = True
    m = match("i need help", score=0.99, margin=0.9)
    d = decide("i need kelp", m, s, None)
    assert d["action"] == "speak"
    assert d["text"] == "i need help"
    assert d["override"] == "i need help"


def test_vocab_override_respects_its_thresholds(s):
    s["vocab_override"] = True
    m = match("i need help", score=0.60, margin=0.9)
    assert decide("i need kelp", m, s, None)["override"] is None


def test_vocab_override_never_speaks_the_stop_phrase(s):
    s["vocab_override"] = True
    m = match(norm_text(STOP), score=0.99, margin=0.9)
    d = decide("something else entirely", m, s, None)
    assert d.get("override") is None


def test_speak_carries_the_preview_delay(s):
    s["speak_delay_s"] = 1.5
    assert decide("hello", None, s, None)["delay_s"] == pytest.approx(1.5)


def test_settings_round_trip(tmp_path):
    st = Settings(tmp_path)
    st.update({"stop_phrase": "green tractor", "speak_delay_s": "2"})
    assert Settings(tmp_path).get()["stop_phrase"] == "green tractor"
    assert Settings(tmp_path).get()["speak_delay_s"] == pytest.approx(2.0)


def test_settings_rejects_unknown_keys(tmp_path):
    st = Settings(tmp_path)
    st.update({"definitely_not_a_setting": 1})
    assert "definitely_not_a_setting" not in st.get()
