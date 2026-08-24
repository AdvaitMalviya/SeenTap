"""Ten verbs, fuzzy matched. A miss must produce nothing, never the wrong verb."""
import pytest

from seentap import config, parse


def test_the_vocabulary_is_the_ten_verbs_from_the_report():
    assert config.VOCAB == [
        "click", "double click", "right click", "select",
        "scroll up", "scroll down", "drag", "drop", "cancel", "recalibrate",
    ]


@pytest.mark.parametrize("verb", config.VOCAB)
def test_every_verb_matches_itself(verb):
    assert parse.parse(verb)[0] == verb


@pytest.mark.parametrize("said,expect", [
    ("Click.", "click"),
    ("  CLICK  ", "click"),
    ("duble click", "double click"),
    ("scrol down", "scroll down"),
    ("recalibrat", "recalibrate"),
])
def test_misrecognitions_still_land_on_the_right_verb(said, expect):
    assert parse.parse(said)[0] == expect


@pytest.mark.parametrize("said,expect", [
    ("double click", "double click"),
    ("right click", "right click"),
    ("click", "click"),
])
def test_longer_verbs_are_not_swallowed_by_their_prefix(said, expect):
    """'double click' must not degrade into 'click'."""
    assert parse.parse(said)[0] == expect


@pytest.mark.parametrize("said", ["banana", "what time is it", "", "   "])
def test_out_of_vocabulary_speech_is_refused(said):
    verb, score = parse.parse(said)
    assert verb is None, "a misrecognition must do nothing, not something wrong"


def test_threshold_is_honoured():
    assert parse.parse("click", threshold=101)[0] is None


# --- C2 baseline: numbered tiles -------------------------------------------

@pytest.mark.parametrize("said,tile,verb", [
    ("seven click", 7, "click"),
    ("3 double click", 3, "double click"),
    ("tile eleven select", 11, "select"),
    ("twelve scroll up", 12, "scroll up"),
])
def test_numbered_tile_commands_split_into_tile_and_verb(said, tile, verb):
    t, v, _ = parse.parse_numbered(said, n_tiles=12)
    assert (t, v) == (tile, verb)


def test_numbered_rejects_a_tile_outside_the_grid():
    assert parse.parse_numbered("ninety click", n_tiles=12)[0] is None


def test_numbered_rejects_a_missing_verb():
    assert parse.parse_numbered("seven", n_tiles=12)[1] is None
