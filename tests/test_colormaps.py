import pytest

from pyphenix._colormaps import CHANNEL_COLORS, DEFAULT_COLORS, channel_color


@pytest.mark.parametrize("name,expected", [
    ("DAPI", "cyan"),
    ("Hoechst", "cyan"),
    ("Brightfield", "gray"),
    ("Alexa 488", "green"),
    ("GFP", "green"),
    ("EGFP", "green"),
    ("Alexa 555", "yellow"),
    ("Alexa 568", "yellow"),
    ("Cy3", "yellow"),
    ("mCherry", "magenta"),
    ("mStrawberry", "magenta"),
    ("Alexa 647", "magenta"),
    ("Cy5", "magenta"),
])
def test_known_channel_names_map_correctly(name, expected):
    assert channel_color(name, idx=0) == expected


def test_known_channel_name_ignores_idx():
    assert channel_color("DAPI", idx=0) == "cyan"
    assert channel_color("DAPI", idx=5) == "cyan"


def test_unknown_name_falls_back_to_default_by_idx():
    assert channel_color("MyNovelDye", idx=0) == DEFAULT_COLORS[0]
    assert channel_color("MyNovelDye", idx=2) == DEFAULT_COLORS[2]


def test_unknown_name_wraps_idx_around_default_list():
    n = len(DEFAULT_COLORS)
    assert channel_color("MyNovelDye", idx=n) == DEFAULT_COLORS[0]
    assert channel_color("MyNovelDye", idx=n + 3) == DEFAULT_COLORS[3]


def test_case_insensitive_substring_matching():
    assert channel_color("dapi", idx=0) == "cyan"
    assert channel_color("Dapi Signal", idx=0) == "cyan"
    assert channel_color("Channel: ALEXA 488 nm", idx=0) == "green"


def test_substring_match_picks_first_listed_entry_on_ambiguity():
    # The dict is iterated in insertion order; the first key whose
    # lowercase form is a substring of the channel name wins. This is
    # the historical behavior we want to preserve.
    expected = next(
        color for key, color in CHANNEL_COLORS.items()
        if key.lower() in "dapi channel".lower()
    )
    assert channel_color("DAPI channel", idx=0) == expected


def test_returns_string():
    assert isinstance(channel_color("DAPI", idx=0), str)
    assert isinstance(channel_color("Unknown", idx=0), str)
