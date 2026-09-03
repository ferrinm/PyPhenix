import pytest

from pyphenix._colormaps import (
    CHANNEL_COLORS,
    COLORMAP_RGB,
    DEFAULT_COLORS,
    channel_color,
    channel_color_ome,
)


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


# ----------------------------------------------------------------------
# OME Channel/@Color encoding
# ----------------------------------------------------------------------

def test_every_returnable_colormap_has_an_rgb():
    """channel_color_ome indexes COLORMAP_RGB by name, so a colour added to
    CHANNEL_COLORS or DEFAULT_COLORS without an RGB entry would KeyError at
    save time rather than here."""
    returnable = set(CHANNEL_COLORS.values()) | set(DEFAULT_COLORS)
    assert returnable <= set(COLORMAP_RGB)


@pytest.mark.parametrize("name,idx,expected", [
    ("DAPI", 0, 0x00FFFFFF),                    # cyan
    ("Alexa 488", 0, 0x00FF00FF),               # green
    ("Alexa 555", 0, 0xFFFF00FF - (1 << 32)),   # yellow, past int32 max
    ("mCherry", 0, 0xFF00FFFF - (1 << 32)),     # magenta, past int32 max
    ("Brightfield", 0, 0xFFFFFFFF - (1 << 32)),  # gray -> white
    # An unknown name falls back to DEFAULT_COLORS by index, and the encoding
    # must follow that colour rather than the name.
    ("MyNovelDye", 0, 0x00FFFFFF),              # DEFAULT_COLORS[0], cyan
    ("MyNovelDye", 5, 0x0000FFFF),              # DEFAULT_COLORS[5], blue
])
def test_ome_color_encodes_rgba_big_endian(name, idx, expected):
    assert channel_color_ome(name, idx) == expected


def test_ome_color_fits_signed_int32():
    """Channel/@Color is an xsd:int; an out-of-range value is invalid OME."""
    for name in list(CHANNEL_COLORS) + ["MyNovelDye"]:
        for idx in range(len(DEFAULT_COLORS)):
            assert -(1 << 31) <= channel_color_ome(name, idx) < (1 << 31)
