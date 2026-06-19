"""Tests for the FFC public API: FFCProfile, ffc_correction_images,
apply_ffc(dtype=...), and FFCCoverageWarning."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import pyphenix
from pyphenix import FFCCoverageWarning, FFCProfile, OperaPhenixReader
from pyphenix.errors import FFCCoverageWarning as FFCCoverageWarning_via_errors


def _make_profile(
    channel_id: int,
    *,
    mean: float = 1.0,
    coeff: float = 1.0,
    real: bool = True,
    profile_type: str = "Identity",
) -> FFCProfile:
    """Build an FFCProfile with a uniform polynomial of value `coeff`.

    When real=True, character is 'NonFlat' and the polynomial is a constant
    of value `coeff`, so has_correction() returns True. When real=False,
    character is 'Null' and the profile type is 'Identity' (i.e. nothing
    to correct), matching the most common shape produced by Harmony when
    no correction was computed for a channel.
    """
    if real:
        ffc_data = {
            "Character": "NonFlat",
            "Mean": mean,
            "Profile": {
                "Type": "Polynomial",
                # Only the n=0, k=0 term — uniform correction value = coeff
                "Coefficients": [[coeff]],
                "Dims": [10, 10],
                "Origin": [5.0, 5.0],
                "Scale": [1.0, 1.0],
            },
        }
    else:
        ffc_data = {
            "Character": "Null",
            "Mean": mean,
            "Profile": {"Type": profile_type},
        }
    return FFCProfile(channel_id, ffc_data)


def _make_reader_stub(ffc_profiles, image_size=(8, 8), channel_ids=None):
    """Build a minimal OperaPhenixReader instance for unit testing.

    Bypasses __init__ because we don't need to parse an XML on disk.
    """
    reader = OperaPhenixReader.__new__(OperaPhenixReader)
    reader.ffc_profiles = dict(ffc_profiles)
    if channel_ids is None:
        channel_ids = sorted(ffc_profiles.keys())
    reader.metadata = SimpleNamespace(
        image_size=image_size,
        channel_ids=list(channel_ids),
    )
    return reader


# ---------------------------------------------------------------------------
# Public-surface imports
# ---------------------------------------------------------------------------

def test_public_imports_top_level():
    assert pyphenix.FFCProfile is FFCProfile
    assert pyphenix.FFCCoverageWarning is FFCCoverageWarning
    assert pyphenix.OperaPhenixReader is OperaPhenixReader


def test_warning_class_also_importable_from_errors_module():
    assert FFCCoverageWarning_via_errors is FFCCoverageWarning


def test_ffc_coverage_warning_is_user_warning_subclass():
    assert issubclass(FFCCoverageWarning, UserWarning)


# ---------------------------------------------------------------------------
# Warning behaviour
# ---------------------------------------------------------------------------

def test_full_coverage_emits_no_warning():
    reader = _make_reader_stub({
        1: _make_profile(1),
        2: _make_profile(2),
    })
    data = np.full((1, 2, 1, 8, 8), 100, dtype=np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FFCCoverageWarning)
        reader.apply_ffc(data, [1, 2], verbose=False)
        reader.ffc_correction_images()


def test_no_profiles_is_silent_passthrough():
    reader = _make_reader_stub({})
    data = np.full((1, 2, 1, 8, 8), 100, dtype=np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FFCCoverageWarning)
        out = reader.apply_ffc(data, [1, 2], verbose=False)
        tiles = reader.ffc_correction_images(channel_ids=[1, 2])
    # Passthrough returns input unchanged (same object — current contract).
    assert out is data
    assert tiles == {}


def test_partial_coverage_fires_warning_with_channel_lists():
    reader = _make_reader_stub({
        1: _make_profile(1),
        2: _make_profile(2),
    })
    data = np.full((1, 4, 1, 8, 8), 100, dtype=np.uint16)
    with pytest.warns(FFCCoverageWarning) as caught:
        reader.apply_ffc(data, [1, 2, 3, 4], verbose=False)
    assert len(caught) == 1
    msg = str(caught[0].message)
    assert "[1, 2]" in msg
    assert "[3, 4]" in msg
    assert "channel 3: no profile" in msg
    assert "channel 4: no profile" in msg


def test_mixed_has_correction_fires_warning():
    reader = _make_reader_stub({
        1: _make_profile(1),
        2: _make_profile(2, real=False),  # Identity — has_correction()==False
    })
    data = np.full((1, 2, 1, 8, 8), 100, dtype=np.uint16)
    with pytest.warns(FFCCoverageWarning) as caught:
        reader.apply_ffc(data, [1, 2], verbose=False)
    msg = str(caught[0].message)
    assert "[1]" in msg
    assert "[2]" in msg
    assert "type=Identity" in msg


def test_warning_fires_once_per_call():
    reader = _make_reader_stub({1: _make_profile(1)})
    data = np.full((1, 3, 1, 8, 8), 100, dtype=np.uint16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader.apply_ffc(data, [1, 2, 3], verbose=False)
    ffc_warnings = [w for w in caught if issubclass(w.category, FFCCoverageWarning)]
    assert len(ffc_warnings) == 1


def test_ffc_correction_images_emits_warning_on_partial_coverage():
    reader = _make_reader_stub({1: _make_profile(1)})
    with pytest.warns(FFCCoverageWarning):
        reader.ffc_correction_images(channel_ids=[1, 2])


# ---------------------------------------------------------------------------
# ffc_correction_images contract
# ---------------------------------------------------------------------------

def test_ffc_correction_images_defaults_to_metadata():
    reader = _make_reader_stub(
        {1: _make_profile(1), 2: _make_profile(2)},
        image_size=(8, 8),
        channel_ids=[1, 2],
    )
    tiles = reader.ffc_correction_images()
    assert set(tiles.keys()) == {1, 2}
    assert tiles[1].shape == (8, 8)
    assert tiles[1].dtype == np.float32


def test_ffc_correction_images_respects_explicit_shape_and_channels():
    reader = _make_reader_stub({1: _make_profile(1), 2: _make_profile(2)})
    tiles = reader.ffc_correction_images(shape=(16, 32), channel_ids=[1])
    assert list(tiles.keys()) == [1]
    assert tiles[1].shape == (16, 32)


def test_ffc_correction_images_omits_channels_without_real_profile():
    reader = _make_reader_stub({
        1: _make_profile(1),
        2: _make_profile(2, real=False),
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FFCCoverageWarning)
        tiles = reader.ffc_correction_images(channel_ids=[1, 2, 3])
    assert set(tiles.keys()) == {1}


# ---------------------------------------------------------------------------
# apply_ffc dtype param
# ---------------------------------------------------------------------------

def test_apply_ffc_default_dtype_is_float32():
    reader = _make_reader_stub({1: _make_profile(1, coeff=1.0)})
    data = np.full((1, 1, 1, 8, 8), 100, dtype=np.uint16)
    out = reader.apply_ffc(data, [1], verbose=False)
    assert out.dtype == np.float32


def test_apply_ffc_float32_default_matches_legacy_behaviour():
    # The dtype="float32" path divides by illumination and returns float32 —
    # bytewise identical to a hand-rolled equivalent that does the same.
    reader = _make_reader_stub({1: _make_profile(1, coeff=2.0)})
    data = np.full((1, 1, 1, 4, 4), 100, dtype=np.uint16)
    out_default = reader.apply_ffc(data, [1], verbose=False)
    out_explicit = reader.apply_ffc(data, [1], verbose=False, dtype="float32")
    expected = (data.astype(np.float32) / 2.0)
    np.testing.assert_array_equal(out_default, expected)
    np.testing.assert_array_equal(out_explicit, expected)


def test_apply_ffc_uint16_round_trip_within_one_lsb():
    # Uniform-illumination fixture: correction = mean everywhere → divide
    # then multiply cancels exactly. Round-trip should preserve the input
    # (uint16) within ±1 LSB.
    for value in (0, 1, 100, 12345, 65535):
        reader = _make_reader_stub({1: _make_profile(1, coeff=2.5, mean=2.5)})
        data = np.full((1, 1, 1, 8, 8), value, dtype=np.uint16)
        out = reader.apply_ffc(data, [1], verbose=False, dtype="uint16")
        assert out.dtype == np.uint16
        diff = np.abs(out.astype(np.int32) - data.astype(np.int32))
        assert diff.max() <= 1, f"value={value}: max diff {diff.max()}"


def test_apply_ffc_uint16_clips_overflow():
    # If the data * mean / correction exceeds 65535, output must clip there.
    reader = _make_reader_stub({1: _make_profile(1, coeff=1.0, mean=100.0)})
    data = np.full((1, 1, 1, 4, 4), 1000, dtype=np.uint16)
    out = reader.apply_ffc(data, [1], verbose=False, dtype="uint16")
    assert out.dtype == np.uint16
    assert (out == 65535).all()


def test_apply_ffc_rejects_unknown_dtype():
    reader = _make_reader_stub({1: _make_profile(1)})
    data = np.full((1, 1, 1, 4, 4), 100, dtype=np.uint16)
    with pytest.raises(ValueError, match="dtype must be"):
        reader.apply_ffc(data, [1], verbose=False, dtype="int8")


def test_apply_ffc_apply_false_returns_input_unchanged():
    reader = _make_reader_stub({1: _make_profile(1)})
    data = np.full((1, 1, 1, 4, 4), 100, dtype=np.uint16)
    out = reader.apply_ffc(data, [1], apply=False, verbose=False)
    assert out is data
