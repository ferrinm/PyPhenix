import math
import random
import re
import threading
from collections import OrderedDict, UserList
import numpy as np
import pytest
from pathlib import Path
import xml.etree.ElementTree as ET

from pyphenix import (
    napari_get_reader,
    OperaPhenixReader,
    UnsupportedHarmonyVersionError,
)
from pyphenix._reader import HARMONY_NAMESPACES, LazyImageArray


def _write_mock_experiment(tmp_path, harmony_version):
    """Create a minimal mock Opera Phenix experiment directory structure."""
    # Create directory structure
    images_dir = tmp_path / "Images"
    images_dir.mkdir()

    # Create XML with structure matching real Opera Phenix data.
    # The namespace distinguishes Harmony versions.
    ns = HARMONY_NAMESPACES[harmony_version]

    root = ET.Element("EvaluationInputData")
    root.set("xmlns", ns)
    root.set("xmlns:xsd", "http://www.w3.org/2001/XMLSchema")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("Version", "2")
    
    # Add basic info
    user = ET.SubElement(root, "User")
    user.text = "TEST USER"
    
    instrument = ET.SubElement(root, "InstrumentType")
    instrument.text = "Phenix"
    
    # Add Plates section
    plates = ET.SubElement(root, "Plates")
    plate = ET.SubElement(plates, "Plate")
    
    plate_id = ET.SubElement(plate, "PlateID")
    plate_id.text = "TEST001"
    
    plate_rows = ET.SubElement(plate, "PlateRows")
    plate_rows.text = "2"
    
    plate_cols = ET.SubElement(plate, "PlateColumns")
    plate_cols.text = "2"
    
    # Add wells in Plates section
    well_plate = ET.SubElement(plate, "Well", id="0101")
    
    # Add Wells section
    wells = ET.SubElement(root, "Wells")
    well = ET.SubElement(wells, "Well")
    
    well_id = ET.SubElement(well, "id")
    well_id.text = "0101"
    
    well_row = ET.SubElement(well, "Row")
    well_row.text = "1"
    
    well_col = ET.SubElement(well, "Col")
    well_col.text = "1"
    
    # Add image references
    image_ref = ET.SubElement(well, "Image", id="0101K1F1P1R1")
    
    # Add Maps section with channel metadata
    maps = ET.SubElement(root, "Maps")
    map_elem = ET.SubElement(maps, "Map")
    
    entry = ET.SubElement(map_elem, "Entry", ChannelID="1")
    
    ch_name = ET.SubElement(entry, "ChannelName")
    ch_name.text = "DAPI"
    
    img_type = ET.SubElement(entry, "ImageType")
    img_type.text = "Signal"
    
    img_res_x = ET.SubElement(entry, "ImageResolutionX", Unit="m")
    img_res_x.text = "2.96688132474701E-07"
    
    img_res_y = ET.SubElement(entry, "ImageResolutionY", Unit="m")
    img_res_y.text = "2.96688132474701E-07"
    
    img_size_x = ET.SubElement(entry, "ImageSizeX")
    img_size_x.text = "100"
    
    img_size_y = ET.SubElement(entry, "ImageSizeY")
    img_size_y.text = "100"
    
    exc_wave = ET.SubElement(entry, "MainExcitationWavelength", Unit="nm")
    exc_wave.text = "375"
    
    em_wave = ET.SubElement(entry, "MainEmissionWavelength", Unit="nm")
    em_wave.text = "456"
    
    obj_mag = ET.SubElement(entry, "ObjectiveMagnification", Unit="")
    obj_mag.text = "40"
    
    obj_na = ET.SubElement(entry, "ObjectiveNA", Unit="")
    obj_na.text = "1.1"
    
    exp_time = ET.SubElement(entry, "ExposureTime", Unit="s")
    exp_time.text = "0.1"
    
    # Add Images section with complete metadata
    images = ET.SubElement(root, "Images")
    image = ET.SubElement(images, "Image", Version="1")
    
    img_id = ET.SubElement(image, "id")
    img_id.text = "0101K1F1P1R1"
    
    state = ET.SubElement(image, "State")
    state.text = "Ok"
    
    url = ET.SubElement(image, "URL")
    url.text = "r01c01f01p01-ch1sk1fk1fl1.tiff"
    
    img_row = ET.SubElement(image, "Row")
    img_row.text = "1"
    
    img_col = ET.SubElement(image, "Col")
    img_col.text = "1"
    
    field_id = ET.SubElement(image, "FieldID")
    field_id.text = "1"
    
    plane_id = ET.SubElement(image, "PlaneID")
    plane_id.text = "1"
    
    timepoint_id = ET.SubElement(image, "TimepointID")
    timepoint_id.text = "1"
    
    channel_id = ET.SubElement(image, "ChannelID")
    channel_id.text = "1"
    
    # Add position information (THIS WAS MISSING!)
    pos_x = ET.SubElement(image, "PositionX", Unit="m")
    pos_x.text = "0.0003204"
    
    pos_y = ET.SubElement(image, "PositionY", Unit="m")
    pos_y.text = "0.0003204"
    
    pos_z = ET.SubElement(image, "PositionZ", Unit="m")
    pos_z.text = "-2E-06"
    
    abs_pos_z = ET.SubElement(image, "AbsPositionZ", Unit="m")
    abs_pos_z.text = "0.135366693"
    
    # Write XML
    tree = ET.ElementTree(root)
    index_path = images_dir / "Index.xml"
    tree.write(index_path, encoding='utf-8', xml_declaration=True)
    
    # Create test image file
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    from PIL import Image
    img = Image.fromarray(test_image)
    img.save(images_dir / "r01c01f01p01-ch1sk1fk1fl1.tiff")

    return tmp_path


@pytest.fixture
def mock_phenix_experiment(tmp_path):
    """A mock experiment for tests that don't care about the Harmony version."""
    return _write_mock_experiment(tmp_path, "HarmonyV7")


def test_get_reader_valid_directory(mock_phenix_experiment):
    """Test that reader is returned for valid Phenix directory."""
    reader = napari_get_reader(str(mock_phenix_experiment))
    assert reader is not None
    assert callable(reader)


def test_get_reader_invalid_directory(tmp_path):
    """Test that reader returns None for invalid directory."""
    reader = napari_get_reader(str(tmp_path))
    assert reader is None


def test_get_reader_file_path(tmp_path):
    """Test that reader returns None for file paths."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    reader = napari_get_reader(str(test_file))
    assert reader is None


def test_reader_returns_layer_data(mock_phenix_experiment):
    """Test that reader returns proper layer data structure."""
    reader = napari_get_reader(str(mock_phenix_experiment))
    layer_data_list = reader(str(mock_phenix_experiment))
    
    assert isinstance(layer_data_list, list)
    assert len(layer_data_list) > 0
    
    # Check first layer
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple)
    assert len(layer_data_tuple) == 3  # (data, metadata, layer_type)
    
    data, metadata, layer_type = layer_data_tuple
    assert isinstance(data, np.ndarray)
    assert isinstance(metadata, dict)
    assert layer_type == 'image'


def test_reader_metadata_structure(mock_phenix_experiment):
    """Test that reader returns proper metadata structure."""
    reader = napari_get_reader(str(mock_phenix_experiment))
    layer_data_list = reader(str(mock_phenix_experiment))
    
    data, metadata, layer_type = layer_data_list[0]
    
    # Check required metadata fields
    assert 'name' in metadata
    assert 'colormap' in metadata
    assert 'blending' in metadata
    assert 'scale' in metadata
    assert 'contrast_limits' in metadata


def test_opera_phenix_reader_initialization(mock_phenix_experiment):
    """Test OperaPhenixReader initialization."""
    reader = OperaPhenixReader(str(mock_phenix_experiment))
    
    assert reader.experiment_path == Path(mock_phenix_experiment)
    assert reader.metadata is not None
    assert hasattr(reader.metadata, 'plate_id')
    assert hasattr(reader.metadata, 'channels')


def test_opera_phenix_reader_missing_directory():
    """Test that reader raises error for missing directory."""
    with pytest.raises(FileNotFoundError):
        OperaPhenixReader("/nonexistent/path")


def test_reader_handles_list_input(mock_phenix_experiment):
    """Test that reader handles list of paths."""
    reader = napari_get_reader([str(mock_phenix_experiment)])
    assert reader is not None
    assert callable(reader)


@pytest.mark.parametrize("harmony_version", sorted(HARMONY_NAMESPACES))
def test_reader_reports_detected_harmony_version(tmp_path, harmony_version):
    """Verify namespace read from the file"""
    experiment_path = _write_mock_experiment(tmp_path, harmony_version)
    reader = OperaPhenixReader(str(experiment_path), verbose=False)

    assert reader.harmony_version == harmony_version
    assert reader.ns == {"ns": HARMONY_NAMESPACES[harmony_version]}
    # Parsing actually resolved through that namespace.
    assert reader.metadata.plate_id == "TEST001"
    assert reader.metadata.channels[1]["name"] == "DAPI"


def test_unsupported_namespace_raises(tmp_path):
    """An unknown namespace fails loudly instead of parsing to nothing."""
    images_dir = tmp_path / "Images"
    images_dir.mkdir()
    root = ET.Element("EvaluationInputData")
    root.set("xmlns", "http://www.perkinelmer.com/PEHH/HarmonyV99")
    ET.ElementTree(root).write(
        images_dir / "Index.xml", encoding="utf-8", xml_declaration=True
    )

    with pytest.raises(UnsupportedHarmonyVersionError, match="HarmonyV99"):
        OperaPhenixReader(str(tmp_path), verbose=False)


def test_unnamespaced_index_raises(tmp_path):
    """An index XML with no namespace at all is also rejected."""
    images_dir = tmp_path / "Images"
    images_dir.mkdir()
    ET.ElementTree(ET.Element("EvaluationInputData")).write(
        images_dir / "Index.xml", encoding="utf-8", xml_declaration=True
    )

    with pytest.raises(UnsupportedHarmonyVersionError, match="no recognised"):
        OperaPhenixReader(str(tmp_path), verbose=False)
@pytest.mark.parametrize(
    "url, expected",
    [
        (r"r01c01\f.tiff", "r01c01/f.tiff"),
        ("r01c01f01p01-ch1sk1fk1fl1.tiff", "r01c01f01p01-ch1sk1fk1fl1.tiff"),
        ("r01c01/f.tiff", "r01c01/f.tiff"),
    ],
    ids=["backslash", "flat", "forward-slash"],
)
def test_construct_image_path_normalises_separators(url, expected):
    """Harmony writes <URL> with Windows separators; POSIX would otherwise read
    the backslash as part of the filename and no image would resolve."""
    reader = OperaPhenixReader.__new__(OperaPhenixReader)  # skip __init__
    reader.images_path = Path("/base/Images")
    reader.structure_type = "export"

    assert reader._construct_image_path(url, 1, 1) == Path("/base/Images") / expected


# ---------------------------------------------------------------------------
# LazyImageArray indexing
#
# The wrapper stands in for a numpy array on the way to napari and the Save
# path, so it has to index like one. Rather than restate numpy's rules,
# these tests compare against an eager array built from the same images.
# ---------------------------------------------------------------------------

LAZY_SHAPE = (3, 2, 4, 6, 5)  # (T, C, Z, Y, X)


def _lazy_and_eager(tmp_path, shape=LAZY_SHAPE):
    """A LazyImageArray over real files, and the equivalent eager array."""
    from PIL import Image as PILImage

    image_shape = shape[-2:]
    expected = np.zeros(shape, dtype=np.uint16)
    image_paths = {}

    for n, tcz in enumerate(np.ndindex(*shape[:-2])):
        pixels = (
            np.arange(math.prod(image_shape), dtype=np.uint16).reshape(image_shape)
            + n * 1000
        )
        url = "p" + "_".join(str(i) for i in tcz) + ".tiff"
        PILImage.fromarray(pixels).save(tmp_path / url)
        image_paths[tcz] = {"url": url, "row": 1, "col": 1}
        expected[tcz] = pixels

    lazy = LazyImageArray(
        shape=shape,
        dtype=np.uint16,
        image_paths=image_paths,
        images_path=tmp_path,
        construct_path_func=lambda url, row, col: tmp_path / url,
    )
    return lazy, expected


@pytest.fixture
def lazy_array(tmp_path):
    return _lazy_and_eager(tmp_path)


def test_lazy_array_exposes_the_layer_data_protocol(lazy_array):
    """napari's LayerDataProtocol (enforced for multiscale data) wants
    shape, ndim, dtype and size."""
    lazy, expected = lazy_array

    assert lazy.shape == expected.shape
    assert lazy.ndim == expected.ndim
    assert lazy.size == expected.size
    assert len(lazy) == len(expected)
    assert lazy.dtype == expected.dtype
    assert isinstance(lazy.dtype, np.dtype)



# The keys below are the ones that record a decision: what drops an axis,
# what adds one, and which read path a key lands on. Breadth is left to
# test_lazy_array_indexing_matches_numpy_for_random_keys.
@pytest.mark.parametrize(
    "key",
    [
        (1, 1, 2),                                  # one Field image
        (slice(None), 1),                           # one channel over time
        (1, 1, 2, slice(1, 4), slice(None)),        # a cropped image
        (1, 1, 2, 3, 4),                            # every axis: a scalar
        (1, 1, 2, Ellipsis, 3, 4),                  # ... : a 0-d array
        (slice(0, 0), 1, 2),                        # an empty result
        (None, 1, 1, 2),                            # None adds an axis
        (True, 1, 1, 2),                            # so does a scalar mask
        ([0, 2], 1, 2),                             # chosen timepoints
        (slice(None), [0, 1], 0),                   # chosen channels
        (np.array([True, False, True]), 0, 0),      # a mask over T
        (UserList([[True, False], [True, True], [False, False]]),),
        (0, 0, 0, [0, 2], 1),                       # index array in (Y, X)
        (np.ones(LAZY_SHAPE[:4], dtype=bool),),     # mask spanning the split
        ([0, 2], 1, 2, [0, 2], slice(None)),        # broadcast across (Y, X)
        (slice(None), slice(None), slice(None), slice(None), [1, 0]),
        (slice(None), [1, 0], slice(None), slice(None), 1),
    ],
    ids=[
        "one-field-image", "channel-over-time", "cropped-image",
        "every-axis-scalar", "ellipsis-gives-a-0d-array", "empty-result",
        "newaxis", "scalar-mask", "timepoint-list", "channel-list",
        "mask-over-t", "userlist-mask", "index-array-in-yx",
        "mask-spanning-the-split", "broadcast-across-yx",
        "yx-list-after-slices", "yx-integer-joins-the-broadcast",
    ],
)
def test_lazy_array_indexing_matches_numpy(lazy_array, key):
    """Integers drop their axis, slices and index arrays keep it, ``None``
    adds one, index arrays broadcast. Type is compared too: every axis
    indexed gives a numpy scalar, and the same key with an ``...`` in it a
    0-d array."""
    lazy, expected = lazy_array

    result = lazy[key]

    assert result.shape == expected[key].shape
    assert isinstance(result, type(expected[key]))
    assert np.array_equal(result, expected[key])


def _index_entries(shape):
    """Index entries to draw random keys from, for an array of ``shape``."""
    return [
        0, 1, -1, np.int64(0), True, False, None, Ellipsis,
        slice(None), slice(0, 2), slice(1, None, 2), slice(None, None, -1),
        slice(0, 0), [0, 1], [1, 0], [], range(2),
        np.array(0), np.array([0, 1]), np.array([[0, 1], [1, 0]]),
        np.ones(shape[0], dtype=bool),      # a mask over one file axis
        np.ones(shape[:2], dtype=bool),     # over two, spanning the split
        np.ones(shape, dtype=bool),         # over every axis
        np.ones(shape[-2:], dtype=bool),    # over (Y, X)
        1.5, "a", [[0, 1], [2]], object(),  # keys numpy rejects
    ]


def _key_repr(key):
    """A key, short enough to read in a failure message."""
    def one(entry):
        if isinstance(entry, np.ndarray):
            return f"{entry.dtype}{list(entry.shape)}"
        return repr(entry)
    return "(" + ", ".join(one(e) for e in key) + ")"


@pytest.mark.parametrize("shape", [LAZY_SHAPE, (4, 6, 5)], ids=["tczyx", "zyx"])
def test_lazy_array_indexing_matches_numpy_for_random_keys(tmp_path, shape):
    """Generated keys, compared down to the text of numpy's own exception.

    The named cases above cover the decisions; these cover the combinations
    nobody thinks to write. numpy's 0-d answer to a zero-width ``...`` was
    found here, by a suite every named case passed."""
    lazy, expected = _lazy_and_eager(tmp_path, shape)
    entries = _index_entries(shape)
    rng = random.Random(0)

    for _ in range(1000):
        key = tuple(
            rng.choice(entries) for _ in range(rng.randint(1, len(shape)))
        )
        described = _key_repr(key)
        try:
            want = expected[key]
        except Exception as numpy_error:
            # repr carries the type and the message, and both must match.
            raised = None
            try:
                lazy[key]
            except Exception as wrapper_error:
                raised = wrapper_error
            assert repr(raised) == repr(numpy_error), described
            continue

        result = lazy[key]
        assert np.shape(result) == np.shape(want), described
        assert isinstance(result, type(want)), described
        assert np.array_equal(result, want), described


@pytest.mark.parametrize(
    "key",
    [
        (LAZY_SHAPE[0], 0, 0),
        (0, 0, 0, LAZY_SHAPE[3], 0),
        ([0, LAZY_SHAPE[0]], 1, 2),
        (np.array([True, False]), 0, 0),
        (0, 0, 0, 0, 0, 0),
        (Ellipsis, Ellipsis, 0),
    ],
    ids=[
        "file-axis-out-of-bounds", "yx-out-of-bounds",
        "index-array-out-of-bounds", "mask-length", "too-many",
        "two-ellipses",
    ],
)
def test_lazy_array_bad_key_fails_exactly_as_numpy_does(lazy_array, key):
    """A bad key raises numpy's own IndexError, verbatim -- including this
    array's axis number, not that of the single image numpy sees. Returning
    zeros for an out-of-range index would hide the caller's bug."""
    lazy, expected = lazy_array

    with pytest.raises(IndexError) as eager_error:
        expected[key]

    with pytest.raises(IndexError, match=re.escape(str(eager_error.value))):
        lazy[key]


@pytest.mark.parametrize(
    "key, path",
    [
        ((1, 1, 2), "file"),                            # one Field image
        ((0, 0), "file"),                               # its Z-stack
        ((slice(None), 1), "file"),                     # one channel over time
        ((0, 0, [0, 3]), "file"),                       # chosen Z-planes
        (([0, 2], 1, 2), "file"),                       # chosen timepoints
        ((slice(None),), "file"),                       # the whole Well
        ((0, 0, 0, slice(2, 5), slice(1, 3)), "file"),  # a cropped image
        ((0, 0, 0, 1, 2), "file"),                      # one pixel
        ((0, 0, 0, Ellipsis, slice(None), slice(None)), "file"),  # 0-width ...
        ((0, 0, 0, [0, 2], 1), "coordinates"),          # index array in (Y, X)
        ((np.ones(LAZY_SHAPE[:4], dtype=bool),), "coordinates"),  # spans split
        (([0, 2], 1, 2, [0, 2], slice(None)), "coordinates"),     # broadcast
        ((slice(None), [1, 0], slice(None), slice(None), 1), "coordinates"),
    ],
    ids=[
        "field-image", "z-stack", "channel-over-time", "chosen-z",
        "chosen-timepoints", "whole-well", "cropped-image", "one-pixel",
        "zero-width-ellipsis", "index-array-in-yx", "mask-spanning-the-split",
        "broadcast-across-yx", "yx-integer-joins-the-broadcast",
    ],
)
def test_lazy_array_reads_the_everyday_keys_file_by_file(
    lazy_array, monkeypatch, key, path
):
    """The work people actually do must stay on the path that allocates
    only the result.

    Both paths agree with numpy, so nothing about a result reveals which one
    ran -- only memory does, and only on data too big for a test, so the
    routing is asserted directly."""
    lazy, _ = lazy_array
    coordinates = []
    real = LazyImageArray._read_by_coordinates
    monkeypatch.setattr(
        LazyImageArray, "_read_by_coordinates",
        lambda self, k: (coordinates.append(k), real(self, k))[1],
    )

    lazy[key]

    assert len(coordinates) == (0 if path == "file" else 1)


def test_lazy_array_reads_only_the_requested_images(lazy_array, monkeypatch):
    """The point of the wrapper: one image indexed, one file opened.

    Counted as opens, not cache misses: the extra read this replaced never
    showed up as one."""
    from PIL import Image as PILImage

    lazy, _ = lazy_array
    opened = []
    real_open = PILImage.open
    monkeypatch.setattr(
        PILImage, "open",
        lambda path, *a, **kw: (opened.append(path), real_open(path, *a, **kw))[1],
    )

    lazy[1, 1, 2]
    assert len(opened) == 1
    assert lazy.get_cache_info()["cache_misses"] == 1

    opened.clear()
    lazy[1, 1]  # one T and one C index, every Z image
    assert len(opened) == LAZY_SHAPE[2] - 1  # (1, 1, 2) is already cached


@pytest.mark.parametrize("absence", ["not-indexed", "no-file"])
def test_lazy_array_missing_image_reads_as_zeros_once(lazy_array, absence):
    """An image with no file reads as zeros rather than raising, so a
    partial acquisition still opens in the viewer, and its absence is
    remembered -- otherwise every pass through the Z slider re-stats it.
    Those lookups are not cache hits: counting them would report a fine hit
    rate for a **Well** whose images are mostly missing."""
    lazy, _ = lazy_array
    if absence == "not-indexed":
        del lazy.image_paths[(0, 0, 0)]
    else:
        (lazy.images_path / "p0_0_0.tiff").unlink()
    constructed = []
    real_construct = lazy.construct_path_func
    lazy.construct_path_func = lambda url, row, col: (
        constructed.append(url), real_construct(url, row, col)
    )[1]

    for _ in range(3):
        assert not lazy[0, 0, 0].any()

    assert len(constructed) <= 1, "the absent path was resolved more than once"
    info = lazy.get_cache_info()
    assert info["missing_images"] == 1
    assert (info["cache_misses"], info["cache_hits"]) == (0, 0)


def test_lazy_array_cache_drops_the_least_recently_used_image(lazy_array):
    """The cache is bounded, and keeps what was used most recently -- so
    re-reading an image has to count as use, not just as an answer."""
    lazy, _ = lazy_array
    lazy._max_cache_size = 2

    for tcz in [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 2)]:
        lazy[tcz]

    assert set(lazy._image_cache) == {(0, 0, 0), (0, 0, 2)}


def test_lazy_array_load_survives_an_eviction_by_another_reader(lazy_array):
    """napari slices on a worker thread, so a second read can evict the
    image a first one is in the middle of claiming. That window used to
    raise KeyError; losing the race should just mean reading again."""
    lazy, expected = lazy_array
    lazy._max_cache_size = 1
    lazy[0, 0, 0]  # the image the read below will go looking for

    raced = []

    class EvictsDuringTheClaim(OrderedDict):
        """Lets another read evict, once, as the claim is being made."""

        def pop(self, key, *default):
            if not raced:
                raced.append(True)
                evicting = threading.Thread(
                    target=lambda: lazy[0, 0, 1]  # evicts (0, 0, 0)
                )
                evicting.start()
                evicting.join()
            return super().pop(key, *default)

    lazy._image_cache = EvictsDuringTheClaim(lazy._image_cache)

    assert np.array_equal(lazy[0, 0, 0], expected[0, 0, 0])


def test_lazy_array_coordinate_read_reads_each_image_once(
    lazy_array, monkeypatch
):
    """Each image the coordinates path touches is opened once, however
    scattered through the result its elements are.

    Held before that path grouped the result by image, and has to keep
    holding: grouping makes reading per run of neighbouring elements the
    easy wrong turn, which would reopen every image the key returns to."""
    from PIL import Image as PILImage

    lazy, expected = lazy_array
    lazy._max_cache_size = 0  # nothing is kept, so a reopen would show
    opened = []
    real_open = PILImage.open
    monkeypatch.setattr(
        PILImage, "open",
        lambda path, *a, **kw: (opened.append(path), real_open(path, *a, **kw))[1],
    )

    # Alternating timepoints with an index array in (Y, X): two images, and
    # each one's elements land in two separate runs of the result.
    key = ([0, 1, 0, 1], 0, 0, [0, 1, 2, 3], 0)

    assert np.array_equal(lazy[key], expected[key])
    assert len(opened) == 2
    assert len(set(opened)) == 2


def test_lazy_array_bulk_read_reads_past_the_cache(lazy_array):
    """A read holding more images than the cache could keep does not fill it.

    It would only evict its own images as it went. Images already cached
    are still used."""
    lazy, expected = lazy_array
    lazy._max_cache_size = 4

    lazy[0, 0, 0]  # one image: worth keeping
    assert lazy.get_cache_info()["cached_images"] == 1
    hits = lazy.get_cache_info()["cache_hits"]

    assert np.array_equal(lazy[:], expected)  # 24 images: over the cache

    info = lazy.get_cache_info()
    assert info["cached_images"] == 1, "the bulk read filled the cache"
    assert info["cache_hits"] == hits + 1, "the cached image was re-read"


def test_lazy_array_materializing_does_not_fill_the_cache(lazy_array):
    """Materializing hands the caller every image, so caching them too holds
    the **Well** twice -- measurably, at twice the peak memory of the Save
    path. This is so even when they would fit, as here."""
    lazy, expected = lazy_array
    assert math.prod(LAZY_SHAPE[:3]) < lazy._max_cache_size

    assert np.array_equal(np.asarray(lazy), expected)

    assert lazy.get_cache_info()["cached_images"] == 0


def test_lazy_array_clear_cache_also_forgets_absences(lazy_array):
    """Remembered absences are only right while the **Export** is what it
    was, so clearing the cache re-checks them."""
    lazy, _ = lazy_array
    del lazy.image_paths[(0, 0, 0)]
    lazy[0, 0, 0]
    lazy[0, 0, 1]
    before = lazy.get_cache_info()
    assert (before["cached_images"], before["missing_images"]) == (1, 1)

    lazy.clear_cache()

    info = lazy.get_cache_info()
    assert (info["cached_images"], info["missing_images"]) == (0, 0)
    assert (info["cache_hits"], info["cache_misses"]) == (0, 0)


def test_lazy_array_repr_summarises_the_cache(lazy_array):
    """The repr is how the cache gets inspected from a notebook, so it has
    to hold up before anything has been read -- no hit rate to divide."""
    lazy, _ = lazy_array

    assert "shape=(3, 2, 4, 6, 5)" in repr(lazy)
    assert "cached=0/100" in repr(lazy)

    lazy[0, 0, 0]

    assert "cached=1/100" in repr(lazy)


def test_lazy_array_image_disagreeing_with_declared_shape_raises(lazy_array):
    """The (Y, X) shape comes from the experiment metadata; a file that
    disagrees must say so, not surface as a bare numpy broadcast error."""
    from PIL import Image as PILImage

    lazy, _ = lazy_array
    bigger = np.zeros((LAZY_SHAPE[3] + 2, LAZY_SHAPE[4] + 2), dtype=np.uint16)
    PILImage.fromarray(bigger).save(lazy.images_path / "p0_0_0.tiff")

    with pytest.raises(ValueError, match="metadata declares"):
        lazy[0, 0, 0]


def test_lazy_array_converts_to_numpy_in_full(lazy_array):
    """np.asarray() is how the Save path materializes the array, and
    tifffile passes a dtype positionally when it does. ``copy=False`` can
    only be refused: reading every image into a new array is a copy."""
    lazy, expected = lazy_array

    assert np.array_equal(np.asarray(lazy), expected)
    assert np.asarray(lazy, dtype=np.float32).dtype == np.float32
    with pytest.raises(ValueError, match="without copying"):
        np.array(lazy, copy=False)


@pytest.mark.parametrize("shape", [(4, 6, 5), (2, 3, 6, 5)], ids=["zyx", "tzyx"])
@pytest.mark.parametrize(
    "key",
    [(0,), (slice(None),), (Ellipsis, [1, 0])],
    ids=["scalar", "bare-slice", "yx-list"],
)
def test_lazy_array_treats_the_last_two_axes_as_the_image(tmp_path, shape, key):
    """The file axes are however many precede (Y, X), so the wrapper is not
    pinned to the 5-D (T, C, Z, Y, X) shape the reader happens to build."""
    lazy, expected = _lazy_and_eager(tmp_path, shape)

    result = lazy[key]

    assert result.shape == expected[key].shape
    assert np.array_equal(result, expected[key])
