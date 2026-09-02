import importlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import tifffile

from pyphenix._save import (
    OME_SUFFIX,
    ome_tiff_path,
    save,
    save_numpy,
    save_ome_tiff,
)

from .test_reader import _write_mock_experiment

OME_NS = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}


def _metadata(n_channels=2, z_step=1.5e-6, time_increment=300.0):
    """A metadata dict shaped like OperaPhenixReader.read_data returns."""
    channel_names = ['DAPI', 'Alexa 488', 'mCherry']
    excitations = [405, 488, 561]
    emissions = [450, 525, 610]
    # Deliberately non-sequential channel IDs: C-axis order must follow
    # insertion order, not a sort of the keys.
    channel_ids = [3, 1, 2][:n_channels]
    return {
        'plate_id': 'MOCK001',
        'plate_layout': {'rows': 2, 'columns': 2},
        'well': 'A02',
        'well_numeric': 'r01c02',
        'shape': {'description': 'T, C, Z, Y, X', 'dimensions': None},
        'fields': [1],
        'timepoints': [1, 2],
        'timepoint_offsets': [0.0, 300.0],
        'time_increment': time_increment,
        'time_unit': 's',
        'channels': {
            ch_id: {
                'name': channel_names[i],
                'excitation': str(excitations[i]),
                'emission': str(emissions[i]),
            }
            for i, ch_id in enumerate(channel_ids)
        },
        'z_slices': [1],
        'pixel_size': {'x': 2.99e-7, 'y': 2.99e-7, 'unit': 'm'},
        'z_step': z_step,
        'stitched': False,
    }


def _pixels(path):
    """Return the OME-XML <Pixels> attributes of a written file."""
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome, f"{path} is not an OME-TIFF"
        root = ET.fromstring(tif.ome_metadata)
    return root.find('.//ome:Pixels', OME_NS).attrib


def _image(path):
    """Return the OME-XML <Image> element of a written file."""
    with tifffile.TiffFile(path) as tif:
        root = ET.fromstring(tif.ome_metadata)
    return root.find('.//ome:Image', OME_NS)


# ----------------------------------------------------------------------
# Filename normalisation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("given", [
    "well_A02",
    "well_A02.tiff",
    "well_A02.tif",
    "well_A02.ome.tiff",
    "well_A02.ome.tif",
    "well_A02.npy",
])
def test_ome_path_normalised(tmp_path, given):
    assert ome_tiff_path(tmp_path / given).name == "well_A02" + OME_SUFFIX


def test_ome_path_preserves_dotted_stem(tmp_path):
    # A dot in the stem must not be mistaken for an extension.
    assert ome_tiff_path(tmp_path / "plate.1_A02").name == (
        "plate.1_A02" + OME_SUFFIX
    )


def test_save_writes_normalised_path(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(), tmp_path / "well_A02.tiff")
    assert written.name == "well_A02" + OME_SUFFIX
    assert written.exists()
    assert not (tmp_path / "well_A02.tiff").exists()


# ----------------------------------------------------------------------
# OME-XML content
# ----------------------------------------------------------------------

def test_written_file_is_ome(tmp_path):
    data = np.zeros((2, 2, 3, 8, 8), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(), tmp_path / "out")
    with tifffile.TiffFile(written) as tif:
        assert tif.is_ome
        assert tif.ome_metadata is not None


@pytest.mark.parametrize("shape", [
    (2, 2, 3, 8, 8),   # full 5-D
    (1, 2, 1, 4, 4),   # single timepoint, single plane — the common case
    (1, 1, 1, 4, 4),   # everything singleton
    (3, 2, 1, 4, 4),   # time series, single plane
])
def test_sizes_match_source_shape(tmp_path, shape):
    """Regression guard: tifffile squeezes singleton axes, so a naive
    ``axes='TCZYX'`` declaration raises "axes do not match stored shape".
    The stored array may be squeezed, but the OME-XML sizes must not be."""
    t, c, z, y, x = shape
    data = np.zeros(shape, dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(n_channels=c), tmp_path / "out")
    pixels = _pixels(written)
    assert int(pixels['SizeT']) == t
    assert int(pixels['SizeC']) == c
    assert int(pixels['SizeZ']) == z
    assert int(pixels['SizeY']) == y
    assert int(pixels['SizeX']) == x


def test_physical_pixel_size_in_microns(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(), tmp_path / "out")
    pixels = _pixels(written)
    # 2.99e-7 m == 0.299 µm
    assert float(pixels['PhysicalSizeX']) == pytest.approx(0.299)
    assert float(pixels['PhysicalSizeY']) == pytest.approx(0.299)
    assert pixels['PhysicalSizeXUnit'] == 'µm'
    assert pixels['PhysicalSizeYUnit'] == 'µm'


def test_z_step_present_when_set(tmp_path):
    data = np.zeros((1, 2, 2, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(z_step=1.5e-6), tmp_path / "out")
    pixels = _pixels(written)
    assert float(pixels['PhysicalSizeZ']) == pytest.approx(1.5)
    assert pixels['PhysicalSizeZUnit'] == 'µm'


def test_z_step_absent_when_none(tmp_path):
    """A single-plane acquisition has no Z spacing; claiming one would lie."""
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(z_step=None), tmp_path / "out")
    assert 'PhysicalSizeZ' not in _pixels(written)


def test_time_increment_present_for_multiple_timepoints(tmp_path):
    data = np.zeros((2, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(
        data, _metadata(time_increment=300.0), tmp_path / "out"
    )
    pixels = _pixels(written)
    assert float(pixels['TimeIncrement']) == pytest.approx(300.0)
    assert pixels['TimeIncrementUnit'] == 's'


def test_time_increment_absent_for_single_timepoint(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(
        data, _metadata(time_increment=None), tmp_path / "out"
    )
    assert 'TimeIncrement' not in _pixels(written)


def test_channel_names_follow_c_axis_order(tmp_path):
    """Channel IDs are 3, 1, 2 — names must follow insertion order."""
    data = np.zeros((1, 3, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(n_channels=3), tmp_path / "out")
    with tifffile.TiffFile(written) as tif:
        root = ET.fromstring(tif.ome_metadata)
    names = [c.get('Name') for c in root.findall('.//ome:Channel', OME_NS)]
    assert names == ['DAPI', 'Alexa 488', 'mCherry']


def test_channel_wavelengths_written(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(), tmp_path / "out")
    with tifffile.TiffFile(written) as tif:
        root = ET.fromstring(tif.ome_metadata)
    channels = root.findall('.//ome:Channel', OME_NS)
    assert [float(c.get('ExcitationWavelength')) for c in channels] == [405, 488]
    assert [float(c.get('EmissionWavelength')) for c in channels] == [450, 525]


def test_provenance_embedded_as_json(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(), tmp_path / "out")
    image = _image(written)
    assert image.get('Name') == 'MOCK001 A02'
    provenance = json.loads(image.find('ome:Description', OME_NS).text)
    assert provenance['plate_id'] == 'MOCK001'
    assert provenance['well'] == 'A02'
    assert provenance['fields'] == [1]
    assert provenance['stitched'] is False


def test_provenance_omits_keys_ome_already_covers(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(), tmp_path / "out")
    provenance = json.loads(
        _image(written).find('ome:Description', OME_NS).text
    )
    for key in ('channels', 'pixel_size', 'z_step', 'time_increment', 'shape'):
        assert key not in provenance


def test_pixels_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4095, (2, 2, 3, 8, 8), dtype=np.uint16)
    written = save_ome_tiff(data, _metadata(), tmp_path / "out")
    read_back = tifffile.imread(written)
    assert read_back.dtype == data.dtype
    assert np.array_equal(read_back.reshape(data.shape), data)


def test_rejects_non_5d_data(tmp_path):
    with pytest.raises(ValueError, match="5-D"):
        save_ome_tiff(np.zeros((4, 4)), _metadata(), tmp_path / "out")


# ----------------------------------------------------------------------
# Sidecar behaviour
# ----------------------------------------------------------------------

def test_ome_tiff_writes_no_sidecar(tmp_path):
    """An OME-TIFF Save is self-describing (ADR 0002)."""
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    save_ome_tiff(data, _metadata(), tmp_path / "well_A02")
    assert list(tmp_path.glob("*.json")) == []


def test_numpy_still_writes_sidecar(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    array_path, sidecar_path = save_numpy(
        data, _metadata(), tmp_path / "well_A02"
    )
    assert array_path.with_suffix('.npy').exists()
    assert sidecar_path.name == "well_A02.json"
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar['plate_id'] == 'MOCK001'
    assert sidecar['pixel_size']['x'] == pytest.approx(2.99e-7)
    assert np.array_equal(np.load(array_path.with_suffix('.npy')), data)


def test_save_dispatches_on_format(tmp_path):
    data = np.zeros((1, 2, 1, 4, 4), dtype=np.uint16)
    ome = save(data, _metadata(), tmp_path / "a", 'ome-tiff')
    npy = save(data, _metadata(), tmp_path / "b", 'numpy')
    assert ome.name.endswith(OME_SUFFIX)
    assert npy.with_suffix('.npy').exists()


def test_save_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unknown save format"):
        save(np.zeros((1, 1, 1, 4, 4)), _metadata(), tmp_path / "a", 'jpeg')


# ----------------------------------------------------------------------
# Lazy loading
# ----------------------------------------------------------------------

class _StubLazyArray:
    """Mimics LazyImageArray's duck-type contract: no ndarray inheritance."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.dtype = array.dtype

    def __array__(self, dtype=None):
        return self._array if dtype is None else self._array.astype(dtype)


def test_lazy_array_saves_identically_to_eager(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.integers(0, 4095, (2, 2, 2, 4, 4), dtype=np.uint16)
    eager = save_ome_tiff(data, _metadata(), tmp_path / "eager")
    lazy = save_ome_tiff(
        _StubLazyArray(data), _metadata(), tmp_path / "lazy"
    )
    assert np.array_equal(tifffile.imread(eager), tifffile.imread(lazy))
    assert _pixels(eager) == _pixels(lazy)


# ----------------------------------------------------------------------
# Integration with the reader, and the single-implementation guarantee
# ----------------------------------------------------------------------

def test_reader_save_output_writes_ome_tiff(tmp_path):
    """The reader's save path goes through the same writer."""
    from pyphenix import OperaPhenixReader

    experiment = tmp_path / "experiment"
    experiment.mkdir()
    _write_mock_experiment(experiment, "HarmonyV7")
    reader = OperaPhenixReader(str(experiment), verbose=False)

    out = tmp_path / "from_reader"
    reader.read_data(output_file=str(out), output_format='ome-tiff',
                     verbose=False)

    written = out.with_name("from_reader" + OME_SUFFIX)
    assert written.exists()
    with tifffile.TiffFile(written) as tif:
        assert tif.is_ome
    assert list(tmp_path.glob("from_reader*.json")) == []


def test_reader_numpy_save_keeps_sidecar(tmp_path):
    from pyphenix import OperaPhenixReader

    experiment = tmp_path / "experiment"
    experiment.mkdir()
    _write_mock_experiment(experiment, "HarmonyV7")
    reader = OperaPhenixReader(str(experiment), verbose=False)

    out = tmp_path / "from_reader.npy"
    reader.read_data(output_file=str(out), output_format='numpy',
                     verbose=False)

    assert out.exists()
    assert (tmp_path / "from_reader.json").exists()


def test_save_logic_has_a_single_implementation():
    """_widget_backup held a third stale copy of the save logic."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('pyphenix._widget_backup')


def test_tifffile_is_a_hard_dependency():
    """No ImportError guard, no silent fallback to numpy."""
    import pyphenix._save as save_module

    assert save_module.tifffile is tifffile
