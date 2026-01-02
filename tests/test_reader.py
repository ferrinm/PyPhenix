import numpy as np
import pytest
from pathlib import Path
import xml.etree.ElementTree as ET

from pyphenix import napari_get_reader, OperaPhenixReader


@pytest.fixture
def mock_phenix_experiment(tmp_path):
    """Create a minimal mock Opera Phenix experiment directory structure."""
    # Create directory structure
    images_dir = tmp_path / "Images"
    images_dir.mkdir()
    
    # Create a minimal Index.xml with proper structure matching what the reader expects
    root = ET.Element("OME")
    root.set("xmlns", "http://www.openmicroscopy.org/Schemas/OME/2016-06")
    
    # Register namespace
    ns = "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"
    ET.register_namespace('ns', ns)
    
    # Add plate info - match exact element names from your reader
    plate = ET.SubElement(root, f"{{{ns}}}Plate")
    
    # PlateID as child element
    plate_id = ET.SubElement(plate, f"{{{ns}}}PlateID")
    plate_id.text = "TEST001"
    
    # PlateRows (not just "Rows")
    rows_elem = ET.SubElement(plate, f"{{{ns}}}PlateRows")
    rows_elem.text = "2"
    
    # PlateColumns (not just "Columns")
    cols_elem = ET.SubElement(plate, f"{{{ns}}}PlateColumns")
    cols_elem.text = "2"
    
    # Add a well with Row and Column as child elements
    well = ET.SubElement(plate, f"{{{ns}}}Well")
    well_row = ET.SubElement(well, f"{{{ns}}}Row")
    well_row.text = "01"
    well_col = ET.SubElement(well, f"{{{ns}}}Column")
    well_col.text = "01"
    
    # Add channel info
    channels = ET.SubElement(root, f"{{{ns}}}Channels")
    channel = ET.SubElement(channels, f"{{{ns}}}Channel")
    ch_id = ET.SubElement(channel, f"{{{ns}}}ChannelID")
    ch_id.text = "1"
    ch_name = ET.SubElement(channel, f"{{{ns}}}ChannelName")
    ch_name.text = "DAPI"
    ex_wave = ET.SubElement(channel, f"{{{ns}}}ExcitationWavelength")
    ex_wave.text = "405"
    em_wave = ET.SubElement(channel, f"{{{ns}}}EmissionWavelength")
    em_wave.text = "450"
    
    # Add image metadata
    image = ET.SubElement(root, f"{{{ns}}}Image")
    img_row = ET.SubElement(image, f"{{{ns}}}Row")
    img_row.text = "01"
    img_col = ET.SubElement(image, f"{{{ns}}}Column")
    img_col.text = "01"
    field_id = ET.SubElement(image, f"{{{ns}}}FieldID")
    field_id.text = "1"
    plane_id = ET.SubElement(image, f"{{{ns}}}PlaneID")
    plane_id.text = "1"
    time_id = ET.SubElement(image, f"{{{ns}}}TimepointID")
    time_id.text = "1"
    img_ch_id = ET.SubElement(image, f"{{{ns}}}ChannelID")
    img_ch_id.text = "1"
    url = ET.SubElement(image, f"{{{ns}}}URL")
    url.text = "r01c01f01p01-ch1sk1fk1fl1.tiff"
    
    pixels = ET.SubElement(image, f"{{{ns}}}Pixels")
    size_x = ET.SubElement(pixels, f"{{{ns}}}SizeX")
    size_x.text = "100"
    size_y = ET.SubElement(pixels, f"{{{ns}}}SizeY")
    size_y.text = "100"
    phys_x = ET.SubElement(pixels, f"{{{ns}}}PhysicalSizeX")
    phys_x.text = "0.00065"
    phys_y = ET.SubElement(pixels, f"{{{ns}}}PhysicalSizeY")
    phys_y.text = "0.00065"
    
    # Write XML
    tree = ET.ElementTree(root)
    index_path = images_dir / "Index.xml"
    tree.write(index_path, encoding='utf-8', xml_declaration=True)
    
    # Create a minimal test image
    test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    from PIL import Image
    img = Image.fromarray(test_image)
    img.save(images_dir / "r01c01f01p01-ch1sk1fk1fl1.tiff")
    
    return tmp_path


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
