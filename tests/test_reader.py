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
    
    # Create XML with structure matching real Opera Phenix data
    # Note: The real XML uses "EvaluationInputData" as root, not "OME"
    ns = "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"
    
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
    
    # Add wells in Plates section (just the id attribute)
    well_plate = ET.SubElement(plate, "Well", id="0101")
    
    # Add Wells section (detailed well information)
    wells = ET.SubElement(root, "Wells")
    well = ET.SubElement(wells, "Well")
    
    well_id = ET.SubElement(well, "id")
    well_id.text = "0101"
    
    well_row = ET.SubElement(well, "Row")
    well_row.text = "1"
    
    well_col = ET.SubElement(well, "Col")
    well_col.text = "1"
    
    # Add image references (K=timepoint, F=field, P=plane, R=channel)
    # Format: wellK1F1P1R1 means well, timepoint 1, field 1, plane 1, channel 1
    image_ref = ET.SubElement(well, "Image", id="0101K1F1P1R1")
    
    # Add Images section with detailed metadata
    images = ET.SubElement(root, "Images")
    image = ET.SubElement(images, "Image")
    
    img_id = ET.SubElement(image, "ImageID")
    img_id.text = "0101K1F1P1R1"
    
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
    
    url = ET.SubElement(image, "URL")
    url.text = "r01c01f01p01-ch1sk1fk1fl1.tiff"
    
    # Add pixel information
    pixels = ET.SubElement(image, "Pixels")
    
    size_x = ET.SubElement(pixels, "SizeX")
    size_x.text = "100"
    
    size_y = ET.SubElement(pixels, "SizeY")
    size_y.text = "100"
    
    phys_x = ET.SubElement(pixels, "PhysicalSizeX")
    phys_x.text = "0.00065"
    
    phys_y = ET.SubElement(pixels, "PhysicalSizeY")
    phys_y.text = "0.00065"
    
    # Add Channels section
    channels = ET.SubElement(root, "Channels")
    channel = ET.SubElement(channels, "Channel")
    
    ch_id = ET.SubElement(channel, "ChannelID")
    ch_id.text = "1"
    
    ch_name = ET.SubElement(channel, "ChannelName")
    ch_name.text = "DAPI"
    
    ex_wave = ET.SubElement(channel, "ExcitationWavelength")
    ex_wave.text = "405"
    
    em_wave = ET.SubElement(channel, "EmissionWavelength")
    em_wave.text = "450"
    
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
