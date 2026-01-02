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
    
    # Create a minimal Index.xml
    root = ET.Element("OME", attrib={
        "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xmlns:ns": "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"
    })
    
    # Add plate info
    plate = ET.SubElement(root, "ns:Plate", attrib={
        "PlateID": "TEST001",
        "Rows": "2",
        "Columns": "2"
    })
    
    # Add a well
    well = ET.SubElement(plate, "ns:Well", attrib={
        "Row": "01",
        "Column": "01"
    })
    
    # Add channel info
    channels = ET.SubElement(root, "ns:Channels")
    channel = ET.SubElement(channels, "ns:Channel", attrib={
        "ChannelID": "1",
        "ChannelName": "DAPI",
        "ExcitationWavelength": "405",
        "EmissionWavelength": "450"
    })
    
    # Add image metadata
    image = ET.SubElement(root, "ns:Image", attrib={
        "Row": "01",
        "Column": "01",
        "FieldID": "1",
        "PlaneID": "1",
        "TimepointID": "1",
        "ChannelID": "1",
        "URL": "r01c01f01p01-ch1sk1fk1fl1.tiff"
    })
    
    pixels = ET.SubElement(image, "ns:Pixels", attrib={
        "SizeX": "100",
        "SizeY": "100",
        "PhysicalSizeX": "0.00065",
        "PhysicalSizeY": "0.00065"
    })
    
    # Write XML
    tree = ET.ElementTree(root)
    index_path = images_dir / "Index.xml"
    tree.write(index_path)
    
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
