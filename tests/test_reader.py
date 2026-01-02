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
