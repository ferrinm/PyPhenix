import pytest
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image


@pytest.fixture
def sample_phenix_data():
    """Generate sample image data for testing."""
    # Create multi-dimensional test data (T, C, Z, Y, X)
    data = np.random.randint(0, 4095, (2, 3, 5, 100, 100), dtype=np.uint16)
    return data


@pytest.fixture
def sample_metadata():
    """Generate sample metadata for testing."""
    return {
        'plate_id': 'TEST001',
        'well': 'r01c01',
        'field': 1,
        'channels': {
            1: {'name': 'DAPI', 'wavelength': 405},
            2: {'name': 'GFP', 'wavelength': 488},
            3: {'name': 'mCherry', 'wavelength': 594}
        },
        'pixel_size': {'x': 6.5e-7, 'y': 6.5e-7},
        'z_step': 1e-6,
        'timepoints': [1, 2],
        'fields': [1],
        'stitched': False
    }


@pytest.fixture
def complete_phenix_experiment(tmp_path):
    """
    Create a more complete mock experiment with multiple wells,
    fields, and channels.
    """
    images_dir = tmp_path / "Images"
    images_dir.mkdir()
    
    # Create Index.xml with more complete structure
    root = ET.Element("OME", attrib={
        "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xmlns:ns": "43B2A954-E3C3-47E1-B392-6635266B0DD3/HarmonyV7"
    })
    
    plate = ET.SubElement(root, "ns:Plate", attrib={
        "PlateID": "TEST001",
        "Rows": "2",
        "Columns": "2"
    })
    
    # Add multiple wells
    for row in ['01', '02']:
        for col in ['01', '02']:
            well = ET.SubElement(plate, "ns:Well", attrib={
                "Row": row,
                "Column": col
            })
    
    # Add channels
    channels = ET.SubElement(root, "ns:Channels")
    channel_info = [
        ("1", "DAPI", "405", "450"),
        ("2", "GFP", "488", "525"),
        ("3", "mCherry", "561", "610")
    ]
    
    for ch_id, ch_name, ex_wave, em_wave in channel_info:
        ET.SubElement(channels, "ns:Channel", attrib={
            "ChannelID": ch_id,
            "ChannelName": ch_name,
            "ExcitationWavelength": ex_wave,
            "EmissionWavelength": em_wave
        })
    
    # Create sample images
    test_image = np.random.randint(0, 4095, (100, 100), dtype=np.uint16)
    
    # Add images for one well with multiple channels
    for ch_id in ['1', '2', '3']:
        image = ET.SubElement(root, "ns:Image", attrib={
            "Row": "01",
            "Column": "01",
            "FieldID": "1",
            "PlaneID": "1",
            "TimepointID": "1",
            "ChannelID": ch_id,
            "URL": f"r01c01f01p01-ch{ch_id}sk1fk1fl1.tiff"
        })
        
        pixels = ET.SubElement(image, "ns:Pixels", attrib={
            "SizeX": "100",
            "SizeY": "100",
            "PhysicalSizeX": "0.00065",
            "PhysicalSizeY": "0.00065"
        })
        
        # Save image file
        img = Image.fromarray(test_image)
        img.save(images_dir / f"r01c01f01p01-ch{ch_id}sk1fk1fl1.tiff")
    
    # Write XML
    tree = ET.ElementTree(root)
    tree.write(images_dir / "Index.xml")
    
    return tmp_path
