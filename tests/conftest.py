import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def make_napari_viewer():
    """Fixture to create a napari viewer for testing."""
    from napari import Viewer
    
    viewers = []
    
    def _make_viewer(*args, **kwargs):
        kwargs.setdefault('show', False)  # Don't show window during tests
        viewer = Viewer(*args, **kwargs)
        viewers.append(viewer)
        return viewer
    
    yield _make_viewer
    
    # Cleanup
    for viewer in viewers:
        viewer.close()


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
