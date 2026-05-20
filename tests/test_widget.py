import numpy as np
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from pyphenix._widget import PhenixDataLoaderWidget, CollapsibleSection
from qtpy.QtWidgets import QListWidget, QAbstractItemView


@pytest.fixture
def widget_with_viewer(make_napari_viewer, qtbot):
    """Create widget with napari viewer."""
    viewer = make_napari_viewer()
    widget = PhenixDataLoaderWidget(viewer)
    qtbot.addWidget(widget)
    return widget, viewer


def test_widget_initialization(widget_with_viewer):
    """Test that widget initializes properly."""
    widget, viewer = widget_with_viewer
    
    assert widget.viewer == viewer
    assert widget.reader is None
    assert widget.metadata is None
    
    # Check that UI elements exist
    assert hasattr(widget, 'path_input')
    assert hasattr(widget, 'well_combo')
    assert hasattr(widget, 'field_combo')
    assert hasattr(widget, 'visualize_btn')


def test_widget_controls_disabled_initially(widget_with_viewer):
    """Test that controls are disabled before loading experiment."""
    widget, _ = widget_with_viewer
    
    assert not widget.well_combo.isEnabled()
    assert not widget.field_combo.isEnabled()
    assert not widget.visualize_btn.isEnabled()


def test_collapsible_section(qtbot):
    """Test CollapsibleSection functionality."""
    from pyphenix._widget import CollapsibleSection
    from qtpy.QtWidgets import QVBoxLayout, QLabel
    from qtpy.QtCore import Qt
    
    section = CollapsibleSection("Test Section")
    qtbot.addWidget(section)
    section.show()  # Ensure widget and its children become visible
    qtbot.waitExposed(section)  # Waits until widget is shown, if supported by your qtbot
    
    # Check initial state - should be expanded
    assert not section.is_collapsed
    assert section.content_widget.isVisible()
    assert section.toggle_button.arrowType() == Qt.DownArrow
    
    # Add some content to test
    content_layout = QVBoxLayout()
    test_label = QLabel("Test Content")
    content_layout.addWidget(test_label)
    section.setContentLayout(content_layout)
    
    # Verify content is set
    assert section.content_widget.layout() is not None
    
    # Test collapsing
    section.toggle()
    qtbot.wait(250)  # Wait for animation to complete (200ms + buffer)
    
    assert section.is_collapsed
    assert section.toggle_button.arrowType() == Qt.RightArrow
    # Content widget height should be 0 or very small after animation
    assert section.content_widget.height() < 10
    
    # Test expanding
    section.toggle()
    qtbot.wait(250)  # Wait for animation
    
    assert not section.is_collapsed
    assert section.toggle_button.arrowType() == Qt.DownArrow
    assert section.content_widget.isVisible()


def test_multiple_collapsible_sections(qtbot):
    """Test that multiple CollapsibleSections work independently."""
    from pyphenix._widget import CollapsibleSection
    from qtpy.QtWidgets import QVBoxLayout, QLabel, QWidget
    from qtpy.QtCore import Qt
    
    # Create container with multiple sections
    container = QWidget()
    qtbot.addWidget(container)
    layout = QVBoxLayout(container)
    
    section1 = CollapsibleSection("Section 1")
    section2 = CollapsibleSection("Section 2")
    
    layout.addWidget(section1)
    layout.addWidget(section2)
    
    section1.show()  # Ensure widget and its children become visible
    qtbot.waitExposed(section1)  # Waits until widget is shown, if supported by your qtbot
    section2.show()  # Ensure widget and its children become visible
    qtbot.waitExposed(section2)  # Waits until widget is shown, if supported by your qtbot

    # Add content to both
    for section in [section1, section2]:
        content_layout = QVBoxLayout()
        content_layout.addWidget(QLabel("Test Content"))
        section.setContentLayout(content_layout)
    
    # Both should start expanded
    assert not section1.is_collapsed
    assert not section2.is_collapsed
    
    # Collapse first section only
    section1.toggle()
    qtbot.wait(250)
    
    assert section1.is_collapsed
    assert not section2.is_collapsed  # Second should still be expanded
    
    # Collapse second section
    section2.toggle()
    qtbot.wait(250)
    
    assert section1.is_collapsed
    assert section2.is_collapsed
    
    # Expand first section only
    section1.toggle()
    qtbot.wait(250)
    
    assert not section1.is_collapsed
    assert section2.is_collapsed  # Second should still be collapsed


def test_collapsible_section_content_layout(qtbot):
    """Test that setContentLayout properly adds content."""
    from pyphenix._widget import CollapsibleSection
    from qtpy.QtWidgets import QVBoxLayout, QLabel, QPushButton
    
    section = CollapsibleSection("Test Section")
    qtbot.addWidget(section)

    section.show()  # Ensure widget and its children become visible
    qtbot.waitExposed(section)  # Waits until widget is shown, if supported by your qtbot
    
    # Create a layout with multiple widgets
    content_layout = QVBoxLayout()
    label1 = QLabel("Label 1")
    label2 = QLabel("Label 2")
    button = QPushButton("Test Button")
    
    content_layout.addWidget(label1)
    content_layout.addWidget(label2)
    content_layout.addWidget(button)
    
    # Set the content
    section.setContentLayout(content_layout)
    
    # Verify the content widget has the layout
    assert section.content_widget.layout() is not None
    
    # Verify widgets are in the content widget
    # Find all child widgets of content_widget
    children = section.content_widget.findChildren(QLabel)
    assert len(children) == 2
    
    buttons = section.content_widget.findChildren(QPushButton)
    assert len(buttons) == 1


def test_select_all_helper(widget_with_viewer, qtbot):
    """Test _select_all helper method."""
    widget, _ = widget_with_viewer
    
    # Add some items to a list widget
    test_list = QListWidget()
    test_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    qtbot.addWidget(test_list)
    test_list.addItems(["Item 1", "Item 2", "Item 3"])
    
    # Initially nothing selected
    assert len(test_list.selectedItems()) == 0
    
    # Select all
    widget._select_all(test_list)
    assert len(test_list.selectedItems()) == 3


def test_clear_all_helper(widget_with_viewer, qtbot):
    """Test _clear_all helper method."""
    widget, _ = widget_with_viewer
    
    # Add and select items
    test_list = QListWidget()
    test_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    qtbot.addWidget(test_list)
    test_list.addItems(["Item 1", "Item 2", "Item 3"])
    
    # Select all first
    widget._select_all(test_list)
    assert len(test_list.selectedItems()) == 3
    
    # Clear selection
    widget._clear_all(test_list)
    assert len(test_list.selectedItems()) == 0


def test_get_selected_indices(widget_with_viewer, qtbot):
    """Test _get_selected_indices helper method."""
    widget, _ = widget_with_viewer
    
    test_list = QListWidget()
    test_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    qtbot.addWidget(test_list)
    test_list.addItems(["Item 1", "Item 2", "Item 3", "Item 4"])
    
    # Select specific items
    test_list.item(0).setSelected(True)
    test_list.item(2).setSelected(True)
    
    indices = widget._get_selected_indices(test_list)
    assert sorted(indices) == [0, 2]


def test_stitch_checkbox_toggles_field_combo(widget_with_viewer):
    """Test that stitch checkbox enables/disables field combo."""
    widget, _ = widget_with_viewer
    
    # Enable controls first
    widget._set_controls_enabled(True)
    
    # Initially field combo should be enabled
    assert widget.field_combo.isEnabled()
    
    # Check stitch checkbox
    widget.stitch_checkbox.setChecked(True)
    assert not widget.field_combo.isEnabled()
    
    # Uncheck stitch checkbox
    widget.stitch_checkbox.setChecked(False)
    assert widget.field_combo.isEnabled()


def test_browse_experiment_sets_path(widget_with_viewer, tmp_path, monkeypatch):
    """Test that browse button sets path correctly."""
    widget, _ = widget_with_viewer
    
    # Mock QFileDialog to return tmp_path
    from qtpy.QtWidgets import QFileDialog
    monkeypatch.setattr(
        QFileDialog,
        'getExistingDirectory',
        lambda *args, **kwargs: str(tmp_path)
    )
    
    widget._browse_experiment()
    assert widget.path_input.text() == str(tmp_path)


def test_load_experiment_warning_for_empty_path(widget_with_viewer):
    """Test that loading with empty path shows warning."""
    widget, _ = widget_with_viewer
    
    widget.path_input.setText("")
    
    with patch('pyphenix._widget.notifications.show_warning') as mock_warning:
        widget._load_experiment()
        mock_warning.assert_called_once()


def test_visualize_warning_for_no_reader(widget_with_viewer):
    """Test that visualizing without reader shows warning."""
    widget, _ = widget_with_viewer
    
    assert widget.reader is None
    
    with patch('pyphenix._widget.notifications.show_warning') as mock_warning:
        widget._visualize_data()
        mock_warning.assert_called_once()


def test_add_layers_creates_image_layers(widget_with_viewer):
    """Test that _add_layers_to_viewer creates proper image layers."""
    widget, viewer = widget_with_viewer
    
    # Create mock data
    data = np.random.randint(0, 255, (1, 2, 5, 100, 100), dtype=np.uint16)
    
    metadata = {
        'channels': {
            1: {'name': 'DAPI', 'wavelength': 405},
            2: {'name': 'GFP', 'wavelength': 488}
        },
        'pixel_size': {'x': 6.5e-7, 'y': 6.5e-7},
        'z_step': 1e-6,
        'well': 'r01c01',
        'plate_id': 'TEST001',
        'stitched': False,
        'fields': [1]
    }
    
    widget._add_layers_to_viewer(viewer, data, metadata)
    
    # Check that layers were added
    assert len(viewer.layers) == 2
    
    # Check first layer properties
    layer = viewer.layers[0]
    assert layer.name.startswith('Ch1')
    assert 'DAPI' in layer.name
    assert layer.colormap.name == 'cyan'


def test_set_controls_enabled(widget_with_viewer):
    """Test _set_controls_enabled method."""
    widget, _ = widget_with_viewer

    # Enable all controls
    widget._set_controls_enabled(True)

    assert widget.well_combo.isEnabled()
    assert widget.field_combo.isEnabled()
    assert widget.time_list.isEnabled()
    assert widget.channel_list.isEnabled()
    assert widget.plate_overview_btn.isEnabled()

    # Disable all controls
    widget._set_controls_enabled(False)

    assert not widget.well_combo.isEnabled()
    assert not widget.field_combo.isEnabled()
    assert not widget.time_list.isEnabled()
    assert not widget.channel_list.isEnabled()
    assert not widget.plate_overview_btn.isEnabled()


def test_plate_overview_button_disabled_initially(widget_with_viewer):
    """Plate overview button starts disabled until an experiment is loaded."""
    widget, _ = widget_with_viewer
    assert not widget.plate_overview_btn.isEnabled()


def test_plate_overview_warns_without_experiment(widget_with_viewer):
    """Clicking with no reader loaded shows a warning, not a crash."""
    widget, _ = widget_with_viewer
    assert widget.reader is None

    with patch('pyphenix._widget.notifications.show_warning') as mock_warn:
        widget._generate_plate_overview()
        mock_warn.assert_called_once()


def test_plate_overview_button_wiring(widget_with_viewer, tmp_path,
                                      monkeypatch):
    """Mock QFileDialog and the OS-open call, verify wiring to
    generate_plate_overview() including the overview-section params."""
    widget, _ = widget_with_viewer

    # Pretend an experiment is loaded with a minimal metadata mock.
    widget.reader = Mock()
    mock_meta = Mock()
    mock_meta.channel_ids = [1]
    mock_meta.channels = {1: {"name": "DAPI"}}
    mock_meta.timepoints = [1]
    mock_meta.planes = [0]
    mock_meta.fields = [1]
    widget.metadata = mock_meta
    widget.path_input.setText("/some/experiment")

    # Populate the overview controls as _populate_selectors would have.
    widget.ov_channel_list.addItem("Ch1: DAPI")
    widget._select_all(widget.ov_channel_list)
    widget.ov_timepoint_combo.addItem("Timepoint 1")
    widget.ov_z_list.addItem("Z-plane 0")
    widget._select_all(widget.ov_z_list)

    chosen_dir = str(tmp_path)
    dialog_mock = Mock(return_value=chosen_dir)
    monkeypatch.setattr(
        'pyphenix._widget.QFileDialog.getExistingDirectory', dialog_mock
    )

    with patch('pyphenix._widget.generate_plate_overview') as mock_gen, \
         patch('pyphenix._widget.subprocess.run') as mock_run:
        widget._generate_plate_overview()

        # Dialog was opened defaulting to the experiment directory.
        args, _kwargs = dialog_mock.call_args
        assert "/some/experiment" in args

        # generate_plate_overview called with the experiment path,
        # output dir, and the overview-section parameters.
        mock_gen.assert_called_once()
        call_args, call_kwargs = mock_gen.call_args
        assert call_args == ("/some/experiment", chosen_dir)
        assert call_kwargs["field"] is None
        assert call_kwargs["channels"] == [1]
        assert call_kwargs["timepoint"] == 1
        assert call_kwargs["z_slices"] is None
        assert call_kwargs["well_px"] == 300
        assert call_kwargs["scalebar_um"] is None
        assert call_kwargs["apply_ffc"] is True

        # OS file viewer was invoked with the chosen directory.
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert chosen_dir in cmd


def test_plate_overview_cancelled_dialog_does_nothing(widget_with_viewer,
                                                     monkeypatch):
    """If the user cancels the directory dialog, nothing is generated."""
    widget, _ = widget_with_viewer
    widget.reader = Mock()
    widget.path_input.setText("/some/experiment")

    monkeypatch.setattr(
        'pyphenix._widget.QFileDialog.getExistingDirectory',
        lambda *a, **kw: ""  # user cancelled
    )

    with patch('pyphenix._widget.generate_plate_overview') as mock_gen, \
         patch('pyphenix._widget.subprocess.run') as mock_run:
        widget._generate_plate_overview()
        mock_gen.assert_not_called()
        mock_run.assert_not_called()
