import napari
from napari.utils import notifications
import numpy as np
from pathlib import Path
from typing import List
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QComboBox, QListWidget, QLabel, QCheckBox,
                            QGroupBox, QAbstractItemView, QLineEdit, QFileDialog,
                            QScrollArea)
from qtpy.QtCore import Qt

from ._reader import OperaPhenixReader

class CollapsibleGroupBox(QGroupBox):
    """A collapsible group box."""
    
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggle)
        
    def _on_toggle(self, checked):
        """Show/hide content when toggled."""
        for child in self.findChildren(QWidget):
            child.setVisible(checked)

class PhenixDataLoaderWidget(QWidget):
    """Interactive widget for loading and visualizing Opera Phenix data in Napari."""
    
    def __init__(self, napari_viewer):
        """
        Initialize the data loader widget.
        
        Parameters
        ----------
        napari_viewer : napari.Viewer
            The napari viewer instance
        """
        super().__init__()
        
        self.viewer = napari_viewer
        self.reader = None
        self.metadata = None
        self.timepoint_overlay = None
        self.current_metadata = None  # Store metadata from last loaded data
        self.current_data = None  # Store currently loaded data
        
        # Build the widget UI
        self._build_ui()
        
    def _build_ui(self):
        """Build the user interface."""
        # Create main layout for the widget
        main_layout = QVBoxLayout()
        
        # Create a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create a container widget for all the controls
        container = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>Opera Phenix Data Loader</h2>")
        layout.addWidget(title)
        
        # Experiment path selector
        path_group = CollapsibleGroupBox("Experiment Selection")
        path_layout = QVBoxLayout()
        
        path_input_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Path to experiment directory...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_experiment)
        path_input_layout.addWidget(self.path_input)
        path_input_layout.addWidget(self.browse_btn)
        path_layout.addLayout(path_input_layout)
        
        self.load_exp_btn = QPushButton("Load Experiment")
        self.load_exp_btn.clicked.connect(self._load_experiment)
        path_layout.addWidget(self.load_exp_btn)
        
        self.exp_info_label = QLabel("")
        path_layout.addWidget(self.exp_info_label)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Well selector
        well_group = CollapsibleGroupBox("Well Selection")
        well_layout = QVBoxLayout()
        
        self.well_combo = QComboBox()
        self.well_combo.currentTextChanged.connect(self._on_well_changed)
        well_layout.addWidget(QLabel("Select Well:"))
        well_layout.addWidget(self.well_combo)
        
        well_group.setLayout(well_layout)
        layout.addWidget(well_group)
        
        # Field selector
        field_group = CollapsibleGroupBox("Field Selection")
        field_layout = QVBoxLayout()
        
        self.stitch_checkbox = QCheckBox("Stitch all fields")
        self.stitch_checkbox.stateChanged.connect(self._on_stitch_changed)
        field_layout.addWidget(self.stitch_checkbox)
        
        field_layout.addWidget(QLabel("Select Field:"))
        self.field_combo = QComboBox()
        field_layout.addWidget(self.field_combo)
        
        field_group.setLayout(field_layout)
        layout.addWidget(field_group)
        
        # Timepoint selector
        time_group = CollapsibleGroupBox("Timepoint Selection")
        time_layout = QVBoxLayout()
        
        time_buttons = QHBoxLayout()
        self.time_select_all_btn = QPushButton("Select All")
        self.time_select_all_btn.clicked.connect(lambda: self._select_all(self.time_list))
        self.time_clear_all_btn = QPushButton("Clear All")
        self.time_clear_all_btn.clicked.connect(lambda: self._clear_all(self.time_list))
        time_buttons.addWidget(self.time_select_all_btn)
        time_buttons.addWidget(self.time_clear_all_btn)
        time_layout.addLayout(time_buttons)
        
        self.time_list = QListWidget()
        self.time_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.time_list.setMaximumHeight(100)  # Limit height
        time_layout.addWidget(self.time_list)
        
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # Channel selector
        channel_group = CollapsibleGroupBox("Channel Selection")
        channel_layout = QVBoxLayout()
        
        channel_buttons = QHBoxLayout()
        self.channel_select_all_btn = QPushButton("Select All")
        self.channel_select_all_btn.clicked.connect(lambda: self._select_all(self.channel_list))
        self.channel_clear_all_btn = QPushButton("Clear All")
        self.channel_clear_all_btn.clicked.connect(lambda: self._clear_all(self.channel_list))
        channel_buttons.addWidget(self.channel_select_all_btn)
        channel_buttons.addWidget(self.channel_clear_all_btn)
        channel_layout.addLayout(channel_buttons)
        
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.channel_list.setMaximumHeight(120)  # Limit height
        channel_layout.addWidget(self.channel_list)
        
        channel_group.setLayout(channel_layout)
        layout.addWidget(channel_group)
        
        # Z-slice selector
        z_group = CollapsibleGroupBox("Z-slice Selection")
        z_layout = QVBoxLayout()
        
        z_buttons = QHBoxLayout()
        self.z_select_all_btn = QPushButton("Select All")
        self.z_select_all_btn.clicked.connect(lambda: self._select_all(self.z_list))
        self.z_clear_all_btn = QPushButton("Clear All")
        self.z_clear_all_btn.clicked.connect(lambda: self._clear_all(self.z_list))
        z_buttons.addWidget(self.z_select_all_btn)
        z_buttons.addWidget(self.z_clear_all_btn)
        z_layout.addLayout(z_buttons)
        
        self.z_list = QListWidget()
        self.z_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.z_list.setMaximumHeight(100)  # Limit height
        z_layout.addWidget(self.z_list)
        
        z_group.setLayout(z_layout)
        layout.addWidget(z_group)
        
        # Display options group
        display_group = CollapsibleGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.timestamp_checkbox = QCheckBox("Show timepoint timestamp")
        self.timestamp_checkbox.setChecked(False)
        self.timestamp_checkbox.stateChanged.connect(self._on_timestamp_toggle)
        display_layout.addWidget(self.timestamp_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Save options
        save_group = CollapsibleGroupBox("Save Options")
        save_layout = QVBoxLayout()
        
        save_path_layout = QHBoxLayout()
        self.save_path_input = QLineEdit()
        self.save_path_input.setPlaceholderText("Output file path...")
        self.save_browse_btn = QPushButton("Browse")
        self.save_browse_btn.clicked.connect(self._browse_save_path)
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(self.save_browse_btn)
        save_layout.addLayout(save_path_layout)
        
        save_layout.addWidget(QLabel("Save format:"))
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["ome-tiff", "numpy"])
        save_layout.addWidget(self.save_format_combo)
        
        self.save_btn = QPushButton("Save Current Data")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        self.save_btn.clicked.connect(self._save_data)
        self.save_btn.setEnabled(False)
        save_layout.addWidget(self.save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Set the layout to the container
        container.setLayout(layout)
        
        # Set the container as the scroll area's widget
        scroll.setWidget(container)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll)
        
        # Visualize button (outside scroll area, always visible)
        self.visualize_btn = QPushButton("Visualize Data")
        self.visualize_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.visualize_btn.clicked.connect(self._visualize_data)
        self.visualize_btn.setEnabled(False)
        main_layout.addWidget(self.visualize_btn)
        
        self.setLayout(main_layout)
        
        # Disable controls until experiment is loaded
        self._set_controls_enabled(False)
    
    def _browse_experiment(self):
        """Open directory dialog for experiment selection."""
        exp_path = QFileDialog.getExistingDirectory(
            self,
            "Select Opera Phenix Experiment Directory"
        )
        
        if exp_path:
            self.path_input.setText(exp_path)
    
    def _browse_save_path(self):
        """Open file dialog for save path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Data",
            "",
            "TIFF files (*.tiff *.tif);;Numpy files (*.npy)"
        )
        
        if file_path:
            self.save_path_input.setText(file_path)
    
    def _load_experiment(self):
        """Load the selected experiment."""
        exp_path = self.path_input.text()
        
        if not exp_path:
            notifications.show_warning("Please select an experiment directory")
            return
        
        try:
            self.reader = OperaPhenixReader(exp_path)
            self.metadata = self.reader.metadata
            
            # Update UI with experiment info
            self.exp_info_label.setText(
                f"<b>Loaded:</b> {Path(exp_path).name}<br>"
                f"Wells: {len(self.metadata.wells)} | "
                f"Channels: {len(self.metadata.channels)}"
            )
            
            # Populate selectors
            self._populate_selectors()
            
            # Enable controls
            self._set_controls_enabled(True)
            self.visualize_btn.setEnabled(True)
            
            notifications.show_info("Experiment loaded successfully!")
            
        except Exception as e:
            notifications.show_error(f"Error loading experiment: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _populate_selectors(self):
        """Populate all selector widgets with experiment data."""
        # Wells
        self.well_combo.clear()
        self.well_combo.addItems([f"r{w[:2]}c{w[2:]}" for w in self.metadata.wells])
        
        # Update fields for first well
        self._update_field_selector()
        
        # Timepoints
        self.time_list.clear()
        self.time_list.addItems([f"Timepoint {t}" for t in self.metadata.timepoints])
        self.time_list.item(0).setSelected(True)
        
        # Channels
        self.channel_list.clear()
        for ch_id in self.metadata.channel_ids:
            ch_name = self.metadata.channels[ch_id]['name']
            self.channel_list.addItem(f"Ch{ch_id}: {ch_name}")
        self._select_all(self.channel_list)
        
        # Z-slices
        self.z_list.clear()
        self.z_list.addItems([f"Z-plane {z}" for z in self.metadata.planes])
        self._select_all(self.z_list)
    
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all control widgets."""
        self.well_combo.setEnabled(enabled)
        self.field_combo.setEnabled(enabled)
        self.stitch_checkbox.setEnabled(enabled)
        self.time_list.setEnabled(enabled)
        self.channel_list.setEnabled(enabled)
        self.z_list.setEnabled(enabled)
        self.time_select_all_btn.setEnabled(enabled)
        self.time_clear_all_btn.setEnabled(enabled)
        self.channel_select_all_btn.setEnabled(enabled)
        self.channel_clear_all_btn.setEnabled(enabled)
        self.z_select_all_btn.setEnabled(enabled)
        self.z_clear_all_btn.setEnabled(enabled)
    
    def _on_well_changed(self):
        """Handle well selection change."""
        if self.metadata is not None:
            self._update_field_selector()
    
    def _on_stitch_changed(self):
        """Handle stitch checkbox change."""
        self.field_combo.setEnabled(not self.stitch_checkbox.isChecked())
    
    def _update_field_selector(self):
        """Update field selector based on selected well."""
        if self.metadata is None:
            return
            
        well_str = self.well_combo.currentText()
        if not well_str:
            return
            
        row = int(well_str[1:3])
        col = int(well_str[4:6])
        
        available_fields = self.reader.well_field_map.get((row, col), self.metadata.fields)
        
        self.field_combo.clear()
        self.field_combo.addItems([f"Field {f}" for f in available_fields])
    
    def _select_all(self, list_widget: QListWidget):
        """Select all items in a list widget."""
        for i in range(list_widget.count()):
            list_widget.item(i).setSelected(True)
    
    def _clear_all(self, list_widget: QListWidget):
        """Clear all selections in a list widget."""
        list_widget.clearSelection()
    
    def _get_selected_indices(self, list_widget: QListWidget) -> List[int]:
        """Get list of selected indices from a list widget."""
        return [i.row() for i in list_widget.selectedIndexes()]
    
    def _on_timestamp_toggle(self, state):
        """Handle timestamp overlay toggle."""
        if state == Qt.Checked:
            self._add_timestamp_overlay()
        else:
            self._remove_timestamp_overlay()
    
    def _add_timestamp_overlay(self):
        """Add timestamp text overlay to viewer."""
        if self.current_metadata is None:
            notifications.show_warning("Please load data first")
            self.timestamp_checkbox.setChecked(False)
            return
        
        if 'timepoint_offsets' not in self.current_metadata or not self.current_metadata['timepoint_offsets']:
            notifications.show_warning("No timepoint information available")
            self.timestamp_checkbox.setChecked(False)
            return
        
        # Remove existing overlay if present
        self._remove_timestamp_overlay()
        
        # Create text overlay
        try:
            self.timepoint_overlay = self.viewer.text_overlay
            self.timepoint_overlay.visible = True
            
            # Connect to viewer dims change event
            self.viewer.dims.events.current_step.connect(self._update_timestamp)
            
            # Initial update
            self._update_timestamp()
            
        except Exception as e:
            notifications.show_error(f"Error adding timestamp overlay: {str(e)}")
            self.timestamp_checkbox.setChecked(False)
    
    def _remove_timestamp_overlay(self):
        """Remove timestamp text overlay from viewer."""
        if self.timepoint_overlay is not None:
            try:
                # Disconnect event
                self.viewer.dims.events.current_step.disconnect(self._update_timestamp)
                
                # Hide overlay
                self.viewer.text_overlay.visible = False
                self.viewer.text_overlay.text = ""
                
                self.timepoint_overlay = None
                
            except Exception as e:
                pass  # Silently handle if already disconnected
    
    def _update_timestamp(self, event=None):
        """Update timestamp overlay with current timepoint."""
        if self.current_metadata is None or self.timepoint_overlay is None:
            return
        
        # Get current timepoint index from viewer
        current_step = self.viewer.dims.current_step
        
        # Determine which axis is time (should be first axis if T > 1)
        if len(current_step) > 0 and len(self.current_metadata['timepoint_offsets']) > 1:
            time_idx = int(current_step[0])
            
            # Get time offset for current timepoint
            if time_idx < len(self.current_metadata['timepoint_offsets']):
                seconds = self.current_metadata['timepoint_offsets'][time_idx]
                
                # Format as HH:MM:SS
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                
                timestamp_str = f"{hours:02d}:{minutes:02d}:{secs:02d} (h:m:s)"
                
                # Update overlay
                self.viewer.text_overlay.text = timestamp_str
                self.viewer.text_overlay.color = 'white'
                self.viewer.text_overlay.font_size = 12
            else:
                self.viewer.text_overlay.text = "--:--:-- (h:m:s)"
        else:
            # Single timepoint
            if len(self.current_metadata['timepoint_offsets']) == 1:
                seconds = self.current_metadata['timepoint_offsets'][0]
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                timestamp_str = f"{hours:02d}:{minutes:02d}:{secs:02d} (h:m:s)"
                self.viewer.text_overlay.text = timestamp_str
            else:
                self.viewer.text_overlay.text = ""
    
    def _save_data(self):
        """Save the currently loaded data to file."""
        if self.current_data is None or self.current_metadata is None:
            notifications.show_warning("No data loaded to save")
            return
        
        save_path = self.save_path_input.text()
        if not save_path:
            notifications.show_warning("Please specify a save path")
            return
        
        save_format = self.save_format_combo.currentText()
        
        try:
            output_path = Path(save_path)
            
            if save_format == 'numpy':
                np.save(output_path, self.current_data)
                metadata_path = output_path.with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(self.current_metadata, f, indent=2, default=str)
                notifications.show_info(f"Saved numpy array to: {output_path}")
                print(f"Saved metadata to: {metadata_path}")
            
            elif save_format == 'ome-tiff':
                try:
                    import tifffile
                    tifffile.imwrite(output_path, self.current_data, photometric='minisblack')
                    metadata_path = output_path.with_suffix('.json')
                    import json
                    with open(metadata_path, 'w') as f:
                        json.dump(self.current_metadata, f, indent=2, default=str)
                    notifications.show_info(f"Saved OME-TIFF to: {output_path}")
                    print(f"Saved metadata to: {metadata_path}")
                except ImportError:
                    notifications.show_error("tifffile not available. Please install it or use numpy format.")
            
        except Exception as e:
            notifications.show_error(f"Error saving data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _visualize_data(self):
        """Load and visualize selected data."""
        if self.reader is None:
            notifications.show_warning("Please load an experiment first")
            return
        
        # Get selections
        well_str = self.well_combo.currentText()
        row = int(well_str[1:3])
        col = int(well_str[4:6])
        
        stitch = self.stitch_checkbox.isChecked()
        
        if not stitch:
            field_str = self.field_combo.currentText()
            field = int(field_str.split()[1])
        else:
            field = None
        
        time_indices = self._get_selected_indices(self.time_list)
        if not time_indices:
            notifications.show_warning("No timepoints selected")
            return
        timepoints = [self.metadata.timepoints[i] for i in time_indices]
        
        channel_indices = self._get_selected_indices(self.channel_list)
        if not channel_indices:
            notifications.show_warning("No channels selected")
            return
        channels = [self.metadata.channel_ids[i] for i in channel_indices]
        
        z_indices = self._get_selected_indices(self.z_list)
        if not z_indices:
            notifications.show_warning("No Z-slices selected")
            return
        z_slices = [self.metadata.planes[i] for i in z_indices]
        
        # Show loading message
        notifications.show_info(f"Loading data for well {well_str}...")
        
        try:
            # Load data (without saving)
            data, metadata = self.reader.read_data(
                row=row,
                column=col,
                field=field,
                stitch_fields=stitch,
                timepoints=timepoints,
                channels=channels,
                z_slices=z_slices
            )
            
            # Store data and metadata for saving later
            self.current_data = data
            self.current_metadata = metadata
            
            # Enable save button now that data is loaded
            self.save_btn.setEnabled(True)
            
            # Clear existing layers and remove overlay
            self._remove_timestamp_overlay()
            self.viewer.layers.clear()
            
            # Visualize data
            self._add_layers_to_viewer(data, metadata)
            
            # Re-enable timestamp if checkbox is checked
            if self.timestamp_checkbox.isChecked():
                self._add_timestamp_overlay()
            
            notifications.show_info("Data loaded successfully!")
            
        except Exception as e:
            notifications.show_error(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _add_layers_to_viewer(self, data, metadata):
        """Add data layers to the viewer."""
        # Extract metadata
        channels_info = metadata['channels']
        pixel_size_x = metadata['pixel_size']['x'] * 1e6
        pixel_size_y = metadata['pixel_size']['y'] * 1e6
        z_step = metadata['z_step'] * 1e6 if metadata['z_step'] is not None else 1.0
        
        # Color mapping
        color_map = {
            'Brightfield': 'gray',
            'DAPI': 'blue',
            'Hoechst': 'blue',
            'Alexa 488': 'green',
            'GFP': 'green',
            'EGFP': 'green',
            'Alexa 555': 'yellow',
            'mCherry': 'magenta',
            'mStrawberry': 'magenta',
            'Alexa 647': 'magenta',
            'Cy5': 'magenta',
            'Cy3': 'yellow',
        }
        default_colors = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']
        
        # Add each channel
        for ch_idx, (ch_id, ch_info) in enumerate(channels_info.items()):
            ch_name = ch_info['name']
            
            # Select color
            color = None
            for key, value in color_map.items():
                if key.lower() in ch_name.lower():
                    color = value
                    break
            if color is None:
                color = default_colors[ch_idx % len(default_colors)]
            
            # Get channel data
            if data.shape[0] > 1:
                channel_data = data[:, ch_idx, :, :, :]
                scale = (1, z_step, pixel_size_y, pixel_size_x)
            else:
                channel_data = data[0, ch_idx, :, :, :]
                scale = (z_step, pixel_size_y, pixel_size_x)
            
            # Calculate contrast limits
            nonzero_data = channel_data[channel_data > 0]
            if len(nonzero_data) > 0:
                contrast_limits = [0, np.percentile(nonzero_data, 99.5)]
            else:
                contrast_limits = [0, 1]
            
            # Add to viewer
            self.viewer.add_image(
                channel_data,
                name=f"Ch{ch_id}: {ch_name}",
                colormap=color,
                blending='additive',
                scale=scale,
                contrast_limits=contrast_limits
            )
        
        # Update viewer title
        well = metadata['well']
        if metadata['stitched']:
            title = f"{metadata['plate_id']} - {well} - Stitched"
        else:
            title = f"{metadata['plate_id']} - {well} - Field {metadata['fields'][0]}"
        self.viewer.title = title
        
        # Enable scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "µm"
        
        # Reset view
        self.viewer.reset_view()
