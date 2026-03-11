import napari
from napari.utils import notifications
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QComboBox, QListWidget, QLabel, QCheckBox,
                            QFrame, QToolButton, QSizePolicy,
                            QGroupBox, QAbstractItemView, QLineEdit, QFileDialog,
                            QScrollArea)
from qtpy.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve

from ._reader import OperaPhenixReader


class CollapsibleSection(QWidget):
    """A collapsible section with arrow indicator."""

    def __init__(self, title="", parent=None, animation_duration=200):
        super().__init__(parent)

        self.animation_duration = animation_duration
        self.is_collapsed = False

        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header frame with background
        header_frame = QFrame()
        header_frame.setFrameShape(QFrame.StyledPanel)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: palette(midlight);
                border: 1px solid palette(mid);
                border-radius: 3px;
            }
            QFrame:hover {
                background-color: palette(light);
            }
        """)

        # Header layout
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(5, 3, 5, 3)

        # Toggle button with arrow
        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toggle_button.setArrowType(Qt.DownArrow)  # Start expanded
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self.toggle)

        # Title label
        self.title_label = QLabel(title)
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)

        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        # Make entire header clickable
        header_frame.mousePressEvent = lambda event: self.toggle()
        header_frame.setCursor(Qt.PointingHandCursor)

        # Content widget
        self.content_widget = QWidget()

        # Add to main layout
        main_layout.addWidget(header_frame)
        main_layout.addWidget(self.content_widget)

        # Animation for smooth collapse/expand
        self.animation = QPropertyAnimation(self.content_widget, b"maximumHeight")
        self.animation.setDuration(self.animation_duration)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        # Start expanded
        self.content_widget.setVisible(True)
        self.content_widget.setMaximumHeight(16777215)  # Max int

    def toggle(self):
        """Toggle the collapsed state with animation."""
        self.is_collapsed = not self.is_collapsed

        if self.is_collapsed:
            # Collapse
            self.toggle_button.setArrowType(Qt.RightArrow)
            self.animation.setStartValue(self.content_widget.height())
            self.animation.setEndValue(0)
            self.animation.start()
        else:
            # Expand
            self.toggle_button.setArrowType(Qt.DownArrow)
            self.animation.setStartValue(0)
            # Calculate actual content height
            self.content_widget.setMaximumHeight(16777215)
            content_height = self.content_widget.sizeHint().height()
            self.animation.setEndValue(content_height)
            self.animation.start()

    def setContentLayout(self, layout):
        """
        Set the content layout for this collapsible section.

        Parameters
        ----------
        layout : QLayout
            The layout to set as the content layout
        """
        self.content_widget.setLayout(layout)


class MetadataFilterWidget(QWidget):
    """
    Widget that displays column filters from an experimental metadata CSV.

    Each column in the CSV becomes a multi-select list. Selecting values
    from multiple columns performs AND filtering to narrow down matching wells.

    Signals
    -------
    wells_filtered : Signal(list)
        Emitted when the filtered well list changes. Carries list of well strings.
    """

    wells_filtered = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metadata_df: Optional[pd.DataFrame] = None
        self._well_column: Optional[str] = None
        self._filter_columns: List[str] = []
        self._filter_lists: Dict[str, QListWidget] = {}

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder label shown before CSV is loaded
        self._placeholder = QLabel("<i>No metadata loaded</i>")
        self._layout.addWidget(self._placeholder)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_csv(self, csv_path: str, well_column: str = "well"):
        """
        Load an experimental metadata CSV and build filter widgets.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        well_column : str
            Name of the column containing well identifiers (e.g. ``"well"``
            or ``"destination_well"``).

        Raises
        ------
        ValueError
            If *well_column* is not found in the CSV.
        """
        df = pd.read_csv(csv_path)

        # Try to auto-detect the well column if the given name is not present
        if well_column not in df.columns:
            candidates = [c for c in df.columns if "well" in c.lower()]
            # Prefer columns whose name is exactly "well" (case-insensitive)
            exact = [c for c in candidates if c.lower() == "well"]
            # Also consider "destination_well" which appears in plate-map CSVs
            dest = [c for c in candidates if "destination" in c.lower()]

            if exact:
                well_column = exact[0]
            elif dest:
                well_column = dest[0]
            elif candidates:
                well_column = candidates[0]
            else:
                raise ValueError(
                    f"Could not find a well column in CSV. "
                    f"Available columns: {list(df.columns)}"
                )

        self._well_column = well_column
        self._metadata_df = df

        # Determine filterable columns (everything except the well column)
        self._filter_columns = [
            c for c in df.columns if c != well_column
        ]

        self._build_filter_widgets()

    def clear(self):
        """Remove all filter widgets and reset state."""
        self._metadata_df = None
        self._well_column = None
        self._filter_columns = []
        self._filter_lists = {}
        self._clear_layout()
        self._placeholder = QLabel("<i>No metadata loaded</i>")
        self._layout.addWidget(self._placeholder)

    def get_metadata_for_well(self, well: str) -> Optional[Dict]:
        """
        Return a dict of metadata values for a specific well.

        Parameters
        ----------
        well : str
            Well identifier (e.g. ``"D04"``).

        Returns
        -------
        dict or None
            Metadata key/value pairs, or *None* if no metadata is loaded or
            the well is not found.
        """
        if self._metadata_df is None or self._well_column is None:
            return None
        rows = self._metadata_df[self._metadata_df[self._well_column] == well]
        if rows.empty:
            return None
        # Return first matching row as dict
        return rows.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_layout(self):
        """Remove all child widgets from layout."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_filter_widgets(self):
        """Create a QListWidget for each filterable column."""
        self._clear_layout()
        self._filter_lists = {}

        if self._metadata_df is None:
            return

        info_label = QLabel(
            f"<b>Metadata columns:</b> {len(self._filter_columns)} &nbsp;|&nbsp; "
            f"<b>Wells:</b> {self._metadata_df[self._well_column].nunique()}"
        )
        self._layout.addWidget(info_label)

        # Select / Clear all filters buttons
        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All Filters")
        select_all_btn.clicked.connect(self._select_all_filters)
        clear_all_btn = QPushButton("Clear All Filters")
        clear_all_btn.clicked.connect(self._clear_all_filters)
        btn_row = QHBoxLayout()
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(clear_all_btn)
        self._layout.addLayout(btn_row)

        for col in self._filter_columns:
            # Skip columns where every value is the same (not useful for filtering)
            unique_vals = self._metadata_df[col].dropna().unique()
            if len(unique_vals) <= 1:
                continue

            group = CollapsibleSection(col)
            group_layout = QVBoxLayout()

            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            # Sort values for readability
            sorted_vals = sorted(unique_vals, key=lambda v: str(v))
            for val in sorted_vals:
                list_widget.addItem(str(val))

            # Limit height based on item count
            max_visible = min(len(sorted_vals), 6)
            list_widget.setMaximumHeight(max_visible * 22 + 4)

            # Connect selection change → refilter
            list_widget.itemSelectionChanged.connect(self._on_filter_changed)

            group_layout.addWidget(list_widget)
            group.setContentLayout(group_layout)

            self._layout.addWidget(group)
            self._filter_lists[col] = list_widget

        # Filtered wells label
        self._filtered_label = QLabel("")
        self._layout.addWidget(self._filtered_label)

        # Initial state: no filters active → all wells
        self._on_filter_changed()

    def _on_filter_changed(self):
        """Recompute the set of wells matching current filter selections."""
        if self._metadata_df is None:
            return

        mask = pd.Series(True, index=self._metadata_df.index)

        for col, list_widget in self._filter_lists.items():
            selected_items = list_widget.selectedItems()
            if not selected_items:
                # No selection in this column → no constraint from this column
                continue
            selected_vals = {item.text() for item in selected_items}
            mask &= self._metadata_df[col].astype(str).isin(selected_vals)

        filtered_df = self._metadata_df[mask]
        filtered_wells = sorted(filtered_df[self._well_column].unique().tolist())

        self._filtered_label.setText(
            f"<b>Matching wells:</b> {len(filtered_wells)}"
        )

        self.wells_filtered.emit(filtered_wells)

    def _select_all_filters(self):
        """Select all items in every filter list."""
        for list_widget in self._filter_lists.values():
            list_widget.blockSignals(True)
            for i in range(list_widget.count()):
                list_widget.item(i).setSelected(True)
            list_widget.blockSignals(False)
        self._on_filter_changed()

    def _clear_all_filters(self):
        """Clear selections in every filter list."""
        for list_widget in self._filter_lists.values():
            list_widget.blockSignals(True)
            list_widget.clearSelection()
            list_widget.blockSignals(False)
        self._on_filter_changed()


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
        self.current_metadata = None
        self.current_data = None

        # Mapping from display well strings to (row, col) tuples
        self._well_display_to_rc: Dict[str, tuple] = {}
        # All wells available from the experiment (display strings)
        self._all_experiment_wells: List[str] = []

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

        # ── Experiment path selector ──────────────────────────────────
        path_group = CollapsibleSection("Experiment Selection")
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

        path_group.setContentLayout(path_layout)
        layout.addWidget(path_group)

        # ── Experimental metadata (CSV) ───────────────────────────────
        meta_csv_group = CollapsibleSection("Experimental Metadata (optional)")
        meta_csv_layout = QVBoxLayout()

        csv_path_layout = QHBoxLayout()
        self.csv_path_input = QLineEdit()
        self.csv_path_input.setPlaceholderText("Path to metadata .csv file...")
        self.csv_browse_btn = QPushButton("Browse")
        self.csv_browse_btn.clicked.connect(self._browse_metadata_csv)
        csv_path_layout.addWidget(self.csv_path_input)
        csv_path_layout.addWidget(self.csv_browse_btn)
        meta_csv_layout.addLayout(csv_path_layout)

        # Well column name override (optional)
        well_col_layout = QHBoxLayout()
        well_col_layout.addWidget(QLabel("Well column:"))
        self.well_col_input = QLineEdit()
        self.well_col_input.setPlaceholderText("auto-detect")
        self.well_col_input.setToolTip(
            "Name of the CSV column containing well identifiers.\n"
            "Leave blank to auto-detect (looks for 'well' or 'destination_well')."
        )
        well_col_layout.addWidget(self.well_col_input)
        meta_csv_layout.addLayout(well_col_layout)

        self.load_csv_btn = QPushButton("Load Metadata CSV")
        self.load_csv_btn.clicked.connect(self._load_metadata_csv)
        meta_csv_layout.addWidget(self.load_csv_btn)

        self.csv_info_label = QLabel("")
        meta_csv_layout.addWidget(self.csv_info_label)

        # The dynamic filter widget
        self.metadata_filter = MetadataFilterWidget()
        self.metadata_filter.wells_filtered.connect(self._on_metadata_wells_filtered)
        meta_csv_layout.addWidget(self.metadata_filter)

        meta_csv_group.setContentLayout(meta_csv_layout)
        layout.addWidget(meta_csv_group)

        # ── Well selector ─────────────────────────────────────────────
        well_group = CollapsibleSection("Well Selection")
        well_layout = QVBoxLayout()

        self.well_combo = QComboBox()
        self.well_combo.currentTextChanged.connect(self._on_well_changed)
        well_layout.addWidget(QLabel("Select Well:"))
        well_layout.addWidget(self.well_combo)

        # Label to show metadata for the selected well
        self.well_meta_label = QLabel("")
        self.well_meta_label.setWordWrap(True)
        self.well_meta_label.setStyleSheet("QLabel { color: palette(text); font-size: 11px; }")
        well_layout.addWidget(self.well_meta_label)

        well_group.setContentLayout(well_layout)
        layout.addWidget(well_group)

        # ── Field selector ────────────────────────────────────────────
        field_group = CollapsibleSection("Field Selection")
        field_layout = QVBoxLayout()

        self.stitch_checkbox = QCheckBox("Stitch all fields")
        self.stitch_checkbox.stateChanged.connect(self._on_stitch_changed)
        field_layout.addWidget(self.stitch_checkbox)

        field_layout.addWidget(QLabel("Select Field:"))
        self.field_combo = QComboBox()
        field_layout.addWidget(self.field_combo)

        field_group.setContentLayout(field_layout)
        layout.addWidget(field_group)

        # ── Timepoint selector ────────────────────────────────────────
        time_group = CollapsibleSection("Timepoint Selection")
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
        self.time_list.setMaximumHeight(100)
        time_layout.addWidget(self.time_list)

        time_group.setContentLayout(time_layout)
        layout.addWidget(time_group)

        # ── Channel selector ──────────────────────────────────────────
        channel_group = CollapsibleSection("Channel Selection")
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
        self.channel_list.setMaximumHeight(120)
        channel_layout.addWidget(self.channel_list)

        channel_group.setContentLayout(channel_layout)
        layout.addWidget(channel_group)

        # ── Z-slice selector ──────────────────────────────────────────
        z_group = CollapsibleSection("Z-slice Selection")
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
        self.z_list.setMaximumHeight(100)
        z_layout.addWidget(self.z_list)

        z_group.setContentLayout(z_layout)
        layout.addWidget(z_group)

        # ── Display options ───────────────────────────────────────────
        display_group = CollapsibleSection("Display Options")
        display_layout = QVBoxLayout()

        self.ffc_checkbox = QCheckBox("Apply flat field correction")
        self.ffc_checkbox.setChecked(True)
        self.ffc_checkbox.setToolTip(
            "Apply flat field correction to remove vignetting and illumination non-uniformity.\n"
            "Requires FFC profiles in the experiment directory.\n"
            "Not compatible with lazy loading mode."
        )
        display_layout.addWidget(self.ffc_checkbox)

        self.lazy_loading_checkbox = QCheckBox("Use lazy loading (load images on-demand)")
        self.lazy_loading_checkbox.setChecked(False)
        self.lazy_loading_checkbox.setToolTip(
            "Load images on-demand instead of loading all into memory at once.\n"
            "Faster initial loading for large datasets, but not compatible with:\n"
            "  • Flat field correction\n"
            "  • Field stitching\n\n"
            "Images are cached in memory as accessed for faster repeat viewing."
        )
        self.lazy_loading_checkbox.stateChanged.connect(self._on_lazy_loading_changed)
        display_layout.addWidget(self.lazy_loading_checkbox)

        self.timestamp_checkbox = QCheckBox("Show timepoint timestamp")
        self.timestamp_checkbox.setChecked(False)
        self.timestamp_checkbox.stateChanged.connect(self._on_timestamp_toggle)
        display_layout.addWidget(self.timestamp_checkbox)

        display_group.setContentLayout(display_layout)
        layout.addWidget(display_group)

        # ── Save options ──────────────────────────────────────────────
        save_group = CollapsibleSection("Save Options")
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

        save_group.setContentLayout(save_layout)
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

    # ------------------------------------------------------------------
    # Metadata CSV handling
    # ------------------------------------------------------------------

    def _browse_metadata_csv(self):
        """Open file dialog to select a metadata CSV."""
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Experimental Metadata CSV",
            "",
            "CSV files (*.csv);;All files (*)"
        )
        if csv_path:
            self.csv_path_input.setText(csv_path)

    def _load_metadata_csv(self):
        """Load the selected metadata CSV into the filter widget."""
        csv_path = self.csv_path_input.text()
        if not csv_path:
            notifications.show_warning("Please select a metadata CSV file")
            return

        well_col = self.well_col_input.text().strip() or "well"

        try:
            self.metadata_filter.load_csv(csv_path, well_column=well_col)
            self.csv_info_label.setText(
                f"<b>Loaded:</b> {Path(csv_path).name}"
            )
            notifications.show_info("Metadata CSV loaded successfully!")
        except Exception as e:
            notifications.show_error(f"Error loading metadata CSV: {str(e)}")
            import traceback
            traceback.print_exc()

    def _on_metadata_wells_filtered(self, filtered_wells: List[str]):
        """
        Handle the signal from MetadataFilterWidget when filtered wells change.

        Updates the well combo box to show only the intersection of wells
        that are (a) present in the experiment and (b) match the metadata
        filter criteria.

        Parameters
        ----------
        filtered_wells : list of str
            Well identifiers from the CSV that match active filters.
        """
        if not self._all_experiment_wells:
            return

        if not filtered_wells:
            # No filter constraint → show all experiment wells
            display_wells = self._all_experiment_wells
        else:
            # Normalise the CSV well names so they match the display format.
            # CSV may use "D04" while display uses "D04" – they should match
            # directly, but we also handle leading-zero differences.
            csv_set = set()
            for w in filtered_wells:
                csv_set.add(self._normalise_well_str(w))

            display_wells = [
                w for w in self._all_experiment_wells
                if self._normalise_well_str(w) in csv_set
            ]

        # Preserve currently selected well if still in the filtered set
        current = self.well_combo.currentText()

        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        self.well_combo.addItems(display_wells)

        if current in display_wells:
            self.well_combo.setCurrentText(current)
        self.well_combo.blockSignals(False)

        # Trigger well-change logic for the (possibly new) selection
        self._on_well_changed()

    @staticmethod
    def _normalise_well_str(well: str) -> str:
        """
        Normalise a well string to a canonical form ``<letter><2-digit col>``.

        Handles inputs like ``"D4"``, ``"D04"``, ``"d04"``.
        """
        well = well.strip()
        if not well:
            return well
        letter = well[0].upper()
        col_str = well[1:].lstrip("0") or "0"
        return f"{letter}{int(col_str):02d}"

    # ------------------------------------------------------------------
    # Experiment loading & selector population
    # ------------------------------------------------------------------

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

    def _on_lazy_loading_changed(self, state):
        """Handle lazy loading checkbox change."""
        if state == Qt.Checked:
            if self.stitch_checkbox.isChecked():
                self.stitch_checkbox.setChecked(False)
                notifications.show_warning(
                    "Field stitching disabled - not compatible with lazy loading mode"
                )
            if self.ffc_checkbox.isChecked():
                self.ffc_checkbox.setChecked(False)
                notifications.show_warning(
                    "Flat field correction disabled - not compatible with lazy loading mode"
                )
            self.stitch_checkbox.setEnabled(False)
            self.ffc_checkbox.setEnabled(False)
        else:
            self.stitch_checkbox.setEnabled(True)
            if self.reader and self.reader.ffc_profiles:
                self.ffc_checkbox.setEnabled(True)

    def _load_experiment(self):
        """Load the selected experiment."""
        exp_path = self.path_input.text()

        if not exp_path:
            notifications.show_warning("Please select an experiment directory")
            return

        try:
            # Reset experimental metadata filters (keep CSV path for convenience)
            self.metadata_filter.clear()
            self.csv_info_label.setText("")
            self.well_meta_label.setText("")

            self.reader = OperaPhenixReader(exp_path)
            self.metadata = self.reader.metadata

            # Check if FFC profiles were loaded
            ffc_status = ""
            if self.reader.ffc_profiles:
                n_profiles = len(self.reader.ffc_profiles)
                n_with_correction = sum(
                    1 for p in self.reader.ffc_profiles.values()
                    if p.has_correction()
                )
                ffc_status = f"<br>FFC profiles: {n_with_correction}/{n_profiles} channels"
                self.ffc_checkbox.setEnabled(True)
                self.ffc_checkbox.setToolTip(
                    f"Apply flat field correction to remove vignetting.\n"
                    f"Found corrections for {n_with_correction} channel(s)."
                )
            else:
                ffc_status = "<br><i>No FFC profiles found</i>"
                self.ffc_checkbox.setEnabled(False)
                self.ffc_checkbox.setChecked(False)
                self.ffc_checkbox.setToolTip(
                    "No flat field correction profiles found in experiment"
                )

            self.exp_info_label.setText(
                f"<b>Loaded:</b> {Path(exp_path).name}<br>"
                f"Wells: {len(self.metadata.wells)} | "
                f"Channels: {len(self.metadata.channels)}"
                f"{ffc_status}"
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
        # Wells - convert to letter notation and build lookup maps
        self.well_combo.clear()
        self._well_display_to_rc = {}
        self._all_experiment_wells = []

        for w in self.metadata.wells:
            row_num = int(w[:2])
            col_num = int(w[2:])
            row_letter = self.reader.row_to_letter(row_num)
            display = f"{row_letter}{col_num:02d}"
            self._well_display_to_rc[display] = (row_num, col_num)
            self._all_experiment_wells.append(display)

        self.well_combo.addItems(self._all_experiment_wells)

        # Update fields for first well
        self._update_field_selector()

        # Timepoints
        self.time_list.clear()
        self.time_list.addItems(
            [f"Timepoint {t}" for t in self.metadata.timepoints]
        )
        self.time_list.item(0).setSelected(True)

        # Channels
        self.channel_list.clear()
        for ch_id in self.metadata.channel_ids:
            ch_name = self.metadata.channels[ch_id]['name']
            self.channel_list.addItem(f"Ch{ch_id}: {ch_name}")
        self._select_all(self.channel_list)

        # Z-slices
        self.z_list.clear()
        self.z_list.addItems(
            [f"Z-plane {z}" for z in self.metadata.planes]
        )
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

        if enabled and self.reader and self.reader.ffc_profiles:
            self.ffc_checkbox.setEnabled(True)
        else:
            self.ffc_checkbox.setEnabled(False)

    def _on_well_changed(self):
        """Handle well selection change."""
        if self.metadata is not None:
            self._update_field_selector()
            self._update_well_metadata_label()

    def _update_well_metadata_label(self):
        """Show experimental metadata for the currently selected well."""
        well_str = self.well_combo.currentText()
        if not well_str:
            self.well_meta_label.setText("")
            return

        meta = self.metadata_filter.get_metadata_for_well(well_str)
        if meta is None:
            self.well_meta_label.setText("")
            return

        # Build a compact summary of the metadata
        lines = []
        for key, val in meta.items():
            # Skip the well column itself
            if self.metadata_filter._well_column and key == self.metadata_filter._well_column:
                continue
            lines.append(f"<b>{key}:</b> {val}")

        self.well_meta_label.setText("<br>".join(lines))

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

        rc = self._well_display_to_rc.get(well_str)
        if rc is None:
            return
        row_num, col_num = rc

        available_fields = self.reader.well_field_map.get(
            (row_num, col_num), self.metadata.fields
        )

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

    # ------------------------------------------------------------------
    # Timestamp overlay
    # ------------------------------------------------------------------

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

        if ('timepoint_offsets' not in self.current_metadata
                or not self.current_metadata['timepoint_offsets']):
            notifications.show_warning("No timepoint information available")
            self.timestamp_checkbox.setChecked(False)
            return

        self._remove_timestamp_overlay()

        try:
            self.timepoint_overlay = self.viewer.text_overlay
            self.timepoint_overlay.visible = True
            self.viewer.dims.events.current_step.connect(self._update_timestamp)
            self._update_timestamp()
        except Exception as e:
            notifications.show_error(f"Error adding timestamp overlay: {str(e)}")
            self.timestamp_checkbox.setChecked(False)

    def _remove_timestamp_overlay(self):
        """Remove timestamp text overlay from viewer."""
        if self.timepoint_overlay is not None:
            try:
                self.viewer.dims.events.current_step.disconnect(
                    self._update_timestamp
                )
                self.viewer.text_overlay.visible = False
                self.viewer.text_overlay.text = ""
                self.timepoint_overlay = None
            except Exception:
                pass

    def _update_timestamp(self, event=None):
        """Update timestamp overlay with current timepoint."""
        if self.current_metadata is None or self.timepoint_overlay is None:
            return

        current_step = self.viewer.dims.current_step

        if (len(current_step) > 0
                and len(self.current_metadata['timepoint_offsets']) > 1):
            time_idx = int(current_step[0])

            if time_idx < len(self.current_metadata['timepoint_offsets']):
                seconds = self.current_metadata['timepoint_offsets'][time_idx]
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)

                timestamp_str = f"{hours:02d}:{minutes:02d}:{secs:02d} (HH:MM:SS)"

                self.viewer.text_overlay.text = timestamp_str
                self.viewer.text_overlay.color = 'white'
                self.viewer.text_overlay.font_size = 12
                self.viewer.text_overlay.position = 'top_right'
                self.viewer.text_overlay.visible = True
            else:
                self.viewer.text_overlay.text = "--:--:--"
                self.viewer.text_overlay.position = 'top_right'
                self.viewer.text_overlay.visible = True
        else:
            if len(self.current_metadata['timepoint_offsets']) == 1:
                seconds = self.current_metadata['timepoint_offsets'][0]
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                timestamp_str = (
                    f"Single timepoint\n{hours:02d}:{minutes:02d}:{secs:02d}"
                )
                self.viewer.text_overlay.text = timestamp_str
                self.viewer.text_overlay.position = 'top_right'
                self.viewer.text_overlay.visible = True
            else:
                self.viewer.text_overlay.text = ""
                self.viewer.text_overlay.visible = False

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

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
                    tifffile.imwrite(
                        output_path, self.current_data,
                        photometric='minisblack'
                    )
                    metadata_path = output_path.with_suffix('.json')
                    import json
                    with open(metadata_path, 'w') as f:
                        json.dump(
                            self.current_metadata, f, indent=2, default=str
                        )
                    notifications.show_info(
                        f"Saved OME-TIFF to: {output_path}"
                    )
                    print(f"Saved metadata to: {metadata_path}")
                except ImportError:
                    notifications.show_error(
                        "tifffile not available. "
                        "Please install it or use numpy format."
                    )

        except Exception as e:
            notifications.show_error(f"Error saving data: {str(e)}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------

    def _visualize_data(self):
        """Load and visualize selected data."""
        if self.reader is None:
            notifications.show_warning("Please load an experiment first")
            return

        well_str = self.well_combo.currentText()
        if not well_str:
            notifications.show_warning("No well selected")
            return

        rc = self._well_display_to_rc.get(well_str)
        if rc is None:
            notifications.show_warning(f"Well {well_str} not recognised")
            return
        row, col = rc

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

        apply_ffc = self.ffc_checkbox.isChecked()
        lazy_loading = self.lazy_loading_checkbox.isChecked()

        ffc_msg = " (with FFC)" if apply_ffc else " (without FFC)"
        lazy_msg = " [lazy loading]" if lazy_loading else ""
        notifications.show_info(
            f"Loading data for well {well_str}{ffc_msg}{lazy_msg}..."
        )

        try:
            data, metadata = self.reader.read_data(
                row=row,
                column=col,
                field=field,
                stitch_fields=stitch,
                timepoints=timepoints,
                channels=channels,
                z_slices=z_slices,
                apply_ffc=apply_ffc,
                lazy_loading=lazy_loading
            )

            self.current_data = data
            self.current_metadata = metadata

            if lazy_loading:
                self.save_btn.setToolTip(
                    "Note: Saving will load all images into memory"
                )
            else:
                self.save_btn.setToolTip("")

            self.save_btn.setEnabled(True)

            self._remove_timestamp_overlay()
            self.viewer.layers.clear()

            self._add_layers_to_viewer(data, metadata)

            if self.timestamp_checkbox.isChecked():
                self._add_timestamp_overlay()

            if lazy_loading:
                from ._reader import LazyImageArray
                if isinstance(data, LazyImageArray):
                    notifications.show_info(
                        f"Lazy loading enabled. Images will load as you navigate. "
                        f"Cache size: {data._max_cache_size} images"
                    )

            notifications.show_info("Data loaded successfully!")

        except Exception as e:
            notifications.show_error(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()

    def _add_layers_to_viewer(self, data, metadata):
        """Add data layers to the viewer."""
        channels_info = metadata['channels']
        pixel_size_x = metadata['pixel_size']['x'] * 1e6
        pixel_size_y = metadata['pixel_size']['y'] * 1e6
        z_step = (
            metadata['z_step'] * 1e6
            if metadata['z_step'] is not None
            else 1.0
        )

        color_map = {
            'Brightfield': 'gray',
            'DAPI': 'cyan',
            'Hoechst': 'cyan',
            'Alexa 488': 'green',
            'GFP': 'green',
            'EGFP': 'green',
            'Alexa 555': 'yellow',
            'Alexa 568': 'yellow',
            'mCherry': 'magenta',
            'mStrawberry': 'magenta',
            'Alexa 647': 'magenta',
            'Cy5': 'magenta',
            'Cy3': 'yellow',
        }
        default_colors = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']

        for ch_idx, (ch_id, ch_info) in enumerate(channels_info.items()):
            ch_name = ch_info['name']

            color = None
            for key, value in color_map.items():
                if key.lower() in ch_name.lower():
                    color = value
                    break
            if color is None:
                color = default_colors[ch_idx % len(default_colors)]

            if data.shape[0] > 1:
                channel_data = data[:, ch_idx, :, :, :]
                scale = (1, z_step, pixel_size_y, pixel_size_x)
            else:
                channel_data = data[0, ch_idx, :, :, :]
                scale = (z_step, pixel_size_y, pixel_size_x)

            nonzero_data = channel_data[channel_data > 0]
            if len(nonzero_data) > 0:
                contrast_limits = [0, np.percentile(nonzero_data, 99.5)]
            else:
                contrast_limits = [0, 1]

            self.viewer.add_image(
                channel_data,
                name=f"Ch{ch_id}: {ch_name}",
                colormap=color,
                blending='additive',
                scale=scale,
                contrast_limits=contrast_limits
            )

        well = metadata['well']
        if metadata['stitched']:
            title = f"{metadata['plate_id']} - {well} - Stitched"
        else:
            title = (
                f"{metadata['plate_id']} - {well} - "
                f"Field {metadata['fields'][0]}"
            )

        # Append experimental metadata to title if available
        exp_meta = self.metadata_filter.get_metadata_for_well(well)
        if exp_meta:
            # Pick a few key columns to show (compound, cell_line, condition, etc.)
            summary_keys = [
                k for k in ['compound', 'cell_line', 'condition',
                             'final_concentration']
                if k in exp_meta
                   and k != self.metadata_filter._well_column
            ]
            if summary_keys:
                summary_parts = [str(exp_meta[k]) for k in summary_keys]
                title += " | " + " / ".join(summary_parts)

        self.viewer.title = title

        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "µm"

        self.viewer.reset_view()
