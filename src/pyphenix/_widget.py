import napari
from napari.utils import notifications
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                            QComboBox, QListWidget, QLabel, QCheckBox,
                            QFrame, QToolButton, QSizePolicy,
                            QGroupBox, QAbstractItemView, QLineEdit, QFileDialog,
                            QScrollArea, QRadioButton, QButtonGroup)
from qtpy.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve

from ._reader import OperaPhenixReader

def _normalise_well_str(well: str) -> str:
    """
    Normalise a well string to a canonical form ``<letter><2-digit col>``.

    Handles inputs like:
    - ``"D4"``, ``"D04"``, ``"d04"`` (letter + column)
    - ``"2_3"``, ``"02_03"`` (numeric row_col)

    Parameters
    ----------
    well : str
        Well identifier in any supported format.

    Returns
    -------
    str
        Canonical well string, e.g. ``"B03"``.
    """
    well = well.strip()
    if not well:
        return well

    # Check for "row_col" numeric format (e.g. "2_3", "02_03")
    if '_' in well:
        parts = well.split('_', 1)
        try:
            row_num = int(parts[0])
            col_num = int(parts[1])
            letter = chr(ord('A') + row_num - 1)
            return f"{letter}{col_num:02d}"
        except (ValueError, IndexError):
            pass

    # Standard letter+column format
    letter = well[0].upper()
    if letter.isalpha():
        col_str = well[1:].lstrip("0") or "0"
        try:
            return f"{letter}{int(col_str):02d}"
        except ValueError:
            pass

    return well

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
            If no well column can be identified in the CSV.
        """
        df = pd.read_csv(csv_path)

        # Try to auto-detect the well column if the given name is not present
        if well_column not in df.columns:
            candidates = [c for c in df.columns if "well" in c.lower()]
            exact = [c for c in candidates if c.lower() == "well"]
            dest = [c for c in candidates
                    if "destination" in c.lower()]

            if exact:
                well_column = exact[0]
            elif dest:
                well_column = dest[0]
            elif candidates:
                well_column = candidates[0]
            else:
                # Last resort: synthesise a well column from Row + Column
                row_cols = [c for c in df.columns
                            if c.lower() == "row"]
                col_cols = [c for c in df.columns
                            if c.lower() == "column"]

                if row_cols and col_cols:
                    row_col_name = row_cols[0]
                    col_col_name = col_cols[0]

                    # Build "row_col" string, then normalise to letter format
                    df["_well"] = (
                        df[row_col_name].astype(str)
                        + "_"
                        + df[col_col_name].astype(str)
                    )
                    df["_well"] = df["_well"].apply(_normalise_well_str)
                    well_column = "_well"
                else:
                    raise ValueError(
                        f"Could not find a well column in CSV. "
                        f"Available columns: {list(df.columns)}"
                    )

        self._well_column = well_column
        self._metadata_df = df

        self._filter_columns = [
            c for c in df.columns
            if c != well_column and not c.startswith("_")
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
            Well identifier in any supported format (e.g. ``"D04"``,
            ``"4_2"``, ``"d4"``).

        Returns
        -------
        dict or None
            Metadata key/value pairs, or *None* if no metadata is loaded or
            the well is not found.
        """
        if self._metadata_df is None or self._well_column is None:
            return None

        # Try exact match first (fast path)
        rows = self._metadata_df[self._metadata_df[self._well_column] == well]
        if not rows.empty:
            return rows.iloc[0].to_dict()

        # Fall back to normalised comparison
        normalised = _normalise_well_str(well)
        for _, row in self._metadata_df.iterrows():
            csv_well = str(row[self._well_column])
            if _normalise_well_str(csv_well) == normalised:
                return row.to_dict()

        return None

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
            unique_vals = self._metadata_df[col].dropna().unique()
            if len(unique_vals) <= 1:
                continue

            group = CollapsibleSection(col)
            group_layout = QVBoxLayout()

            list_widget = QListWidget()
            list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
            sorted_vals = sorted(unique_vals, key=lambda v: str(v))
            for val in sorted_vals:
                list_widget.addItem(str(val))

            max_visible = min(len(sorted_vals), 6)
            list_widget.setMaximumHeight(max_visible * 22 + 4)

            list_widget.itemSelectionChanged.connect(self._on_filter_changed)

            group_layout.addWidget(list_widget)
            group.setContentLayout(group_layout)

            self._layout.addWidget(group)
            self._filter_lists[col] = list_widget

        self._filtered_label = QLabel("")
        self._layout.addWidget(self._filtered_label)

        self._on_filter_changed()

    def _on_filter_changed(self):
        """Recompute the set of wells matching current filter selections."""
        if self._metadata_df is None:
            return

        mask = pd.Series(True, index=self._metadata_df.index)

        for col, list_widget in self._filter_lists.items():
            selected_items = list_widget.selectedItems()
            if not selected_items:
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

    # Visualization mode constants
    MODE_REPLACE = "replace"
    MODE_NEW_WINDOW = "new_window"
    MODE_SIDE_BY_SIDE = "side_by_side"

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

        # ── Multi-well state ──────────────────────────────────────────
        # Track additional viewer windows opened via "New Window" mode
        self._extra_viewers: List[napari.Viewer] = []
        # Track tile layout for side-by-side mode.
        # Each entry: {'label': str, 'translate_y': float, 'translate_x': float,
        #              'extent_y': float, 'extent_x': float, 'layers': list[str]}
        self._tiles: List[Dict] = []
        # Points layer used for text annotations in side-by-side mode
        self._annotation_layers: List[str] = []

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

        self.well_meta_label = QLabel("")
        self.well_meta_label.setWordWrap(True)
        self.well_meta_label.setStyleSheet(
            "QLabel { color: palette(text); font-size: 11px; }"
        )
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
        self.time_select_all_btn.clicked.connect(
            lambda: self._select_all(self.time_list)
        )
        self.time_clear_all_btn = QPushButton("Clear All")
        self.time_clear_all_btn.clicked.connect(
            lambda: self._clear_all(self.time_list)
        )
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
        self.channel_select_all_btn.clicked.connect(
            lambda: self._select_all(self.channel_list)
        )
        self.channel_clear_all_btn = QPushButton("Clear All")
        self.channel_clear_all_btn.clicked.connect(
            lambda: self._clear_all(self.channel_list)
        )
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
        self.z_select_all_btn.clicked.connect(
            lambda: self._select_all(self.z_list)
        )
        self.z_clear_all_btn = QPushButton("Clear All")
        self.z_clear_all_btn.clicked.connect(
            lambda: self._clear_all(self.z_list)
        )
        z_buttons.addWidget(self.z_select_all_btn)
        z_buttons.addWidget(self.z_clear_all_btn)
        z_layout.addLayout(z_buttons)

        self.z_list = QListWidget()
        self.z_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.z_list.setMaximumHeight(100)
        z_layout.addWidget(self.z_list)

        z_group.setContentLayout(z_layout)
        layout.addWidget(z_group)

        # ── Visualization mode ────────────────────────────────────────
        vis_group = CollapsibleSection("Visualization Mode")
        vis_layout = QVBoxLayout()

        self._vis_mode_group = QButtonGroup(self)

        self._radio_replace = QRadioButton("Replace current view")
        self._radio_replace.setToolTip(
            "Clear the viewer and show only the new data selection."
        )
        self._radio_replace.setChecked(True)

        self._radio_new_window = QRadioButton("Open in new window")
        self._radio_new_window.setToolTip(
            "Keep the current viewer untouched and open a\n"
            "second napari window for the new data selection."
        )

        self._radio_side_by_side = QRadioButton("Add side-by-side")
        self._radio_side_by_side.setToolTip(
            "Add the new data next to previously loaded data\n"
            "in the same viewer (tiled layout)."
        )

        self._vis_mode_group.addButton(self._radio_replace)
        self._vis_mode_group.addButton(self._radio_new_window)
        self._vis_mode_group.addButton(self._radio_side_by_side)

        vis_layout.addWidget(self._radio_replace)
        vis_layout.addWidget(self._radio_new_window)
        vis_layout.addWidget(self._radio_side_by_side)

        # Padding control for side-by-side
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("Tile gap (%):"))
        self._tile_gap_input = QLineEdit("5")
        self._tile_gap_input.setToolTip(
            "Gap between tiled datasets as a percentage\n"
            "of the image width. Only used in side-by-side mode."
        )
        self._tile_gap_input.setMaximumWidth(50)
        padding_layout.addWidget(self._tile_gap_input)
        padding_layout.addStretch()
        vis_layout.addLayout(padding_layout)

        # Reset tiles button
        self._reset_tiles_btn = QPushButton("Reset Tiled Layout")
        self._reset_tiles_btn.setToolTip(
            "Clear all tiled datasets and start fresh."
        )
        self._reset_tiles_btn.clicked.connect(self._reset_tiles)
        vis_layout.addWidget(self._reset_tiles_btn)

        vis_group.setContentLayout(vis_layout)
        layout.addWidget(vis_group)

        # ── Display options ───────────────────────────────────────────
        display_group = CollapsibleSection("Display Options")
        display_layout = QVBoxLayout()

        self.ffc_checkbox = QCheckBox("Apply flat field correction")
        self.ffc_checkbox.setChecked(True)
        self.ffc_checkbox.setToolTip(
            "Apply flat field correction to remove vignetting and "
            "illumination non-uniformity.\n"
            "Requires FFC profiles in the experiment directory.\n"
            "Not compatible with lazy loading mode."
        )
        display_layout.addWidget(self.ffc_checkbox)

        self.lazy_loading_checkbox = QCheckBox(
            "Use lazy loading (load images on-demand)"
        )
        self.lazy_loading_checkbox.setChecked(False)
        self.lazy_loading_checkbox.setToolTip(
            "Load images on-demand instead of loading all into memory at once.\n"
            "Faster initial loading for large datasets, but not compatible "
            "with:\n"
            "  • Flat field correction\n"
            "  • Field stitching\n\n"
            "Images are cached in memory as accessed for faster repeat viewing."
        )
        self.lazy_loading_checkbox.stateChanged.connect(
            self._on_lazy_loading_changed
        )
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
    # Visualization mode helpers
    # ------------------------------------------------------------------

    def _get_vis_mode(self) -> str:
        """Return the currently selected visualization mode string."""
        if self._radio_new_window.isChecked():
            return self.MODE_NEW_WINDOW
        if self._radio_side_by_side.isChecked():
            return self.MODE_SIDE_BY_SIDE
        return self.MODE_REPLACE

    def _get_tile_gap_fraction(self) -> float:
        """Return the tile gap as a fraction (0–1) of image width."""
        try:
            pct = float(self._tile_gap_input.text())
        except ValueError:
            pct = 5.0
        return max(0.0, pct / 100.0)

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
            notifications.show_error(
                f"Error loading metadata CSV: {str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _on_metadata_wells_filtered(self, filtered_wells: List[str]):
        """
        Handle the signal from MetadataFilterWidget when filtered wells change.
        """
        if not self._all_experiment_wells:
            return

        if not filtered_wells:
            display_wells = self._all_experiment_wells
        else:
            csv_set = set()
            for w in filtered_wells:
                csv_set.add(_normalise_well_str(w))

            display_wells = [
                w for w in self._all_experiment_wells
                if _normalise_well_str(w) in csv_set
            ]

        current = self.well_combo.currentText()

        self.well_combo.blockSignals(True)
        self.well_combo.clear()
        self.well_combo.addItems(display_wells)

        if current in display_wells:
            self.well_combo.setCurrentText(current)
        self.well_combo.blockSignals(False)

        self._on_well_changed()

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
                    "Field stitching disabled - not compatible with "
                    "lazy loading mode"
                )
            if self.ffc_checkbox.isChecked():
                self.ffc_checkbox.setChecked(False)
                notifications.show_warning(
                    "Flat field correction disabled - not compatible with "
                    "lazy loading mode"
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
            notifications.show_warning(
                "Please select an experiment directory"
            )
            return

        try:
            # Reset experimental metadata filters
            self.metadata_filter.clear()
            self.csv_info_label.setText("")
            self.well_meta_label.setText("")

            self.reader = OperaPhenixReader(exp_path)
            self.metadata = self.reader.metadata

            ffc_status = ""
            if self.reader.ffc_profiles:
                n_profiles = len(self.reader.ffc_profiles)
                n_with_correction = sum(
                    1 for p in self.reader.ffc_profiles.values()
                    if p.has_correction()
                )
                ffc_status = (
                    f"<br>FFC profiles: "
                    f"{n_with_correction}/{n_profiles} channels"
                )
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

            self._populate_selectors()

            self._set_controls_enabled(True)
            self.visualize_btn.setEnabled(True)

            # Reset tiled layout state when loading new experiment
            self._tiles = []
            self._annotation_layers = []

            notifications.show_info("Experiment loaded successfully!")

        except Exception as e:
            notifications.show_error(
                f"Error loading experiment: {str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _populate_selectors(self):
        """Populate all selector widgets with experiment data."""
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

        self._update_field_selector()

        self.time_list.clear()
        self.time_list.addItems(
            [f"Timepoint {t}" for t in self.metadata.timepoints]
        )
        self.time_list.item(0).setSelected(True)

        self.channel_list.clear()
        for ch_id in self.metadata.channel_ids:
            ch_name = self.metadata.channels[ch_id]['name']
            self.channel_list.addItem(f"Ch{ch_id}: {ch_name}")
        self._select_all(self.channel_list)

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

        lines = []
        for key, val in meta.items():
            if (self.metadata_filter._well_column
                    and key == self.metadata_filter._well_column):
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
            self.viewer.dims.events.current_step.connect(
                self._update_timestamp
            )
            self._update_timestamp()
        except Exception as e:
            notifications.show_error(
                f"Error adding timestamp overlay: {str(e)}"
            )
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

                timestamp_str = (
                    f"{hours:02d}:{minutes:02d}:{secs:02d} (HH:MM:SS)"
                )

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
                    f"Single timepoint\n"
                    f"{hours:02d}:{minutes:02d}:{secs:02d}"
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
                    json.dump(
                        self.current_metadata, f, indent=2, default=str
                    )
                notifications.show_info(
                    f"Saved numpy array to: {output_path}"
                )
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
    # Build a human-readable label for the current selection
    # ------------------------------------------------------------------

    def _build_selection_label(self, metadata: Dict) -> str:
        """
        Build a concise label string for the current data selection.

        Parameters
        ----------
        metadata : dict
            Metadata dict returned by ``reader.read_data``.

        Returns
        -------
        str
            E.g. ``"D04 – Field 1"`` or ``"D04 – Stitched"``.
        """
        well = metadata['well']
        if metadata['stitched']:
            field_str = "Stitched"
        else:
            field_str = f"Field {metadata['fields'][0]}"

        label = f"{well} – {field_str}"

        # Append experimental metadata summary if available
        exp_meta = self.metadata_filter.get_metadata_for_well(well)
        if exp_meta:
            summary_keys = [
                k for k in [
                    'compound', 'cell_line', 'condition',
                    'final_concentration',
                ]
                if k in exp_meta
                   and k != self.metadata_filter._well_column
            ]
            if summary_keys:
                parts = [str(exp_meta[k]) for k in summary_keys]
                label += "\n" + " / ".join(parts)

        return label

    # ------------------------------------------------------------------
    # Tiled (side-by-side) layout helpers
    # ------------------------------------------------------------------

    def _reset_tiles(self):
        """Clear all tiles and reset the viewer for a fresh start."""
        self._tiles = []
        self._annotation_layers = []
        self._remove_timestamp_overlay()
        self.viewer.layers.clear()
        self.current_data = None
        self.current_metadata = None
        self.save_btn.setEnabled(False)
        notifications.show_info("Tiled layout reset.")

    def _compute_next_tile_origin(
        self, new_extent_y: float, new_extent_x: float
    ) -> Tuple[float, float]:
        """
        Compute the (translate_y, translate_x) for the next tile.

        Tiles are laid out left-to-right in a single row.  The gap between
        tiles is controlled by the user's *tile gap %* input.

        Parameters
        ----------
        new_extent_y : float
            Height of the new dataset in scaled (physical) units.
        new_extent_x : float
            Width of the new dataset in scaled (physical) units.

        Returns
        -------
        tuple of float
            ``(translate_y, translate_x)`` for the new tile.
        """
        if not self._tiles:
            return (0.0, 0.0)

        gap_frac = self._get_tile_gap_fraction()

        # Find the right edge of the rightmost existing tile
        max_right = max(
            t['translate_x'] + t['extent_x'] for t in self._tiles
        )

        # Use the average tile width as the reference for the gap
        avg_width = np.mean([t['extent_x'] for t in self._tiles])
        gap = avg_width * gap_frac

        translate_x = max_right + gap
        translate_y = 0.0  # Keep all tiles top-aligned

        return (translate_y, translate_x)

    def _add_annotation_to_viewer(
        self,
        target_viewer: napari.Viewer,
        label_text: str,
        position_yx: Tuple[float, float],
        scale: tuple,
        has_time_dim: bool,
    ):
        """
        Add a text annotation as a Points layer with text properties.

        Parameters
        ----------
        target_viewer : napari.Viewer
            The viewer to add the annotation to.
        label_text : str
            Text to display.
        position_yx : tuple of float
            (Y, X) position in *world* (physical) coordinates for the anchor.
        scale : tuple
            Scale tuple matching the image layers — only used to determine
            dimensionality, not applied to the points layer itself.
        has_time_dim : bool
            Whether the data has a leading time dimension in the viewer.
        """
        y_pos, x_pos = position_yx

        # Build the point coordinate in world space.
        # We set scale=(1,1,...) on the points layer so that the coordinates
        # we supply ARE the world coordinates directly — no double-scaling.
        if has_time_dim:
            # 4-D viewer: (T, Z, Y, X) — place at t=0, z=0
            point_coord = np.array([[0, 0, y_pos, x_pos]])
            pts_scale = (1, 1, 1, 1)
        else:
            # 3-D viewer: (Z, Y, X) — place at z=0
            point_coord = np.array([[0, y_pos, x_pos]])
            pts_scale = (1, 1, 1)

        layer_name = f"label: {label_text.splitlines()[0]}"

        text_props = {
            'text': [label_text],
        }

        # napari >=0.5 renamed edge_color → border_color and removed
        # the old name.  Try the new API first, fall back to old.
        points_kwargs = dict(
            data=point_coord,
            name=layer_name,
            text=text_props,
            size=0,  # invisible point
            face_color='transparent',
            scale=pts_scale,
        )

        try:
            target_viewer.add_points(
                border_color='transparent', **points_kwargs
            )
        except TypeError:
            # Older napari that still uses edge_color
            target_viewer.add_points(
                edge_color='transparent', **points_kwargs
            )

        # Style the text after adding — napari sets defaults on add
        layer = target_viewer.layers[layer_name]
        layer.text.color = 'white'
        layer.text.size = 14
        layer.text.anchor = 'upper_left'

        return layer_name

    # ------------------------------------------------------------------
    # Visualize – main entry point
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

            # Dispatch to the correct visualization mode
            mode = self._get_vis_mode()
            if mode == self.MODE_REPLACE:
                self._visualize_replace(data, metadata)
            elif mode == self.MODE_NEW_WINDOW:
                self._visualize_new_window(data, metadata)
            elif mode == self.MODE_SIDE_BY_SIDE:
                self._visualize_side_by_side(data, metadata)

            if lazy_loading:
                from ._reader import LazyImageArray
                if isinstance(data, LazyImageArray):
                    notifications.show_info(
                        f"Lazy loading enabled. Images will load as you "
                        f"navigate. Cache size: {data._max_cache_size} images"
                    )

            notifications.show_info("Data loaded successfully!")

        except Exception as e:
            notifications.show_error(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Mode 1: Replace
    # ------------------------------------------------------------------

    def _visualize_replace(self, data, metadata):
        """Clear the viewer and display the new data (original behaviour)."""
        self._remove_timestamp_overlay()
        self.viewer.layers.clear()
        self._tiles = []
        self._annotation_layers = []

        self._add_layers_to_viewer(self.viewer, data, metadata)

        if self.timestamp_checkbox.isChecked():
            self._add_timestamp_overlay()

    # ------------------------------------------------------------------
    # Mode 2: New Window
    # ------------------------------------------------------------------

    def _visualize_new_window(self, data, metadata):
        """Open the data in a new napari viewer window."""
        # If the primary viewer is empty, use it instead of creating new
        if len(self.viewer.layers) == 0:
            self._add_layers_to_viewer(self.viewer, data, metadata)
            label = self._build_selection_label(metadata)
            scale = self._get_scale_tuple(metadata, data)
            has_time = data.shape[0] > 1
            self._add_annotation_to_viewer(
                self.viewer, label, (0.0, 0.0), scale, has_time
            )
            if self.timestamp_checkbox.isChecked():
                self._add_timestamp_overlay()
            return

        # Add label annotation to the primary viewer if it doesn't have one yet
        if not self._annotation_layers:
            if self.current_metadata is not None:
                # We need to figure out what the *previous* data was.
                # Use the viewer title which was set by the previous load.
                prev_label = self.viewer.title
                # Add a simple annotation for the existing data
                first_layer = self.viewer.layers[0]
                scale = first_layer.scale
                has_time = len(first_layer.data.shape) >= 4
                name = self._add_annotation_to_viewer(
                    self.viewer, prev_label, (0.0, 0.0),
                    tuple(scale[-3:]), has_time
                )
                self._annotation_layers.append(name)

        # Create new viewer
        new_viewer = napari.Viewer(
            title=self._build_selection_label(metadata).splitlines()[0]
        )
        self._extra_viewers.append(new_viewer)

        self._add_layers_to_viewer(new_viewer, data, metadata)

        # Add annotation to new window
        label = self._build_selection_label(metadata)
        scale = self._get_scale_tuple(metadata, data)
        has_time = data.shape[0] > 1
        self._add_annotation_to_viewer(
            new_viewer, label, (0.0, 0.0), scale, has_time
        )

        new_viewer.scale_bar.visible = True
        new_viewer.scale_bar.unit = "µm"
        new_viewer.reset_view()

    # ------------------------------------------------------------------
    # Mode 3: Side-by-side
    # ------------------------------------------------------------------

    def _visualize_side_by_side(self, data, metadata):
        """Add the new data next to existing data in the same viewer."""
        scale = self._get_scale_tuple(metadata, data)
        has_time = data.shape[0] > 1

        # Calculate physical extents of the new data
        pixel_size_x = metadata['pixel_size']['x'] * 1e6
        pixel_size_y = metadata['pixel_size']['y'] * 1e6
        extent_y = data.shape[-2] * pixel_size_y
        extent_x = data.shape[-1] * pixel_size_x

        # If this is the first tile and there is already data in the viewer,
        # register the existing data as tile 0.
        if not self._tiles and len(self.viewer.layers) > 0:
            self._register_existing_as_tile_zero()

        # Compute where to place the new tile
        translate_y, translate_x = self._compute_next_tile_origin(
            extent_y, extent_x
        )

        # Record the tile
        tile_label = self._build_selection_label(metadata)
        tile_info = {
            'label': tile_label,
            'translate_y': translate_y,
            'translate_x': translate_x,
            'extent_y': extent_y,
            'extent_x': extent_x,
            'layers': [],
        }

        # Add channel layers with translation
        layer_names = self._add_layers_to_viewer(
            self.viewer, data, metadata,
            translate_yx=(translate_y, translate_x),
            name_prefix=f"[{metadata['well']}] ",
        )
        tile_info['layers'] = layer_names

        self._tiles.append(tile_info)

        # Add text annotation for this tile
        anno_name = self._add_annotation_to_viewer(
            self.viewer,
            tile_label,
            (translate_y, translate_x),
            scale,
            has_time,
        )
        self._annotation_layers.append(anno_name)

        if self.timestamp_checkbox.isChecked():
            self._add_timestamp_overlay()

        self.viewer.reset_view()

    def _register_existing_as_tile_zero(self):
        """
        Retroactively register layers that are already in the viewer as
        the first tile so the tiling logic knows about them.
        """
        if not self.viewer.layers:
            return

        # Find the bounding box of existing layers
        min_y = float('inf')
        min_x = float('inf')
        max_y = float('-inf')
        max_x = float('-inf')

        layer_names = []
        for layer in self.viewer.layers:
            layer_names.append(layer.name)
            # Get the extent in world coordinates
            extent = layer.extent
            # extent.world is ((min_dims...), (max_dims...))
            world_min = extent.world[0]
            world_max = extent.world[1]
            # Last two dims are Y, X
            min_y = min(min_y, world_min[-2])
            min_x = min(min_x, world_min[-1])
            max_y = max(max_y, world_max[-2])
            max_x = max(max_x, world_max[-1])

        extent_y = max_y - min_y
        extent_x = max_x - min_x

        # Build label from current viewer title
        prev_label = self.viewer.title or "Previous data"

        tile_info = {
            'label': prev_label,
            'translate_y': min_y,
            'translate_x': min_x,
            'extent_y': extent_y,
            'extent_x': extent_x,
            'layers': layer_names,
        }
        self._tiles.append(tile_info)

        # Add annotation for the first tile
        first_layer = self.viewer.layers[0]
        scale = tuple(first_layer.scale[-3:])
        has_time = len(first_layer.data.shape) >= 4
        anno_name = self._add_annotation_to_viewer(
            self.viewer, prev_label, (min_y, min_x), scale, has_time
        )
        self._annotation_layers.append(anno_name)

    # ------------------------------------------------------------------
    # Core layer-adding logic (shared by all modes)
    # ------------------------------------------------------------------

    def _get_scale_tuple(self, metadata: Dict, data) -> tuple:
        """
        Return the (z, y, x) scale tuple in µm.

        Parameters
        ----------
        metadata : dict
            Metadata dict from reader.
        data : array-like
            The image data (used to check shape).

        Returns
        -------
        tuple
            ``(z_step_µm, pixel_y_µm, pixel_x_µm)``
        """
        pixel_size_x = metadata['pixel_size']['x'] * 1e6
        pixel_size_y = metadata['pixel_size']['y'] * 1e6
        z_step = (
            metadata['z_step'] * 1e6
            if metadata['z_step'] is not None
            else 1.0
        )
        return (z_step, pixel_size_y, pixel_size_x)

    def _add_layers_to_viewer(
        self,
        target_viewer: napari.Viewer,
        data,
        metadata: Dict,
        translate_yx: Tuple[float, float] = (0.0, 0.0),
        name_prefix: str = "",
    ) -> List[str]:
        """
        Add image channel layers to a viewer.

        Parameters
        ----------
        target_viewer : napari.Viewer
            Which viewer to add to.
        data : array-like
            5-D image data ``(T, C, Z, Y, X)``.
        metadata : dict
            Metadata dict from ``reader.read_data``.
        translate_yx : tuple of float
            ``(translate_y, translate_x)`` offset in physical (µm) units.
        name_prefix : str
            Optional prefix prepended to every layer name (useful for
            distinguishing tiles).

        Returns
        -------
        list of str
            Names of the layers that were added.
        """
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
        default_colors = [
            'cyan', 'magenta', 'yellow', 'green', 'red', 'blue',
        ]

        ty, tx = translate_yx
        added_names = []

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
                translate = (0, 0, ty, tx)
            else:
                channel_data = data[0, ch_idx, :, :, :]
                scale = (z_step, pixel_size_y, pixel_size_x)
                translate = (0, ty, tx)

            nonzero_data = channel_data[channel_data > 0]
            if len(nonzero_data) > 0:
                contrast_limits = [0, np.percentile(nonzero_data, 99.5)]
            else:
                contrast_limits = [0, 1]

            layer_name = f"{name_prefix}Ch{ch_id}: {ch_name}"

            target_viewer.add_image(
                channel_data,
                name=layer_name,
                colormap=color,
                blending='additive',
                scale=scale,
                translate=translate,
                contrast_limits=contrast_limits,
            )

            added_names.append(layer_name)

        # Set viewer title
        well = metadata['well']
        if metadata['stitched']:
            title = f"{metadata['plate_id']} - {well} - Stitched"
        else:
            title = (
                f"{metadata['plate_id']} - {well} - "
                f"Field {metadata['fields'][0]}"
            )

        exp_meta = self.metadata_filter.get_metadata_for_well(well)
        if exp_meta:
            summary_keys = [
                k for k in [
                    'compound', 'cell_line', 'condition',
                    'final_concentration',
                ]
                if k in exp_meta
                   and k != self.metadata_filter._well_column
            ]
            if summary_keys:
                summary_parts = [str(exp_meta[k]) for k in summary_keys]
                title += " | " + " / ".join(summary_parts)

        target_viewer.title = title

        target_viewer.scale_bar.visible = True
        target_viewer.scale_bar.unit = "µm"

        target_viewer.reset_view()

        return added_names
