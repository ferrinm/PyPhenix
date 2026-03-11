try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader, OperaPhenixReader

# napari and Qt dependencies are optional — only import the widget and
# launch_viewer helper when they are actually installed.
try:
    from ._widget import PhenixDataLoaderWidget as PhenixDataLoaderWidget
except (ImportError, ModuleNotFoundError):
    PhenixDataLoaderWidget = None

def launch_viewer(experiment_path=None):
    """
    Launch napari viewer with pyphenix widget.

    Parameters
    ----------
    experiment_path : str, optional
        Path to Opera Phenix experiment directory.
        If provided, the experiment will be automatically loaded.

    Returns
    -------
    viewer : napari.Viewer
        The napari viewer instance
    widget : PhenixDataLoaderWidget
        The widget instance

    Raises
    ------
    ImportError
        If napari is not installed.

    Examples
    --------
    >>> from pyphenix import launch_viewer
    >>> viewer, widget = launch_viewer('/path/to/experiment')
    """
    try:
        import napari
    except ImportError:
        raise ImportError(
            "napari is required for launch_viewer(). "
            "Install it with:  pip install 'pyphenix[napari]'  "
            "or  pip install napari"
        )

    if PhenixDataLoaderWidget is None:
        raise ImportError(
            "The pyphenix widget could not be imported. "
            "Make sure napari and qtpy are installed."
        )

    viewer = napari.Viewer()

    widget = PhenixDataLoaderWidget(viewer)
    viewer.window.add_dock_widget(widget, name='Opera Phenix Loader', area='right')

    if experiment_path:
        widget.path_input.setText(experiment_path)
        widget._load_experiment()

    return viewer, widget


__all__ = [
    "launch_viewer",
    "napari_get_reader",
    "OperaPhenixReader",
]

# Only advertise the widget class when it was successfully imported
if PhenixDataLoaderWidget is not None:
    __all__.append("PhenixDataLoaderWidget")
