try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader, OperaPhenixReader
from ._widget import PhenixDataLoaderWidget

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
    
    Examples
    --------
    >>> from pyphenix import launch_viewer
    >>> viewer, widget = launch_viewer('/path/to/experiment')
    """
    # Create viewer
    viewer = napari.Viewer()
    
    # Add widget
    widget = PhenixDataLoaderWidget(viewer)
    viewer.window.add_dock_widget(widget, name='Opera Phenix Loader', area='right')
    
    # Load experiment if path provided
    if experiment_path:
        widget.path_input.setText(experiment_path)
        widget._load_experiment()
    
    return viewer, widget

__all__ = (
    "launch_viewer",
    "napari_get_reader",
    "OperaPhenixReader",
    "PhenixDataLoaderWidget",
)
