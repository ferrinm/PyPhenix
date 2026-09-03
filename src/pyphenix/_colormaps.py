"""Canonical channel→colormap mapping shared by the napari widget,
the drag-and-drop reader, and the plate overview generator.

There must be exactly one such mapping in the codebase; adding a
fluorophore here propagates everywhere it's rendered.
"""

# Order matters: lookup is first-match on case-insensitive substring,
# so more specific names (e.g. "Alexa 488") should come before broader
# ones that might substring-match them.
CHANNEL_COLORS = {
    'Brightfield': 'gray',
    'DAPI': 'cyan',
    'Hoechst': 'cyan',
    'Alexa 488': 'green',
    'GFP': 'green',
    'EGFP': 'green',
    'Alexa 555': 'yellow',
    'Alexa 568': 'yellow',
    'Cy3': 'yellow',
    'mCherry': 'magenta',
    'mStrawberry': 'magenta',
    'Alexa 647': 'magenta',
    'Cy5': 'magenta',
}

DEFAULT_COLORS = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']

# RGB for every colormap name this module can return. OME-TIFF has no way to
# reference a colormap by name — its Channel/@Color is a literal colour — so a
# **Save** needs the values, not the names.
# 'gray' is (255, 255, 255) here because napari's gray ramps black to white
COLORMAP_RGB = {
    'gray': (255, 255, 255),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'yellow': (255, 255, 0),
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
}


def channel_color(name, idx):
    """Return the napari colormap name for a channel.

    Parameters
    ----------
    name : str
        Channel name as reported by Harmony (e.g. ``"DAPI"``,
        ``"Alexa 488"``). Matched case-insensitively as a substring
        against the keys of :data:`CHANNEL_COLORS`.
    idx : int
        Zero-based channel index. Used as the fallback when *name*
        matches nothing in :data:`CHANNEL_COLORS`, indexing into
        :data:`DEFAULT_COLORS` with wrap-around.

    Returns
    -------
    str
        A napari colormap name (e.g. ``"green"``, ``"magenta"``).
    """
    name_lower = name.lower()
    for key, color in CHANNEL_COLORS.items():
        if key.lower() in name_lower:
            return color
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]


def channel_color_ome(name, idx):
    """Return the OME-TIFF ``Channel/@Color`` for a channel.

    The colour is the one :func:`channel_color` picks, encoded the way the OME
    schema wants it: RGBA packed big-endian into a *signed* 32-bit int, since
    ``Channel/@Color`` is an ``xsd:int``. Alpha is always opaque.

    Parameters
    ----------
    name : str
        Channel name as reported by Harmony, as for :func:`channel_color`.
    idx : int
        Zero-based channel index, as for :func:`channel_color`.

    Returns
    -------
    int
        Signed int32 RGBA, e.g. ``16777215`` for cyan, ``-16711681`` for
        magenta.
    """
    red, green, blue = COLORMAP_RGB[channel_color(name, idx)]
    rgba = (red << 24) | (green << 16) | (blue << 8) | 0xFF
    # Values above 2**31 - 1 wrap to their negative two's-complement form.
    return rgba - (1 << 32) if rgba >= (1 << 31) else rgba
