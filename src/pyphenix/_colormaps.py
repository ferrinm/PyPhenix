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
