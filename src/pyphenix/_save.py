"""Canonical Save implementation shared by the reader and the napari widget.

There must be exactly one place that writes a **Save**; the reader's
``_save_output`` and the widget's save button both delegate here so the two
cannot drift.

An OME-TIFF **Save** is self-describing: the standard OME fields carry pixel
size, Z step, time increment, channel identity (and color), and the Phenix
provenance that OME has no standard home for (plate, well, fields, stitched)
goes into the OME-XML image description. It therefore gets no **Sidecar**. A
numpy **Save** still needs one, because a bare ``.npy`` has nowhere to hold any
of this. See ``docs/adr/0002-ome-tiff-saves-are-self-describing.md``.
"""

import json
from pathlib import Path

import numpy as np
import tifffile

from ._colormaps import channel_color_ome

OME_SUFFIX = '.ome.tiff'

# Stripped (longest first) before OME_SUFFIX is appended, so that a caller
# passing "well_A02", "well_A02.tiff" or "well_A02.ome.tiff" all land on the
# same file rather than accumulating extensions.
_STRIPPED_SUFFIXES = ('.ome.tiff', '.ome.tif', '.tiff', '.tif', '.npy')

# Metadata keys with no standard OME home. Everything else in the metadata dict
# (shape, channels, pixel_size, z_step, time_increment) is already expressed in
# the OME-XML, so embedding it again would just be duplication.
PROVENANCE_KEYS = (
    'plate_id',
    'plate_layout',
    'well',
    'well_numeric',
    'fields',
    'timepoints',
    'z_slices',
    'timepoint_offsets',
    'time_unit',
    'stitched',
)


def ome_tiff_path(output_file):
    """Return *output_file* normalised to end in ``.ome.tiff``.

    Parameters
    ----------
    output_file : str or pathlib.Path
        Path the caller asked to write to, with or without an extension.

    Returns
    -------
    pathlib.Path
        The same location with any image/array extension replaced by
        ``.ome.tiff``.
    """
    path = Path(output_file)
    name = path.name
    lowered = name.lower()
    for suffix in _STRIPPED_SUFFIXES:
        if lowered.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return path.with_name(name + OME_SUFFIX)


def _provenance(metadata):
    """Extract the Phenix-specific keys that OME cannot express."""
    return {
        key: metadata[key]
        for key in PROVENANCE_KEYS
        if key in metadata
    }


def _channel_metadata(channels):
    """Build the OME ``Channel`` list from the reader's channel dict.

    ``channels`` is ordered by construction: the reader builds it from the
    selected channel list, so iteration order is C-axis order.

    One dict per channel, not one dict of lists. tifffile accepts either, but
    only the per-channel form lets a single channel omit an attribute the
    others carry -- and Harmony reports an emission of 0 for Brightfield, which
    OME cannot express (see :func:`_as_wavelength`). Under the shared form that
    one channel would drop ``EmissionWavelength`` for the whole plate, taking
    every real emission wavelength with it.
    """
    channel_list = []
    for idx, (ch_id, info) in enumerate(channels.items()):
        name = info.get('name') or f"Channel {ch_id}"
        entry = {'Name': name, 'Color': channel_color_ome(name, idx)}
        # Harmony reports these as XML text, and not every channel has one OME
        # can carry, so each is written only for the channels that do.
        for key, source in (('Excitation', 'excitation'),
                            ('Emission', 'emission')):
            wavelength = _as_wavelength(info.get(source))
            if wavelength is not None:
                entry[f'{key}Wavelength'] = wavelength
                entry[f'{key}WavelengthUnit'] = 'nm'
        channel_list.append(entry)
    return channel_list


def _as_float(value):
    """Coerce a Harmony XML text value to float, or None if it isn't one."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_wavelength(value):
    """Coerce a Harmony wavelength to float, or None if OME cannot carry it.

    OME types ``ExcitationWavelength`` and ``EmissionWavelength`` as
    ``PositiveFloat``, so only a value Harmony reports as greater than zero can
    be written. The rule is about representability, not about which channel it
    came from: Brightfield keeps its excitation, because Harmony reports a real
    positive ``MainExcitationWavelength`` for the transmitted-light source, and
    loses only its emission, which Harmony reports as 0 for a channel with no
    emission filter.

    That 0 is what makes this worth guarding. Writing it through leaves the
    whole OME-XML schema-invalid, at which point Bio-Formats abandons the OME
    reader and opens the file as a plain TIFF stack. Dropping
    the attribute costs only the channel that has no value to report; see
    :func:`_channel_metadata`.
    """
    number = _as_float(value)
    if number is None or number <= 0:
        return None
    return number


def ome_metadata(metadata):
    """Translate a reader metadata dict into tifffile's OME metadata mapping.

    Only keys tifffile recognises are emitted — it silently ignores unknown
    ones, so anything not in this function will not reach the file.

    Parameters
    ----------
    metadata : dict
        As returned by ``OperaPhenixReader.read_data``.

    Returns
    -------
    dict
        Suitable for ``tifffile.imwrite(..., ome=True, metadata=...)``.
    """
    ome = {'axes': 'TCZYX'}

    pixel_size = metadata.get('pixel_size') or {}
    # The reader records pixel size in metres; OME is conventionally µm.
    size_x = _as_float(pixel_size.get('x'))
    size_y = _as_float(pixel_size.get('y'))
    if size_x:
        ome['PhysicalSizeX'] = size_x * 1e6
        ome['PhysicalSizeXUnit'] = 'µm'
    if size_y:
        ome['PhysicalSizeY'] = size_y * 1e6
        ome['PhysicalSizeYUnit'] = 'µm'

    # Absent for single-plane acquisitions — omit rather than writing a
    # placeholder that would claim a Z spacing the acquisition never had.
    z_step = _as_float(metadata.get('z_step'))
    if z_step:
        ome['PhysicalSizeZ'] = z_step * 1e6
        ome['PhysicalSizeZUnit'] = 'µm'

    # Only meaningful with more than one timepoint, where the reader sets it.
    time_increment = _as_float(metadata.get('time_increment'))
    if time_increment:
        ome['TimeIncrement'] = time_increment
        ome['TimeIncrementUnit'] = metadata.get('time_unit') or 's'

    channels = metadata.get('channels') or {}
    if channels:
        ome['Channel'] = _channel_metadata(channels)

    plate_id = metadata.get('plate_id')
    well = metadata.get('well')
    if plate_id and well:
        ome['Name'] = f"{plate_id} {well}"
    elif well:
        ome['Name'] = str(well)

    provenance = _provenance(metadata)
    if provenance:
        ome['Description'] = json.dumps(provenance, default=str)

    return ome


def save_ome_tiff(data, metadata, output_file):
    """Write *data* as a self-describing OME-TIFF. No **Sidecar** is written.

    Data is `zlib` DEFLATE-compressed, which is lossless and requires no dependency.

    Parameters
    ----------
    data : numpy.ndarray or LazyImageArray
        Image data with shape ``(T, C, Z, Y, X)``. A lazily loaded array is
        materialised by the write.
    metadata : dict
        As returned by ``OperaPhenixReader.read_data``.
    output_file : str or pathlib.Path
        Destination; normalised to ``.ome.tiff``.

    Returns
    -------
    pathlib.Path
        The path actually written.
    """
    if getattr(data, 'ndim', None) != 5:
        raise ValueError(
            "OME-TIFF save expects 5-D (T, C, Z, Y, X) data, got shape "
            f"{getattr(data, 'shape', None)}."
        )

    output_path = ome_tiff_path(output_file)
    # photometric is load-bearing: without it tifffile refuses a 5-axis
    # declaration whenever T or Z is 1 (the common single-timepoint,
    # single-plane case) with "axes do not match stored shape". The stored
    # array is squeezed either way, but SizeT/SizeC/SizeZ survive in the XML.
    tifffile.imwrite(
        output_path,
        data,
        photometric='minisblack',
        ome=True,
        compression='zlib',
        metadata=ome_metadata(metadata),
    )
    return output_path


def save_numpy(data, metadata, output_file):
    """Write *data* as ``.npy`` plus a JSON **Sidecar** holding *metadata*.

    Returns
    -------
    tuple of pathlib.Path
        The array path and the sidecar path.
    """
    output_path = Path(output_file)
    np.save(output_path, np.asarray(data))
    sidecar_path = output_path.with_suffix('.json')
    with open(sidecar_path, 'w') as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return output_path, sidecar_path


def save(data, metadata, output_file, output_format):
    """Write a **Save** in *output_format*.

    Parameters
    ----------
    output_format : {'ome-tiff', 'numpy'}

    Returns
    -------
    pathlib.Path
        The primary artifact written (the OME-TIFF, or the ``.npy``).
    """
    if output_format == 'ome-tiff':
        return save_ome_tiff(data, metadata, output_file)
    if output_format == 'numpy':
        array_path, _ = save_numpy(data, metadata, output_file)
        return array_path
    raise ValueError(
        f"Unknown save format {output_format!r}; expected 'ome-tiff' or 'numpy'."
    )
